"""激波选择性滤波（对应 OCFD_filtering.cu 的 Filter_Fopt_shock，366-412 行）。

Ducros 式传感 + 选择性耗散，逐方向独立应用：

    dp0 = 0.25*(-p[i+1] + 2p[i] - p[i-1])            （中心二阶差分）
    dp1 = 0.25*(-p[i+2] + 2p[i+1] - p[i])            （右移）
    dp2 = 0.25*(-p[i] + 2p[i-1] - p[i-2])            （左移）
    ri  = 0.5*((dp0-dp1)^2 + (dp0-dp2)^2)/p[i]^2 + 1e-16
    hh  = max(0, 1 - rth/ri)                          （1 → 强激波区）
    Sc1 = 0.5*(hh[i] + hh[i+1])*s0
    Sc2 = 0.5*(hh[i] + hh[i-1])*s0
    f[i] -= Sc1*(c1*(f[i+1]-f[i]) + c2*(f[i+2]-f[i-1]))
          - Sc2*(c1*(f[i]-f[i-1]) + c2*(f[i+1]-f[i-2]))

c1 = -0.2103830, c2 = 0.0396170（Bogey & Bailly 型 4 点耗散系数）。
区域掩码（Filter_para 的矩形区）通过 mask 参数传入（内区布尔场，
False 处不滤波）。周期/出流鬼胞由调用方填充 p 与 f 的鬼胞。
"""

import jax.numpy as jnp

from .boundary import (LAP, NVARS, OUTFLOW, PERIODIC, fill_halos_x,
                       fill_halos_y, fill_halos_z)

_C1 = -0.2103830
_C2 = 0.0396170


def _shock_filter_axis(f, p, axis, s0, rth, periodic=True):
    """沿 axis 的滤波（moved 视图由外部保证：p/求解方向在末轴）。"""
    nx = f.shape[-1] - 2 * LAP
    lo, hi = 0, nx                      # 输出（内区 cell）窗口，紧凑索引

    # p 的三个二阶差分（0.25 系数）
    def d2(shift):
        """取平移 ``shift`` 后的压力二阶差分窗口，用于构造激波传感器。"""
        a = p[..., lo + shift + 1 + LAP: hi + shift + 1 + LAP]
        b = p[..., lo + shift + LAP: hi + shift + LAP]
        c = p[..., lo + shift - 1 + LAP: hi + shift - 1 + LAP]
        return 0.25 * (-a + 2.0 * b - c)

    dp0 = d2(0)
    dp1 = d2(1)
    dp2 = d2(-1)
    p00 = p[..., lo + LAP: hi + LAP]
    ri = 0.5 * ((dp0 - dp1) ** 2 + (dp0 - dp2) ** 2) / (p00 * p00) + 1.0e-16
    ri = 1.0 - rth / ri
    hh = 0.5 * (ri + jnp.abs(ri))       # = max(0, ri)

    # hh 的相邻平均（Sc1 用 hh[i+1]、Sc2 用 hh[i-1]；hh 全长与 p 对齐）。
    # PERIODIC 环绕；其余边界端点复制（edge pad，避免对端环绕值泄入）。
    if periodic:
        hh_p1 = jnp.concatenate([hh[..., 1:], hh[..., :1]], axis=-1)
        hh_m1 = jnp.concatenate([hh[..., -1:], hh[..., :-1]], axis=-1)
    else:
        hh_p1 = jnp.pad(hh, [(0, 0)] * (hh.ndim - 1) + [(0, 1)], mode='edge')[..., 1:]
        hh_m1 = jnp.pad(hh, [(0, 0)] * (hh.ndim - 1) + [(1, 0)], mode='edge')[..., :-1]
    sc1 = 0.5 * (hh + hh_p1) * s0
    sc2 = 0.5 * (hh + hh_m1) * s0

    def sh(k):
        """取相对当前内区格点偏移 ``k`` 的守恒量窗口。"""
        return f[..., lo + k + LAP: hi + k + LAP]

    diss = (sc1 * (_C1 * (sh(1) - sh(0)) + _C2 * (sh(2) - sh(-1)))
            - sc2 * (_C1 * (sh(0) - sh(-1)) + _C2 * (sh(1) - sh(-2))))
    return f[..., lo + LAP: hi + LAP] - diss


def shock_filter(f, p, grid, s0=0.01, rth=0.3, bcs=None, axes=(0, 1, 2),
                 mask=None):
    """全场选择性滤波。f/p: (NVARS, nz+2L, ny+2L, nx+2L)（含鬼胞，鬼胞已填）。

    mask: 内区布尔场 (nz,ny,nx)，None = 全区；False 处保持原值。
    返回滤波后的 f（鬼胞区保持原值）。三方向经 moveaxis 统一到末轴，
    与 solver._rhs_along_axis 同一模式。
    """
    if bcs is None:
        bcs = (PERIODIC,) * 3
    from .solver import _inner_slice_moved
    out = f
    nxi = {0: grid.nx, 1: grid.ny, 2: grid.nz}
    if p.ndim == 3:
        p = p[None]                                # 标量场升 4 维（var 轴）
    for axis in axes:
        moved_f = jnp.moveaxis(f, 3 - axis, -1)
        moved_p = jnp.moveaxis(p, 3 - axis, -1)
        fm = fill_halos_x(moved_f, bcs[axis], nxi[axis])
        pm = fill_halos_x(moved_p, bcs[axis], nxi[axis])
        filt = _shock_filter_axis(fm, pm, axis, s0, rth,
                                  periodic=(bcs[axis] == PERIODIC))
        # 切掉非末轴鬼胞（末轴 filt 已是内区长度）
        inner_moved = _inner_slice_moved(axis, grid)
        filt_inner = filt[(slice(None),) + inner_moved[1:-1] + (slice(None),)]
        if mask is not None:
            orig = moved_f[(slice(None),) + inner_moved[1:]]
            filt_inner = jnp.where(mask, filt_inner, orig)
        full_moved = moved_f.at[inner_moved].set(filt_inner)
        out = jnp.moveaxis(full_moved, -1, 3 - axis)
    return out
