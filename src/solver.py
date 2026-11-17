"""求解器：无粘右端项、CFL 时间步长与三阶 Runge–Kutta 时间推进。

无粘部分基于守恒量、Steger–Warming 通量分裂和可选重构格式，支持直角网格
以及由 ``mesh.py`` 提供的曲线坐标度量；粘性项由 ``viscous.py`` 单独组装。

du 的符号约定（OCFD_Schemes.cu:306 核实）：
    du[c] = -Ajac * (F_{c+1/2} - F_{c-1/2}) / h
直角均匀网格 Ajac=1，h 为该轴物理步长（参考空间 == 物理空间）。

实现说明：借 moveaxis 把求解方向统一挪到 axis=-1，stencil 代码只写一份；
每个方向先算出 moved 视图下的内区 du，铺回带鬼胞全场后 moveaxis 还原，
与总 du 相加。分裂是逐点运算，鬼胞只在"填充之后被 UP7 读到"这一路径上
有意义——每轴只需填该轴鬼胞（每轴 stencil 只沿该轴）。
"""

import jax
import jax.numpy as jnp

from .fields import (LAP, NVARS, OUTFLOW, PERIODIC, WALL, fill_halos_x,
                     fill_halos_y, fill_halos_z, fill_wall_y,
                     spatial_array_axis, zero_nonperiodic_ghosts)
from .schemes import (SOLVE_SCHEMES, steger_warming, up7_rhs_along_last_axis,
                    weno7_symbo_rhs_along_last_axis)
from .viscous import dif_bound_for_axis


def sound_speed(cons, gamma):
    """全场声速（逐点，鬼胞区按当前值计算）。"""
    d, rhou, rhov, rhow, rhoE = cons
    vv2 = (rhou ** 2 + rhov ** 2 + rhow ** 2) / d ** 2
    p = (gamma - 1.0) * (rhoE - 0.5 * d * vv2)
    return jnp.sqrt(gamma * p / d)


def _inner_slice_moved(axis, grid):
    """moved 视图（求解方向在末轴）下的内区 slice，含变量轴。"""
    dims_inner = (grid.nz, grid.ny, grid.nx)
    # moved 轴序：axis=0 -> (z,y,x)；axis=1 -> (z,x,y)；axis=2 -> (y,x,z)
    if axis == 0:
        order = (0, 1, 2)
    elif axis == 1:
        order = (0, 2, 1)
    else:
        order = (1, 2, 0)
    slices = tuple(slice(LAP, LAP + dims_inner[i]) for i in order)
    return (slice(None),) + slices


def _rhs_along_axis(cons, axis, grid, gamma, epsl_sw, bcs, dxs,
                    scheme="UP7", weno_tv=None, bound=None, metric=None,
                    zero_flux_ghosts=False, zero_state_ghosts=False):
    """单方向无粘 RHS：分裂 → 鬼胞填充 → 面通量 → 差分 → 铺回全场。

    scheme: "UP7" 或 "WENO7-SYMBO"；weno_tv: None 或 (limiter, tv_max)。
    bound: None 或 (lo,hi)，该方向两侧是否走原码面通量边界带（DIF_BOUND）。
    metric: None 或 (A1,A2,A3)，该方向的余子式场（随 cons 同形状）；
            给出时按原码 `Stager_Warming` 把面面积 ss=|A| 装进分裂通量。
    zero_flux_ghosts: 调试开关，强制清除非周期方向的分裂通量鬼层；
    这不符合 OCFD 的通量分裂流程，平板对齐时应保持 False。
    """
    # 原码声速和分裂内核覆盖含鬼层区域，先准备状态，再逐点计算分裂通量。
    cons = fill_halos_x(cons, bcs[0], grid.nx)
    if bcs[1] == WALL:
        cons = fill_wall_y(cons, grid.ny)
    else:
        cons = fill_halos_y(cons, bcs[1], grid.ny)
    cons = fill_halos_z(cons, bcs[2], grid.nz)
    if zero_state_ghosts:
        # 当前单块平板口径：保留入口状态，清零其余非周期状态鬼层。
        cons = zero_nonperiodic_ghosts(cons, bcs, preserve_inflow=True)
    moved = jnp.moveaxis(cons, 1 + spatial_array_axis(axis), -1)
    nxi = (grid.nx, grid.ny, grid.nz)[axis]

    moved_metric = None
    if metric is not None:
        # metric 是 3 维标量场（轴序 (z,y,x)，无变量轴）：2-axis 是它在数组里的轴
        moved_metric = tuple(jnp.moveaxis(m, spatial_array_axis(axis), -1) for m in metric)
    fp, fm = steger_warming(moved, gamma, epsl_sw, axis, metric=moved_metric)

    # 仅供诊断：原码不会在分裂后统一清零非周期方向的通量鬼层。
    if zero_flux_ghosts and bcs[axis] != PERIODIC:
        fp = fp.at[..., :LAP].set(0.0).at[..., -LAP:].set(0.0)
        fm = fm.at[..., :LAP].set(0.0).at[..., -LAP:].set(0.0)

    du_vars = []
    for var in range(NVARS):
        if scheme == "UP7":
            du_var = up7_rhs_along_last_axis(fp[var], fm[var], dxs[axis], bound=bound)
        elif scheme == "WENO7-SYMBO":
            tv = weno_tv if weno_tv is not None else (None, None)
            du_var = weno7_symbo_rhs_along_last_axis(
                fp[var], fm[var], dxs[axis], tv_limiter=tv[0], tv_max_cap=tv[1],
                bound=bound)
        elif scheme in SOLVE_SCHEMES:
            du_var = SOLVE_SCHEMES[scheme](fp[var], fm[var], dxs[axis], bound=bound)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        du_vars.append(du_var)
    du_moved = jnp.stack(du_vars)                # (5, 非求解轴全长, 求解轴内区)
    # 非求解轴切到内区；求解轴在 up7 输出中已是内区，保持不动
    inner_spatial = _inner_slice_moved(axis, grid)[1:-1]
    du_moved_inner = du_moved[(slice(None),) + inner_spatial + (slice(None),)]

    # 只返回内区（轴序已还原），由调用方 .at[inner].add 累加：
    # 原先每轴"zeros(全场)+set+add"要 3 次全场写，现在只写内区 3 次。
    return jnp.moveaxis(du_moved_inner, -1, 1 + spatial_array_axis(axis))


def inviscid_rhs(cons, grid, gamma=1.4, epsl_sw=0.0, bcs=None, axes=(0,), dx=None,
                scheme="UP7", weno_tv=None, metrics=None, dif_bound=None,
                zero_flux_ghosts=False, zero_state_ghosts=False, dif_axis_map=(0, 1, 2)):
    """无粘 RHS。bcs 与 axes 均按计算方向 (i,j,k) 排列；
    dx: 该方向的物理步长（标量或三元组，与 axes 中的方向对应取用；
    非求解方向填 0 占位即可）。

    metrics: 可选，`mesh.compute_metrics_from_coords` 的输出。
    按原码 `put_du_p_kernel`（OCFD_Schemes.cu:306）的口径：
        du = -Ajac * (F̂_{i+1/2} - F̂_{i-1/2}) / h
    其中 **F̂ 是面积加权通量**：原码 `Stager_Warming_ker` 用该方向的余子式
    （Akx/Aky/Akz 等）当"法向量"，`E1 = A1u+A2v+A3w`、`ss=|A|` 进 E1/E2/E3，
    于是分裂通量自带面面积 ss。JAX 侧把 (A1,A2,A3) 传进 `steger_warming`
    复现这一点；**漏掉 ss 会让 du 小 ss 倍**（槽道上 ss≈30，实测线性场
    JAX/OCFD ≈ 1/30）。给 Ajac=1、A=单位向量的直角度量时与 metrics=None
    逐位相同。

    dif_bound: 原码 DIF_BOUND 六标志（i-/i+/j-/j+/k-/k+）；
    某侧为 1 时该侧边界按 OCFD 的 boundp/boundm 带重构（只读内点）。

    1D 算例用 axes=(0,)；3D 用 (0,1,2)。返回与 cons 同 shape 的 du，
    鬼胞区为 0。
    """
    if bcs is None:
        bcs = (OUTFLOW,) * 3
    if dx is None:
        raise ValueError("inviscid_rhs requires dx (physical step size)")
    dxs = (dx, dx, dx) if isinstance(dx, (int, float)) else dx
    du = jnp.zeros(grid.shape, dtype=cons.dtype)
    sl = grid.inner()
    for axis in axes:
        bound = dif_bound_for_axis(dif_bound, axis, dif_axis_map) if dif_bound else None
        # 该方向的余子式（原码 Akx/Aky/Akz、Aix/Aiy/Aiz、Asx/Asy/Asz）
        metric = None
        if metrics is not None:
            keys = (("Akx", "Aky", "Akz"), ("Aix", "Aiy", "Aiz"),
                    ("Asx", "Asy", "Asz"))[axis]
            metric = tuple(metrics[k] for k in keys)
        du = du.at[sl].add(_rhs_along_axis(cons, axis, grid, gamma, epsl_sw, bcs, dxs,
                                           scheme, weno_tv, bound, metric=metric,
                                           zero_flux_ghosts=zero_flux_ghosts,
                                           zero_state_ghosts=zero_state_ghosts))
    if metrics is not None:
        # 只看内区；鬼胞区 du 恒为 0，乘任何有限 Ajac 仍是 0
        du = du * metrics["Ajac"]
    return du


def compute_dt(cons, grid, gamma, cfl, dx):
    """CFL 时间步长：dt = cfl / Σ_axis((|u_n| + c) / h)。内区取各轴最大。

    维度分裂格式的谱半径按各方向叠加（dt·Σλi ≤ C，SSP-RK3 对流 C≈1），
    因此多维求和；单轴时求和退化为单轴 max。**注意：求和式是移植版自加
    的逻辑，原版用固定 dt（parameters.c:56），无原码可对拍。**

    dx 为标量（各向同性）或三元组 (dx, dy, dz)。某方向 h<=0 表示该方向
    不求解（1D 算例的 y/z）。
    """
    d = cons[0]
    u = cons[1] / d
    v = cons[2] / d
    w = cons[3] / d
    cc = sound_speed(cons, gamma)
    sl = grid.inner()[1:]   # 分量场为 3 维，去掉变量轴
    dxs = (dx, dx, dx) if isinstance(dx, (int, float)) else dx
    lams = []
    for vel, h in zip((u, v, w), dxs):
        if h <= 0.0:
            continue
        lams.append(jnp.max((jnp.abs(vel) + cc)[sl] / h))
    lam_sum = jnp.sum(jnp.stack(lams))
    return cfl / lam_sum


def rk3_step_stages(cons, dt, rhs_fn, bc_apply=None):
    """SSP-RK3 一步，同时返回三个子步的 RHS。

    返回 (q_new, (du1, du2, du3))，其中 du1 = L(Q^n)、du2 = L(Q^*)、
    du3 = L(Q^{**})——与原码主循环每个 RK 子步 dump 的 pdu 一一对应
    （OCFD_time.cu 的 ker1/2/3 系数：1、3/4+1/4、1/3+2/3）。
    对拍工具（compare.py jax-run）用它逐子步存 du；rk3_step 只是丢掉了 du。
    """
    if bc_apply is not None:
        cons = bc_apply(cons)
    # stage1: Q*  = Q^n + dt*L(Q^n)
    du1 = rhs_fn(cons)
    q1 = cons + dt * du1
    if bc_apply is not None:
        q1 = bc_apply(q1)
    # stage2: Q** = 3/4*Q^n + 1/4*(Q* + dt*L(Q*))
    du2 = rhs_fn(q1)
    q2 = 0.75 * cons + 0.25 * (q1 + dt * du2)
    if bc_apply is not None:
        q2 = bc_apply(q2)
    # stage3: Q^{n+1} = 1/3*Q^n + 2/3*(Q** + dt*L(Q**))
    du3 = rhs_fn(q2)
    q3 = cons / 3.0 + (2.0 / 3.0) * (q2 + dt * du3)
    if bc_apply is not None:
        q3 = bc_apply(q3)
    return q3, (du1, du2, du3)


def rk3_step(cons, dt, rhs_fn, bc_apply=None):
    """SSP-RK3 一步（只返回推进后的状态；子步 RHS 见 rk3_step_stages）。"""
    q3, _ = rk3_step_stages(cons, dt, rhs_fn, bc_apply=bc_apply)
    return q3
