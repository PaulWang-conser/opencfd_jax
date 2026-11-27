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
import numpy as np

from .boundary import (LAP, NONREFLECTING, NVARS, OUTFLOW, PERIODIC, WALL, fill_faces,
                       fill_halos_i, fill_halos_j, fill_halos_k, fill_wall_j,
                       zero_nonperiodic_ghosts)
from .fields import array_axis
from .schemes import (_BAND_EP, _WENO5_EPS, _WENO7_EPS, SOLVE_SCHEMES,
                    steger_warming, up7_rhs_along_last_axis,
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
    dims_inner = (grid.nk, grid.nj, grid.ni)
    # moved 计算轴序：axis=0 -> (k,j,i)；axis=1 -> (k,i,j)；axis=2 -> (j,i,k)
    if axis == 0:
        order = (0, 1, 2)
    elif axis == 1:
        order = (0, 2, 1)
    else:
        order = (1, 2, 0)
    slices = tuple(slice(LAP, LAP + dims_inner[i]) for i in order)
    return (slice(None),) + slices


def scales_from_sizes(dims) -> tuple:
    """由全局顶点数 (Ni, Nj, Nk) 算各方向的分裂通量尺度 s_axis = ∏_{m≠axis}(N_m - 1)。

    单块按自己的网格尺寸调用，见 :func:`flux_scales_for`；多块路径按各块的尺寸
    调用它，给参考尺度的下界校验用。
    """
    return tuple(int(np.prod([dims[m] - 1 for m in range(3) if m != axis]))
                 for axis in range(3))


def flux_scales_for(grid, dxs, metrics):
    """按度量口径推断各方向的分裂通量尺度 s_axis（不换算时返回 None）。

    CGNS 单块（dxs=(1,1,1) 且带度量）下，JAX 的计算坐标就是物理坐标本身，
    而 OCFD 把坐标归一到 [0,1] 用 hx=1/(N-1)；两端分裂通量差
    s_axis = ∏_{m≠axis}(N_m - 1)。直角算例（无度量）与 channel
    （dxs=1/(N-1)，与 OCFD hx 同单位）通量单位一致，不换算。
    """
    if metrics is None or tuple(float(d) for d in dxs) != (1.0, 1.0, 1.0):
        return None
    return scales_from_sizes((grid.ni, grid.nj, grid.nk))


def _rhs_along_axis(cons, axis, grid, gamma, epsl_sw, bcs, dxs,
                    scheme="UP7", weno_tv=None, bound=None, metric=None,
                    zero_flux_ghosts=False, zero_state_ghosts=False,
                    farfield=None, faces=None, fill_state_halos=True,
                    flux_scale=1.0, nonref=(False, False)):
    """单方向无粘 RHS：分裂 → 鬼胞填充 → 面通量 → 差分 → 铺回全场。

    scheme: "UP7" 或 "WENO7-SYMBO"；weno_tv: None 或 (limiter, tv_max)。
    bound: None 或 (lo,hi)，该方向两侧是否走原码面通量边界带（DIF_BOUND）。
    metric: None 或 (A1,A2,A3)，该方向的余子式场（随 cons 同形状）；
            给出时按原码 `Stager_Warming` 把面面积 ss=|A| 装进分裂通量。
    zero_flux_ghosts: 调试开关，强制清除非周期方向的分裂通量鬼层；
    这不符合 OCFD 的通量分裂流程，平板对齐时应保持 False。
    faces: 可选的六侧描述；给出时按面填常规鬼层，支持同轴两侧不同 kind。
    fill_state_halos: False 表示调用方已经在 prepare 中填好状态，跳过状态填充。
    flux_scale: 该方向 JAX/OCFD 分裂通量的单位比 s（见 flux_scales_for）。
            β 是通量二次式，WENO5/WENO7 主模板、所有边界带 epsilon 和
            OMP6 极值判据阈值都除以 s²，使权重与 OCFD 同口径（Step 4.9
            定点复算：差 1.04 → 2.65e-13）。UP7 无 epsilon、只换算边界带；
            WENO7-SYMBO 未接入换算，s≠1 时明确拒绝。
    nonref: (低侧, 高侧) 是否为无反射面（原码 NON_REFLETION）。由 inviscid_rhs
            按六侧描述算好传入；WENO7-SYMBO 不支持，该轴开无反射时明确拒绝。
    """
    # 原码声速和分裂内核覆盖含鬼层区域，先准备状态，再逐点计算分裂通量。
    if fill_state_halos:
        if faces is not None:
            cons = fill_faces(cons, faces, grid)
        else:
            cons = fill_halos_i(cons, bcs[0], grid.ni, farfield=farfield)
            if bcs[1] == WALL:
                cons = fill_wall_j(cons, grid.nj)
            else:
                cons = fill_halos_j(cons, bcs[1], grid.nj, farfield=farfield)
            cons = fill_halos_k(cons, bcs[2], grid.nk, farfield=farfield)
        if zero_state_ghosts:
            # 当前单块平板口径：保留入口状态，清零其余非周期状态鬼层。
            cons = zero_nonperiodic_ghosts(cons, bcs, preserve_inflow=True)
    moved = jnp.moveaxis(cons, 1 + array_axis(axis), -1)
    nxi = (grid.ni, grid.nj, grid.nk)[axis]

    moved_metric = None
    if metric is not None:
        # metric 是 3 维标量场（计算轴序 (k,j,i)，无变量轴）；2-axis 是数组轴
        moved_metric = tuple(jnp.moveaxis(m, array_axis(axis), -1) for m in metric)
    fp, fm = steger_warming(moved, gamma, epsl_sw, axis, metric=moved_metric)

    # 仅供诊断：原码不会在分裂后统一清零非周期方向的通量鬼层。
    if zero_flux_ghosts and bcs[axis] != PERIODIC:
        fp = fp.at[..., :LAP].set(0.0).at[..., -LAP:].set(0.0)
        fm = fm.at[..., :LAP].set(0.0).at[..., -LAP:].set(0.0)

    du_vars = []
    # epsilon 尺度换算：WENO5/WENO7 主模板与所有边界带按 s² 换算，OMP6 的
    # 极值判据阈值是通量平方单位同样换算。FP32 额外传 stable=True：
    # 权重分子分母同除 min(ε+S)^k（代数等价）——曾实测真实 plate 出口
    # 模板的未归一化权重 ~1e88 超 FP32 上限、平坦模板 ε⁶ 下溢（Astra 复审）。
    eps_kwargs = {}
    if flux_scale != 1.0:
        s2 = float(flux_scale) ** 2
        eps_kwargs = {"band_eps": _BAND_EP / s2}
        if scheme == "WENO7":
            eps_kwargs["eps"] = _WENO7_EPS / s2
        elif scheme == "WENO5":
            eps_kwargs["eps"] = _WENO5_EPS / s2
        elif scheme.startswith("OMP6"):
            eps_kwargs["thr"] = 1.0e-10 / s2
    stable = fp.dtype == jnp.float32
    for var in range(NVARS):
        if scheme == "UP7":
            du_var = up7_rhs_along_last_axis(fp[var], fm[var], dxs[axis], bound=bound,
                                             band_eps=eps_kwargs.get("band_eps"),
                                             stable=stable, nonref=nonref)
        elif scheme == "WENO7-SYMBO":
            if nonref[0] or nonref[1]:
                raise ValueError(
                    "WENO7-SYMBO 尚未支持无反射（nonreflecting）边界："
                    "该轴的低侧或高侧标了无反射，请改用 WENO7 等注册格式")
            if flux_scale != 1.0:
                raise ValueError(
                    "WENO7-SYMBO 尚未接入 flux_scale 换算（主模板 ε、"
                    "tv_max_cap 与边界带 band_eps 均未按 s 换算）；"
                    "该格式与 CGNS 度量口径组合待其自身对拍启动时补齐")
            tv = weno_tv if weno_tv is not None else (None, None)
            du_var = weno7_symbo_rhs_along_last_axis(
                fp[var], fm[var], dxs[axis], tv_limiter=tv[0], tv_max_cap=tv[1],
                bound=bound)
        elif scheme in SOLVE_SCHEMES:
            du_var = SOLVE_SCHEMES[scheme](fp[var], fm[var], dxs[axis], bound=bound,
                                           stable=stable, nonref=nonref,
                                           **eps_kwargs)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        du_vars.append(du_var)
    du_moved = jnp.stack(du_vars)                # (5, 非求解轴全长, 求解轴内区)
    # 非求解轴切到内区；求解轴在 up7 输出中已是内区，保持不动
    inner_spatial = _inner_slice_moved(axis, grid)[1:-1]
    du_moved_inner = du_moved[(slice(None),) + inner_spatial + (slice(None),)]

    # 只返回内区（轴序已还原），由调用方 .at[inner].add 累加：
    # 原先每轴"zeros(全场)+set+add"要 3 次全场写，现在只写内区 3 次。
    return jnp.moveaxis(du_moved_inner, -1, 1 + array_axis(axis))


def _nonref_per_axis(bcs, faces):
    """各轴 (低侧, 高侧) 是否为无反射面。给出六侧描述时按面判断，否则按轴级 bcs。"""
    out = []
    for axis in range(3):
        if faces:
            lo = any(f.axis == axis and f.kind == NONREFLECTING and f.side.endswith("-")
                     for f in faces)
            hi = any(f.axis == axis and f.kind == NONREFLECTING and f.side.endswith("+")
                     for f in faces)
        else:
            lo = hi = bcs[axis] == NONREFLECTING
        out.append((bool(lo), bool(hi)))
    return tuple(out)


def inviscid_rhs(cons, grid, gamma=1.4, epsl_sw=0.0, bcs=None, axes=(0,), dx=None,
                scheme="UP7", weno_tv=None, metrics=None, dif_bound=None,
                zero_flux_ghosts=False, zero_state_ghosts=False,
                dif_axis_map=(0, 1, 2), farfield=None, faces=None,
                fill_state_halos=True, flux_scale=None):
    """无粘 RHS。bcs 与 axes 均按计算方向 (i,j,k) 排列；
    dx: 该方向的计算空间步长（直角算例可与物理步长一致；标量或三元组，
    与 axes 中的方向对应取用；
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

    farfield: 通用 FARFIELD 的 5 分量来流状态；按面填充时只写远场那一侧。
    faces: 可选六侧描述；给出时按面填状态鬼层，避免轴级 bcs 把对侧也当成远场。
    fill_state_halos: False 表示状态已在 ``BoundaryExecution.prepare`` 中准备好。
    flux_scale: 各方向 JAX/OCFD 分裂通量的单位比（标量或三元组，按 axes
    中的方向取用；None 或 1 表示不换算）。见 flux_scales_for 与
    ``_rhs_along_axis`` 的 docstring。

    1D 算例用 axes=(0,)；3D 用 (0,1,2)。返回与 cons 同 shape 的 du，
    鬼胞区为 0。
    """
    if bcs is None:
        bcs = (OUTFLOW,) * 3
    if dx is None:
        raise ValueError("inviscid_rhs requires dx (physical step size)")
    dxs = (dx, dx, dx) if isinstance(dx, (int, float)) else dx
    if flux_scale is None:
        flux_scale = 1.0
    scales = (flux_scale, flux_scale, flux_scale) if isinstance(
        flux_scale, (int, float)) else tuple(flux_scale)
    if fill_state_halos and faces is not None:
        cons = fill_faces(cons, faces, grid)
        if zero_state_ghosts:
            cons = zero_nonperiodic_ghosts(cons, bcs, preserve_inflow=True)
        fill_state_halos = False
    nonref_axes = _nonref_per_axis(bcs, faces)
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
                                           zero_state_ghosts=zero_state_ghosts,
                                           farfield=farfield, faces=faces,
                                           fill_state_halos=fill_state_halos,
                                           flux_scale=scales[axis],
                                           nonref=nonref_axes[axis]))
    if metrics is not None:
        # 只看内区；鬼胞区 du 恒为 0，乘任何有限 Ajac 仍是 0
        du = du * metrics["Ajac"]
    return du


def rk3_q1(cons, dt, du1):
    """RK3 第一阶段的组合：整步起点 ``cons`` 加上 ``dt`` 乘第一阶段的右端项。"""
    return cons + dt * du1


def rk3_q2(cons, dt, q1, du2):
    """RK3 第二阶段的组合；``q1`` 是准备之前的第一阶段态。"""
    return 0.75 * cons + 0.25 * (q1 + dt * du2)


def rk3_q3(cons, dt, q2, du3):
    """RK3 第三阶段的组合；``q2`` 是准备之前的第二阶段态。"""
    return cons / 3.0 + (2.0 / 3.0) * (q2 + dt * du3)


def rk3_step_stages(cons, dt, rhs_fn):
    """SSP-RK3 一步，同时返回三个子步的 RHS。

    返回 (q_new, (du1, du2, du3))，其中 du1 = L(Q^n)、du2 = L(Q*)、
    du3 = L(Q**)——与原码主循环每个 RK 子步 dump 的 pdu 一一对应
    （OCFD_time.cu 的 ker1/2/3 系数：1、3/4+1/4、1/3+2/3）。
    对拍工具（compare.py jax-run）用它逐子步存 du；rk3_step 只是丢掉了 du。

    三个阶段的组合是 :func:`rk3_q1`、:func:`rk3_q2`、:func:`rk3_q3`。多块路径
    分阶段逐块求右端项，不能把 RHS 写成一个回调，直接调用这三个函数。
    边界在每个子步的 ``rhs_fn`` 里由物理准备走一遍，这里不收边界回调。
    """
    # stage1: Q*  = Q^n + dt*L(Q^n)
    du1 = rhs_fn(cons)
    q1 = rk3_q1(cons, dt, du1)
    # stage2: Q** = 3/4*Q^n + 1/4*(Q* + dt*L(Q*))
    du2 = rhs_fn(q1)
    q2 = rk3_q2(cons, dt, q1, du2)
    # stage3: Q^{n+1} = 1/3*Q^n + 2/3*(Q** + dt*L(Q**))
    du3 = rhs_fn(q2)
    q3 = rk3_q3(cons, dt, q2, du3)
    return q3, (du1, du2, du3)


def rk3_step(cons, dt, rhs_fn):
    """SSP-RK3 一步（只返回推进后的状态；子步 RHS 见 rk3_step_stages）。"""
    q3, _ = rk3_step_stages(cons, dt, rhs_fn)
    return q3
