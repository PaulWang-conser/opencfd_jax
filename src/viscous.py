"""粘性项：CD6/CD8 导数场 → 应力/能量通量 → 通量散度 → du[1..4]。

支持直角网格和曲线坐标度量；核心本构公式对应 OpenCFD 原码：

- 导数（OCFD_Schemes.cu:1237 CD6 / 1333 CD8）：
    CD6: (45(f1-f-1) - 9(f2-f-2) + (f3-f-3)) / 60h
    CD8: (672(f1-f-1) - 168(f2-f-2) + 32(f3-f-3) - 3(f4-f-4)) / 840h
  阶数已数值验证（Σc*k^3 = Σc*k^5 = Σc*k^7 = 0）。

- 应力与能量通量（OCFD_NS_Jacobian3d.cu:80-236）：
    s11 = (2*ux - 2/3*div)*mu, s22 = (2*vy - 2/3*div)*mu,
    s33 = (2*wz - 2/3*div)*mu
    s12 = (uy+vx)*mu, s13 = (uz+wx)*mu, s23 = (vz+wy)*mu
    E1 = u*s11 + v*s12 + w*s13 + mu*(Cp/Pr)*Tx   （Tx = dT/dx）
    E2 = u*s12 + v*s22 + w*s23 + mu*(Cp/Pr)*Ty
    E3 = u*s13 + v*s23 + w*s33 + mu*(Cp/Pr)*Tz
  x 方向粘性通量 Ev = Ajac*(s11, s12, s13, E1)（vis_flus_ev_ker，
  直角坐标 Ajac=1），对 Ev 沿该方向再求导后正号累加进 du[1..4]
  （YF_Pe_XF，commen_kernel.cu:158，du += Ajac*vis）。

- 无量纲关系（commen_kernel.cu:95-152 与 OCFD_init.cu:380-392）：
    T = (rhoE - 0.5*rho*V^2) / (rho*Cv),  Cv = 1/(gamma*(gamma-1)*Ama^2)
    p = T*rho/(gamma*Ama^2),  cc = sqrt(T)/Ama
    mu = amu_C0 * T^1.5 / (Tsb + T)   （Sutherland）
    amu_C0 = (1 + 110.4/Ref_T)/Re,  Tsb = 110.4/Ref_T
"""

import jax.numpy as jnp

from .boundary import (LAP, PLATE, WALL, _EV_WALL_SIGNS,
                       _fill_axis, _fill_ev_plate_j, _fill_ev_wall_j,
                       fill_halos_i, fill_halos_j, fill_halos_k,
                       fill_wall_j, zero_nonperiodic_ghosts)
from .fields import spatial_array_axis

# 历史测试和旧调用方仍用旧内部名；与 canonical 函数同一对象。
_fill_ev_wall_y = _fill_ev_wall_j
_fill_ev_plate_y = _fill_ev_plate_j

_CD6_A = (45.0, -9.0, 1.0)
_CD6_DEN = 60.0
_CD8_A = (672.0, -168.0, 32.0, -3.0)
_CD8_DEN = 840.0


class PhysParams:
    """无量纲物理参数与派生量（对应 OCFD_init.cu:380-392 的 opencfd_para_init）。"""

    def __init__(self, gamma=1.4, Ama=1.0, Re=1.0e3, Ref_T=1.0, Pr=0.7):
        """保存基本物性，并预计算 Cv、Cp 和 Sutherland 粘性公式所需系数。"""
        self.gamma = gamma
        self.Ama = Ama
        self.Re = Re
        self.Ref_T = Ref_T
        self.Pr = Pr
        self.Cv = 1.0 / (gamma * (gamma - 1.0) * Ama * Ama)
        self.Cp = 1.0 / ((gamma - 1.0) * Ama * Ama)
        self.Tsb = 110.4 / Ref_T
        self.amu_C0 = (1.0 / Re) * (1.0 + 110.4 / Ref_T)
        self.cp_over_pr = self.Cp / Pr   # vis_flux_init_c


def band_one_sided(d, f, axis, h, lo=True, hi=True):
    """把 d 沿 axis 的**边界带**（每侧 LAP 层）覆盖成 OCFD 的单侧差商。

    公式照抄 OCFD_bound_Scheme.cu:79-112（D0bound1 / D0bound2 宏），与
    OCFD_Dx0_bound_kernel_m/p（:189-228）逐项相同：

        i=0 : (f1 - f0) / h                                    （1 阶）
        i=1 : (f2 - f0) / (2h)                                 （2 阶）
        i=2 : (f0 - 8f1 + 8f3 - f4) / (12h)                    （4 阶单侧）
        i=3 : (f6 - f0 - 9(f5 - f1) + 45(f4 - f2)) / (60h)     （6 阶单侧）

    顶部（末 4 层）按 tmp = n+LAP-1 对称取。f 与 d 都含 LAP 层鬼胞，
    内区索引为 LAP..LAP+n-1；只覆盖内区，鬼胞不管。

    这是 OCFD 的核心设计：**物理边界带不用鬼胞，换单侧公式**。原码里
    `JAC_BOUND`（度量）与 `DIF_BOUND`（导数/面通量）控制哪一侧启用。
    """
    d = jnp.moveaxis(d, axis, -1)
    f = jnp.moveaxis(f, axis, -1)
    n = f.shape[-1] - 2 * LAP
    if n < 7:
        raise ValueError(f"边界带宽 {n} < 7，无法用小侧单侧公式")

    st = LAP
    if lo:
        f0, f1, f2, f3, f4, f5, f6 = (f[..., st + k] for k in range(7))
        d = d.at[..., st + 0].set((f1 - f0) / h)
        d = d.at[..., st + 1].set((f2 - f0) / (2.0 * h))
        d = d.at[..., st + 2].set((f0 - 8.0 * f1 + 8.0 * f3 - f4) / (12.0 * h))
        d = d.at[..., st + 3].set(
            (f6 - f0 - 9.0 * (f5 - f1) + 45.0 * (f4 - f2)) / (60.0 * h))
    if hi:
        e = st + n - 1
        g0, g1, g2, g3, g4, g5, g6 = (f[..., e - k] for k in range(7))
        d = d.at[..., e - 0].set((g0 - g1) / h)
        d = d.at[..., e - 1].set((g0 - g2) / (2.0 * h))
        d = d.at[..., e - 2].set((g4 - 8.0 * g3 + 8.0 * g1 - g0) / (12.0 * h))
        d = d.at[..., e - 3].set(
            (g0 - g6 - 9.0 * (g1 - g5) + 45.0 * (g2 - g4)) / (60.0 * h))
    return jnp.moveaxis(d, -1, axis)


def cd_along_axis(g, axis, h, order=6, bound=(0, 0)):
    """沿 shape 轴 axis 的 CD6/CD8 中心差分，输出与输入同形。

    物理内区结果与原码一致；鬼胞位置的值用边界复制 pad 生成
    （原码只输出物理区，鬼胞导数无人读，此处仅为形状一致）。

    bound=(lo, hi)：任一为 1 时，把该侧 LAP 层覆盖成 OCFD 的单侧公式
    （`band_one_sided`）。原码 `OCFD_dy0(..., D0_bound[...])` 的语义。
    """
    if order == 6:
        coeffs, den, m = _CD6_A, _CD6_DEN, 3
    else:
        coeffs, den, m = _CD8_A, _CD8_DEN, 4
    moved = jnp.moveaxis(g, axis, -1)
    n = moved.shape[-1]
    gp = jnp.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(m, m)], mode="edge")
    # pad 后位置 p 对应原 moved 索引 p - m；差分 (g[i+k] - g[i-k])
    acc = 0.0
    for k, c in enumerate(coeffs, start=1):
        plus = gp[..., m + k: m + k + n]        # moved[i+k]
        minus = gp[..., m - k: m - k + n]       # moved[i-k]
        acc = acc + c * (plus - minus)
    d = jnp.moveaxis(acc / (den * h), -1, axis)
    if bound[0] or bound[1]:
        # band_one_sided 的输入/输出都在原始轴序，不能再搬一次
        d = band_one_sided(d, g, axis, h, lo=bool(bound[0]), hi=bool(bound[1]))
    return d


def dif_bound_for_axis(dif_bound, axis, dif_axis_map=(0, 1, 2)):
    """按 JAX 计算方向及其对应的原码计算方向取 DIF_BOUND 两侧标志。"""
    if not dif_bound:
        return (0, 0)
    i = 2 * dif_axis_map[axis]
    j = i + 1
    return (int(dif_bound[i]), int(dif_bound[j]))


# Ev 与状态壁面的实现都在 boundary.py；本模块仅保留旧测试使用的别名导入。


def viscous_rhs(cons, grid, params, bcs, axes=(0,), dx=None, order=6, metrics=None,
                dif_bound=None, zero_state_ghosts=False, dif_axis_map=(0, 1, 2),
                boundary_execution=None):
    """粘性 RHS。cons 为含鬼胞尺寸但鬼胞未填的守恒量；内部先填鬼胞。

    dx: 按计算 i/j/k 排列的三元组（代码名为 dx/dy/dz）。返回与 cons 同 shape 的 du（鬼胞区为 0）。
    metrics: 可选，`mesh.compute_metrics_from_coords` 的输出。给出时
    导数换成协变形式 ∂f/∂x_l = f_ξ·Ak_l + f_η·Ai_l + f_ζ·As_l，散度
    乘 Ajac（原码 OCFD_NS_Jacobian3d.cu:33-52 的口径）。直角度量时与
    metrics=None **逐位相同**。
    dif_bound: 原码计算方向的六标志 (i-,i+,j-,j+,k-,k+)。为 1 的一侧，
    LAP 层导数改走单侧公式（`band_one_sided`），即原码
    `OCFD_dy0(u, uy, ..., D0_bound[...])` 的行为；None 等价全 0。
    """
    if dx is None:
        raise ValueError("viscous_rhs requires dx")
    dxs = (dx, dx, dx) if isinstance(dx, (int, float)) else dx
    # 计算方向统一为 (i,j,k)；标量数组的空间轴为 (k,j,i)。

    # 0. cons 填计算 i/j/k 三轴鬼胞；当前计算 j 方向支持 wall
    f = fill_halos_i(cons, bcs[0], grid.ni)
    if bcs[1] == WALL:
        f = fill_wall_j(f, grid.nj)
    else:
        f = fill_halos_j(f, bcs[1], grid.nj)
    f = fill_halos_k(f, bcs[2], grid.nk)
    if zero_state_ghosts:
        # 原码语义：非周期物理边界的鬼层是 0（没人给它赋值）
        f = zero_nonperiodic_ghosts(f, bcs, preserve_inflow=True)

    d, rhou, rhov, rhow, rhoE = f
    # 鬼层按原码口径被清 0 时（zero_state_ghosts），这里会出现 0/0；只在鬼层取值，
    # 随后 Ev 鬼层也会被清 0，所以不受影响的用保护值兜住即可。
    dsafe = jnp.where(d > 0.0, d, 1.0)
    u = rhou / dsafe
    v = rhov / dsafe
    w = rhow / dsafe
    E_int = rhoE / dsafe - 0.5 * (u * u + v * v + w * w)
    T = E_int / params.Cv
    T = jnp.where(T > 0.0, T, 1.0)
    if zero_state_ghosts:
        # 派生温度的物理鬼层在原码中为零，导数必须读取这个值。
        T = zero_nonperiodic_ghosts(T, bcs, preserve_inflow=True)
    mu = params.amu_C0 * T ** 1.5 / (params.Tsb + T)

    # 1. 导数场（12 个）：cd 沿各轴输出全长。
    # 与 compute_dt 同一约定：dxs[axis] <= 0 表示该方向不求解（如 2D 算例
    # 的 z），该轴导数置 0（场沿该轴均匀时物理导数本为 0）。
    def cd_or_zero(g, axis):
        """计算指定计算方向的中心差分；非求解方向直接返回零导数场。"""
        h = dxs[axis]
        if h <= 0.0:
            return jnp.zeros_like(g)
        return cd_along_axis(g, spatial_array_axis(axis), h, order,
                             bound=dif_bound_for_axis(dif_bound, axis, dif_axis_map))

    if metrics is None:
        ux = cd_or_zero(u, 0)
        uy = cd_or_zero(u, 1)
        uz = cd_or_zero(u, 2)
        vx = cd_or_zero(v, 0)
        vy = cd_or_zero(v, 1)
        vz = cd_or_zero(v, 2)
        wx = cd_or_zero(w, 0)
        wy = cd_or_zero(w, 1)
        wz = cd_or_zero(w, 2)
        Tx = cd_or_zero(T, 0)
        Ty = cd_or_zero(T, 1)
        Tz = cd_or_zero(T, 2)
    else:
        # 协变导数（原码 vis_flux_s_ker，OCFD_NS_Jacobian3d.cu:126-128 / :190）：
        #   ∂f/∂x_l = f_ξ * Ak_l + f_η * Ai_l + f_ζ * As_l
        # f_ξ/f_η/f_ζ 是参考空间（ξ∈[0,1]，步长 dxs）上的导数；
        # 原码键 Ak/Ai/As 分别对应计算 i/j/k（ξ/η/ζ）方向的余子式向量。
        # 直角度量（A*k=1, A*i=A*s=0）时逐位退化为上面的物理导数。
        def cov(f):
            """由三个参考方向导数和网格度量合成物理坐标导数 ``(f_x,f_y,f_z)``。"""
            fk = cd_or_zero(f, 0)
            fi = cd_or_zero(f, 1)
            fs = cd_or_zero(f, 2)
            return tuple(
                fk * metrics[ck] + fi * metrics[ci] + fs * metrics[cs]
                for (ck, ci, cs) in (("Akx", "Aix", "Asx"),
                                     ("Aky", "Aiy", "Asy"),
                                     ("Akz", "Aiz", "Asz")))
        ux, uy, uz = cov(u)
        vx, vy, vz = cov(v)
        wx, wy, wz = cov(w)
        Tx, Ty, Tz = cov(T)

    # 2. 应力（vis_flux_s_ker）
    div = ux + vy + wz
    s11 = (2.0 * ux - 2.0 / 3.0 * div) * mu
    s22 = (2.0 * vy - 2.0 / 3.0 * div) * mu
    s33 = (2.0 * wz - 2.0 / 3.0 * div) * mu
    s12 = (uy + vx) * mu
    s13 = (uz + wx) * mu
    s23 = (vz + wy) * mu

    # 3. 能量通量（vis_flux_e_ker）
    muc = mu * params.cp_over_pr
    E1 = u * s11 + v * s12 + w * s13 + muc * Tx
    E2 = u * s12 + v * s22 + w * s23 + muc * Ty
    E3 = u * s13 + v * s23 + w * s33 + muc * Tz

    # 4. 每方向：Ev 组装 → 填鬼胞 → 再求导 → du[1..4] 正号累加（内区）
    # 直角（metrics=None）：Ev 分量就是应力本身（Ajac=1）。
    # 曲线（metrics）：照原码 vis_flus_ev_ker（OCFD_NS_Jacobian3d.cu:222-236）：
    #   akx = Ajac*Akx（对应方向那一组面积分量），Ev1 = akx*s11+aky*s12+akz*s13
    # 即 Ev = J⁻¹(Ak·S)，与讲义 `V̂1 = J⁻¹(ξ_x V1 + ξ_y V2)` 完全一致。
    if metrics is None:
        ev_by_axis = {
            0: (s11, s12, s13, E1),
            1: (s12, s22, s23, E2),
            2: (s13, s23, s33, E3),
        }
    else:
        aj = metrics["Ajac"]
        ak = {0: ("Akx", "Aky", "Akz"), 1: ("Aix", "Aiy", "Aiz"),
              2: ("Asx", "Asy", "Asz")}

        def evv(axis):
            """把一个计算方向的应力/能量通量投影到曲线坐标的面积加权通量。"""
            ax, ay, az = (aj * metrics[c] for c in ak[axis])
            return (ax * s11 + ay * s12 + az * s13,
                    ax * s12 + ay * s22 + az * s23,
                    ax * s13 + ay * s23 + az * s33,
                    ax * E1 + ay * E2 + az * E3)
        ev_by_axis = {0: evv(0), 1: evv(1), 2: evv(2)}
    nxi_by_axis = {0: grid.ni, 1: grid.nj, 2: grid.nk}
    du = jnp.zeros(grid.shape, dtype=cons.dtype)
    sl_inner = grid.inner()
    for axis in axes:
        Evs = ev_by_axis[axis]
        Ev_stack = jnp.stack(Evs)                  # (4, z, y, x)
        mixed_axis = (boundary_execution is not None and
                      any(f.axis == axis for f in boundary_execution.mixed_outflow))
        if mixed_axis:
            Ev_stack = boundary_execution.fill_mixed_outflow(Ev_stack, axis)
        else:
            Ev_stack = _fill_axis(Ev_stack, axis, bcs[axis], nxi_by_axis[axis],
                                  is_ev=True)
        if zero_state_ghosts:
            # 原码粘性通量 kernel 同样只跑内区，Ev 鬼层没被算过（是 0）；
            # 顶部（z+ 未打开 DIF_BOUND）那个中心差分会读到它 —— 这正是原码
            # 顶部 k=255 出现 ±2.2e6 的来源。
            Ev_stack = zero_nonperiodic_ghosts(Ev_stack, bcs)
        for i, var in enumerate((1, 2, 3, 4)):
            if dxs[axis] <= 0.0:
                continue                           # 该方向不求解
            # 原码 du_viscous_Jacobian3d_y_final（OCFD_NS_Jacobian3d.cu:363-366）
            # 的 Ev 求导同样带 D0_bound[2i]/[2i+1]：边界带换单侧
            dEv = cd_along_axis(Ev_stack[i], spatial_array_axis(axis), dxs[axis], order,
                                bound=dif_bound_for_axis(dif_bound, axis, dif_axis_map))
            add = dEv[sl_inner[1:]]                # 内区（3 维）
            du = du.at[(var,) + sl_inner[1:]].add(add)
    if metrics is not None:
        # 原码 vis_flux 的散度同样只带 Ajac（与无粘同口径）
        du = du * metrics["Ajac"]
    return du
