"""统一边界层：通用鬼胞填充、壁面镜像、粘性 Ev 规则与执行配置。

本模块负责**所有通用边界**的数值实现：

- 周期/出流沿计算 i/j/k 的鬼胞填充；
- 无反射出流（原码 NON_REFLETION）：鬼层与出流相同，区别在无粘 RHS 的
  边界内点只留一半分裂通量差（实现见 ``schemes._nonref_or_sum``）；
- 无滑移壁面的守恒量镜像（``fill_wall_j``）；
- 粘性通量 Ev 的壁面/平板镜像（``fill_ev_wall_j`` / ``fill_ev_plate_j``）；
- 非周期鬼层清零（``zero_nonperiodic_ghosts``）。

本模块只使用计算方向 i/j/k 的数组索引和 ``ni/nj/nk``，不设置、也不读取
物理坐标 ``(x, y, z)``。物理几何（平板前缘/尾缘、通道高度、壁面位置）在
``cases.py`` 生成网格和初值时确定；曲线坐标场来自
``mesh.read_ocfd_mesh`` 的 ``Axx/Ayy/Azz``，并由
``mesh.compute_metrics_from_coords`` 换算成度量键，边界模块不直接接触这些
坐标值。原码的 ``Axx``/``Ayy``/``Azz`` 分别表示物理 ``x``/``y``/``z``
坐标场，形状为计算轴序 ``(nk, nj, ni)``；``Axx`` 里的 ``xx`` 是原码字段
名，不是度量键 ``Akx/Aix/Asx``。

槽道三点压力外推、平板两点压力外推及入口/顶部 Dirichlet 也在此实现；
算例模块保留兼容入口，并通过 :class:`BoundaryExecution` 接入主循环。
这样通用填充只有一份实现，特殊边界也只有一个明确的 case 侧入口。

边界带（DIF_BOUND 面通量、粘性单侧导数）的设计与核对笔记已移到
``docs/boundary_scheme_notes.md``；本模块不承接那部分历史材料。

扩展位置：

- 新增**通用**边界：在 ``_GENERIC_AXIS_FILLERS`` 注册计算 i/j/k 三个填充函数，
  并把类型加入 ``GENERIC_KINDS``；六侧 ``FaceBoundary`` 会按面得到对应 kind。
  同一轴两侧可以不同 kind；轴级 ``bcs`` 只保留代表值，不再决定整轴填充。
- 新增**case 特殊**边界：在 case 侧显式提供带 ``name``、``faces``、``apply``
  的 ``special_boundary`` 对象，并把 ``apply`` 同时设为 ``bc_apply``；
  ``mesh_data.from_case`` 和 ``validate_special_sides`` 在 JIT 前校验六侧、
  回调与轴级 bcs 一致。``plate`` / ``channel`` 是同一入口的两个内置实现。
"""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp

from .axes import SIDES, grid_extent

LAP = 4
NVARS = 5

# 边界类型。INFLOW/PLATE 只用于非对称 case 回调边界；FARFIELD 是通用单面
# 边界：只写被标记的那个面，不把同轴对侧也当成远场。
PERIODIC = "periodic"
OUTFLOW = "outflow"
INFLOW = "inflow"
PLATE = "plate"
WALL = "wall"
FARFIELD = "farfield"
# 无反射出流（原码 NON_REFLETION 口径）：鬼层、黏性通量鬼层、边界带都与
# 出流相同；区别只在无粘 RHS——边界内点沿该轴只保留一半分裂通量差
# （低侧丢正分裂、高侧丢负分裂，见 schemes._nonref_or_sum）。
NONREFLECTING = "nonreflecting"

GENERIC_KINDS = (PERIODIC, OUTFLOW, NONREFLECTING, WALL, FARFIELD, INFLOW)
SIX_SIDES = SIDES

# 守恒量 var 轴符号：(rho, rho*u, rho*v, rho*w, rho*E)
_WALL_SIGN = (1.0, -1.0, -1.0, -1.0, 1.0)
WALL_SIGN = _WALL_SIGN

# 粘性通量 Ev 的镜像符号。x/y 是不同物理分量，不能共用一套。
EV_WALL_SIGNS = {
    0: (+1.0, -1.0, +1.0, -1.0),   # x 方向 Ev=(s11,s12,s13,E1)
    1: (-1.0, +1.0, -1.0, -1.0),   # y 方向 Ev=(s12,s22,s23,E2)：原码符号
}
_EV_WALL_SIGNS = EV_WALL_SIGNS

# 周期面复制对侧；入口由边界赋值；出口保留已有状态鬼层。
# 原版 INIT_STAT=1 时出口初值为零，INIT_STAT=0 时可为非零。
# 壁面和远场鬼层清零，边界点另行赋值。
HALO_PERIODIC = "periodic"
HALO_COPY = "copy"
HALO_KEEP = "keep"
HALO_ZERO = "zero"

# 壁面执行来源里，会把“壁面点”写掉的那些：这类壁面的鬼层没有物理意义，
# 差分必须走边界带（否则中心模板读到 0）。
WALL_BAND_EXECUTORS = ("callback", "callback+halo")


def halo_mode(kind):
    """状态鬼层：周期复制，入口与出口保持，壁面与远场清零。"""
    if kind == PERIODIC:
        return HALO_PERIODIC
    if kind in (OUTFLOW, NONREFLECTING):
        return HALO_KEEP
    if kind == INFLOW:
        return HALO_KEEP
    # 壁面与远场：鬼层清零，边界点单独赋值，该侧开单侧差分带。
    return HALO_ZERO


def face_halo_mode(face):
    """状态鬼层模式：``face.params["halo"]`` 显式给出时优先，否则按 kind 派生。

    目前只用于把出口从"保持原值"（原码口径，初始为零）切成零梯度延拓
    （``copy``）。黏性通量 Ev 的鬼层走 :func:`_fill_ev_face`，出口恒清零，
    不受这个开关影响。
    """
    mode = face.params.get("halo") or halo_mode(face.kind)
    if mode not in (HALO_PERIODIC, HALO_COPY, HALO_KEEP, HALO_ZERO):
        raise ValueError(
            f"{face.side} 的 halo={mode!r}；可选 periodic/copy/keep/zero")
    return mode


def dif_band(kind, executor="halo"):
    """该面是否开边界单侧带（原码 DIF_BOUND 的逐侧开关）。

    当前缺省为壁面和远场开带，周期、入口和出口不开带。
    """
    del executor
    return kind in (WALL, FARFIELD)



def dif_bound_from_faces(faces, dif_axis_map=(0, 1, 2)):
    """把六侧 ``dif_band`` 汇总成原码次序 ``(x-,x+,y-,y+,z-,z+)`` 的六标志。"""
    flags = [0] * 6
    for face in tuple(faces):
        original = dif_axis_map[int(face.axis)]
        flags[2 * original + (1 if face.side.endswith("+") else 0)] = (
            1 if face.dif_band else 0)
    return tuple(flags)


def _farfield_layer(f, farfield):
    """把 5 分量来流状态扩成可写入 f 的广播形状。

    只在通用 FARFIELD 分支使用；``farfield`` 的形状和数值由
    ``build_execution`` 在 JIT 之前校验，这里不把 Python 校验带进 trace。
    """
    if farfield is None:
        raise ValueError("FARFIELD 边界需要 case['farfield'] 来流状态")
    value = jnp.asarray(farfield, dtype=f.dtype)
    if value.shape != (NVARS,):
        raise ValueError(
            f"farfield 形状必须是 ({NVARS},)，得到 {value.shape}")
    return value.reshape((NVARS,) + (1,) * (f.ndim - 1))


def fill_halos_i(f, bc, ni, farfield=None):
    """沿计算 i（ξ，数组 axis=-1）填充四层鬼胞。f: (..., ni+2*LAP)。

    FARFIELD 在旧的轴级入口会把同轴两侧都写成同一个来流状态；新的按面
    入口 ``fill_faces`` 只写被标记的那一面。两者不混用：主链走按面入口。
    """
    if bc == PERIODIC:
        f = f.at[..., :LAP].set(f[..., ni:ni + LAP])
        f = f.at[..., LAP + ni:].set(f[..., LAP:LAP + LAP])
    elif bc in (OUTFLOW, NONREFLECTING):
        pass
    elif bc == FARFIELD:
        layer = _farfield_layer(f, farfield)
        f = f.at[..., :LAP + 1].set(layer)
        f = f.at[..., LAP + ni - 1:].set(layer)
    elif bc in (INFLOW, PLATE):
        pass   # 由 case 回调写入入口剖面/壁面；单轴填充只保留原值
    else:
        raise ValueError(f"unknown bc: {bc}")
    return f


def fill_halos_j(f, bc, nj, farfield=None):
    """沿计算 j（η，数组 axis=-2）填充四层鬼胞。"""
    if bc == PERIODIC:
        f = f.at[..., :LAP, :].set(f[..., nj:nj + LAP, :])
        f = f.at[..., LAP + nj:, :].set(f[..., LAP:LAP + LAP, :])
    elif bc in (OUTFLOW, NONREFLECTING):
        pass
    elif bc == FARFIELD:
        layer = _farfield_layer(f, farfield)
        f = f.at[..., :LAP + 1, :].set(layer)
        f = f.at[..., LAP + nj - 1:, :].set(layer)
    elif bc in (INFLOW, PLATE):
        pass   # 由 case 回调写入入口/顶部/壁面；单轴填充只保留原值
    else:
        raise ValueError(f"unknown bc: {bc}")
    return f


def fill_halos_k(f, bc, nk, farfield=None):
    """沿计算 k（ζ，数组 axis=-3）填充四层鬼胞。"""
    if bc == PERIODIC:
        f = f.at[:, :LAP, :, :].set(f[:, nk:nk + LAP, :, :])
        f = f.at[:, LAP + nk:, :, :].set(f[:, LAP:LAP + LAP, :, :])
    elif bc in (OUTFLOW, NONREFLECTING):
        pass
    elif bc == FARFIELD:
        layer = _farfield_layer(f, farfield)
        f = f.at[:, :LAP + 1, :, :].set(layer)
        f = f.at[:, LAP + nk - 1:, :, :].set(layer)
    elif bc in (INFLOW, PLATE):
        pass   # 由 case 回调写入；单轴填充只保留原值
    else:
        raise ValueError(f"unknown bc: {bc}")
    return f


_GENERIC_AXIS_FILLERS = {
    0: fill_halos_i,
    1: fill_halos_j,
    2: fill_halos_k,
}


def fill_halos_axis(f, axis, bc, n, farfield=None):
    """按计算方向调用单轴通用填充（f 的最后三轴为 (k,j,i)）。"""
    return _GENERIC_AXIS_FILLERS[axis](f, bc, n, farfield=farfield)


def _mirror_wall_halos(f, side, nj):
    """只镜像被标记的那一侧壁面鬼层（rho/rhoE 偶、三个动量奇）。

    仅供没有算例回调的通用 WALL 使用（如 poiseuille 验证算例）；平板/槽道
    这类有回调的壁面走“鬼层清零 + 回调写壁面点”，不镜像。
    """
    sign = jnp.array(_WALL_SIGN).reshape(5, 1, 1, 1)
    if side == "j-":
        mir = f[:, :, 2 * LAP:LAP:-1, :]
        return f.at[:, :, :LAP, :].set(mir * sign)
    if side == "j+":
        mir = f[:, :, nj + LAP - 2:nj + LAP - 6:-1, :]
        return f.at[:, :, nj + LAP:nj + 2 * LAP, :].set(mir * sign)
    raise ValueError(f"通用 WALL 不支持 {side}（仅 j-/j+）")


def fill_wall_j(f, nj):
    """沿计算 j 的无滑移壁面状态镜像：rho/rhoE 偶、三个动量奇。

    下壁 ``ghost[p] = f[2*LAP-p] * sign``；上壁
    ``ghost[nj+LAP+q] = f[nj+LAP-2-q] * sign``。
    """
    sign = jnp.array(_WALL_SIGN).reshape(5, 1, 1, 1)
    mir_lo = f[:, :, 2 * LAP:LAP:-1, :]
    f = f.at[:, :, :LAP, :].set(mir_lo * sign)
    mir_hi = f[:, :, nj + LAP - 2:nj + LAP - 6:-1, :]
    f = f.at[:, :, nj + LAP:nj + 2 * LAP, :].set(mir_hi * sign)
    return f


def primitive(cons, phys):
    """守恒量 → 原始变量 (rho, u(3), T)，对应原码 cons_to_pri_kernel。

    除法用 `rho>0` 保护：新 alloc 的场鬼胞是 0，直接除会出 NaN/inf。
    真实计算里调用前 i/k 鬼胞已由周期填充（见 apply_wall_bc 说明）。
    """
    rho = cons[0]
    rho_safe = jnp.where(rho > 0, rho, 1.0)
    u = cons[1:4] / rho_safe
    T = (cons[4] - 0.5 * rho_safe * jnp.sum(u * u, axis=0)) / (rho_safe * phys.Cv)
    return rho, u, T


def _wall_layer(rho, T, j, step, tw):
    """壁面那一层：返回 (rho_wall, T_wall)，均为 (nz, nx) 形状的场。

    j: 壁面所在计算 j 索引；step=+1 表示向 +j（下壁），-1 表示向 −j（上壁）。
    """
    j1, j2, j3 = j + step, j + 2 * step, j + 3 * step
    pt1 = rho[:, j1, :] * T[:, j1, :]
    pt2 = rho[:, j2, :] * T[:, j2, :]
    pt3 = rho[:, j3, :] * T[:, j3, :]
    pw = (18.0 * pt1 - 9.0 * pt2 + 2.0 * pt3) / 11.0
    if tw > 0.0:
        Tw = jnp.full_like(pw, tw)          # 等温（冷）壁
    else:
        Tw = (18.0 * T[:, j1, :] - 9.0 * T[:, j2, :]
              + 2.0 * T[:, j3, :]) / 11.0   # 绝热壁：T 也三点外推
    # 角鬼胞（i/k 同时是鬼胞）上 ρT=0，绝热外推会给出 Tw=0；用保护分母避免 NaN，
    # 这些位置随后会被 i/k 周期填充覆盖，不影响内区。
    return pw / jnp.where(Tw > 0, Tw, 1.0), Tw


def apply_channel_wall_bc(cons, grid, phys, Tw_bottom=1.0, Tw_top=1.0):
    """把 IBC=200 的双壁条件写进 cons（壁面层 + 镜像鬼胞），返回新 cons。

    调用时机与原码一致：每个 RK 子步 `du_comput` **之前**（等价于原码
    `get_duvwT()`+`OCFD_bc()` 在上一子步末尾做的事）。i/k 鬼胞应已由周期
    填充（`fields.fill_halos_i/k`）；本函数只管计算 j 向壁面与镜像。
    """
    nj = grid_extent(grid, 1)
    rho, _, T = primitive(cons, phys)
    Cv = phys.Cv

    for j, step, tw in ((LAP, +1, Tw_bottom), (LAP + nj - 1, -1, Tw_top)):
        rho_w, Tw = _wall_layer(rho, T, j, step, tw)
        cons = cons.at[0, :, j, :].set(rho_w)                       # rho
        cons = cons.at[1:4, :, j, :].set(0.0)                      # 无滑移
        cons = cons.at[4, :, j, :].set(rho_w * Cv * Tw)            # ρE = ρ Cv T

    # 镜像鬼胞：ρ/ρE 偶、ρu/ρv/ρw 奇（与原码 for i in 0..LAP-1 / 1..LAP 同索引）
    return fill_wall_j(cons, nj)


def apply_plate_bc(cons, grid, phys, inflow=None, far=None, Tw=1.0):
    """只写下壁的物理边界点，不碰任何鬼层。

    cons: (5, nk+8, nj+8, ni+8)。入口（i-）由通用 INFLOW 写（边界点 + 四层
    鬼层剖面），出口（i+）由 OUTFLOW 不赋值，顶部（j+）由通用 FARFIELD 写
    边界点。壁面的鬼层在 ``fill_faces`` 里已被清零，这里不镜像。

    ``inflow`` / ``far`` 仅为旧调用兼容保留，不再使用。
    """
    del inflow, far
    ni, nj, nk = (grid_extent(grid, 0), grid_extent(grid, 1),
                  grid_extent(grid, 2))
    del ni, nk
    kz, _, ix = _interior_window(grid)

    # 下壁边界点：rho 由 ρT 两点外推、无滑移、ρE = ρ Cv Tw（IBC=124 系数）
    rho, _, T = primitive(cons, phys)
    rho_w = (4.0 * rho[:, LAP + 1, :] * T[:, LAP + 1, :]
             - rho[:, LAP + 2, :] * T[:, LAP + 2, :]) / (3.0 * Tw)
    cons = cons.at[0, kz, LAP, ix].set(rho_w[kz, ix])
    cons = cons.at[1:4, kz, LAP, ix].set(0.0)
    cons = cons.at[4, kz, LAP, ix].set(rho_w[kz, ix] * phys.Cv * Tw)
    return cons


def fill_halos(f, bcs, ni, nj, nk, farfield=None):
    """依次填充计算 i/j/k 鬼胞；j 的 WALL 走状态镜像。"""
    f = fill_halos_i(f, bcs[0], ni, farfield=farfield)
    if bcs[1] == WALL:
        f = fill_wall_j(f, nj)
    else:
        f = fill_halos_j(f, bcs[1], nj, farfield=farfield)
    f = fill_halos_k(f, bcs[2], nk, farfield=farfield)
    return f


def _face_extent(face, grid):
    """取该面沿计算轴的内点数。"""
    return (grid_extent(grid, 0), grid_extent(grid, 1),
            grid_extent(grid, 2))[face.axis]


def _fill_face_halos(f, face, grid):
    """按面的规则填状态鬼层；出口缺省保持已有值（face.params["halo"] 可覆盖）。"""
    mode = face_halo_mode(face)
    if mode == HALO_KEEP:
        return f
    axis = int(face.axis)
    n = _face_extent(face, grid)
    moved = jnp.moveaxis(f, 3 - axis, -1)
    low = face.side.endswith("-")
    if mode == HALO_PERIODIC:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., n:n + LAP])
        else:
            moved = moved.at[..., LAP + n:].set(moved[..., LAP:LAP + LAP])
    elif mode == HALO_COPY:
        # 零梯度 / 常数延拓：四层鬼层 = 边界点那一层。
        node = 0 if low else n - 1
        src = moved[..., LAP + node:LAP + node + 1]
        if low:
            moved = moved.at[..., :LAP].set(src)
        else:
            moved = moved.at[..., LAP + n:].set(src)
    else:
        if low:
            moved = moved.at[..., :LAP].set(0.0)
        else:
            moved = moved.at[..., LAP + n:].set(0.0)
    return jnp.moveaxis(moved, -1, 3 - axis)


def fill_faces(f, faces, grid):
    """按面填状态鬼层：先清零，再周期复制；入口保持，出口缺省保持。

    入口剖面由 :func:`write_faces` 写入。其他面的操作可能作用于交界鬼层。
    出口显式设 ``params["halo"]="copy"`` 时按零梯度延拓。
    """
    faces = tuple(faces)
    if not faces:
        return f
    order = (HALO_ZERO, HALO_PERIODIC, HALO_COPY, HALO_KEEP)
    for mode in order:
        for face in faces:
            if face_halo_mode(face) == mode:
                f = _fill_face_halos(f, face, grid)
    return f


def _interior_window(grid):
    """返回内区在 (k, j, i) 三轴上的开区间。"""
    return (slice(LAP, LAP + grid_extent(grid, 2)),
            slice(LAP, LAP + grid_extent(grid, 1)),
            slice(LAP, LAP + grid_extent(grid, 0)))


def _write_inflow_face(cons, face, grid, layer):
    """写 INFLOW：边界点 + 沿法向的四层鬼层（剖面在内区横向范围内广播）。"""
    kz, jy, ix = _interior_window(grid)
    if face.side == "i-":
        return cons.at[:, kz, jy, :LAP + 1].set(layer[:, :, :, None])
    if face.side == "i+":
        return cons.at[:, kz, jy, LAP + grid_extent(grid, 0) - 1:].set(
            layer[:, :, :, None])
    if face.side == "j-":
        return cons.at[:, kz, :LAP + 1, ix].set(layer[:, :, None, :])
    if face.side == "j+":
        return cons.at[:, kz, LAP + grid_extent(grid, 1) - 1:, ix].set(
            layer[:, :, None, :])
    if face.side == "k-":
        return cons.at[:, :LAP + 1, jy, ix].set(layer[:, None, :, :])
    if face.side == "k+":
        return cons.at[:, LAP + grid_extent(grid, 2) - 1:, jy, ix].set(
            layer[:, None, :, :])
    raise ValueError(f"unknown inflow side: {face.side}")


def _write_wall_point(cons, face, grid, phys):
    """通用 WALL 的壁面点：无滑移 + 等温/绝热，ρ 由 ρT 两点外推（IBC=124）。

    ``face.thermal == "isothermal"`` 用 ``params["Tw"]`` 作壁温；``"adiabatic"``
    把 T 也两点外推（Tw<=0 的写法与 ``_wall_layer`` 一致）。鬼层不在这里处理，
    由 :func:`fill_faces` 的镜像/清零负责。
    """
    if face.axis != 1:
        raise ValueError("通用 WALL 只支持计算 j 方向（axis=1）")
    tw = _as_float(face.params.get("Tw", 1.0), "wall.Tw")
    nj = grid_extent(grid, 1)
    full_k, _, full_i = _face_lateral(face, grid)
    step = 1 if face.side.endswith("-") else -1
    j = LAP if step > 0 else LAP + nj - 1
    j1, j2 = j + step, j + 2 * step
    rho, _, T = primitive(cons, phys)
    pt1 = rho[:, j1, :] * T[:, j1, :]
    pt2 = rho[:, j2, :] * T[:, j2, :]
    if tw > 0.0:
        Tw = jnp.full_like(pt1, tw)
        denom = Tw
    else:
        Tw = 2.0 * T[:, j1, :] - T[:, j2, :]
        denom = jnp.where(Tw > 0, Tw, 1.0)
    rho_w = (4.0 * pt1 - pt2) / (3.0 * denom)
    cons = cons.at[0, full_k, j, full_i].set(rho_w[full_k, full_i])
    cons = cons.at[1:4, full_k, j, full_i].set(0.0)
    cons = cons.at[4, full_k, j, full_i].set(
        rho_w[full_k, full_i] * phys.Cv * Tw[full_k, full_i])
    return cons


def _face_lateral(face, grid):
    """边界点仅写横向内区，交界鬼层由相邻面处理。"""
    del face
    return _interior_window(grid)


def write_faces(cons, faces, grid, inflow=None, farfield=None, phys=None):
    """写边界点：INFLOW（含鬼层）、FARFIELD 与声明了热条件的通用 WALL。

    出口/周期面不赋值；没有回调的 WALL 若没声明 ``thermal`` 就只镜像鬼层（原码
    验证算例口径），声明了才写壁面点（``phys`` 必须给出）。
    """
    for face in tuple(faces):
        kz, jy, ix = _interior_window(grid)
        full_k, full_j, full_i = _face_lateral(face, grid)
        if face.kind == INFLOW:
            if inflow is not None:
                layer = jnp.asarray(inflow, dtype=cons.dtype)
                cons = _write_inflow_face(cons, face, grid, layer)
            elif farfield is not None:
                # 没有剖面文件时按自由来流常数给入口
                layer = jnp.asarray(_farfield_state({"farfield": farfield}),
                                    dtype=cons.dtype).reshape(NVARS, 1, 1)
                cons = _write_inflow_face(cons, face, grid, layer)
            # 两者都缺席的旧式特殊入口：不写，交给算例回调
        elif face.kind == FARFIELD:
            layer = _farfield_layer(cons, farfield)
            if face.side == "i-":
                cons = cons.at[:, kz, jy, LAP:LAP + 1].set(layer)
            elif face.side == "i+":
                cons = cons.at[:, kz, jy,
                                LAP + grid_extent(grid, 0) - 1:
                                LAP + grid_extent(grid, 0)].set(layer)
            elif face.side == "j-":
                cons = cons.at[:, full_k, LAP:LAP + 1, full_i].set(layer)
            elif face.side == "j+":
                cons = cons.at[:, full_k, LAP + grid_extent(grid, 1) - 1:
                                LAP + grid_extent(grid, 1), full_i].set(layer)
            elif face.side == "k-":
                cons = cons.at[:, LAP:LAP + 1, full_j, full_i].set(layer)
            elif face.side == "k+":
                cons = cons.at[:, LAP + grid_extent(grid, 2) - 1:
                                LAP + grid_extent(grid, 2), full_j, full_i].set(layer)
            else:
                raise ValueError(f"unknown farfield side: {face.side}")
        elif (face.kind == WALL and face.executor == "halo"
              and face.thermal is not None and phys is not None):
            cons = _write_wall_point(cons, face, grid, phys)
    return cons


def zero_nonperiodic_ghosts(f, bcs, preserve_inflow=False):
    """清除非周期方向的鬼层，周期方向保留交换值。

    ``preserve_inflow=True`` 时保留 INFLOW 低端鬼层（平板入口状态）。
    f 的最后三轴为计算 (k, j, i)，bcs 按计算 (i, j, k) 排列。
    """
    for axis, bc in enumerate(bcs):
        if bc == PERIODIC:
            continue
        ax = f.ndim - 1 - axis
        f = jnp.moveaxis(f, ax, -1)
        if not (preserve_inflow and bc == INFLOW):
            f = f.at[..., :LAP].set(0.0)
        f = f.at[..., -LAP:].set(0.0)
        f = jnp.moveaxis(f, -1, ax)
    return f


def fill_ev_wall_j(f, nj, signs):
    """粘性通量 Ev（4 分量堆叠，(4,k,j,i)）的计算 j 双壁镜像。"""
    sign = jnp.array(signs).reshape(4, 1, 1, 1)
    mir_lo = f[:, :, 2 * LAP:LAP:-1, :]
    f = f.at[:, :, :LAP, :].set(mir_lo * sign)
    mir_hi = f[:, :, nj + LAP - 2:nj + LAP - 6:-1, :]
    f = f.at[:, :, nj + LAP:nj + 2 * LAP, :].set(mir_hi * sign)
    return f


def fill_ev_plate_j(f, nj, signs):
    """平板 Ev：下壁镜像（同槽道下壁）+ 顶部零梯度（自由来流侧）。"""
    sign = jnp.array(signs).reshape(4, 1, 1, 1)
    mir_lo = f[:, :, 2 * LAP:LAP:-1, :]
    f = f.at[:, :, :LAP, :].set(mir_lo * sign)
    f = f.at[:, :, nj + LAP:, :].set(f[:, :, nj + LAP - 1:nj + LAP, :])
    return f


def fill_ev_axis(f, axis, bc, nxi, is_ev=False):
    """Ev 场的单轴填充；壁面/平板规则与守恒量状态规则显式分开。

    wall 目前只在计算 j 方向支持；i/k 的 WALL 走通用填充会按未知边界报错，
    避免把槽道镜像误用于其他方向。FARFIELD 的粘性通量按零梯度处理：远场
    状态是 Dirichlet 常量，Ev 的鬼层直接复制边界内层。
    """
    if bc in (OUTFLOW, NONREFLECTING, INFLOW):
        moved = jnp.moveaxis(f, 3 - axis, -1)
        moved = moved.at[..., :LAP].set(0.0)
        moved = moved.at[..., LAP + nxi:].set(0.0)
        return jnp.moveaxis(moved, -1, 3 - axis)
    if bc == FARFIELD:
        if axis == 0:
            f = f.at[..., :LAP].set(f[..., LAP:LAP + 1])
            f = f.at[..., LAP + nxi:].set(f[..., LAP + nxi - 1:LAP + nxi])
        elif axis == 1:
            f = f.at[..., :LAP, :].set(f[..., LAP:LAP + 1, :])
            f = f.at[..., LAP + nxi:, :].set(f[..., LAP + nxi - 1:LAP + nxi, :])
        else:
            f = f.at[:, :LAP, :, :].set(f[:, LAP:LAP + 1, :, :])
            f = f.at[:, LAP + nxi:, :, :].set(
                f[:, LAP + nxi - 1:LAP + nxi, :, :])
        return f
    if axis == 0:
        return fill_halos_i(f, bc, nxi)
    if axis == 1:
        if bc == WALL:
            assert is_ev, "cons-level wall fill 应走 fill_wall_j（var 镜像）"
            return fill_ev_wall_j(f, nxi, EV_WALL_SIGNS[axis])
        if bc == PLATE:
            assert is_ev, "cons-level plate fill 由 plate.apply 写入"
            return fill_ev_plate_j(f, nxi, EV_WALL_SIGNS[axis])
        return fill_halos_j(f, bc, nxi)
    return fill_halos_k(f, bc, nxi)


def _mirror_ev_wall_halos(f, side, n, signs):
    """只镜像被标记那一侧的 Ev 鬼层（与状态镜像共用 ``_EV_WALL_SIGNS``）。"""
    sign = jnp.array(signs).reshape(4, 1, 1, 1)
    if side == "j-":
        mir = f[:, :, 2 * LAP:LAP:-1, :]
        return f.at[:, :, :LAP, :].set(mir * sign)
    if side == "j+":
        mir = f[:, :, n + LAP - 2:n + LAP - 6:-1, :]
        return f.at[:, :, n + LAP:n + 2 * LAP, :].set(mir * sign)
    raise ValueError(f"通用 WALL 的 Ev 不支持 {side}（仅 j-/j+）")


def _fill_ev_face(f, face, grid, signs=None):
    """填 Ev 鬼层；入口和出口按本次原版参考初始化为零。"""
    mode = HALO_ZERO if face.kind in (OUTFLOW, NONREFLECTING, INFLOW) else halo_mode(face.kind)
    if mode == HALO_KEEP:
        return f
    axis = int(face.axis)
    n = _face_extent(face, grid)
    moved = jnp.moveaxis(f, 3 - axis, -1)
    low = face.side.endswith("-")
    if mode == HALO_PERIODIC:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., n:n + LAP])
        else:
            moved = moved.at[..., LAP + n:].set(moved[..., LAP:LAP + LAP])
    elif mode == HALO_COPY:
        node = 0 if low else n - 1
        src = moved[..., LAP + node:LAP + node + 1]
        if low:
            moved = moved.at[..., :LAP].set(src)
        else:
            moved = moved.at[..., LAP + n:].set(src)
    else:
        if low:
            moved = moved.at[..., :LAP].set(0.0)
        else:
            moved = moved.at[..., LAP + n:].set(0.0)
    return jnp.moveaxis(moved, -1, 3 - axis)


def fill_ev_faces(f, faces, grid):
    """按六侧描述填 Ev 鬼层：先清零，再周期复制。"""
    faces = tuple(faces)
    if not faces:
        return f
    for mode in (HALO_ZERO, HALO_PERIODIC, HALO_COPY):
        for face in faces:
            if (HALO_ZERO if face.kind in (OUTFLOW, NONREFLECTING, INFLOW) else halo_mode(face.kind)) == mode:
                f = _fill_ev_face(f, face, grid)
    return f


@dataclass(frozen=True)
class BoundaryExecution:
    """JIT 前冻结的边界执行配置（描述、轴级 bcs 与 case 回调同源）。

    mode: ``generic`` / ``channel`` / ``plate`` / ``custom``。
    callback: case 提供的 ``bc_apply``；generic 必须为 None。
    faces: 六侧只读描述，用于校验和文档化。
    channel_bc / plate: 特殊边界的参数包或几何/剖面字典；custom 由 case
    通过 ``special_boundary`` 对象提供 ``name``、``faces`` 和 ``bc_apply``。
    """

    mode: str
    bcs: Tuple[str, str, str]
    callback: Optional[Callable]
    faces: Tuple[Any, ...] = ()
    channel_bc: Any = None
    plate: Any = None
    mixed_outflow: Tuple[Any, ...] = ()
    grid: Any = None
    farfield: Any = None
    inflow: Any = None
    band: Optional[Tuple[int, ...]] = None
    phys: Any = None

    def apply(self, q):
        """兼容入口：执行一次 ``prepare``，不再单独应用特殊回调。"""
        return self.prepare(q)

    def prepare(self, q, zero_state_ghosts=False):
        """按“边界点赋值 → 鬼层 → 特殊回调”准备一次状态。

        ``write_faces`` 先写边界点，``fill_faces`` 再处理周期和清零面；
        出口鬼层保持原值。特殊回调最后覆盖它接管的局部区域。

        ``zero_state_ghosts`` 仅为旧调用兼容保留：鬼层规则已经统一到 halo，
        不再按算例开关。
        """
        del zero_state_ghosts
        if self.faces:
            q = write_faces(q, self.faces, self.grid,
                            inflow=self.inflow, farfield=self.farfield,
                            phys=self.phys)
            q = fill_faces(q, self.faces, self.grid)
        elif self.grid is not None:
            q = fill_halos(q, self.bcs, grid_extent(self.grid, 0),
                           grid_extent(self.grid, 1), grid_extent(self.grid, 2),
                           farfield=self.farfield)
        return q if self.callback is None else self.callback(q)

    def fill_mixed_outflow(self, q, axis=None):
        """清零指定方向普通出口的粘性通量鬼层。"""
        for face in self.mixed_outflow:
            if axis is None or face.axis == axis:
                q = _fill_ev_face(q, face, self.grid)
        return q


def _as_float(value, name):
    """把边界参数读成有限浮点，拒绝 NaN/inf。"""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是浮点数，得到 {value!r}") from exc
    if not (out == out and abs(out) != float("inf")):
        raise ValueError(f"{name} 含非有限值: {out}")
    return out


def _farfield_state(case):
    """从 case 读取并校验通用远场的 5 分量守恒状态。

    远场状态必须在 JIT 之前固定；这里做形状、有限值和密度检查，
    避免把非法状态带进编译后的边界填充。
    """
    raw = case.get("farfield")
    if raw is None:
        raise ValueError(
            "通用 FARFIELD 需要 case['farfield']（5 分量守恒量）")
    state = jnp.asarray(raw)
    if state.shape != (NVARS,):
        raise ValueError(f"case['farfield'] 形状必须是 ({NVARS},)，得到 {state.shape}")
    if not bool(jnp.all(jnp.isfinite(state))):
        raise ValueError(f"case['farfield'] 含 NaN/inf: {state}")
    if float(state[0]) <= 0.0:
        raise ValueError(f"case['farfield'] 的密度必须为正，得到 {float(state[0])}")
    return state


def _face_map(desc):
    """把描述里的六侧转成 side -> FaceBoundary 的映射。"""
    faces = tuple(getattr(desc, "faces", ()))
    if len(faces) != 6 or tuple(f.side for f in faces) != SIX_SIDES:
        raise ValueError(f"边界描述必须按 {SIX_SIDES} 顺序包含六侧")
    return {f.side: f for f in faces}


def _require_callback(case, label):
    """取 case 的 bc_apply 并确认可调用。"""
    callback = case.get("bc_apply")
    if not callable(callback):
        raise ValueError(
            f"{label} 需要可调用的 case['bc_apply'] 回调；"
            f"得到 {callback!r}")
    return callback


def validate_special_sides(faces, bcs):
    """校验 case 特殊边界的六侧结构与方向限制，并返回原元组。

    六侧 faces 是权威描述；轴级 ``bcs`` 只是低侧代表值（供旧调用方读取），
    **不再**要求逐面 kind 与之一致——CGNS 的同轴异类正负面
    （如 i- 出流、i+ 远场）是合法配置，曾在此被误拒（终审 P2）。
    """
    if len(faces) != 6 or tuple(f.side for f in faces) != SIX_SIDES:
        raise ValueError(
            f"special_boundary.faces 必须按 {SIX_SIDES} 顺序给出六侧")
    for axis, bc in enumerate(bcs):
        axis_faces = [f for f in faces if f.axis == axis]
        callback_faces = [f for f in axis_faces
                          if f.executor in ("callback", "callback+halo")]
        if callback_faces and bc not in (INFLOW, PLATE):
            raise ValueError(
                f"special_boundary axis={axis} 有回调面，轴级 bcs 必须是 "
                f"INFLOW/PLATE（no-op 占位），得到 {bc!r}")
        if not callback_faces and bc not in (PERIODIC, OUTFLOW, NONREFLECTING,
                                             FARFIELD, WALL):
            # WALL 无回调面是合法组合（通用 WALL 的状态鬼层由 fill_faces 清零、
            # Ev 侧另有规则；Poiseuille 派生六侧曾在此被误拒——2026-11-23 Astra 复审）
            raise ValueError(
                f"special_boundary axis={axis} 没有回调面，轴级 bcs 只能是 "
                f"PERIODIC/OUTFLOW/NONREFLECTING/FARFIELD/WALL，得到 {bc!r}")
        if bc == WALL and axis != 1:
            raise ValueError("special_boundary 的 WALL 只支持计算 j 方向")
    for face in faces:
        if face.kind == WALL and face.axis != 1:
            raise ValueError("special_boundary 的 WALL 只支持计算 j 方向")
    return faces


def _check_faces(faces, kinds, executors, label):
    """逐侧核对语义与执行来源；不匹配时在 JIT 前失败。"""
    expected = {
        "i-": kinds[0], "i+": kinds[1],
        "j-": kinds[2], "j+": kinds[3],
        "k-": kinds[4], "k+": kinds[5],
    }
    exec_expected = {
        "i-": executors[0], "i+": executors[1],
        "j-": executors[2], "j+": executors[3],
        "k-": executors[4], "k+": executors[5],
    }
    for side in SIX_SIDES:
        face = faces[side]
        if face.kind != expected[side]:
            raise ValueError(
                f"{label} 的 {side} 描述是 {face.kind!r}，但执行配置要求 "
                f"{expected[side]!r}")
        if face.executor != exec_expected[side]:
            raise ValueError(
                f"{label} 的 {side} 执行来源是 {face.executor!r}，但要求 "
                f"{exec_expected[side]!r}")


def build_execution(case: MappingABC, desc=None) -> BoundaryExecution:
    """在 JIT 前把 case、六侧描述和轴级 bcs 收敛成一份执行配置。

    现有组合只接受三类：通用（周期/出流/壁面/远场）、channel（IBC=200）、
    plate（IBC=124）。任何“回调存在但没有专属标记”“标记存在但回调缺失”
    “固定六侧与 bcs 矛盾”都在这里报错，绝不猜一个能跑的组合。
    """
    if not isinstance(case, MappingABC):
        raise ValueError("case 必须是映射")
    if desc is None:
        from .mesh_data import from_case
        desc = from_case(case)
    bcs = tuple(getattr(desc, "bcs", ()))
    if len(bcs) != 3:
        raise ValueError(f"边界执行配置需要三个轴级 bcs，得到 {bcs}")
    # case 可能在描述构造后被原地修改；两份轴级 bcs 必须仍是同一份决策。
    if "bcs" in case and tuple(case["bcs"]) != bcs:
        raise ValueError(
            f"case['bcs']={tuple(case['bcs'])} 与已构造描述的 bcs={bcs} 不一致；"
            "请在修改 bcs 后重新构造描述")

    phys = case.get("params")
    faces = _face_map(desc)
    face_tuple = tuple(faces.values())
    # 边界带由 kind 派生（非周期且非 INFLOW 才开），不再由算例手写 dif_bound。
    band = dif_bound_from_faces(face_tuple,
                                getattr(desc, "dif_axis_map", (0, 1, 2)))
    inflow = case.get("inflow")
    callback = case.get("bc_apply")
    has_plate = case.get("plate") is not None
    has_channel = case.get("channel_bc") is not None
    has_special = case.get("special_boundary") is not None
    if sum(x is not None for x in (case.get("plate"), case.get("channel_bc"),
                                   case.get("special_boundary"))) > 1:
        raise ValueError("case 同时带多种特殊边界标记，无法确定执行策略")

    if has_channel:
        callback = _require_callback(case, "channel 特殊边界")
        if bcs != (PERIODIC, WALL, PERIODIC):
            raise ValueError(
                f"channel 的固定六侧要求 bcs=(periodic, wall, periodic)，"
                f"得到 {bcs}；不能把周期轴改成其他边界")
        _check_faces(
            faces,
            (PERIODIC, PERIODIC, WALL, WALL, PERIODIC, PERIODIC),
            ("halo", "halo", "callback+halo", "callback+halo", "halo", "halo"),
            "channel")
        channel_bc = case["channel_bc"]
        for attr in ("Tw_bottom", "Tw_top"):
            if not hasattr(channel_bc, attr):
                raise ValueError(f"channel_bc 缺少 {attr}")
            _as_float(getattr(channel_bc, attr), f"channel_bc.{attr}")
        for side in ("j-", "j+"):
            face = faces[side]
            if face.thermal not in ("isothermal", "adiabatic"):
                raise ValueError(f"channel {side} 缺少热条件描述")
            if "Tw" not in face.params:
                raise ValueError(f"channel {side} 缺少 Tw 参数")
            tw = _as_float(getattr(channel_bc, "Tw_bottom" if side == "j-" else "Tw_top"),
                           f"channel_bc.{side}.Tw")
            if face.params["Tw"] != tw or face.thermal != ("isothermal" if tw > 0 else "adiabatic"):
                raise ValueError(f"channel {side} 的 Tw 壁温与描述不一致；请重新构造描述")
        return BoundaryExecution("channel", bcs, callback, face_tuple,
                                 channel_bc=channel_bc, grid=desc.grid,
                                 band=band, inflow=inflow, phys=phys)

    if has_plate:
        callback = _require_callback(case, "plate 特殊边界")
        if bcs != (INFLOW, PLATE, PERIODIC):
            raise ValueError(
                f"plate 的固定六侧要求 bcs=(inflow, plate, periodic)，"
                f"得到 {bcs}；入口/壁面/顶部不能换成通用轴级边界")
        _check_faces(
            faces,
            (INFLOW, OUTFLOW, WALL, FARFIELD, PERIODIC, PERIODIC),
            ("callback", "callback", "callback", "callback", "halo", "halo"),
            "plate")
        plate_marker = case["plate"]
        if not isinstance(plate_marker, MappingABC):
            raise ValueError("case['plate'] 必须是映射")
        for side in ("j-", "j+"):
            face = faces[side]
            if side == "j-" and face.thermal != "isothermal":
                raise ValueError("plate j- 必须描述为等温壁")
        needs_farfield = any(f.kind == FARFIELD for f in faces.values())
        farfield = _farfield_state(case) if needs_farfield else None
        return BoundaryExecution("plate", bcs, callback, face_tuple,
                                 plate=plate_marker, grid=desc.grid,
                                 farfield=farfield, band=band, inflow=inflow,
                                 phys=phys)

    if has_special:
        callback = _require_callback(case, "special_boundary 特殊边界")
        special = case["special_boundary"]
        if not isinstance(getattr(special, "name", None), str):
            raise ValueError("special_boundary 需要字符串 name")
        marker_faces = tuple(getattr(special, "faces", ()))
        if not marker_faces:
            # faces 可选：缺省继承按算例派生的六侧描述（2026-11-23 ⑥ 入口；
            # 脚本只写 bc_apply 时不必抄六侧——给出时仍须逐侧一致）
            marker_faces = tuple(faces.values())
        validate_special_sides(marker_faces, bcs)
        if marker_faces != tuple(faces.values()):
            raise ValueError("special_boundary.faces 与已构造的六侧描述不一致")
        mixed_outflow = tuple(f for f in marker_faces if f.executor == "halo" and
                              bcs[f.axis] in (INFLOW, PLATE)
                              and f.kind in (OUTFLOW, NONREFLECTING))
        needs_farfield = any(f.kind == FARFIELD for f in marker_faces)
        farfield = _farfield_state(case) if needs_farfield else None
        return BoundaryExecution("custom", bcs, callback, marker_faces,
                                 mixed_outflow=mixed_outflow, grid=desc.grid,
                                 farfield=farfield, band=band, inflow=inflow,
                                 phys=phys)

    if callback is not None:
        raise ValueError(
            "case 提供了 bc_apply，但没有 channel_bc / plate 特殊边界标记；"
            "未知特殊边界必须在 case 侧显式选择并补六侧描述")
    for face in faces.values():
        if face.executor != "halo":
            raise ValueError(
                f"通用边界 {face.side} 的执行来源必须是 halo，"
                f"得到 {face.executor!r}")
        if face.kind not in GENERIC_KINDS:
            raise ValueError(
                f"通用边界不支持 {face.side}={face.kind!r}")
        if face.kind == WALL and face.axis != 1:
            raise ValueError(
                f"通用 WALL 只支持计算 j 方向（axis=1），得到 axis={face.axis}")
    needs_farfield = any(f.kind == FARFIELD for f in faces.values())
    farfield = _farfield_state(case) if needs_farfield else None
    return BoundaryExecution("generic", bcs, None, face_tuple,
                             grid=desc.grid, farfield=farfield,
                             band=band, inflow=inflow, phys=phys)


def kinds_for(bcs):
    """把轴级 bcs 展开成六侧语义序列（i-,i+,j-,j+,k-,k+）。"""
    return (bcs[0], bcs[0], bcs[1], bcs[1], bcs[2], bcs[2])
