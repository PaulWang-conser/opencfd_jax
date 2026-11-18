"""统一边界层：通用鬼胞填充、壁面镜像、粘性 Ev 规则与执行配置。

本模块负责**所有通用边界**的数值实现：

- 周期/出流沿计算 i/j/k 的鬼胞填充；
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

GENERIC_KINDS = (PERIODIC, OUTFLOW, WALL, FARFIELD)
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
    elif bc == OUTFLOW:
        f = f.at[..., :LAP].set(f[..., LAP:LAP + 1])
        f = f.at[..., LAP + ni:].set(f[..., LAP + ni - 1:LAP + ni])
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
    elif bc == OUTFLOW:
        f = f.at[..., :LAP, :].set(f[..., LAP:LAP + 1, :])
        f = f.at[..., LAP + nj:, :].set(f[..., LAP + nj - 1:LAP + nj, :])
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
    elif bc == OUTFLOW:
        f = f.at[:, :LAP, :, :].set(f[:, LAP:LAP + 1, :, :])
        f = f.at[:, LAP + nk:, :, :].set(f[:, LAP + nk - 1:LAP + nk, :, :])
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


def apply_plate_bc(cons, grid, phys, inflow, far, Tw=1.0):
    """写入平板的三类鬼胞/壁面层，返回新 cons。

    cons: (5, nk+8, nj+8, ni+8)；入口、壁面和顶层均包含物理点。
    inflow: (5, nk, nj) 入口守恒量剖面；far: (5,) 顶部来流常数。

    物理平板的 x 前缘/尾缘和高度 H 已在 ``cases.py`` 建网格时确定；本函数
    只按计算 i/j 索引写入口、下壁和顶部，不读取物理坐标 ``Axx``。
    """
    ni, nj, nk = (grid_extent(grid, 0), grid_extent(grid, 1),
                  grid_extent(grid, 2))
    ny_in = LAP + nj
    nz_in = LAP + nk

    # 1) i 两侧：左 = 入口 Dirichlet（含首个物理点），右 = 零梯度鬼层
    cons = cons.at[:, LAP:nz_in, LAP:ny_in, :LAP + 1].set(inflow[:, :, :, None])
    cons = cons.at[:, :, :, LAP + ni:].set(cons[:, :, :, LAP + ni - 1:LAP + ni])

    # 2) 下壁：壁面层（rho 由 ρT 两点外推、无滑移、ρE = ρ Cv Tw）
    rho, _, T = primitive(cons, phys)
    # IBC=124 的等温壁使用两点压力外推，系数与槽道边界不同。
    rho_w = (4.0 * rho[:, LAP + 1, :] * T[:, LAP + 1, :]
             - rho[:, LAP + 2, :] * T[:, LAP + 2, :]) / (3.0 * Tw)
    cons = cons.at[0, :, LAP, :].set(rho_w)
    cons = cons.at[1:4, :, LAP, :].set(0.0)
    cons = cons.at[4, :, LAP, :].set(rho_w * phys.Cv * Tw)

    # 3) 下壁镜像鬼胞：ghost[p] = f[2LAP-p] * sign（速度反号、rho/ρE 同号）
    sign = jnp.array(_WALL_SIGN).reshape(5, 1, 1, 1)
    cons = cons.at[:, :, :LAP, :].set(cons[:, :, 2 * LAP:LAP:-1, :] * sign)

    # 4) 顶部：自由来流 Dirichlet（全 i，含角鬼胞）
    cons = cons.at[:, :, ny_in - 1:, :].set(far.reshape(5, 1, 1, 1))
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


def _fill_face_state(f, face, grid, farfield=None):
    """按单个 ``FaceBoundary`` 填状态鬼层。

    所有面都先按自己的 ``kind`` 填常规值；``callback`` / ``callback+halo``
    只是标记“特殊回调稍后会覆盖”，不在通用填充里跳过。这样特殊脚本覆盖的
    正是已经存在的普通鬼胞值，覆盖范围由脚本自己决定。
    """
    axis = int(face.axis)
    n = _face_extent(face, grid)
    moved = jnp.moveaxis(f, 3 - axis, -1)
    low = face.side.endswith("-")
    if face.kind == PERIODIC:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., n:n + LAP])
        else:
            moved = moved.at[..., LAP + n:].set(moved[..., LAP:LAP + LAP])
    elif face.kind == OUTFLOW:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., LAP:LAP + 1])
        else:
            moved = moved.at[..., LAP + n:].set(
                moved[..., LAP + n - 1:LAP + n])
    elif face.kind == FARFIELD:
        layer = _farfield_layer(moved, farfield)
        if low:
            moved = moved.at[..., :LAP + 1].set(layer)
        else:
            moved = moved.at[..., LAP + n - 1:].set(layer)
    elif face.kind == WALL:
        if axis != 1:
            raise ValueError("通用 WALL 只支持计算 j 方向（axis=1）")
        sign = jnp.asarray(_WALL_SIGN).reshape(
            (NVARS,) + (1,) * (moved.ndim - 1))
        if low:
            moved = moved.at[..., :LAP].set(
                moved[..., 2 * LAP:LAP:-1] * sign)
        else:
            moved = moved.at[..., LAP + n:].set(
                moved[..., LAP + n - 2:LAP + n - 6:-1] * sign)
    elif face.kind in (INFLOW, PLATE):
        pass   # 由 case 回调写入该面；这里不重写
    else:
        raise ValueError(f"unknown bc: {face.kind}")
    return jnp.moveaxis(moved, -1, 3 - axis)


def fill_faces(f, faces, grid, farfield=None):
    """按六侧描述填常规状态鬼层，供 ``BoundaryExecution.prepare`` 使用。

    所有面先按 kind 填普通值；远场面先写自己的边界内点和鬼层，再让周期/出流
    面从已经更新的内点复制，这样单侧远场不会要求对侧也是远场。特殊回调在
    本函数之后执行，覆盖它接管的鬼胞。
    """
    faces = tuple(faces)
    if not faces:
        return f
    for face in faces:
        if face.kind == FARFIELD:
            f = _fill_face_state(f, face, grid, farfield=farfield)
    for face in faces:
        if face.kind == FARFIELD:
            continue
        f = _fill_face_state(f, face, grid, farfield=farfield)
    return f


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


def _fill_ev_face(f, face, grid, signs):
    """按单面填 4 分量 Ev 鬼层；与状态面不同，回调面也要填 Ev。"""
    axis = int(face.axis)
    n = _face_extent(face, grid)
    moved = jnp.moveaxis(f, 3 - axis, -1)
    low = face.side.endswith("-")
    kind = face.kind
    if kind == PERIODIC:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., n:n + LAP])
        else:
            moved = moved.at[..., LAP + n:].set(moved[..., LAP:LAP + LAP])
    elif kind == OUTFLOW:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., LAP:LAP + 1])
        else:
            moved = moved.at[..., LAP + n:].set(
                moved[..., LAP + n - 1:LAP + n])
    elif kind == FARFIELD:
        if low:
            moved = moved.at[..., :LAP].set(moved[..., LAP:LAP + 1])
        else:
            moved = moved.at[..., LAP + n:].set(
                moved[..., LAP + n - 1:LAP + n])
    elif kind == WALL:
        if axis != 1:
            raise ValueError("通用 WALL 的 Ev 只支持计算 j 方向（axis=1）")
        sign = jnp.asarray(signs).reshape((4,) + (1,) * (moved.ndim - 1))
        if low:
            moved = moved.at[..., :LAP].set(
                moved[..., 2 * LAP:LAP:-1] * sign)
        else:
            moved = moved.at[..., LAP + n:].set(
                moved[..., LAP + n - 2:LAP + n - 6:-1] * sign)
    elif kind == PLATE:
        if axis != 1:
            raise ValueError("PLATE 的 Ev 只支持计算 j 方向（axis=1）")
        if low:
            sign = jnp.asarray(signs).reshape((4,) + (1,) * (moved.ndim - 1))
            moved = moved.at[..., :LAP].set(
                moved[..., 2 * LAP:LAP:-1] * sign)
        else:
            moved = moved.at[..., LAP + n:].set(
                moved[..., LAP + n - 1:LAP + n])
    elif kind == INFLOW:
        pass
    else:
        raise ValueError(f"unknown Ev bc: {kind}")
    return jnp.moveaxis(moved, -1, 3 - axis)


def fill_ev_faces(f, faces, grid):
    """按六侧描述填 Ev 鬼层；调用方仍负责 ``zero_state_ghosts``。"""
    for face in tuple(faces):
        signs = EV_WALL_SIGNS.get(face.axis, EV_WALL_SIGNS[1])
        f = _fill_ev_face(f, face, grid, signs)
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

    def apply(self, q):
        """兼容入口：执行一次 ``prepare``，不再单独应用特殊回调。"""
        return self.prepare(q)

    def prepare(self, q, zero_state_ghosts=False):
        """按“常规 halo → 清零（可选）→ 特殊回调”准备一次状态。

        常规填充处理全部六侧；``callback`` / ``callback+halo`` 只是说明该面
        稍后会被特殊回调覆盖，不在常规阶段跳过。清零在回调之前，特殊回调
        最后写入的鬼胞不会被常规填充再次覆盖。
        """
        if self.faces:
            q = fill_faces(q, self.faces, self.grid, farfield=self.farfield)
            if zero_state_ghosts:
                q = zero_nonperiodic_ghosts(q, self.bcs, preserve_inflow=True)
        elif self.grid is not None:
            q = fill_halos(q, self.bcs, grid_extent(self.grid, 0),
                           grid_extent(self.grid, 1), grid_extent(self.grid, 2),
                           farfield=self.farfield)
            if zero_state_ghosts:
                q = zero_nonperiodic_ghosts(q, self.bcs, preserve_inflow=True)
        return q if self.callback is None else self.callback(q)

    def fill_mixed_outflow(self, q, axis=None):
        """对含特殊回调的轴，仅填充其普通出流侧的鬼层。"""
        for face in self.mixed_outflow:
            if axis is not None and face.axis != axis:
                continue
            moved = jnp.moveaxis(q, 3 - face.axis, -1)
            n = (grid_extent(self.grid, 0), grid_extent(self.grid, 1),
                 grid_extent(self.grid, 2))[face.axis]
            if face.side.endswith("-"):
                moved = moved.at[..., :LAP].set(moved[..., LAP:LAP + 1])
            else:
                moved = moved.at[..., LAP + n:].set(moved[..., LAP + n - 1:LAP + n])
            q = jnp.moveaxis(moved, -1, 3 - face.axis)
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
    """校验 case 特殊边界的六侧与轴级 bcs 是否一致，并返回原元组。"""
    if len(faces) != 6 or tuple(f.side for f in faces) != SIX_SIDES:
        raise ValueError(
            f"special_boundary.faces 必须按 {SIX_SIDES} 顺序给出六侧")
    for axis, bc in enumerate(bcs):
        axis_faces = [f for f in faces if f.axis == axis]
        callback_faces = [f for f in axis_faces
                          if f.executor in ("callback", "callback+halo")]
        halo_faces = [f for f in axis_faces if f.executor == "halo"]
        if callback_faces and bc not in (INFLOW, PLATE):
            raise ValueError(
                f"special_boundary axis={axis} 有回调面，轴级 bcs 必须是 "
                f"INFLOW/PLATE（no-op 占位），得到 {bc!r}")
        if not callback_faces and bc not in (PERIODIC, OUTFLOW, FARFIELD):
            raise ValueError(
                f"special_boundary axis={axis} 没有回调面，轴级 bcs 只能是 "
                f"PERIODIC/OUTFLOW/FARFIELD，得到 {bc!r}")
        for face in halo_faces:
            if callback_faces and face.kind == OUTFLOW:
                continue
            if face.kind != bc:
                raise ValueError(
                    f"special_boundary {face.side} 的 halo 描述 {face.kind!r} "
                    f"与轴级 bcs {bc!r} 不一致")
        if bc == WALL and axis != 1:
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

    faces = _face_map(desc)
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
        return BoundaryExecution("channel", bcs, callback, tuple(faces.values()),
                                 channel_bc=channel_bc, grid=desc.grid)

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
        return BoundaryExecution("plate", bcs, callback, tuple(faces.values()),
                                 plate=plate_marker, grid=desc.grid,
                                 farfield=farfield)

    if has_special:
        callback = _require_callback(case, "special_boundary 特殊边界")
        special = case["special_boundary"]
        if not isinstance(getattr(special, "name", None), str):
            raise ValueError("special_boundary 需要字符串 name")
        marker_faces = tuple(getattr(special, "faces", ()))
        validate_special_sides(marker_faces, bcs)
        if marker_faces != tuple(faces.values()):
            raise ValueError("special_boundary.faces 与已构造的六侧描述不一致")
        mixed_outflow = tuple(f for f in marker_faces if f.executor == "halo" and
                              bcs[f.axis] in (INFLOW, PLATE) and f.kind == OUTFLOW)
        needs_farfield = any(f.kind == FARFIELD for f in marker_faces)
        farfield = _farfield_state(case) if needs_farfield else None
        return BoundaryExecution("custom", bcs, callback, marker_faces,
                                 mixed_outflow=mixed_outflow, grid=desc.grid,
                                 farfield=farfield)

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
    return BoundaryExecution("generic", bcs, None, tuple(faces.values()),
                             grid=desc.grid, farfield=farfield)


def kinds_for(bcs):
    """把轴级 bcs 展开成六侧语义序列（i-,i+,j-,j+,k-,k+）。"""
    return (bcs[0], bcs[0], bcs[1], bcs[1], bcs[2], bcs[2])
