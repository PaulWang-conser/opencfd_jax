"""HDF5/CGNS 单块结构网格读取与执行适配。

分层口径：
- ``read_cgns_zone`` 只负责把文件读成 ``CgnsZone``，保留原始 BCType、PointRange
  和 1to1 连接；``validate=True`` 在返回前检查本步可执行子集，``validate=False``
  只做结构校验，供真实样例诊断使用。
- ``to_block_mesh`` 把可执行子集翻译成 ``BlockMesh``；``build_cgns_block`` 再补
  LAP 鬼胞坐标并计算曲线度量。

当前只支持三维 Structured zone、顶点坐标（``GridLocation=Vertex``）和整面边界；
子区、翻转/轴交换 1to1、``Periodic_t``、``CellCenter``、``Rind``、``PointList``
在 reader 或 adapter 阶段明确拒绝。可执行 BCType 包含远场 ``BCFarfield``；
远场来流状态不来自 CGNS 文件（本步不解析 ``BCDataSet``），由算例侧提供。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .axes import AXIS_NAMES, SIDES, normalize_side, side_axis
from .boundary import FARFIELD, LAP, OUTFLOW, PERIODIC, WALL
from .fields import Grid
from .mesh import METRIC_KEYS, compute_metrics_from_coords, pad_coords
from .mesh_data import BlockMesh, FaceBoundary

_BC_OUTFLOW = ("BCExtrapolate", "BCOutflow", "BCOutflowSubsonic",
               "BCOutflowSupersonic")
_BC_WALL = ("BCWall",)
_BC_SYMMETRY = ("BCSymmetryPlane",)
_BC_FARFIELD = ("BCFarfield",)

_BC_EXECUTABLE = _BC_OUTFLOW + _BC_WALL + _BC_SYMMETRY + _BC_FARFIELD


class CgnsError(ValueError):
    """CGNS 结构、索引或执行子集错误；消息带 zone/patch 名。"""


@dataclass(frozen=True)
class CgnsBC:
    """一个 ``ZoneBC_t`` 下的 ``BC_t`` 原始 patch。"""

    name: str
    bc_type: str
    point_range: tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class CgnsConnection:
    """一个 ``GridConnectivity1to1_t``；索引已转为 0-based 闭区间。"""

    name: str
    donor_zone: str
    point_range: tuple[tuple[int, int, int], tuple[int, int, int]]
    point_range_donor: tuple[tuple[int, int, int], tuple[int, int, int]]
    transform: tuple[int, int, int]


@dataclass(frozen=True)
class CgnsZone:
    """一个结构化 CGNS zone 的原始读取结果。

    ``vertex_size`` 保持 CGNS 的 ``(ni,nj,nk)`` 顺序；``coords`` 的数组按
    ``(nk,nj,ni)`` 存放，与仓库现有计算轴序一致。
    """

    name: str
    vertex_size: tuple[int, int, int]
    cell_size: tuple[int, int, int]
    coords: Mapping[str, np.ndarray]
    bcs: tuple[CgnsBC, ...]
    connections: tuple[CgnsConnection, ...]

    @property
    def ni(self) -> int:
        """计算 i 方向离散点数。"""
        return self.vertex_size[0]

    @property
    def nj(self) -> int:
        """计算 j 方向离散点数。"""
        return self.vertex_size[1]

    @property
    def nk(self) -> int:
        """计算 k 方向离散点数。"""
        return self.vertex_size[2]


def _label(obj) -> str:
    """读 h5py 节点的 ``label`` 属性并统一成字符串。"""
    raw = obj.attrs.get("label")
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _text(group) -> str:
    """读 CGNS 字符串节点（``data`` 可能是 int8/bytes/str）。"""
    data = group[" data"][()]
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace").strip()
    arr = np.asarray(data)
    if arr.dtype.kind in ("S", "U"):
        return str(arr.item() if arr.size == 1 else arr.tobytes()).strip()
    if arr.dtype.kind in ("i", "u"):
        return bytes(arr.astype(np.int8).ravel()).decode(
            "utf-8", errors="replace").strip()
    raise CgnsError(f"无法把 {group.name} 的 data 解码成字符串：{data!r}")


def _data(group) -> np.ndarray:
    """读节点下的 `` data`` dataset。"""
    if " data" not in group:
        raise CgnsError(f"{group.name} 缺少 ' data' dataset")
    return np.asarray(group[" data"][()])


def _range(value, name: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """把 SIDS 的 1-based ``IndexRange_t`` 转成 0-based 闭区间。"""
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != (2, 3):
        raise CgnsError(f"{name} 需要 (2,3) 的 PointRange，得到 {arr.shape}")
    if np.any(arr < 1):
        raise CgnsError(f"{name} 的 PointRange 含非正索引：{arr.tolist()}")
    lo = tuple(int(v) - 1 for v in arr[0])
    hi = tuple(int(v) - 1 for v in arr[1])
    return lo, hi


def _check_extent(lo, hi, vertex, name: str) -> None:
    """确认 PointRange 落在 ``VertexSize`` 内；允许 donor 范围倒序。"""
    for axis, (a, b, n) in enumerate(zip(lo, hi, vertex)):
        amin, amax = min(a, b), max(a, b)
        if amax >= n or amin < 0:
            raise CgnsError(
                f"{name} 的 PointRange {lo}..{hi} 超出 {AXIS_NAMES[axis]} 顶点数 {n}")


def _side_of(lo, hi, vertex) -> str | None:
    """判断一个 PointRange 是否恰好覆盖整面；返回 ``i-`` 等，否则 None。"""
    lo = tuple(min(a, b) for a, b in zip(lo, hi))
    hi = tuple(max(a, b) for a, b in zip(lo, hi))
    degenerate = [axis for axis in range(3) if lo[axis] == hi[axis]]
    if len(degenerate) != 1:
        return None
    axis = degenerate[0]
    for other in range(3):
        if other != axis and (lo[other] != 0 or hi[other] != vertex[other] - 1):
            return None
    if lo[axis] == 0:
        return AXIS_NAMES[axis] + "-"
    if lo[axis] == vertex[axis] - 1:
        return AXIS_NAMES[axis] + "+"
    return None


def _check_location(group, name: str) -> None:
    """拒绝网格/BC 上的 ``CellCenter``、``Rind>0`` 和 ``PointList``。"""
    if "GridLocation" in group:
        loc = _text(group["GridLocation"])
        if loc != "Vertex":
            raise CgnsError(
                f"{name} 的 GridLocation={loc!r}；本步只接受 Vertex")
    if "Rind" in group:
        rind = np.asarray(group["Rind"][" data"][()], dtype=np.int64)
        if np.any(rind > 0):
            raise CgnsError(f"{name} 的 Rind={rind.tolist()}；本步不接受鬼胞数据")
    if "PointList" in group:
        raise CgnsError(f"{name} 使用 PointList；本步只接受 PointRange")
    if "Periodic_t" in group:
        raise CgnsError(f"{name} 使用 Periodic_t；本步只接受整面 identity 1to1")


def _zone_nodes(base):
    """列出 ``CGNSBase_t`` 下的 ``Zone_t`` 节点。"""
    return [obj for obj in base.values() if _label(obj) == "Zone_t"]


def _decode_zone_data(arr: np.ndarray, zone_name: str):
    """解析 ``Zone_t`` 的 ``[VertexSize, CellSize, VertexSizeBoundary]``。"""
    arr = np.asarray(arr, dtype=np.int64)
    if arr.shape != (3, 3):
        raise CgnsError(
            f"{zone_name} 的 Zone_t data 需要 (3,3)，得到 {arr.shape}")
    vertex = tuple(int(v) for v in arr[0])
    cell = tuple(int(v) for v in arr[1])
    if any(v <= 1 for v in vertex):
        raise CgnsError(f"{zone_name} 的 VertexSize={vertex}；本步只支持三维 zone")
    if tuple(c + 1 for c in cell) != vertex:
        raise CgnsError(
            f"{zone_name} 的 CellSize={cell} 与 VertexSize={vertex} 不满足 +1")
    return vertex, cell


def _read_bcs(zone, vertex, zone_name: str) -> tuple[CgnsBC, ...]:
    """读取 ``ZoneBC_t`` 下所有 BC 原始 patch。"""
    if "ZoneBC" not in zone:
        return ()
    bcs = []
    for node in zone["ZoneBC"].values():
        if _label(node) != "BC_t":
            continue
        name = node.name.rsplit("/", 1)[-1]
        _check_location(node, f"{zone_name}/{name}")
        if "PointRange" not in node:
            raise CgnsError(f"{zone_name}/{name} 缺少 PointRange")
        bc_type = _text(node)
        lo, hi = _range(_data(node["PointRange"]), f"{zone_name}/{name}")
        _check_extent(lo, hi, vertex, f"{zone_name}/{name}")
        bcs.append(CgnsBC(name=name, bc_type=bc_type, point_range=(lo, hi)))
    return tuple(bcs)


def _read_connections(zone, vertex, zone_name: str) -> tuple[CgnsConnection, ...]:
    """读取 ``ZoneGridConnectivity_t`` 下所有 1to1 连接。"""
    if "ZoneGridConnectivity" not in zone:
        return ()
    out = []
    for node in zone["ZoneGridConnectivity"].values():
        if _label(node) == "GridConnectivity_t":
            raise CgnsError(
                f"{zone_name}/{node.name.rsplit('/', 1)[-1]} 是一般连接；"
                "本步只支持 GridConnectivity1to1_t")
        if _label(node) != "GridConnectivity1to1_t":
            continue
        name = node.name.rsplit("/", 1)[-1]
        _check_location(node, f"{zone_name}/{name}")
        if "PointRange" not in node or "PointRangeDonor" not in node:
            raise CgnsError(f"{zone_name}/{name} 缺少 PointRange/PointRangeDonor")
        donor = _text(node)
        lo, hi = _range(_data(node["PointRange"]), f"{zone_name}/{name}")
        dlo, dhi = _range(_data(node["PointRangeDonor"]),
                          f"{zone_name}/{name}/PointRangeDonor")
        _check_extent(lo, hi, vertex, f"{zone_name}/{name}")
        transform = (1, 2, 3)
        if "Transform" in node:
            raw = np.asarray(node["Transform"][" data"][()], dtype=np.int64)
            if raw.shape != (3,):
                raise CgnsError(
                    f"{zone_name}/{name} 的 Transform 需要 3 个整数，得到 {raw.shape}")
            transform = tuple(int(v) for v in raw)
        out.append(CgnsConnection(
            name=name, donor_zone=donor.strip(), point_range=(lo, hi),
            point_range_donor=(dlo, dhi), transform=transform))
    return tuple(out)


def _read_coords(zone, vertex, zone_name: str) -> Mapping[str, np.ndarray]:
    """读取 ``CoordinateX/Y/Z`` 并转成 ``(nk,nj,ni)``。"""
    if "GridCoordinates" not in zone:
        raise CgnsError(f"{zone_name} 缺少 GridCoordinates")
    coord = zone["GridCoordinates"]
    _check_location(coord, f"{zone_name}/GridCoordinates")
    arrays = {}
    for key, name in (("Axx", "CoordinateX"), ("Ayy", "CoordinateY"),
                      ("Azz", "CoordinateZ")):
        if name not in coord:
            raise CgnsError(f"{zone_name}/GridCoordinates 缺少 {name}")
        arr = _data(coord[name])
        expect = (vertex[2], vertex[1], vertex[0])
        if arr.shape != expect:
            raise CgnsError(
                f"{zone_name}/{name} shape={arr.shape}，期望 (nk,nj,ni)={expect}")
        arrays[key] = np.asarray(arr, dtype=np.float64)
        arrays[key].setflags(write=False)
    return MappingProxyType(arrays)


def read_cgns_zone(path, zone: str | None = None,
                   validate: bool = True) -> CgnsZone:
    """读取一个 HDF5/CGNS 结构化 zone。

    ``zone`` 为空且文件只有一个 zone 时自动选择；多 zone 文件必须显式给名字。
    ``validate=False`` 只做结构校验，保留所有原始 BCType 和 1to1，供真实样例
    诊断；默认 ``True`` 会在返回前拒绝本步执行子集之外的内容。
    """
    import h5py

    try:
        handle = h5py.File(str(path), "r")
    except (OSError, ValueError) as exc:
        raise CgnsError(f"无法打开 HDF5/CGNS 文件 {path}: {exc}") from exc
    with handle:
        bases = [obj for obj in handle.values() if _label(obj) == "CGNSBase_t"]
        if not bases:
            raise CgnsError(f"{path} 缺少 CGNSBase_t")
        zones = []
        for base in bases:
            dims = np.asarray(base[" data"][()], dtype=np.int64)
            if dims.shape != (2,) or tuple(int(v) for v in dims) != (3, 3):
                raise CgnsError(
                    f"{base.name} 的 CellDimension/PhysicalDimension={dims.tolist()}；"
                    "本步只支持三维")
            zones.extend(_zone_nodes(base))
        if not zones:
            raise CgnsError(f"{path} 没有 Zone_t")
        if zone is None:
            if len(zones) != 1:
                names = [_label_zone_name(z) for z in zones]
                raise CgnsError(
                    f"{path} 有 {len(zones)} 个 zone：{names}；请显式给 zone 名")
            selected = zones[0]
        else:
            wanted = _norm_name(zone)
            matches = [z for z in zones if _norm_name(_zone_name(z)) == wanted]
            if len(matches) != 1:
                names = [_zone_name(z) for z in zones]
                raise CgnsError(f"{path} 找不到唯一 zone {zone!r}；现有 {names}")
            selected = matches[0]

        name = _zone_name(selected)
        vertex, cell = _decode_zone_data(_data(selected), name)
        if "ZoneType" not in selected:
            raise CgnsError(f"{name} 缺少 ZoneType")
        zone_type = _text(selected["ZoneType"])
        if zone_type != "Structured":
            raise CgnsError(f"{name} 的 ZoneType={zone_type!r}；只支持 Structured")
        coords = _read_coords(selected, vertex, name)
        bcs = _read_bcs(selected, vertex, name)
        connections = _read_connections(selected, vertex, name)
        result = CgnsZone(name=name, vertex_size=vertex, cell_size=cell,
                          coords=coords, bcs=bcs, connections=connections)
        if validate:
            _validate_executable(result)
        return result


def _norm_name(name: str) -> str:
    """把 CGNS 名字中的连续空格归一化，便于用户输入 ``Zone 1`` 匹配。"""
    return " ".join(str(name).split())


def _zone_name(node) -> str:
    """去掉 h5py 路径，只保留 zone 名。"""
    return node.name.rsplit("/", 1)[-1].strip()


def _label_zone_name(node) -> str:
    """返回用于错误信息的 zone 名。"""
    return _zone_name(node)


def _bc_kind(bc_type: str) -> str | None:
    """把 SIDS BCType 映射到执行 kind；不支持的返回 None。"""
    if bc_type in _BC_OUTFLOW:
        return OUTFLOW
    if bc_type in _BC_WALL:
        return WALL
    if bc_type in _BC_SYMMETRY:
        return PERIODIC
    if bc_type in _BC_FARFIELD:
        return FARFIELD
    return None


def _collect_sides(zone: CgnsZone) -> dict[str, tuple[str, str]]:
    """把 BC 和整面 identity 1to1 收敛成 side -> (kind, source)。"""
    vertex = zone.vertex_size
    sides: dict[str, tuple[str, str]] = {}

    def put(side: str, kind: str, source: str):
        """登记一个 side；同一侧重复来源直接报错。"""
        if side in sides:
            raise CgnsError(
                f"{zone.name}/{side} 同时由 {sides[side][1]} 和 {source} 定义")
        sides[side] = (kind, source)

    for bc in zone.bcs:
        lo, hi = bc.point_range
        side = _side_of(lo, hi, vertex)
        if side is None:
            raise CgnsError(
                f"{zone.name}/{bc.name} 的 PointRange={lo}..{hi} 不是整面；"
                "本步只支持整面边界")
        kind = _bc_kind(bc.bc_type)
        if kind is None:
            raise CgnsError(
                f"{zone.name}/{bc.name} 的 BCType={bc.bc_type!r} 不在本步执行子集")
        put(side, kind, bc.name)

    for conn in zone.connections:
        if conn.donor_zone != zone.name:
            raise CgnsError(
                f"{zone.name}/{conn.name} 的 donor={conn.donor_zone!r}；"
                "本步只支持同块自连接周期")
        if conn.transform != (1, 2, 3):
            raise CgnsError(
                f"{zone.name}/{conn.name} 的 Transform={conn.transform}；"
                "本步只支持 identity")
        lo, hi = conn.point_range
        side = _side_of(lo, hi, vertex)
        if side is None:
            raise CgnsError(
                f"{zone.name}/{conn.name} 的 PointRange={lo}..{hi} 不是整面；"
                "本步只支持整面 1to1")
        donor_side = _side_of(conn.point_range_donor[0],
                              conn.point_range_donor[1], vertex)
        if donor_side is None:
            raise CgnsError(
                f"{zone.name}/{conn.name} 的 PointRangeDonor 不是整面")
        if side[0] != donor_side[0] or side[1] == donor_side[1]:
            raise CgnsError(
                f"{zone.name}/{conn.name} 的 side={side} donor={donor_side}；"
                "不是同一轴的对侧整面")
        put(side, PERIODIC, conn.name)
    return sides


def _validate_executable(zone: CgnsZone) -> None:
    """校验本步可执行子集；错误信息带 zone/patch 名。"""
    sides = _collect_sides(zone)
    for axis, name in enumerate(AXIS_NAMES):
        lo = sides.get(name + "-")
        hi = sides.get(name + "+")
        if lo is None or hi is None:
            raise CgnsError(
                f"{zone.name} 的 {name} 轴缺少一侧边界：{name}-={lo}, {name}+={hi}")
        if lo[0] != hi[0]:
            raise CgnsError(
                f"{zone.name} 的 {name} 轴两侧 kind 不同：{lo[0]} / {hi[0]}")
        if lo[0] == WALL and axis != 1:
            raise CgnsError(f"{zone.name} 的壁面只支持 j 方向，得到 {name} 轴")


def to_block_mesh(zone: CgnsZone, axes: Iterable[int] = (0, 1, 2),
                  dif_bound=None) -> BlockMesh:
    """把可执行 CGNS 子集翻译成 ``BlockMesh``。

    只接受每侧一个整面 patch、同一轴两侧 kind 相同、壁面只在 j 方向。
    ``bcs`` 顺序固定为计算方向 ``(i,j,k)``；六侧执行来源统一为 ``halo``。
    """
    _validate_executable(zone)
    sides = _collect_sides(zone)
    axes = tuple(int(a) for a in axes)
    if not axes or any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"axes 必须是 0/1/2 的非空子集，得到 {axes}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes 不能重复，得到 {axes}")

    bcs = tuple(sides[AXIS_NAMES[a] + "-"][0] for a in range(3))
    faces = []
    for side in SIDES:
        kind = sides[side][0]
        faces.append(FaceBoundary(
            side=side, axis=side_axis(side), kind=kind, executor="halo",
            dif_band=False, detail=f"CGNS {sides[side][1]}"))
    grid = Grid(ni=zone.ni, nj=zone.nj, nk=zone.nk)
    return BlockMesh(
        grid=grid, dxs=(1.0, 1.0, 1.0), axes=axes, bcs=bcs,
        dif_bound=dif_bound, dif_axis_map=(0, 1, 2), metrics=None,
        coords=zone.coords, faces=tuple(faces))


def build_cgns_block(zone: CgnsZone, axes: Iterable[int] = (0, 1, 2),
                     dif_bound=None) -> BlockMesh:
    """在 ``to_block_mesh`` 基础上补鬼胞坐标并计算曲线度量。"""
    block = to_block_mesh(zone, axes=axes, dif_bound=dif_bound)
    periodic = tuple(kind == PERIODIC for kind in block.bcs)
    padded = {key: pad_coords(value, periodic=periodic)
              for key, value in block.coords.items()}
    metrics = compute_metrics_from_coords(
        padded, dxs=block.dxs, order=6, jac_bound=(1, 1, 1), periodic=periodic)
    return BlockMesh(
        grid=block.grid, dxs=block.dxs, axes=block.axes, bcs=block.bcs,
        dif_bound=block.dif_bound, dif_axis_map=block.dif_axis_map,
        metrics=MappingProxyType({k: metrics[k] for k in METRIC_KEYS}),
        coords=block.coords, faces=block.faces)
