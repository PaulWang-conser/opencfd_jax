"""HDF5/CGNS 单块结构网格读取与执行适配。

分层口径：
- ``read_cgns_zone`` 只负责把文件读成 ``CgnsZone``，保留原始 BCType、PointRange
  和 1to1 连接；``validate=True`` 在返回前检查本步可执行子集，``validate=False``
  只做结构校验，供真实样例诊断使用。
- ``to_block_mesh`` 把可执行子集翻译成 ``BlockMesh``；``build_cgns_block`` 再补
  LAP 鬼胞坐标并计算曲线度量。
- ``read_cgns_zones`` 把一个多 zone 文件的全部 zone 按文件顺序读成 ``CgnsZone``，
  只做结构校验，并把 1to1 连接下 ``Periodic_t`` 的三个数组记进连接。连接是否
  合法由多块路径的拓扑校验负责。``zone_metadata`` 只读 zone 的名字、类型和尺寸。

单块路径只支持三维 Structured zone、顶点坐标（``GridLocation=Vertex``）和整面边界；
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
from .boundary import (FARFIELD, INFLOW, LAP, NONREFLECTING, OUTFLOW, PERIODIC,
                       WALL)
from .fields import Grid
from .mesh import METRIC_KEYS, compute_metrics_from_coords, pad_coords
from .boundary import dif_band
from .mesh_data import BlockMesh, FaceBoundary

_BC_OUTFLOW = ("BCExtrapolate", "BCOutflow", "BCOutflowSubsonic")
# 项目约定：BCOutflowSupersonic 表示原码 NON_REFLETION 口径的无反射出口。
# SIDS 字面是"全部外推"，与此不同；零梯度出流用 outflow 加 outlet_ghost=copy。
_BC_NONREFLECTING = ("BCOutflowSupersonic",)
_BC_WALL = ("BCWall",)
_BC_SYMMETRY = ("BCSymmetryPolar",)
_BC_INFLOW = ("BCInflow",)
_BC_FARFIELD = ("BCFarfield",)

_BC_EXECUTABLE = (_BC_OUTFLOW + _BC_NONREFLECTING + _BC_WALL + _BC_SYMMETRY
                  + _BC_FARFIELD + _BC_INFLOW)


class CgnsError(ValueError):
    """CGNS 结构、索引或执行子集错误；消息带 zone/patch 名。"""


@dataclass(frozen=True)
class CgnsBC:
    """一个 ``ZoneBC_t`` 下的 ``BC_t`` 原始 patch。"""

    name: str
    bc_type: str
    point_range: tuple[tuple[int, int, int], tuple[int, int, int]]
    family: str | None = None
    original_type: str | None = None


@dataclass(frozen=True)
class CgnsZoneMeta:
    """``Zone_t`` 节点的名字、类型和尺寸；不读坐标。

    入口判定用 ``problem``：zone 不是 ``Structured``、或尺寸读不出来时，它记着原因，
    与读取器的报错是同一句话；这时读不出的那几项是 None。
    """

    name: str
    zone_type: str | None
    vertex_size: tuple[int, int, int] | None
    cell_size: tuple[int, int, int] | None
    problem: str | None = None


@dataclass(frozen=True)
class CgnsPeriodic:
    """1to1 连接下 ``Periodic_t`` 的三个数组；文件里缺的记为 None。"""

    rotation_center: np.ndarray | None = None
    rotation_angle: np.ndarray | None = None
    translation: np.ndarray | None = None


@dataclass(frozen=True)
class CgnsConnection:
    """一个 ``GridConnectivity1to1_t``；索引已转为 0-based 闭区间。"""

    name: str
    donor_zone: str
    point_range: tuple[tuple[int, int, int], tuple[int, int, int]]
    point_range_donor: tuple[tuple[int, int, int], tuple[int, int, int]]
    transform: tuple[int, int, int]
    periodic: CgnsPeriodic | None = None


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


def _range(value, name: str, positive: bool = True
           ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """把 SIDS 的 1-based ``IndexRange_t`` 转成 0-based 闭区间。

    ``positive=False`` 时不查下标非正，留给拓扑校验按范围越界报（它要先报
    ``Transform``）；形状不对仍然在这里报。
    """
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != (2, 3):
        raise CgnsError(f"{name} 需要 (2,3) 的 PointRange，得到 {arr.shape}")
    if positive and np.any(arr < 1):
        raise CgnsError(f"{name} 的 PointRange 含非正索引：{arr.tolist()}")
    lo = tuple(int(v) - 1 for v in arr[0])
    hi = tuple(int(v) - 1 for v in arr[1])
    return lo, hi


def check_extent(lo, hi, vertex, name: str) -> None:
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


def face_axes(lo, hi):
    """有向范围是一个面片时返回 (法向轴, 切向轴, 切向轴)，否则返回 None。

    点状、线状、体型的范围都不是面片。面片可以是整面，也可以是整面的一部分。
    """
    flat = [axis for axis in range(3) if lo[axis] == hi[axis]]
    if len(flat) != 1:
        return None
    tangential = [axis for axis in range(3) if axis != flat[0]]
    return flat[0], tangential[0], tangential[1]


def side_of_face(axis, lo, vertex) -> str | None:
    """面片落在块的哪个侧面（``i-`` 等）；落在内部时返回 None。"""
    if lo[axis] == 0:
        return AXIS_NAMES[axis] + "-"
    if lo[axis] == vertex[axis] - 1:
        return AXIS_NAMES[axis] + "+"
    return None


def face_boundary(side: str, kind: str, detail: str) -> FaceBoundary:
    """按 CGNS 的边界种类造一个整面描述。

    通用边界都由鬼层填充执行，边界带按 kind 派生（壁面和远场开带，其余不开）。
    """
    return FaceBoundary(side=side, axis=side_axis(side), kind=kind,
                        executor="halo", dif_band=dif_band(kind, "halo"),
                        detail=detail)


def _check_location(group, name: str, keep_periodic: bool = False) -> None:
    """拒绝网格/BC 上的 ``CellCenter``、``Rind>0`` 和 ``PointList``。

    缺省连 ``Periodic_t`` 一并拒绝；多块读取器打开 ``keep_periodic``，把周期
    信息留给 ``CgnsConnection.periodic`` 记录。
    """
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
    if keep_periodic:
        return
    if "Periodic_t" in group:
        raise CgnsError(f"{name} 使用 Periodic_t；本步只接受整面 identity 1to1")
    # 标准 CGNS 把周期信息挂在 GridConnectivity1to1_t → GridConnectivityProperty_t
    # → Periodic_t，直接子节点查不到；不下钻会漏检带周期的连接（2026-11-23 复审）。
    for prop in (v for v in group.values() if _label(v) == "GridConnectivityProperty_t"):
        if any(_label(v) == "Periodic_t" for v in prop.values()):
            raise CgnsError(
                f"{name} 的 GridConnectivityProperty 下有 Periodic_t；"
                "本步只接受整面 identity 1to1")


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
        original_type = bc_type
        family = None
        if bc_type == "FamilySpecified":
            family_nodes = [v for v in node.values() if _label(v) == "FamilyName_t"]
            if len(family_nodes) != 1:
                raise CgnsError(f"{zone_name}/{name}: FamilySpecified 需要唯一 FamilyName")
            family = _text(family_nodes[0])
            base = zone.parent
            if family not in base or _label(base[family]) != "Family_t":
                raise CgnsError(f"{zone_name}/{name}: 找不到 Family {family!r}")
            family_bcs = [v for v in base[family].values() if _label(v) == "FamilyBC_t"]
            if len(family_bcs) != 1:
                raise CgnsError(f"{zone_name}/{name}: Family {family!r} 需要唯一 FamilyBC")
            bc_type = _text(family_bcs[0])
        lo, hi = _range(_data(node["PointRange"]), f"{zone_name}/{name}")
        check_extent(lo, hi, vertex, f"{zone_name}/{name}")
        bcs.append(CgnsBC(name=name, bc_type=bc_type, point_range=(lo, hi),
                          family=family, original_type=original_type))
    return tuple(bcs)


_PERIODIC_ARRAYS = (("rotation_center", "RotationCenter"),
                    ("rotation_angle", "RotationAngle"),
                    ("translation", "Translation"))


def _read_periodic(node, name: str) -> CgnsPeriodic | None:
    """读 1to1 连接下 ``GridConnectivityProperty_t/Periodic_t`` 的三个数组。

    没有这个节点时返回 None；节点在而某个数组缺时，该数组记为 None。
    """
    if "Periodic_t" in node:
        raise CgnsError(
            f"{name} 的 Periodic_t 直接挂在 GridConnectivity1to1_t 下；"
            "标准位置是 GridConnectivityProperty_t 之下")
    fields = {}
    found = False
    for prop in (v for v in node.values()
                 if _label(v) == "GridConnectivityProperty_t"):
        for per in (v for v in prop.values() if _label(v) == "Periodic_t"):
            found = True
            for field, array in _PERIODIC_ARRAYS:
                if array in per:
                    try:
                        fields[field] = np.asarray(_data(per[array]),
                                                   dtype=np.float64)
                    except (TypeError, ValueError) as exc:
                        raise CgnsError(
                            f"{name} 的 Periodic_t/{array} 不是数值数组：{exc}"
                        ) from exc
    return CgnsPeriodic(**fields) if found else None


def _read_connections(zone, vertex, zone_name: str,
                      keep_periodic: bool = False) -> tuple[CgnsConnection, ...]:
    """读取 ``ZoneGridConnectivity_t`` 下所有 1to1 连接。

    ``keep_periodic=True`` 供多块读取器使用：``Periodic_t`` 读进连接记录，留给
    拓扑校验。供体范围在两种情况下都不在这里验界：供体可以是别的块，它的尺寸
    这里不知道，由拓扑校验去比。多块读取时接收范围也不在这里验界，下标非正也不在
    这里查，都留给拓扑校验，让它排在 ``Transform`` 之后报（方案 KTD3）。
    """
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
        _check_location(node, f"{zone_name}/{name}", keep_periodic=keep_periodic)
        if "PointRange" not in node or "PointRangeDonor" not in node:
            raise CgnsError(f"{zone_name}/{name} 缺少 PointRange/PointRangeDonor")
        donor = _text(node)
        strict = not keep_periodic                 # 多块读取：下标非正也留给拓扑校验
        lo, hi = _range(_data(node["PointRange"]), f"{zone_name}/{name}", positive=strict)
        dlo, dhi = _range(_data(node["PointRangeDonor"]),
                          f"{zone_name}/{name}/PointRangeDonor", positive=strict)
        if strict:
            check_extent(lo, hi, vertex, f"{zone_name}/{name}")
        transform = (1, 2, 3)
        if "Transform" in node:
            raw = np.asarray(node["Transform"][" data"][()], dtype=np.int64)
            if raw.shape != (3,):
                raise CgnsError(
                    f"{zone_name}/{name} 的 Transform 需要 3 个整数，得到 {raw.shape}")
            transform = tuple(int(v) for v in raw)
        out.append(CgnsConnection(
            name=name, donor_zone=donor.strip(), point_range=(lo, hi),
            point_range_donor=(dlo, dhi), transform=transform,
            periodic=(_read_periodic(node, f"{zone_name}/{name}")
                      if keep_periodic else None)))
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


def _open(path):
    """以只读方式打开 CGNS/HDF5 文件，由调用方关闭；打不开转成 ``CgnsError``。"""
    import h5py

    try:
        return h5py.File(str(path), "r")
    except (OSError, ValueError) as exc:
        raise CgnsError(f"无法打开 HDF5/CGNS 文件 {path}: {exc}") from exc


def _check_base(base) -> None:
    """确认 ``CGNSBase_t`` 是三维的。"""
    dims = np.asarray(base[" data"][()], dtype=np.int64)
    if dims.shape != (2,) or tuple(int(v) for v in dims) != (3, 3):
        raise CgnsError(
            f"{base.name} 的 CellDimension/PhysicalDimension={dims.tolist()}；"
            "本步只支持三维")


def _bases(handle, path) -> list:
    """列出 ``CGNSBase_t``，并确认都是三维。"""
    bases = [obj for obj in handle.values() if _label(obj) == "CGNSBase_t"]
    if not bases:
        raise CgnsError(f"{path} 缺少 CGNSBase_t")
    for base in bases:
        _check_base(base)
    return bases


def _structured_problem(zone_name: str, zone_type: str) -> str:
    """zone 不是 ``Structured`` 时的报错句子；读取器和入口判定共用。"""
    return f"{zone_name} 的 ZoneType={zone_type!r}；只支持 Structured"


def _zone_header(node) -> CgnsZoneMeta:
    """读 ``Zone_t`` 的名字、尺寸和类型。"""
    name = _zone_name(node)
    vertex, cell = _decode_zone_data(_data(node), name)
    if "ZoneType" not in node:
        raise CgnsError(f"{name} 缺少 ZoneType")
    return CgnsZoneMeta(name=name, zone_type=_text(node["ZoneType"]),
                        vertex_size=vertex, cell_size=cell)


def _zone_header_or_problem(node) -> CgnsZoneMeta:
    """入口判定用的宽松版：读不出尺寸或类型、不是 ``Structured`` 时不报错，
    把原因记进 ``problem``。

    节点残缺的方式很多（数据集被删、写成了组、引用了不存在的外部文件），逐种枚举
    异常类型堵不住，所以这里一律接住。本函数只为入口选路径，不许它改变单块路径的
    报错；严格读取走 :func:`_zone_header`，与以前相同。
    """
    name = _zone_name(node)
    try:
        zone_type = _text(node["ZoneType"]) if "ZoneType" in node else None
        if zone_type is not None and zone_type != "Structured":
            return CgnsZoneMeta(name=name, zone_type=zone_type, vertex_size=None,
                                cell_size=None,
                                problem=_structured_problem(name, zone_type))
        return _zone_header(node)
    except CgnsError as exc:
        reason = str(exc)                        # 读取器的消息里已经带着 zone 名
    except Exception as exc:                     # 其余残缺：带上 zone 名和原异常
        reason = f"{name} 的 Zone_t 读不出来：{type(exc).__name__}: {exc}"
    return CgnsZoneMeta(name=name, zone_type=None, vertex_size=None,
                        cell_size=None, problem=reason)


def _read_zone(node, keep_periodic: bool = False) -> CgnsZone:
    """读一个 ``Zone_t``：尺寸、类型、坐标、边条和 1to1 连接；不做可执行校验。"""
    header = _zone_header(node)
    if header.zone_type != "Structured":
        raise CgnsError(_structured_problem(header.name, header.zone_type))
    return CgnsZone(
        name=header.name, vertex_size=header.vertex_size,
        cell_size=header.cell_size,
        coords=_read_coords(node, header.vertex_size, header.name),
        bcs=_read_bcs(node, header.vertex_size, header.name),
        connections=_read_connections(node, header.vertex_size, header.name,
                                      keep_periodic=keep_periodic))


def read_cgns_zone(path, zone: str | None = None,
                   validate: bool = True) -> CgnsZone:
    """读取一个 HDF5/CGNS 结构化 zone。

    ``zone`` 为空且文件只有一个 zone 时自动选择；多 zone 文件必须显式给名字。
    ``validate=False`` 只做结构校验，保留所有原始 BCType 和 1to1，供真实样例
    诊断；默认 ``True`` 会在返回前拒绝本步执行子集之外的内容。
    """
    with _open(path) as handle:
        bases = [obj for obj in handle.values() if _label(obj) == "CGNSBase_t"]
        if not bases:
            raise CgnsError(f"{path} 缺少 CGNSBase_t")
        zones = []
        for base in bases:
            # 逐个 Base 先验维数、再列出它的 zone。次序与第六步之前相同：几个 Base 各
            # 坏一处时，先报排在前面那个的错（2026-11-24，Astra 第 20 轮）。
            _check_base(base)
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

        result = _read_zone(selected)
        if validate:
            _validate_executable(result)
        return result


def read_cgns_zones(path) -> tuple[CgnsZone, ...]:
    """读取一个多 zone 文件的全部原始 zone，按文件顺序返回。

    文件只能有一个 ``CGNSBase_t``，所有 zone 都必须是 Structured。只做结构校验：
    1to1 连接下 ``Periodic_t`` 的三个数组记进 ``CgnsConnection.periodic``，供体
    范围不在这里验界；连接是否配对、范围和坐标是否对得上，由多块路径的拓扑
    校验负责。

    只抛 ``CgnsError``。节点残缺的方式枚举不完，所以这里出的其他异常都转成
    ``CgnsError``，带上 zone 名（能知道是哪个 zone 时）和原始异常。Base 下打不开的
    链接（悬空的软链接、缺失的外部文件）点名报错，不跳过：一个 zone 自己就是打不开
    的链接时，跳过等于悄悄少算一块。
    """
    try:
        with _open(path) as handle:
            return _zones_under(handle, path)
    except CgnsError:
        raise
    except Exception as exc:       # 列 Base 这一步出的其他异常：Base 不是组、根下有悬空链接
        raise CgnsError(f"{path} 读不出来：{type(exc).__name__}: {exc}") from exc


def _zones_under(handle, path) -> tuple[CgnsZone, ...]:
    """文件只能有一个 Base；逐个 zone 读取。"""
    bases = _bases(handle, path)
    if len(bases) != 1:
        names = [base.name.rsplit("/", 1)[-1] for base in bases]
        raise CgnsError(
            f"{path} 有 {len(bases)} 个 CGNSBase_t：{names}；只支持单个 Base")
    zones = []
    for name in bases[0]:
        try:
            node = bases[0][name]
        except Exception as exc:
            # 悬空的软链接、指向缺失文件的外部链接。不跳过：少读一个 zone 会悄悄
            # 算错，宁可点名让它修文件。
            raise CgnsError(
                f"{path} 的 Base 下有打不开的链接 {name!r}：{type(exc).__name__}: {exc}"
            ) from exc
        if _label(node) != "Zone_t":
            continue
        try:
            zones.append(_read_zone(node, keep_periodic=True))
        except CgnsError:
            raise                          # 读取器自己的消息里已经带着 zone 名
        except Exception as exc:           # 节点残缺的其余情形：在这里补上 zone 名
            raise CgnsError(
                f"{_zone_name(node)} 读不出来：{type(exc).__name__}: {exc}") from exc
    if not zones:
        raise CgnsError(f"{path} 没有 Zone_t")
    return tuple(zones)


def _openable(group):
    """``group`` 里能打开的子节点；悬空的链接（指向不存在的节点或文件）跳过。"""
    for name in group:
        try:
            yield group[name]
        except Exception:            # h5py 对悬空的软链接、缺失的外部链接抛 KeyError 或 OSError
            continue


def zone_metadata(path) -> tuple[CgnsZoneMeta, ...]:
    """只读每个 zone 的原名、类型和尺寸，按文件顺序返回；不读坐标。

    只抛 ``CgnsError``，而且只在打不开文件、没有 Base 或 Zone 时抛。zone 的内容有问题
    （类型不是 ``Structured``、节点残缺、尺寸读不出来）时不在这里报错，原因记在
    :attr:`CgnsZoneMeta.problem` 里：文件里只有一个 zone 时入口判定交回单块路径，由
    那条路径报；不止一个时由多块路径报。悬空的链接当作不存在。不检查 Base 的个数
    和维数。
    """
    try:
        with _open(path) as handle:
            bases = [obj for obj in _openable(handle) if _label(obj) == "CGNSBase_t"]
            if not bases:
                raise CgnsError(f"{path} 缺少 CGNSBase_t")
            nodes = [node for base in bases for node in _openable(base)
                     if _label(node) == "Zone_t"]
            if not nodes:
                raise CgnsError(f"{path} 没有 Zone_t")
            return tuple(_zone_header_or_problem(node) for node in nodes)
    except CgnsError:
        raise
    except Exception as exc:       # 其余残缺：标成 Base 的节点不是组、zone 名不是 UTF-8 等
        raise CgnsError(
            f"{path} 的元数据读不出来：{type(exc).__name__}: {exc}") from exc


def _norm_name(name: str) -> str:
    """把 CGNS 名字中的连续空格归一化，便于用户输入 ``Zone 1`` 匹配。"""
    return " ".join(str(name).split())


def _zone_name(node) -> str:
    """去掉 h5py 路径，只保留 zone 名。"""
    return node.name.rsplit("/", 1)[-1].strip()


def _label_zone_name(node) -> str:
    """返回用于错误信息的 zone 名。"""
    return _zone_name(node)


def bc_kind(bc_type: str) -> str | None:
    """把 SIDS BCType 映射到执行 kind；不支持的返回 None。"""
    if bc_type in _BC_OUTFLOW:
        return OUTFLOW
    if bc_type in _BC_NONREFLECTING:
        return NONREFLECTING
    if bc_type in _BC_WALL:
        return WALL
    if bc_type in _BC_SYMMETRY:
        return PERIODIC
    if bc_type in _BC_FARFIELD:
        return FARFIELD
    if bc_type in _BC_INFLOW:
        return INFLOW
    return None


def _collect_sides(zone: CgnsZone) -> dict[str, tuple[str, str, str]]:
    """把 BC 和整面 identity 1to1 收敛成 side -> (kind, source, origin)。

    origin 为 ``bc`` / ``conn``：周期配对要求两侧同为 Polar BC 或同为
    自连接 1to1，混配不满足项目"BCSymmetryPolar 同轴整面成对"的约定。
    """
    vertex = zone.vertex_size
    sides: dict[str, tuple[str, str, str]] = {}

    def put(side: str, kind: str, source: str, origin: str):
        """登记一个 side；同一侧重复来源直接报错。"""
        if side in sides:
            raise CgnsError(
                f"{zone.name}/{side} 同时由 {sides[side][1]} 和 {source} 定义")
        sides[side] = (kind, source, origin)

    for bc in zone.bcs:
        lo, hi = bc.point_range
        side = _side_of(lo, hi, vertex)
        if side is None:
            raise CgnsError(
                f"{zone.name}/{bc.name} 的 PointRange={lo}..{hi} 不是整面；"
                "本步只支持整面边界")
        kind = bc_kind(bc.bc_type)
        if kind is None:
            raise CgnsError(
                f"{zone.name}/{bc.name} 的 BCType={bc.bc_type!r} 不在本步执行子集")
        source = f"{bc.name} (Family={bc.family})" if bc.family else bc.name
        put(side, kind, source, "bc")

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
        put(side, PERIODIC, conn.name, "conn")
    return sides


def _validate_executable(zone: CgnsZone) -> None:
    """校验本步可执行子集；错误信息带 zone/patch 名。

    允许同一轴两侧是不同 kind（例如 ``i-`` 出流、``i+`` 远场），执行层
    按面而不是按轴填充；仍然要求每个轴两侧都有整面 patch。
    """
    sides = _collect_sides(zone)
    for axis, name in enumerate(AXIS_NAMES):
        lo = sides.get(name + "-")
        hi = sides.get(name + "+")
        if lo is None or hi is None:
            raise CgnsError(
                f"{zone.name} 的 {name} 轴缺少一侧边界：{name}-={lo}, {name}+={hi}")
        if (lo[0] == PERIODIC) != (hi[0] == PERIODIC):
            raise CgnsError(f"{zone.name} 的 {name} 轴周期边界必须同轴成对：{lo}, {hi}")
        # 周期两侧必须同来源：一面 BCSymmetryPolar、对面 identity 1to1 自连接
        # 的混配曾能通过 kind 配对（2026-11-23 复审反例），不满足 Polar 约定。
        if lo[0] == PERIODIC and hi[0] == PERIODIC and lo[2] != hi[2]:
            raise CgnsError(
                f"{zone.name} 的 {name} 轴周期两侧来源不同（{lo[2]}/{hi[2]}）；"
                "BCSymmetryPolar 必须同轴正负整面成对，不与 1to1 自连接混配")
        if lo[0] == WALL and axis != 1:
            raise CgnsError(f"{zone.name} 的壁面只支持 j 方向，得到 {name} 轴")
        if hi[0] == WALL and axis != 1:
            raise CgnsError(f"{zone.name} 的壁面只支持 j 方向，得到 {name} 轴")


def to_block_mesh(zone: CgnsZone, axes: Iterable[int] = (0, 1, 2),
                  dif_bound=None) -> BlockMesh:
    """把可执行 CGNS 子集翻译成 ``BlockMesh``。

    每侧一个整面 patch；同一轴两侧可以是不同 kind（例如一侧出流、一侧远场），
    执行层按 ``faces`` 逐面填充。``bcs`` 仍保留轴级代表值（取低侧面），供旧
    调用方读取；新执行链以六侧 ``faces`` 为准。壁面仍然只在 j 方向。
    """
    _validate_executable(zone)
    sides = _collect_sides(zone)
    axes = tuple(int(a) for a in axes)
    if not axes or any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"axes 必须是 0/1/2 的非空子集，得到 {axes}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes 不能重复，得到 {axes}")

    # 轴级 bcs 只是代表值；真正的执行语义在六侧 faces。
    bcs = tuple(sides[AXIS_NAMES[a] + "-"][0] for a in range(3))
    # DIF_BOUND 由边条 kind 派生（4.9.13 口径：CGNS 边条驱动，WALL/FARFIELD
    # 开带；plate 的"远场关带"由 cases 的 plate-CGNS 分支按面覆盖）
    faces = []
    for side in SIDES:
        faces.append(face_boundary(side, sides[side][0], f"CGNS {sides[side][1]}"))
    grid = Grid(ni=zone.ni, nj=zone.nj, nk=zone.nk)
    return BlockMesh(
        grid=grid, dxs=(1.0, 1.0, 1.0), axes=axes, bcs=bcs,
        dif_bound=dif_bound, dif_axis_map=(0, 1, 2), metrics=None,
        coords=zone.coords, faces=tuple(faces))


def build_cgns_block(zone: CgnsZone, axes: Iterable[int] = (0, 1, 2),
                     dif_bound=None) -> BlockMesh:
    """在 ``to_block_mesh`` 基础上补鬼胞坐标并计算曲线度量。"""
    block = to_block_mesh(zone, axes=axes, dif_bound=dif_bound)
    periodic = tuple(
        any(f.kind == PERIODIC for f in block.faces if f.axis == axis)
        for axis in range(3))
    padded = {key: pad_coords(value, periodic=periodic)
              for key, value in block.coords.items()}
    metrics = compute_metrics_from_coords(
        padded, dxs=block.dxs, order=6, jac_bound=(1, 1, 1), periodic=periodic)
    return BlockMesh(
        grid=block.grid, dxs=block.dxs, axes=block.axes, bcs=block.bcs,
        dif_bound=block.dif_bound, dif_axis_map=block.dif_axis_map,
        metrics=MappingProxyType({k: metrics[k] for k in METRIC_KEYS}),
        coords=block.coords, faces=block.faces)
