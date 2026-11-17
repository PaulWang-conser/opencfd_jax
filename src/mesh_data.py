"""单块网格与边界描述（第 1 步）。

过渡期约定（见 ``docs/plans/2026-11-17-step1-mesh-boundary-requirements.md``）：
现有算例字典仍是唯一可变来源；本模块只在消费者入口由 ``from_case()`` 即时
构造一个只读快照，不与旧键并列保存，也不驱动新的边界实现。

描述表达的是**现状**：
- ``grid`` / ``dxs`` / ``axes`` / ``bcs`` 复用算例里的既有对象和值；
- 六个面侧只说明该侧当前由哪段既有代码执行（鬼胞填充或算例回调），
  不声称替代壁面状态更新、鬼胞填充或粘性通量处理；
- ``metrics`` / ``coords`` 只在算例真的提供时引用，没有就保持 ``None``。
"""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .fields import INFLOW, LAP, OUTFLOW, PERIODIC, PLATE, WALL, Grid
from .mesh import METRIC_KEYS

AXIS_NAMES = ("x", "y", "z")
SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")

# 描述的边界种类；farfield 只用于平板顶面，原码没有同名枚举。
FARFIELD = "farfield"

_EXEC_HALO = "halo"
_EXEC_CALLBACK = "callback"
_EXEC_CALLBACK_HALO = "callback+halo"

_VALID_KINDS = (PERIODIC, OUTFLOW, WALL, INFLOW, PLATE, FARFIELD)
_COORD_KEYS = ("Axx", "Ayy", "Azz")


@dataclass(frozen=True)
class FaceBoundary:
    """单块的一个面侧。

    side: ``"x-"`` 等六个方向侧。
    axis: 计算方向 0/1/2（x/y/z）。
    kind: 该侧语义（periodic/outflow/wall/inflow/plate/farfield）。
    executor: 当前执行来源，``halo`` = 既有鬼胞填充，``callback`` = 算例
        ``bc_apply``，``callback+halo`` = 两者都参与。
    dif_band: 该侧是否启用原码 DIF_BOUND 面通量/导数边界带。
    detail: 给人看的说明，不参与执行。
    """

    side: str
    axis: int
    kind: str
    executor: str
    dif_band: bool
    detail: str = ""

    def __post_init__(self):
        """拒绝拼错的面名、轴号和来源，避免描述悄悄失真。"""
        if self.side not in SIDES:
            raise ValueError(f"side 必须是 {SIDES} 之一，得到 {self.side!r}")
        if self.side[0] != AXIS_NAMES[self.axis]:
            raise ValueError(f"side={self.side} 与 axis={self.axis} 不一致")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind 必须是 {_VALID_KINDS} 之一，得到 {self.kind!r}")
        if self.executor not in (_EXEC_HALO, _EXEC_CALLBACK, _EXEC_CALLBACK_HALO):
            raise ValueError(f"未知 executor: {self.executor!r}")


@dataclass(frozen=True)
class BlockMesh:
    """一个块的网格、步长、方向与六侧边界描述（只读快照）。"""

    grid: Grid
    dxs: Tuple[float, float, float]
    axes: Tuple[int, ...]
    bcs: Tuple[str, str, str]
    dif_bound: Optional[Tuple[int, ...]]
    dif_axis_map: Tuple[int, int, int]
    metrics: Optional[Mapping]
    coords: Optional[Mapping]
    faces: Tuple[FaceBoundary, ...]

    def __post_init__(self):
        """确认六侧齐全且顺序固定，方便按名字查找。"""
        if len(self.faces) != 6 or tuple(f.side for f in self.faces) != SIDES:
            raise ValueError("faces 必须按 x-/x+/y-/y+/z-/z+ 顺序给出六侧")

    def face(self, side: str) -> FaceBoundary:
        """按 ``"x-"`` 等面名取单侧面描述。"""
        for item in self.faces:
            if item.side == side:
                return item
        raise KeyError(side)

    @property
    def inner_shape(self) -> Tuple[int, int, int]:
        """内区数组形状 ``(nz, ny, nx)``。"""
        return (self.grid.nz, self.grid.ny, self.grid.nx)

    @property
    def has_metrics(self) -> bool:
        """是否真的带曲线度量（而不是直角假设）。"""
        return self.metrics is not None

    @property
    def has_coords(self) -> bool:
        """是否真的保存了内区坐标。"""
        return self.coords is not None


def _as_floats(raw, name, length=3):
    """把算例里的步长读成有限浮点三元组，拒绝长度或类型不对的值。"""
    if raw is None:
        raise ValueError(f"case 缺少 {name}")
    try:
        values = tuple(float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是可转成浮点的长度 3 序列") from exc
    if len(values) != length:
        raise ValueError(f"{name} 需要 {length} 个值，得到 {values}")
    if not all(v == v and abs(v) != float("inf") for v in values):
        raise ValueError(f"{name} 含非有限值: {values}")
    return values


def _as_ints(raw, name, length=None):
    """把整数序列读成元组；给出 length 时检查长度。"""
    try:
        values = tuple(int(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数序列") from exc
    if length is not None and len(values) != length:
        raise ValueError(f"{name} 需要 {length} 个值，得到 {values}")
    return values


def _dif_band(dif_bound, axis, side, dif_axis_map):
    """按原码六标志（i-/i+/j-/j+/k-/k+）和方向映射取该侧开关。"""
    if dif_bound is None:
        return False
    original_axis = dif_axis_map[axis]
    index = 2 * original_axis + (1 if side == "+" else 0)
    return bool(dif_bound[index])


def _face(side, kind, executor, dif_bound, dif_axis_map, detail=""):
    """构造一个面侧描述，统一推导 axis 与 dif_band。"""
    axis = AXIS_NAMES.index(side[0])
    return FaceBoundary(side=side, axis=axis, kind=kind, executor=executor,
                        dif_band=_dif_band(dif_bound, axis, side[1], dif_axis_map),
                        detail=detail)


def _plate_faces(dif_bound, dif_axis_map):
    """平板：x 两侧不对称，y 下壁/上来流，z 周期；全部由 plate.apply 写状态。"""
    return (
        _face("x-", INFLOW, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "入口 Dirichlet 剖面（plate.apply）"),
        _face("x+", OUTFLOW, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "出口零梯度鬼层（plate.apply）"),
        _face("y-", WALL, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "无滑移等温壁 + 镜像鬼胞（plate.apply）"),
        _face("y+", FARFIELD, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "顶部自由来流 Dirichlet（plate.apply）"),
        _face("z-", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("z+", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
    )


def _channel_faces(dif_bound, dif_axis_map):
    """槽道：x/z 周期，y 双壁由 ChannelBC 每子步写状态、fill_wall_y 补镜像。"""
    return (
        _face("x-", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("x+", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("y-", WALL, _EXEC_CALLBACK_HALO, dif_bound, dif_axis_map,
              "下壁无滑移等温壁（ChannelBC.apply + fill_wall_y）"),
        _face("y+", WALL, _EXEC_CALLBACK_HALO, dif_bound, dif_axis_map,
              "上壁无滑移等温壁（ChannelBC.apply + fill_wall_y）"),
        _face("z-", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("z+", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
    )


def _generic_faces(bcs, dif_bound, dif_axis_map):
    """其余内置算例：按轴级 bc 展开到两侧；INFLOW/PLATE 只有专属算例可表达。"""
    faces = []
    for axis, bc in enumerate(bcs):
        for sign in ("-", "+"):
            side = AXIS_NAMES[axis] + sign
            if bc == PERIODIC:
                faces.append(_face(side, PERIODIC, _EXEC_HALO, dif_bound,
                                   dif_axis_map, "周期"))
            elif bc == OUTFLOW:
                faces.append(_face(side, OUTFLOW, _EXEC_HALO, dif_bound,
                                   dif_axis_map, "零梯度出流"))
            elif bc == WALL:
                faces.append(_face(side, WALL, _EXEC_HALO, dif_bound,
                                   dif_axis_map, "壁面镜像鬼胞"))
            else:
                raise ValueError(
                    f"{side} 的轴级边界 {bc!r} 不能由通用单块描述准确表示；"
                    "只有 plate/channel 适配器能表达 INFLOW/PLATE")
    return tuple(faces)


def _wrap_fields(raw, name, keys, shape, shape_label):
    """校验并只读包装一组场；没有真实场时返回 None，不构造虚拟值。"""
    if raw is None:
        return None
    if not isinstance(raw, MappingABC):
        raise ValueError(f"{name} 必须是映射")
    missing = [k for k in keys if k not in raw]
    if missing:
        raise ValueError(f"{name} 缺少键: {missing}")
    for key in keys:
        if tuple(raw[key].shape) != shape:
            raise ValueError(
                f"{name}[{key}] 形状 {raw[key].shape} != {shape_label} {shape}")
    return MappingProxyType({k: raw[k] for k in keys})


def from_case(case: Mapping) -> BlockMesh:
    """由现有算例字典即时构造只读单块描述。

    调用点必须在创建 JIT 闭包之前；闭包建立后不再承诺跟随 case 的后续修改。
    描述不写入 case，避免出现两份可独立修改的配置。
    """
    if not isinstance(case, MappingABC):
        raise ValueError("case 必须是映射（现有算例字典）")

    grid = case.get("grid")
    # 仓库现有的包别名跑法会同时存在 opencfd_jax.fields 与 opencfd_jax.src.fields
    # 两份模块，Grid 类身份不同；这里按结构校验，避免把同一套网格判成非法。
    if grid is None or not all(hasattr(grid, a) for a in
                               ("nx", "ny", "nz", "inner", "alloc")):
        raise ValueError(
            "case['grid'] 必须是 fields.Grid 兼容对象（有 nx/ny/nz/inner/alloc）")

    dxs = _as_floats(case.get("dxs"), "dxs")
    raw_axes = case.get("axes")
    if raw_axes is None:
        raise ValueError("case 缺少 axes")
    axes = _as_ints(raw_axes, "axes")
    if not axes:
        raise ValueError("axes 不能为空")
    if len(set(axes)) != len(axes) or any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"axes 必须是 0/1/2 的无重复子集，得到 {axes}")
    for axis in axes:
        if dxs[axis] <= 0.0:
            raise ValueError(f"求解方向 axis={axis} 的步长必须为正，得到 {dxs[axis]}")

    raw_bcs = case.get("bcs")
    if raw_bcs is None:
        raise ValueError("case 缺少 bcs")
    bcs = tuple(raw_bcs)
    if len(bcs) != 3:
        raise ValueError(f"bcs 需要 3 个值，得到 {bcs}")
    for bc in bcs:
        if bc not in _VALID_KINDS:
            raise ValueError(f"未知边界类型 {bc!r}；可用 {_VALID_KINDS}")

    raw_dif = case.get("dif_bound")
    dif_bound = None
    if raw_dif is not None:
        dif_bound = _as_ints(raw_dif, "dif_bound", 6)
        if any(v not in (0, 1) for v in dif_bound):
            raise ValueError(f"dif_bound 只能是 0/1，得到 {dif_bound}")

    dif_axis_map = _as_ints(case.get("dif_axis_map", (0, 1, 2)),
                            "dif_axis_map", 3)
    if set(dif_axis_map) != {0, 1, 2}:
        raise ValueError(f"dif_axis_map 必须是 0/1/2 的排列，得到 {dif_axis_map}")

    shape = (grid.nz + 2 * LAP, grid.ny + 2 * LAP, grid.nx + 2 * LAP)
    metrics = _wrap_fields(case.get("metrics"), "metrics", METRIC_KEYS,
                           shape, "网格内区含鬼胞")
    coords = _wrap_fields(case.get("coords"), "coords", _COORD_KEYS,
                          (grid.nz, grid.ny, grid.nx), "内区")

    if case.get("plate") is not None:
        faces = _plate_faces(dif_bound, dif_axis_map)
    elif case.get("channel_bc") is not None:
        faces = _channel_faces(dif_bound, dif_axis_map)
    else:
        faces = _generic_faces(bcs, dif_bound, dif_axis_map)

    return BlockMesh(grid=grid, dxs=dxs, axes=axes, bcs=bcs,
                     dif_bound=dif_bound, dif_axis_map=dif_axis_map,
                     metrics=metrics, coords=coords, faces=faces)
