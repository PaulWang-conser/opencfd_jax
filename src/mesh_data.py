"""单块网格与边界描述（第 1 步）。

过渡期约定（见 ``docs/plans/2026-11-17-step1-mesh-boundary-requirements.md``）：
现有算例字典仍是唯一可变来源；本模块只在消费者入口由 ``from_case()`` 即时
构造一个只读快照，不与旧键并列保存。第 2 步起，``boundary.build_execution``
会把这份描述与 case 回调收敛成唯一一份执行配置，``main.py`` / ``compare.py``
都从该配置取轴级 bcs 和回调。

描述表达的是**现状**：
- ``grid`` / ``dxs`` / ``axes`` / ``bcs`` 复用算例里的既有对象和值；
- 六个面侧只说明该侧当前由哪段既有代码执行（鬼胞填充或算例回调），
  不声称替代壁面状态更新、鬼胞填充或粘性通量处理；
- ``metrics`` / ``coords`` 只在算例真的提供时引用，没有就保持 ``None``。
"""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .axes import (AXIS_NAMES, DIMENSION_NAMES, SIDES, grid_extent,
                   normalize_side, side_axis)
from .fields import INFLOW, LAP, OUTFLOW, PERIODIC, PLATE, WALL, Grid
from .boundary import dif_band, validate_special_sides
from .mesh import METRIC_KEYS

# 描述的边界种类；farfield 既用于通用轴级边界，也用于平板顶面回调。
FARFIELD = "farfield"

_EXEC_HALO = "halo"
_EXEC_CALLBACK = "callback"
_EXEC_CALLBACK_HALO = "callback+halo"

_VALID_KINDS = (PERIODIC, OUTFLOW, WALL, INFLOW, PLATE, FARFIELD)
_VALID_THERMAL = (None, "isothermal", "adiabatic")
_COORD_KEYS = ("Axx", "Ayy", "Azz")


@dataclass(frozen=True)
class FaceBoundary:
    """单块的一个面侧。

    side: 规范六侧 ``"i-"``/``"i+"``/``"j-"``/``"j+"``/``"k-"``/``"k+"``。
    axis: 计算方向 0/1/2，即 i/j/k（ξ/η/ζ）。
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
    thermal: Optional[str] = None
    params: Mapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        """拒绝拼错的面名、轴号、来源和热条件，避免描述悄悄失真。"""
        side = normalize_side(self.side)
        if self.axis not in (0, 1, 2):
            raise ValueError(f"axis 必须是 0/1/2，得到 {self.axis!r}")
        if side[0] != AXIS_NAMES[self.axis]:
            raise ValueError(f"side={side} 与 axis={self.axis} 不一致")
        object.__setattr__(self, "side", side)
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind 必须是 {_VALID_KINDS} 之一，得到 {self.kind!r}")
        if self.executor not in (_EXEC_HALO, _EXEC_CALLBACK, _EXEC_CALLBACK_HALO):
            raise ValueError(f"未知 executor: {self.executor!r}")
        if self.thermal not in _VALID_THERMAL:
            raise ValueError(f"未知 thermal: {self.thermal!r}")
        if not isinstance(self.params, MappingABC):
            raise ValueError(f"params 必须是映射，得到 {self.params!r}")
        object.__setattr__(self, "params",
                           MappingProxyType(dict(self.params)))


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
        """确认六侧齐全且按规范顺序排列，方便按名字查找。"""
        if len(self.faces) != 6 or tuple(f.side for f in self.faces) != SIDES:
            raise ValueError(f"faces 必须按 {SIDES} 顺序给出六侧")

    def face(self, side: str) -> FaceBoundary:
        """按规范面名取单侧面描述。"""
        wanted = normalize_side(side)
        for item in self.faces:
            if item.side == wanted:
                return item
        raise KeyError(side)

    @property
    def inner_shape(self) -> Tuple[int, int, int]:
        """内区数组形状 ``(nk, nj, ni)``。"""
        return (grid_extent(self.grid, 2), grid_extent(self.grid, 1),
                grid_extent(self.grid, 0))

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


def _face(side, kind, executor, dif_bound, dif_axis_map, detail="",
          thermal=None, params=None):
    """构造一个面侧描述；显式 dif_bound 优先，缺省按 kind 派生。

    dif_bound 为 None 时按 ``boundary.dif_band(kind)`` 派生（壁面/远场开带）；
    给出时按原码次序逐侧读取——`--plate-dif-bound` 与 in 文件的 DIF_BOUND
    必须真的生效（曾静默被 kind 派生覆盖，2026-11-23 修）。
    """
    side = normalize_side(side)
    axis = side_axis(side)
    if dif_bound is None:
        band = dif_band(kind, executor)
    else:
        band = bool(dif_bound[2 * dif_axis_map[axis]
                               + (1 if side.endswith("+") else 0)])
    return FaceBoundary(side=side, axis=axis, kind=kind, executor=executor,
                        dif_band=band,
                        detail=detail, thermal=thermal,
                        params={} if params is None else params)


def _thermal_for_tw(tw):
    """槽道壁的热条件：Tw>0 等温，Tw<=0 绝热（与 channel._wall_layer 一致）。"""
    return "isothermal" if float(tw) > 0.0 else "adiabatic"


def _plate_faces(dif_bound, dif_axis_map, Tw=1.0):
    """平板：i 两侧不对称，j 下壁/上远场，k 周期。

    鬼层由 ``fill_faces`` 统一处理（非周期清零），边界点按面写：入口剖面交给
    通用 INFLOW（边界点 + 四层鬼层），下壁由 ``plate.apply`` 写壁面点，顶部由
    通用 FARFIELD 写单点，出口不赋值（方式 A：单侧带 + 零鬼层）。
    """
    return (
        _face("i-", INFLOW, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "入口剖面：边界点 + 四层鬼层（通用 INFLOW）",
              params={"profile": "inflow"}),
        _face("i+", OUTFLOW, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "出口不赋值：保持已有状态鬼层"),
        _face("j-", WALL, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "无滑移等温壁：只写壁面点，鬼层清零（plate.apply）",
              thermal="isothermal", params={"Tw": float(Tw)}),
        _face("j+", FARFIELD, _EXEC_CALLBACK, dif_bound, dif_axis_map,
              "顶部远场：只写最后一层内点（通用 FARFIELD）"),
        _face("k-", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("k+", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
    )


def _channel_faces(channel_bc, dif_bound, dif_axis_map):
    """槽道：i/k 周期，j 双壁由 ChannelBC 每子步写状态、fill_wall_j 补镜像。

    每侧热条件来自该侧自己的 Tw：Tw>0 等温，Tw<=0 绝热；上下壁可以不同。
    """
    Tw_bottom = float(channel_bc.Tw_bottom)
    Tw_top = float(channel_bc.Tw_top)
    return (
        _face("i-", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("i+", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("j-", WALL, _EXEC_CALLBACK_HALO, dif_bound, dif_axis_map,
              f"下壁无滑移{_thermal_for_tw(Tw_bottom)}壁：只写壁面层，鬼层清零（ChannelBC.apply）",
              thermal=_thermal_for_tw(Tw_bottom), params={"Tw": Tw_bottom}),
        _face("j+", WALL, _EXEC_CALLBACK_HALO, dif_bound, dif_axis_map,
              f"上壁无滑移{_thermal_for_tw(Tw_top)}壁：只写壁面层，鬼层清零（ChannelBC.apply）",
              thermal=_thermal_for_tw(Tw_top), params={"Tw": Tw_top}),
        _face("k-", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
        _face("k+", PERIODIC, _EXEC_HALO, dif_bound, dif_axis_map, "周期"),
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
                if axis != 1:
                    raise ValueError(
                        f"{side} 的通用 WALL 不可用：壁面镜像只支持计算 j 方向"
                        f"（axis=1）；得到 axis={axis}")
                faces.append(_face(side, WALL, _EXEC_HALO, dif_bound,
                                   dif_axis_map, "壁面镜像鬼胞"))
            elif bc == FARFIELD:
                faces.append(_face(side, FARFIELD, _EXEC_HALO, dif_bound,
                                   dif_axis_map,
                                   "远场 Dirichlet：边界内点 + 两侧鬼层"))
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
    # 按结构校验网格对象的 ni/nj/nk、inner 与 alloc，避免依赖具体 Grid 类身份。
    dimension_names = DIMENSION_NAMES
    has_dimensions = all(hasattr(grid, name) for name in dimension_names)
    if grid is None or not (has_dimensions and hasattr(grid, "inner")
                            and hasattr(grid, "alloc")):
        raise ValueError(
            "case['grid'] 必须是 fields.Grid 兼容对象"
            "（有 ni/nj/nk，以及 inner/alloc）")

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

    shape = (grid_extent(grid, 2) + 2 * LAP,
             grid_extent(grid, 1) + 2 * LAP,
             grid_extent(grid, 0) + 2 * LAP)
    metrics = _wrap_fields(case.get("metrics"), "metrics", METRIC_KEYS,
                           shape, "网格内区含鬼胞")
    coords = _wrap_fields(case.get("coords"), "coords", _COORD_KEYS,
                          (grid_extent(grid, 2), grid_extent(grid, 1),
                           grid_extent(grid, 0)), "内区")

    plate_marker = case.get("plate")
    channel_marker = case.get("channel_bc")
    special_marker = case.get("special_boundary")
    if sum(x is not None for x in (plate_marker, channel_marker, special_marker)) > 1:
        raise ValueError("case 只能选择一种特殊边界：plate / channel_bc / special_boundary")
    if special_marker is not None:
        if not callable(case.get("bc_apply")):
            raise ValueError(
                "special_boundary 需要可调用的 case['bc_apply'] 回调；"
                f"得到 {case.get('bc_apply')!r}")
        if not isinstance(getattr(special_marker, "name", None), str):
            raise ValueError("special_boundary 需要字符串 name")
        raw_faces = tuple(getattr(special_marker, "faces", ()))
        if not raw_faces:
            # faces 可选：优先继承已解析的 case["faces"]（CGNS 路径的六侧
            # 带异类正负面/壁温/逐侧带标志，_generic_faces 会丢——Astra 复审），
            # 无则按 bcs 派生
            faces = tuple(case["faces"]) if case.get("faces") else _generic_faces(
                bcs, dif_bound, dif_axis_map)
        else:
            if len(raw_faces) != 6 or tuple(f.side for f in raw_faces) != SIDES:
                raise ValueError(
                    f"special_boundary.faces 必须按 {SIDES} 顺序给出六侧")
            if not all(isinstance(f, FaceBoundary) for f in raw_faces):
                raise ValueError("special_boundary.faces 必须是 FaceBoundary")
            faces = validate_special_sides(raw_faces, bcs)
    elif plate_marker is not None:
        if not callable(case.get("bc_apply")):
            raise ValueError(
                "plate 特殊边界需要可调用的 case['bc_apply'] 回调；"
                f"得到 {case.get('bc_apply')!r}")
        if bcs != (INFLOW, PLATE, PERIODIC):
            raise ValueError(
                f"plate 固定六侧与 bcs 矛盾：要求 bcs=(inflow, plate, periodic)，"
                f"得到 {bcs}")
        if not isinstance(plate_marker, MappingABC):
            raise ValueError("case['plate'] 必须是映射")
        Tw = float(plate_marker.get("Tw", plate_marker.get("T0", 1.0)))
        if not (Tw == Tw and abs(Tw) != float("inf")) or Tw <= 0.0:
            raise ValueError(f"plate 需要正的有限壁温 Tw，得到 {Tw}")
        faces = _plate_faces(dif_bound, dif_axis_map, Tw=Tw)
    elif channel_marker is not None:
        if not callable(case.get("bc_apply")):
            raise ValueError(
                "channel 特殊边界需要可调用的 case['bc_apply'] 回调；"
                f"得到 {case.get('bc_apply')!r}")
        if bcs != (PERIODIC, WALL, PERIODIC):
            raise ValueError(
                f"channel 固定六侧与 bcs 矛盾：要求 bcs=(periodic, wall, periodic)，"
                f"得到 {bcs}")
        for attr in ("Tw_bottom", "Tw_top"):
            if not hasattr(channel_marker, attr):
                raise ValueError(f"channel_bc 缺少 {attr}")
            value = float(getattr(channel_marker, attr))
            if not (value == value and abs(value) != float("inf")):
                raise ValueError(f"channel_bc.{attr} 含非有限值: {value}")
        faces = _channel_faces(channel_marker, dif_bound, dif_axis_map)
    else:
        if case.get("bc_apply") is not None:
            raise ValueError(
                "case 提供了 bc_apply，但没有 channel_bc / plate 特殊边界标记；"
                "未知特殊边界必须在 case 侧显式选择并补六侧描述")
        raw_faces = case.get("faces")
        if raw_faces is not None:
            faces = tuple(raw_faces)
            if len(faces) != 6 or tuple(f.side for f in faces) != SIDES:
                raise ValueError(f"case['faces'] 必须按 {SIDES} 顺序给出六侧")
            if not all(isinstance(f, FaceBoundary) for f in faces):
                raise ValueError("case['faces'] 必须是 FaceBoundary")
        else:
            faces = _generic_faces(bcs, dif_bound, dif_axis_map)

    return BlockMesh(grid=grid, dxs=dxs, axes=axes, bcs=bcs,
                     dif_bound=dif_bound, dif_axis_map=dif_axis_map,
                     metrics=metrics, coords=coords, faces=faces)
