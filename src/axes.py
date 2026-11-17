"""计算方向 (i,j,k) 与旧 x/y/z 标签的唯一映射。

计算方向 (i,j,k) 对应 (ξ,η,ζ)，数组空间布局始终是 (k,j,i)。本模块只定义
命名和标签映射，不改变任何数值、数组形状或边界公式。
"""

from numbers import Integral

AXIS_NAMES = ("i", "j", "k")
DIMENSION_NAMES = ("ni", "nj", "nk")
LEGACY_DIMENSION_NAMES = ("nx", "ny", "nz")
SIDES = ("i-", "i+", "j-", "j+", "k-", "k+")
LEGACY_AXIS_NAMES = ("x", "y", "z")
LEGACY_SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")

_AXIS_TO_INDEX = {name: axis for axis, name in enumerate(AXIS_NAMES)}
_LEGACY_AXIS_TO_INDEX = {
    name: axis for axis, name in enumerate(LEGACY_AXIS_NAMES)
}
_SIDE_TO_INDEX = {
    label: index // 2 for index, label in enumerate(SIDES)
}
_LEGACY_SIDE_TO_INDEX = {
    label: index // 2 for index, label in enumerate(LEGACY_SIDES)
}

MISSING = object()


def resolve_dimensions(ni=MISSING, nj=MISSING, nk=MISSING, *,
                       nx=MISSING, ny=MISSING, nz=MISSING):
    """逐轴合并规范 ni/nj/nk 与旧 nx/ny/nz；同轴冲突报错。

    只返回一份规范值，不保存旧字段。未给任何一轴的旧/新值时报错。
    """
    values = []
    for canonical, legacy, label in (
            (ni, nx, "i"), (nj, ny, "j"), (nk, nz, "k")):
        if canonical is MISSING and legacy is MISSING:
            raise ValueError(f"缺少网格尺寸 {label}")
        if canonical is MISSING:
            canonical = legacy
        if legacy is not MISSING and canonical != legacy:
            raise ValueError(
                f"网格尺寸 {label} 的新旧值冲突：{canonical!r} != {legacy!r}")
        values.append(canonical)
    return tuple(values)


def normalize_axis(axis):
    """接受 i/j/k、x/y/z 或 0/1/2，返回计算轴编号。"""
    if isinstance(axis, Integral):
        value = int(axis)
        if 0 <= value < len(AXIS_NAMES):
            return value
        raise ValueError(f"计算轴必须是 0/1/2，得到 {axis!r}")
    if axis in _AXIS_TO_INDEX:
        return _AXIS_TO_INDEX[axis]
    if axis in _LEGACY_AXIS_TO_INDEX:
        return _LEGACY_AXIS_TO_INDEX[axis]
    raise ValueError(f"计算轴必须是 i/j/k 或 x/y/z，得到 {axis!r}")


def axis_name(axis):
    """返回规范计算轴名 i/j/k。"""
    return AXIS_NAMES[normalize_axis(axis)]


def legacy_axis_name(axis):
    """返回兼容期旧轴名 x/y/z。"""
    return LEGACY_AXIS_NAMES[normalize_axis(axis)]


def normalize_side(side):
    """接受 i±/j±/k± 或 x±/y±/z±，返回规范六侧标签。"""
    if side in _SIDE_TO_INDEX:
        return side
    if side in _LEGACY_SIDE_TO_INDEX:
        return SIDES[_LEGACY_SIDE_TO_INDEX[side] * 2 +
                     (1 if side.endswith("+") else 0)]
    raise ValueError(f"side 必须是 {SIDES} 或 {LEGACY_SIDES} 之一，得到 {side!r}")


def legacy_side(side):
    """返回规范六侧标签对应的旧 x±/y±/z± 标签。"""
    normalized = normalize_side(side)
    index = _SIDE_TO_INDEX[normalized]
    return LEGACY_SIDES[index * 2 + (1 if normalized.endswith("+") else 0)]


def side_axis(side):
    """返回六侧标签所属的计算轴编号。"""
    return _SIDE_TO_INDEX[normalize_side(side)]


def side_sign(side):
    """返回负侧 0、正侧 1。"""
    return 1 if normalize_side(side).endswith("+") else 0


def array_axis(axis):
    """计算轴编号映射到标量数组 (k,j,i) 的轴。"""
    return 2 - normalize_axis(axis)


def grid_extent(grid, axis):
    """读取网格对象某一计算方向的内区尺寸，兼容新旧属性名。"""
    index = normalize_axis(axis)
    canonical = DIMENSION_NAMES[index]
    legacy = LEGACY_DIMENSION_NAMES[index]
    has_canonical = hasattr(grid, canonical)
    has_legacy = hasattr(grid, legacy)
    if has_canonical and has_legacy \
            and getattr(grid, canonical) != getattr(grid, legacy):
        raise ValueError(
            f"grid 的 {canonical}={getattr(grid, canonical)!r} 与 "
            f"{legacy}={getattr(grid, legacy)!r} 冲突")
    if has_canonical:
        return getattr(grid, canonical)
    if has_legacy:
        return getattr(grid, legacy)
    raise ValueError(f"grid 缺少 {canonical} 或 {legacy} 属性")
