"""计算方向 (i,j,k) 的规范命名与尺寸工具。

计算方向 (i,j,k) 对应 (ξ,η,ζ)，数组空间布局始终是 (k,j,i)。本模块只定义
命名和标签映射，不改变任何数值、数组形状或边界公式。旧 x/y/z 轴名和
nx/ny/nz 尺寸名不再作为入口接受。
"""

from numbers import Integral

AXIS_NAMES = ("i", "j", "k")
DIMENSION_NAMES = ("ni", "nj", "nk")
SIDES = ("i-", "i+", "j-", "j+", "k-", "k+")

_AXIS_TO_INDEX = {name: axis for axis, name in enumerate(AXIS_NAMES)}
_SIDE_TO_INDEX = {
    label: index // 2 for index, label in enumerate(SIDES)
}

MISSING = object()


def resolve_dimensions(ni=MISSING, nj=MISSING, nk=MISSING):
    """校验并返回规范 ni/nj/nk；缺项或非正整数时直接报错。"""
    values = []
    for canonical, label in ((ni, "i"), (nj, "j"), (nk, "k")):
        if canonical is MISSING:
            raise ValueError(f"缺少网格尺寸 {label}")
        if not isinstance(canonical, Integral) or canonical < 1:
            raise ValueError(
                f"网格尺寸 {label} 必须为正整数，得到 {canonical!r}")
        values.append(canonical)
    return tuple(values)


def normalize_axis(axis):
    """接受 i/j/k 或 0/1/2，返回计算轴编号。"""
    if isinstance(axis, Integral):
        value = int(axis)
        if 0 <= value < len(AXIS_NAMES):
            return value
        raise ValueError(f"计算轴必须是 0/1/2，得到 {axis!r}")
    if axis in _AXIS_TO_INDEX:
        return _AXIS_TO_INDEX[axis]
    raise ValueError(f"计算轴必须是 i/j/k 或 0/1/2，得到 {axis!r}")


def axis_name(axis):
    """返回规范计算轴名 i/j/k。"""
    return AXIS_NAMES[normalize_axis(axis)]


def normalize_side(side):
    """接受 i±/j±/k±，返回规范六侧标签；旧 x±/y±/z± 不再接受。"""
    if side in _SIDE_TO_INDEX:
        return side
    raise ValueError(f"side 必须是 {SIDES} 之一，得到 {side!r}")


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
    """读取网格对象某一计算方向的内区尺寸，只接受 ni/nj/nk。"""
    index = normalize_axis(axis)
    canonical = DIMENSION_NAMES[index]
    if hasattr(grid, canonical):
        return getattr(grid, canonical)
    raise ValueError(f"grid 缺少 {canonical} 属性")
