"""多块串行的拓扑、装配、交换与编排（第六步）。

``enter`` 是多块网格的入口：只用 zone 元数据判定要不要走多块路径，并在读取
坐标之前做完放行校验（R31）。走多块路径时它读入全部 zone，交给
``build_topology`` 校验，打印接口报告。单块路径只调用 ``enter`` 做判定。

``build_topology`` 把读进来的全部 zone 变成一个校验过的 :class:`Topology`。
校验按固定的顺序做，前一步有问题就不进入下一步：

1. 连接：逐条按文件顺序查 ``Transform``、所在侧面、范围、切向增量、倒写；再把
   两条互逆的方向记录配成接口对，查几何；
2. 边条：必须是落在某个侧面上的面片，类型在可执行子集内；
3. 覆盖表：每个面元恰好被一个来源覆盖，带面积的重叠和缺口都报错；
4. 面的归类：物理面、接口面、块内周期面，混着的报错；
5. 尺寸：物理面、周期接缝和块内周期面所在的轴至少 7 个顶点，普通切口的供体
   在法向至少 5 个。

``build_blocks`` 把拓扑变成可计算的块：先按交换表把接口旁的坐标换成供体的，
再算度量（普通切口一侧走中心模板，并把切口旁四层换成供体的内区度量），最后
取出物理面和块内周期面，交给 ``boundary.build_block_execution``。

``solve_step`` 推进一个整步：每个 RK 阶段各块做准备，全块同步交换守恒量鬼胞，
各块算无粘右端项，再用准备之前的阶段态做组合。粘性打开时，各块另外算自己的粘性
通量 ``Ev``，全块同步交换一次 ``Ev``，对换过的 ``Ev`` 求导并累加（第七步）。
``run`` 是正式入口的主循环，每个保存步存一个检查点，每块一个数组、带一份块清单；给了
``--restart`` 就从那张检查点接着算（第八步）。
"""

from __future__ import annotations

import json
import numbers
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from . import boundary, config, init, io, mesh_cgns, runtime_config
from .ana import diagnostic_line
from .axes import AXIS_NAMES, SIDES, side_axis
from .boundary import LAP, PERIODIC
from .cases import cons_from_state, physical_params, plate_faces
from .errors import RuntimeExecutionError, SetupError
from .fields import Grid
from .mesh import (COORD_KEYS, METRIC_KEYS, compute_metrics_from_coords,
                   pad_coords)
from .mesh_cgns import (CgnsError, CgnsZone, bc_kind, check_extent, face_axes,
                        face_boundary, side_of_face)
from .solver import (inviscid_rhs, rk3_q1, rk3_q2, rk3_q3, scales_from_sizes)
from .viscous import viscous_divergence, viscous_fluxes

CUT, SEAM, SELF = "cut", "seam", "self"     # 接口段的类别：普通切口、跨块周期接缝、块内自连接
PHYSICAL, INTERFACE, PERIODIC_FACE = "physical", "interface", "periodic"   # 面的归类
GEOMETRY_RTOL = 1e-6        # 几何容差：两侧坐标最大绝对值（不小于 1）的相对值
GEOMETRY_ATOL = 1e-12       # 几何容差的下限
MIN_FACE_VERTICES = 7       # 物理面、周期接缝、块内周期面所在的轴：度量的单侧带
MIN_DONOR_VERTICES = 5      # 普通切口的供体在法向
MAX_VERTICES = 2**31 - 1    # 旁记里全局顶点数的上限：CGNS 的下标范围是 32 位整数
CD_ORDER = {"CD6": 6, "CD8": 8}           # 粘性导数的阶数，与 main.build_rhs 里的表相同；
                                          # runtime_config.VIS_ORDERS 是允许的名字，不是这张表
# 粘性打开而参考温度取缺省值时的提示，与单块 main.run 打印的那一句逐字相同
REF_T_WARNING = ("WARNING: Ref_T 用了缺省 288.15 K（空气）。Ref_T 是有量纲参考温度"
                 "（对应原码 T_REF），只影响 μ 的温度曲线（μ(T=1)=1/Re 与它无关）；"
                 "与原码对拍时必须用 --ref-t 给出 in 文件里的同值。")


@dataclass(frozen=True)
class Interface:
    """一个接口对：接收块的一个侧面接供体的一个侧面。

    由两条互逆的方向记录合计而成，``names`` 是这两条记录的原名
    （``zone 名 / 连接名``）。范围是 0-based 的有向闭区间，保留文件里的书写方向。
    ``translation`` 是周期接缝的平移量（供体坐标减接收坐标）：前一条记录给了
    ``Translation`` 就是它，否则是实测位移在整个面上的中点。``max_gap`` 在普通切口上
    是两侧对应顶点的最大坐标差，在周期接缝上是实测位移在面上的最大偏离（离中点最
    远多少）。块内自连接也用这个类表示，类别是 ``SELF``，不进接口对的计数。
    """

    kind: str
    zone: str
    side: str
    donor_zone: str
    donor_side: str
    point_range: tuple
    point_range_donor: tuple
    names: tuple[str, str]
    points: int
    max_gap: float
    tolerance: float
    translation: np.ndarray | None = None


@dataclass(frozen=True)
class FaceInfo:
    """一个块的一个侧面：归类、边条的类型与来源、落在它上面的接口段。"""

    zone: str
    side: str
    kind: str                         # PHYSICAL / INTERFACE / PERIODIC_FACE
    bc_type: str | None = None        # 物理面的 BCType；块内周期面是边条的类型或 None
    sources: tuple[str, ...] = ()     # 边条 patch 的名字，或块内自连接两条记录的原名
    interfaces: tuple[Interface, ...] = ()


@dataclass(frozen=True)
class Topology:
    """校验过的多块拓扑。"""

    zones: tuple[CgnsZone, ...]
    interfaces: tuple[Interface, ...]      # 跨块的接口对，按第一条记录在文件里的顺序
    faces: dict[tuple[str, str], FaceInfo]  # (zone 名, 侧面) → 面的信息
    report: str


def _record(zone: CgnsZone, conn) -> str:
    """方向记录的原名 ``zone 名 / 连接名``；名字里的连续空格原样保留。"""
    return f"{zone.name} / {conn.name}"


def _bc_side(zone: CgnsZone, bc):
    """边条 patch 落在哪个侧面；不是面片或不在表面上返回 None。"""
    axes = face_axes(*bc.point_range)
    if axes is None:
        return None
    return side_of_face(axes[0], bc.point_range[0], zone.vertex_size)


def _covers_whole_face(tangential, lo, hi, vertex) -> bool:
    """面片是否铺满这个侧面（不管书写的方向）。"""
    return all(min(lo[a], hi[a]) == 0 and max(lo[a], hi[a]) == vertex[a] - 1
               for a in tangential)


def _points(rng) -> int:
    """面片上的顶点数，不管范围是怎么写的方向。"""
    lo, hi = rng
    return ((abs(hi[0] - lo[0]) + 1) * (abs(hi[1] - lo[1]) + 1)
            * (abs(hi[2] - lo[2]) + 1))


def _span(rng) -> tuple:
    """范围不计书写方向：((各轴的最小下标), (各轴的最大下标))。"""
    return (tuple(map(min, rng[0], rng[1])), tuple(map(max, rng[0], rng[1])))


def _patch(coords, rng) -> np.ndarray:
    """按有向范围取出面片上的坐标点，形状 (点数, 3)，点的顺序沿有向范围。"""
    lo, hi = rng
    index = np.ix_(*[np.arange(lo[a], hi[a] + (1 if hi[a] >= lo[a] else -1),
                               1 if hi[a] >= lo[a] else -1) for a in (2, 1, 0)])
    return np.stack([np.asarray(coords[key])[index].reshape(-1)
                     for key in COORD_KEYS], axis=1)


def _file_translation(tag: str, periodic) -> np.ndarray | None:
    """校验一条记录的 ``Periodic_t``，返回文件给的平移量；文件没给时返回 None。"""
    for field, label in (("rotation_center", "RotationCenter"),
                         ("rotation_angle", "RotationAngle"),
                         ("translation", "Translation")):
        value = getattr(periodic, field)
        if value is None:
            if label != "Translation":
                raise CgnsError(f"{tag} 的 Periodic_t 缺少 {label}")
            continue
        if value.shape != (3,):
            raise CgnsError(
                f"{tag} 的 Periodic_t/{label} 长度是 {value.shape}，应为 3")
        if not np.all(np.isfinite(value)):
            raise CgnsError(
                f"{tag} 的 Periodic_t/{label} 含非有限值：{value.tolist()}")
    if np.any(periodic.rotation_angle != 0.0):
        raise CgnsError(
            f"{tag} 的 RotationAngle={periodic.rotation_angle.tolist()}；"
            "第六步不支持带旋转的周期")
    return periodic.translation


def _check_record(zone: CgnsZone, conn, sizes: dict) -> tuple[str, str]:
    """校验一条方向记录，返回它落在接收块的哪个侧面、接供体的哪个侧面。

    ``Periodic_t`` 在这里只查字段；文件给的平移量留到配对时与实测的核对。
    """
    tag = _record(zone, conn)
    if conn.transform != (1, 2, 3):
        raise CgnsError(
            f"{tag} 的 Transform={list(conn.transform)}；第六步暂不支持翻转、"
            "轴交换或分量写成 0 的 Transform，只接受 (1, 2, 3)")
    donor = conn.donor_zone
    if donor not in sizes:
        raise CgnsError(
            f"{tag} 的供体 zone {donor!r} 不存在；文件里的 zone 是 {list(sizes)}")
    lo, hi = conn.point_range
    dlo, dhi = conn.point_range_donor
    check_extent(lo, hi, zone.vertex_size, tag)     # 多块读取时读取器不查，留到这里
    recv = face_axes(lo, hi)
    if recv is None:
        raise CgnsError(
            f"{tag} 的 PointRange={lo}..{hi} 不是面片；连接要落在块的一个侧面上")
    give = face_axes(dlo, dhi)
    check_extent(dlo, dhi, sizes[donor], tag)
    if give is None:
        raise CgnsError(f"{tag} 的 PointRangeDonor={dlo}..{dhi} 不是面片")
    normal, *tangential = recv
    if give[0] != normal:
        raise CgnsError(
            f"{tag} 接在 {AXIS_NAMES[normal]} 侧面上，供体范围却在 "
            f"{AXIS_NAMES[give[0]]} 侧面上；第六步暂不支持轴交换，只接受同轴相接")
    side = side_of_face(normal, lo, zone.vertex_size)
    if side is None:
        raise CgnsError(f"{tag} 的 PointRange={lo}..{hi} 不在 {zone.name} 的表面上")
    donor_side = side_of_face(normal, dlo, sizes[donor])
    if donor_side is None:
        raise CgnsError(
            f"{tag} 的 PointRangeDonor={dlo}..{dhi} 不在供体 {donor} 的表面上")
    if side[-1] == donor_side[-1]:
        raise CgnsError(
            f"{tag} 把 {zone.name} 的 {side} 面接给 {donor} 的 {donor_side} 面；"
            "只能低侧接高侧、高侧接低侧")
    for axis in tangential:
        span, donor_span = hi[axis] - lo[axis], dhi[axis] - dlo[axis]
        if abs(span) != abs(donor_span):
            raise CgnsError(
                f"{tag} 在 {AXIS_NAMES[axis]} 向接收 {abs(span) + 1} 个点、"
                f"供体给出 {abs(donor_span) + 1} 个点；两个切向轴的点数分配必须相同")
        if span != donor_span:
            raise CgnsError(
                f"{tag} 在 {AXIS_NAMES[axis]} 向只倒写了一侧（接收端 {span:+d}，"
                f"供体端 {donor_span:+d}）；要两侧一同倒写")
    if donor == zone.name:
        if conn.periodic is not None:
            raise CgnsError(
                f"{tag} 是同块自连接却带 Periodic_t；块内周期直接写 1to1 自连接，"
                "不带 Periodic_t")
        if not (_covers_whole_face(tangential, lo, hi, zone.vertex_size)
                and _covers_whole_face(tangential, dlo, dhi, sizes[donor])):
            raise CgnsError(
                f"{tag} 是同块自连接，但没有覆盖相对的两个整面；"
                "块内周期只认相对整面的自连接")
    if conn.periodic is not None:
        _file_translation(tag, conn.periodic)
    return side, donor_side


def _pair_interfaces(records, placed, by_name):
    """把方向记录配成接口对，查几何；返回 (跨块的接口对, 块内自连接对)。"""
    used = [False] * len(records)
    cross, in_block = [], []
    back = {}                  # (供体所在的块, 接回的块, 它落在的两个侧面) → 记录的下标
    for j, (other, c) in enumerate(records):
        back.setdefault((other.name, c.donor_zone, placed[j]), []).append(j)
    for index, (zone, conn) in enumerate(records):
        if used[index]:
            continue
        tag = _record(zone, conn)
        side, donor_side = placed[index]
        donor = by_name[conn.donor_zone]
        candidates = back.get((donor.name, zone.name, (donor_side, side)), [])
        if not candidates:
            raise CgnsError(
                f"{tag} 在 {donor.name} 里找不到互逆的方向记录；应当有一条从 "
                f"{donor.name} 的 {donor_side} 面接回 {zone.name} 的 {side} 面的记录")
        # 两条记录各有“两侧一同倒写”的检查，所以只看范围覆盖哪些点，不看怎么写
        mutual = [j for j in candidates
                  if _span(records[j][1].point_range) == _span(conn.point_range_donor)
                  and _span(records[j][1].point_range_donor) == _span(conn.point_range)]
        if not mutual:
            other = records[candidates[0]][1]
            raise CgnsError(
                f"{tag} 与 {_record(donor, other)} 不互逆：{tag} 的供体范围是 "
                f"{conn.point_range_donor}，对方接回的范围却是 "
                f"{other.point_range}..{other.point_range_donor}")
        free = [j for j in mutual if not used[j]]
        if not free:
            raise CgnsError(
                f"{tag} 与 {_record(donor, records[mutual[0]][1])} 重复描述同一个接口")
        if len(free) > 1:
            raise CgnsError(
                f"{tag} 在 {donor.name} 里有 {len(free)} 条互逆的方向记录："
                f"{[_record(donor, records[j][1]) for j in free]}")
        partner = records[free[0]][1]
        used[index] = used[free[0]] = True
        names = (tag, _record(donor, partner))
        common = dict(zone=zone.name, side=side, donor_zone=donor.name,
                      donor_side=donor_side, point_range=conn.point_range,
                      point_range_donor=conn.point_range_donor, names=names,
                      points=_points(conn.point_range))
        if donor is zone:
            in_block.append(Interface(kind=SELF, translation=None, max_gap=0.0,
                                      tolerance=0.0, **common))
            continue
        if (conn.periodic is None) != (partner.periodic is None):
            raise CgnsError(
                f"{names[0]} 与 {names[1]} 的 Periodic_t 不一致；周期接缝的两条"
                "方向记录都要带 Periodic_t")
        receiving = _patch(zone.coords, conn.point_range)
        giving = _patch(donor.coords, conn.point_range_donor)
        if not (np.isfinite(receiving).all() and np.isfinite(giving).all()):
            raise CgnsError(f"{tag} 两侧对应顶点的坐标含非有限值")
        scale = max(1.0, float(np.abs(receiving).max()),
                    float(np.abs(giving).max()))
        tolerance = max(GEOMETRY_ATOL, GEOMETRY_RTOL * scale)
        if conn.periodic is None:
            kind, translation = CUT, None
            gap = float(np.abs(receiving - giving).max())
            if gap > tolerance:
                raise CgnsError(
                    f"{tag} 两侧对应顶点的坐标不重合，最大坐标差 {gap:.3e}"
                    f"（容差 {tolerance:.1e}）")
        else:
            kind = SEAM
            shift = giving - receiving                # 供体减接收
            if not np.isfinite(shift).all():
                raise CgnsError(f"{tag} 两侧对应顶点的坐标差溢出，算不出平移量")
            # 判据只用整面上的对称量。两条互逆的记录可以各自倒写，首点可以是不同的
            # 物理顶点；拿哪一个点当参考都会让结论随 zone 的先后而变（2026-11-24，
            # Astra 第 19 轮）。中点是实测的平移量，半个跨度是它在面上的最大偏离。
            # 两个写法都是为了换序后恰好只差符号、且在 1e308 这样的位移上不溢出：
            # 先减后除（第 20 轮）；中点取“最小值的一半加最大值的一半”，除以 2 和取负
            # 在浮点里是精确的，加法对正负号对称。写成“最小值加半个跨度”时两者会差
            # 最后一位，中点恰好压在容差上时结论就随 zone 的先后而变（第 21 轮）。
            low, high = shift.min(axis=0), shift.max(axis=0)
            half = (high - low) / 2.0
            middle = low / 2.0 + high / 2.0
            gap = float(np.abs(half).max())
            declared = False
            for owner, record, measured in ((zone, conn, shift), (donor, partner, -shift)):
                given = record.periodic.translation
                if given is None:
                    continue                          # 这条记录没给；给了的只用来核对
                declared = True
                residual = float(np.abs(measured - given).max())
                if residual > tolerance:
                    raise CgnsError(
                        f"{_record(owner, record)} 的 Translation={given.tolist()} 与实测的"
                        f"平移量（供体减接收）不符：首点实测 {measured[0].tolist()}，按它"
                        f"平移后的最大残差 {residual:.3e}（容差 {tolerance:.1e}）")
            first, second = conn.periodic.translation, partner.periodic.translation
            if first is not None and second is not None:
                apart = float(np.abs(first + second).max())    # 反向记录的平移量取负
                if apart > tolerance:
                    raise CgnsError(
                        f"{names[0]} 与 {names[1]} 的 Translation 不一致：{first.tolist()} "
                        f"与 {second.tolist()} 相差 {apart:.3e}（容差 {tolerance:.1e}）；"
                        "反向记录的平移量要取负")
            if not declared and gap > tolerance:     # 有声明时，残差合格已经蕴含这一条
                raise CgnsError(
                    f"{tag} 的平移量在整个面上不是常向量，最大偏离 {gap:.3e}"
                    f"（容差 {tolerance:.1e}）；第六步只支持平移周期")
            if np.abs(middle).max() <= tolerance:    # 看实测的位移，不看文件怎么声明
                raise CgnsError(
                    f"{tag} 带 Periodic_t，但两侧的平移量约为零"
                    f"（{middle.tolist()}），坐标其实是重合的；这不是周期接缝，"
                    "请去掉 Periodic_t")
            translation = middle if first is None else first
        cross.append(Interface(kind=kind, translation=translation, max_gap=gap,
                               tolerance=tolerance, **common))
    return cross, in_block


def _check_periodic_origins(zones, in_block):
    """同一块的同一轴上，``BCSymmetryPolar`` 边条和自连接不能各占一面。"""
    for zone in zones:
        for axis, name in enumerate(AXIS_NAMES):
            origins = []
            for side in (name + "-", name + "+"):
                if any(it.zone == zone.name and it.side == side for it in in_block):
                    origins.append("conn")
                elif any(bc_kind(bc.bc_type) == PERIODIC and _bc_side(zone, bc) == side
                         for bc in zone.bcs):
                    origins.append("bc")
                else:
                    origins.append(None)
            if "bc" in origins and "conn" in origins:
                raise CgnsError(
                    f"{zone.name} 的 {name} 轴周期两侧来源不同"
                    f"（{origins[0]}/{origins[1]}）；BCSymmetryPolar 必须同轴正负"
                    "整面成对，不与 1to1 自连接混配")


def _collect_sources(zones, interfaces, in_block):
    """校验边条 patch，并列出每个面上的来源：边条、接口段、自连接。"""
    sources = {(zone.name, side): [] for zone in zones for side in SIDES}
    for zone in zones:
        for bc in zone.bcs:
            tag = f"{zone.name} / {bc.name}"
            lo, hi = bc.point_range
            axes = face_axes(lo, hi)
            if axes is None:
                raise CgnsError(
                    f"{tag} 的 PointRange={lo}..{hi} 不是面片；边条要落在块的一个侧面上")
            side = side_of_face(axes[0], lo, zone.vertex_size)
            if side is None:
                raise CgnsError(
                    f"{tag} 的 PointRange={lo}..{hi} 不在 {zone.name} 的表面上")
            if bc_kind(bc.bc_type) is None:
                raise CgnsError(f"{tag} 的 BCType={bc.bc_type!r} 不在本步执行子集")
            sources[(zone.name, side)].append(("bc", bc))
    for it in interfaces + in_block:
        sources[(it.zone, it.side)].append((it.kind, it))
        sources[(it.donor_zone, it.donor_side)].append((it.kind, it))
    return sources


def _describe(item, zone_name) -> tuple[str, tuple]:
    """一个来源在报错里的说法，和它在这个面上的范围。"""
    kind, obj = item
    if kind == "bc":
        return f"边条 {obj.name}（{obj.bc_type}）", obj.point_range
    if kind == SELF:
        what = f"自连接 {obj.names[0]} ↔ {obj.names[1]}"
    else:
        what = f"接口段 {obj.names[0]} ↔ {obj.names[1]}"
    here = obj.zone == zone_name
    return what, (obj.point_range if here else obj.point_range_donor)


def _check_coverage(zone: CgnsZone, side: str, items):
    """每个面元恰好被一个来源覆盖；带面积的重叠和缺口都报错。"""
    normal = side_axis(side)
    first, second = [axis for axis in range(3) if axis != normal]
    vertex = zone.vertex_size
    owner = np.full((vertex[second] - 1, vertex[first] - 1), -1, dtype=np.int64)
    for number, item in enumerate(items):
        what, (lo, hi) = _describe(item, zone.name)
        cells = (slice(min(lo[second], hi[second]), max(lo[second], hi[second])),
                 slice(min(lo[first], hi[first]), max(lo[first], hi[first])))
        if np.any(owner[cells] >= 0):
            old_what, (old_lo, old_hi) = _describe(
                items[int(owner[cells][owner[cells] >= 0][0])], zone.name)
            raise CgnsError(
                f"{zone.name} 的 {side} 面带面积的重叠：{what}（{lo}..{hi}）与 "
                f"{old_what}（{old_lo}..{old_hi}）盖住了同一片面元")
        owner[cells] = number
    if np.any(owner < 0):
        missing = owner < 0
        rows = np.where(missing.any(axis=1))[0]
        cols = np.where(missing.any(axis=0))[0]
        raise CgnsError(
            f"{zone.name} 的 {side} 面有缺口：{int(missing.sum())} 个面元没有被边条"
            f"或连接覆盖（0-based 顶点范围 {AXIS_NAMES[first]}={cols[0]}..{cols[-1] + 1}，"
            f"{AXIS_NAMES[second]}={rows[0]}..{rows[-1] + 1}）")


def _classify(zone_name, side, items) -> FaceInfo:
    """给一个面归类：全是边条是物理面，全是接口段是接口面，自连接是块内周期面。"""
    where = f"{zone_name} 的 {side} 面"
    bcs = [obj for kind, obj in items if kind == "bc"]
    segments = [obj for kind, obj in items if kind != "bc"]
    if bcs and segments:
        raise CgnsError(
            f"{where}上边条段与接口段并存（边条 {[b.name for b in bcs]}，接口 "
            f"{[s.names[0] for s in segments]}）；一个面要么全是物理边条，要么全是接口")
    if bcs:
        if len({bc.bc_type for bc in bcs}) > 1:
            raise CgnsError(
                f"{where}上边条段的类型不同（"
                f"{[f'{bc.bc_type} {bc.name}' for bc in bcs]}）；"
                "同一个面的边条必须是同一种类型")
        return FaceInfo(zone_name, side, PHYSICAL, bc_type=bcs[0].bc_type,
                        sources=tuple(bc.name for bc in bcs))
    if len({segment.kind for segment in segments}) > 1:
        raise CgnsError(
            f"{where}上普通接口段与周期接口段并存（"
            f"{[s.names[0] for s in segments]}）；第六步不支持")
    if segments[0].kind == SELF:
        return FaceInfo(zone_name, side, PERIODIC_FACE, sources=segments[0].names)
    return FaceInfo(zone_name, side, INTERFACE, interfaces=tuple(segments))


def _is_polar(face: FaceInfo) -> bool:
    """面是不是由 ``BCSymmetryPolar`` 边条铺成的（已经归为块内周期面的也算）。

    自连接铺成的块内周期面没有边条类型，不算：它和 ``BCSymmetryPolar`` 不能各占一面。
    """
    return (face.bc_type is not None and bc_kind(face.bc_type) == PERIODIC
            and face.kind in (PHYSICAL, PERIODIC_FACE))


def _pair_polar_faces(faces):
    """``BCSymmetryPolar`` 的两个面必须同块同轴成对，成对后归为块内周期面。"""
    for (zone_name, side), face in list(faces.items()):
        if face.kind != PHYSICAL or not _is_polar(face):
            continue
        opposite = side[0] + ("+" if side[1] == "-" else "-")
        if not _is_polar(faces[(zone_name, opposite)]):
            raise CgnsError(
                f"{zone_name} 的 {side} 面是 BCSymmetryPolar，对面的 {opposite} 面"
                "不是；周期边条必须同块同轴成对，跨块的周期要用块间 1to1 连接加 "
                "Periodic_t 表达")
        faces[(zone_name, side)] = replace(face, kind=PERIODIC_FACE)


def _check_sizes(faces, sizes):
    """按每个面实际用到的模板诊断块的尺寸。"""
    for (zone_name, side), face in faces.items():
        axis = side_axis(side)
        label = {PHYSICAL: "物理面", PERIODIC_FACE: "块内周期面"}.get(face.kind)
        if face.kind == INTERFACE and face.interfaces[0].kind == SEAM:
            label = "周期接缝面"
        if label is not None:
            if sizes[zone_name][axis] < MIN_FACE_VERTICES:
                raise CgnsError(
                    f"{zone_name} 的 {side} 面是{label}，{AXIS_NAMES[axis]} 向只有 "
                    f"{sizes[zone_name][axis]} 个顶点，至少 {MIN_FACE_VERTICES} 个")
            continue
        for it in face.interfaces:            # 普通切口：看供体在法向的顶点数
            donor = it.donor_zone if it.zone == zone_name else it.zone
            if sizes[donor][axis] < MIN_DONOR_VERTICES:
                raise CgnsError(
                    f"{zone_name} 的 {side} 面通过 {it.names[0]} 接 {donor}，供体在 "
                    f"{AXIS_NAMES[axis]} 向只有 {sizes[donor][axis]} 个顶点，"
                    f"至少 {MIN_DONOR_VERTICES} 个")


def _report(zones, interfaces, faces) -> str:
    """接口报告：接口对只数跨块的，块内周期另列。"""
    cuts = sum(it.kind == CUT for it in interfaces)
    periodic = [(zone.name, name, faces[(zone.name, name + "-")])
                for zone in zones for name in AXIS_NAMES
                if faces[(zone.name, name + "-")].kind == PERIODIC_FACE]
    lines = [f"多块网格：{len(zones)} 个 zone；接口对 {len(interfaces)} 个"
             f"（普通切口 {cuts} 个，周期接缝 {len(interfaces) - cuts} 个）；"
             f"块内周期轴 {len(periodic)} 个"]
    for number, it in enumerate(interfaces, 1):
        line = (f"接口对 {number}：{it.names[0]} ↔ {it.names[1]}；"
                f"{it.zone} 的 {it.side} 面 ↔ {it.donor_zone} 的 {it.donor_side} 面；"
                f"范围 {it.point_range[0]}..{it.point_range[1]} ↔ "
                f"{it.point_range_donor[0]}..{it.point_range_donor[1]}；"
                f"{it.points} 个对应点")
        if it.kind == CUT:
            line += f"；普通切口；最大坐标差 {it.max_gap:.1e}"
        else:
            # 用 Python 的 round：np.round 要先乘 10^12，在 1e297 这样的位移上会溢出
            shift = tuple(round(float(value), 12) for value in it.translation)
            line += (f"；周期接缝；平移（供体减接收）{shift}；"
                     f"实测平移的最大偏离 {it.max_gap:.1e}")
        lines.append(line + f"；几何容差 {it.tolerance:.1e}")
    for zone_name, name, face in periodic:
        if face.bc_type is not None:          # 边条铺成的；自连接的没有 bc_type
            source = f"{face.bc_type} {list(face.sources)}"
        else:
            source = f"{face.sources[0]} ↔ {face.sources[1]}"
        lines.append(f"块内周期：{zone_name} 的 {name} 轴（{source}）")
    return "\n".join(lines)


def build_topology(zones) -> Topology:
    """校验全部 zone 的连接、边条、覆盖、面的归类和尺寸，返回拓扑。

    顺序见模块说明。任何一处不合要求都抛 ``CgnsError``，消息用原名指出错在
    哪个 zone 的哪条连接或边条上。
    """
    zones = tuple(zones)
    if not zones:
        raise CgnsError("多块网格没有 zone")
    sizes = {}
    for zone in zones:
        if zone.name in sizes:
            raise CgnsError(
                f"zone 名 {zone.name!r} 出现了不止一次；供体按原始 zone 名解析，"
                "名字必须唯一")
        sizes[zone.name] = zone.vertex_size
    by_name = {zone.name: zone for zone in zones}

    # 1. 连接：先逐条按文件顺序校验，再配对、查几何
    records = [(zone, conn) for zone in zones for conn in zone.connections]
    placed = [_check_record(zone, conn, sizes) for zone, conn in records]
    interfaces, in_block = _pair_interfaces(records, placed, by_name)
    _check_periodic_origins(zones, in_block)

    # 2. 边条
    sources = _collect_sources(zones, interfaces, in_block)

    # 3. 覆盖表
    for zone in zones:
        for side in SIDES:
            _check_coverage(zone, side, sources[(zone.name, side)])

    # 4. 面的归类
    faces = {key: _classify(*key, items) for key, items in sources.items()}
    _pair_polar_faces(faces)

    # 5. 尺寸
    _check_sizes(faces, sizes)

    return Topology(zones=zones, interfaces=tuple(interfaces), faces=faces,
                    report=_report(zones, interfaces, faces))


# ---- 块的装配：交换表、坐标、度量、面的子集、边界执行器 ----


@dataclass(frozen=True)
class Exchange:
    """一个接口段在一个方向上的交换表。

    ``slots`` 是接收块含鬼胞数组里的 (k, j, i) 下标，``sources`` 是供体块里对应
    的下标，都是 (3, 点数) 的整数数组。点按数组轴 (k, j, i) 的 C 序铺开，法向轴
    上排的是鬼层号。接收槽全在鬼胞，供体源全在内区，两者不相交，所以从一个快照
    里读、往另一个块里写，先后顺序无关。坐标、度量、守恒量、粘性通量 ``Ev`` 共用
    这一张表。

    ``yielded_slots`` / ``yielded_sources`` 是本段让给别的段的顶点：T 形共线处的一条
    棱，阶梯或角点上的一个点。本段本来会往这些槽里写的值取自这些源。运行时不写
    它们，只留给审计核对。
    """

    interface: Interface
    slots: np.ndarray
    sources: np.ndarray
    yielded_slots: np.ndarray
    yielded_sources: np.ndarray


@dataclass(frozen=True)
class Block:
    """一个运行块：自己的网格、坐标、度量、面的子集和边界执行器。

    ``coords`` 和 ``metrics`` 含鬼胞。``faces`` 只含物理面和块内周期面，接口面
    不在其中；它为空时调用方要跳过 ``execution.prepare``。轴级的
    ``execution.bcs`` 是占位值，``faces`` 非空时 ``prepare`` 不读它。
    """

    name: str
    grid: Grid
    dxs: tuple
    axes: tuple
    coords: Mapping
    metrics: Mapping
    faces: tuple
    execution: boundary.BoundaryExecution
    cons0: Any


@dataclass(frozen=True)
class BlockSet:
    """装配好的全部块和它们的交换表。

    ``gamma`` 是物性参数里的比热比；``kernels`` 缓存编好的求解函数，按（参考尺度、
    粘性导数的阶数）分开。
    """

    blocks: tuple[Block, ...]
    tables: tuple[Exchange, ...]
    gamma: float
    kernels: dict = field(default_factory=dict, compare=False, repr=False)

    def __getitem__(self, name: str) -> Block:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)


def _vertex_owners(side: str, segments, vertex) -> list[np.ndarray]:
    """接收面上每个顶点归哪一段接口：返回每一段自己负责的顶点（布尔表，按段的范围排）。

    ``segments`` 是落在接收面上的接口段，每项是 (接口对, 这个面是不是它的接收侧)；
    ``vertex`` 是接收块的顶点数。一个面上有几段时，它们只在棱或角上相接。相接的
    顶点归“两个切向轴上下标都较小的那个相邻面元”所属的段；顶点在面的边上、那一侧
    没有面元时，取有的那个。T 形共线处这就是切向起点较小的段拥有那条棱；长棱对着
    两条短棱的阶梯、四段相接的角点也只剩一个写入者，与段的先后无关。每个面元恰好
    属于一段（覆盖表查过），所以每个顶点恰好归一段。
    """
    first, second = [axis for axis in range(3) if axis != side_axis(side)]
    cells = np.full((vertex[second] - 1, vertex[first] - 1), -1, dtype=np.int64)
    spans = []
    for number, (pair, here) in enumerate(segments):
        rng = pair.point_range if here else pair.point_range_donor
        box = [sorted((rng[0][axis], rng[1][axis])) for axis in (first, second)]
        cells[box[1][0]:box[1][1], box[0][0]:box[0][1]] = number    # 顶点 lo..hi 的面元是 lo..hi-1
        spans.append(box)
    owned = []
    for number, (along_first, along_second) in enumerate(spans):
        cell_first = np.maximum(np.arange(along_first[0], along_first[1] + 1) - 1, 0)
        cell_second = np.maximum(np.arange(along_second[0], along_second[1] + 1) - 1, 0)
        owned.append(cells[cell_second[:, None], cell_first[None, :]] == number)
    return owned


def _expected_slots(name: str, side: str, sizes) -> np.ndarray:
    """接收面 ``name`` 的 ``side`` 上应收的鬼胞槽 (3, 点数)：接口面的内区顶点，
    往面外数四层。"""
    index = [np.arange(LAP, LAP + n) for n in sizes[name]]
    normal = side_axis(side)
    index[normal] = (np.arange(LAP) if side[1] == "-" else
                     np.arange(LAP + sizes[name][normal],
                               LAP + sizes[name][normal] + LAP))
    return np.stack(np.meshgrid(*(index[a] for a in (2, 1, 0)),
                                indexing="ij")).reshape(3, -1)


def _check_single_writer(topology, tables) -> None:
    """建表后的自检：每个接收面上应收的鬼胞槽恰好被一张表写到。

    应收的面从拓扑来（每个跨块接口的两侧），应收的槽按块的尺寸算（见
    :func:`_expected_slots`）。所以除了重复和漏写，写到了别的槽、或者整个面没有
    表，也查得出来。
    """
    sizes = {zone.name: zone.vertex_size for zone in topology.zones}
    faces = {(pair.zone, pair.side) for pair in topology.interfaces}
    faces |= {(pair.donor_zone, pair.donor_side) for pair in topology.interfaces}
    written = {}
    for table in tables:
        written.setdefault((table.interface.zone, table.interface.side),
                           []).append(table.slots)
    stray = sorted(set(written) - faces)
    if stray:
        raise CgnsError(f"交换表写到了不是接口面的 {stray} 上")
    for name, side in sorted(faces):
        wanted = _expected_slots(name, side, sizes)
        slots = (np.concatenate(written[(name, side)], axis=1) if (name, side) in written
                 else np.zeros((3, 0), dtype=np.int64))
        if (slots.shape[1] != wanted.shape[1]
                or not np.array_equal(np.unique(slots, axis=1), np.unique(wanted, axis=1))):
            raise CgnsError(
                f"{name} 的 {side} 面上应收 {wanted.shape[1]} 个鬼胞槽，交换表写到 "
                f"{slots.shape[1]} 个，与应收的槽对不上；每个槽恰好要有一张表写")


def build_exchange_tables(topology) -> tuple[Exchange, ...]:
    """为每个跨块的接口段建交换表，一对接口有来去两张。

    下标映射（R20，鬼层号 r=1..4，供体在法向的内区下标是 0..n-1）：

    ============  ======================  ===========================
    接口类型       接收块的低侧面          接收块的高侧面
                  （接供体的高侧面）      （接供体的低侧面）
    ============  ======================  ===========================
    普通切口       n-1-r                  r
    周期接缝       n-r                    r-1
    ============  ======================  ===========================

    接收范围只取面内区对应的部分，切向的鬼胞不换。一个接收面上有几段时，相接的
    顶点归哪一段见 :func:`_vertex_owners`；建完自检每个接收槽恰好被一张表写到。
    """
    sizes = {zone.name: zone.vertex_size for zone in topology.zones}
    layers = np.arange(1, LAP + 1)
    on_face = {}                # 接收面 → 落在它上面的接口段 (接口对, 这个面是不是它的接收侧)
    for pair in topology.interfaces:
        on_face.setdefault((pair.zone, pair.side), []).append((pair, True))
        on_face.setdefault((pair.donor_zone, pair.donor_side), []).append((pair, False))
    owned = {face: _vertex_owners(face[1], segments, sizes[face[0]])
             for face, segments in on_face.items()}
    tables = []
    for pair in topology.interfaces:
        back = Interface(                       # 反方向：供体接收，原接收块供体
            kind=pair.kind, zone=pair.donor_zone, side=pair.donor_side,
            donor_zone=pair.zone, donor_side=pair.side,
            point_range=pair.point_range_donor, point_range_donor=pair.point_range,
            names=pair.names[::-1], points=pair.points, max_gap=pair.max_gap,
            tolerance=pair.tolerance,
            translation=None if pair.translation is None else -pair.translation)
        for it in (pair, back):
            axis = side_axis(it.side)
            low = it.side.endswith("-")
            n = sizes[it.donor_zone][axis]
            if it.kind == SEAM:
                depth = n - layers if low else layers - 1
            else:
                depth = n - 1 - layers if low else layers
            face = (it.zone, it.side)              # 接收面，以及本段在它上面负责的顶点
            segments = on_face[face]
            keep = owned[face][[pair is segment for segment, _ in segments].index(True)]
            first, second = [a for a in range(3) if a != axis]
            mine, theirs = [], []
            for a in range(3):
                if a == axis:
                    mine.append(-layers if low else sizes[it.zone][axis] - 1 + layers)
                    theirs.append(depth)
                    continue
                lo, hi = sorted((it.point_range[0][a], it.point_range[1][a]))
                mine.append(np.arange(lo, hi + 1))
                # 两条记录的有向增量已经校验相等，倒写与否不影响这个差
                theirs.append(it.point_range_donor[0][a]
                              + (mine[-1] - it.point_range[0][a]))
            grid = np.meshgrid(*(mine[a] for a in (2, 1, 0)), indexing="ij")
            source = np.meshgrid(*(theirs[a] for a in (2, 1, 0)), indexing="ij")
            shape = [1, 1, 1]                      # 布尔表补上法向轴，好广播到四层鬼胞
            shape[2 - second] = keep.shape[0]      # 计算轴 a 对应数组轴 2-a
            shape[2 - first] = keep.shape[1]
            edge = ~np.broadcast_to(keep.reshape(shape), grid[0].shape)
            slots = (np.stack(grid) + LAP).reshape(3, -1)
            sources = (np.stack(source) + LAP).reshape(3, -1)
            edge = edge.reshape(-1)
            slot_depth, source_depth = slots[2 - axis] - LAP, sources[2 - axis] - LAP
            if not (np.all((slot_depth < 0) | (slot_depth >= sizes[it.zone][axis]))
                    and np.all((source_depth >= 0) & (source_depth < n))):
                raise ValueError(f"{it.names[0]} 的交换表越界：接收槽要在鬼胞里，供体源要在内区里")
            tables.append(Exchange(
                interface=it, slots=slots[:, ~edge], sources=sources[:, ~edge],
                yielded_slots=slots[:, edge], yielded_sources=sources[:, edge]))
    _check_single_writer(topology, tables)
    return tuple(tables)


def build_blocks(topology, args, whole=None) -> BlockSet:
    """把校验过的拓扑变成可计算的块。

    ``whole`` 是等价单块的全局顶点数 (Ni, Nj, Nk)，从旁记来；没有时各块按自己
    的轴长选坐标鬼层的规则。每块依次做四件事：

    1. 坐标：按块内周期和 ``whole`` 铺满，再按交换表把接口旁的四层换成供体的；
    2. 度量：用逐侧标志算，普通切口一侧为 0，物理面和周期接缝为 1；
    3. 普通切口旁四层的度量换成供体的内区度量，周期接缝不换；
    4. 取出物理面和块内周期面，按规范的六侧顺序交给 ``build_block_execution``。
    """
    params = physical_params(args)
    inlet = cons_from_state(args.boundary_inflow, params).reshape(5, 1, 1)
    farfield = cons_from_state(args.boundary_farfield, params)
    tables = build_exchange_tables(topology)
    face_of = topology.faces

    def sides(name, axis):
        return (face_of[(name, AXIS_NAMES[axis] + "-")],
                face_of[(name, AXIS_NAMES[axis] + "+")])

    coords = {}
    for zone in topology.zones:
        in_block = tuple(face_of[(zone.name, AXIS_NAMES[a] + "-")].kind == PERIODIC_FACE
                         for a in range(3))
        coords[zone.name] = {
            key: pad_coords(value, periodic=in_block, lengths=whole)
            for key, value in zone.coords.items()}
    for table in tables:
        it = table.interface
        for key in COORD_KEYS:
            coords[it.zone][key][tuple(table.slots)] = (
                coords[it.donor_zone][key][tuple(table.sources)])

    metrics = {}
    for zone in topology.zones:
        flags = tuple(0 if face.kind == INTERFACE and face.interfaces[0].kind == CUT else 1
                      for axis in range(3) for face in sides(zone.name, axis))
        metrics[zone.name] = {
            key: np.array(value) for key, value in compute_metrics_from_coords(
                coords[zone.name], (1.0, 1.0, 1.0), 6,
                side_flags=(flags[0:2], flags[2:4], flags[4:6])).items()}
    for table in tables:
        if table.interface.kind == CUT:
            for key in METRIC_KEYS:
                metrics[table.interface.zone][key][tuple(table.slots)] = (
                    metrics[table.interface.donor_zone][key][tuple(table.sources)])

    blocks = []
    for zone in topology.zones:
        grid = Grid(*zone.vertex_size)
        faces = []
        for side in SIDES:
            face = face_of[(zone.name, side)]
            if face.kind == PHYSICAL:
                faces.append(face_boundary(
                    side, bc_kind(face.bc_type),
                    f"CGNS {face.bc_type} {list(face.sources)}"))
            elif face.kind == PERIODIC_FACE:
                faces.append(face_boundary(side, PERIODIC, "块内周期"))
        faces = plate_faces(tuple(faces), args.boundary_wall_temperature)
        execution = boundary.build_block_execution(
            grid, faces, inflow=inlet, farfield=farfield, phys=params)
        blocks.append(Block(
            name=zone.name, grid=grid, dxs=(1.0, 1.0, 1.0), axes=(0, 1, 2),
            coords=MappingProxyType(coords[zone.name]),
            metrics=MappingProxyType(metrics[zone.name]), faces=faces,
            execution=execution,
            cons0=grid.alloc().at[grid.inner()].set(inlet.reshape(5, 1, 1, 1))))
    return BlockSet(blocks=tuple(blocks), tables=tables, gamma=params.gamma)


def exchange(tables, states) -> dict:
    """从交换前的快照同步交换各块的守恒量鬼胞，返回新的各块状态。

    ``states`` 是 {块名: 准备好的状态}，每个表从供体的内区读、往接收块的鬼胞写，
    读的都是交换前的 ``states``。``tables`` 的先后顺序不影响结果：表只写鬼胞、
    只读内区，不同的表写的槽互不重叠。调用方可以把它连同一个 jit 编译。
    """
    out = dict(states)
    for table in tables:
        it = table.interface
        out[it.zone] = out[it.zone].at[(slice(None),) + tuple(table.slots)].set(
            states[it.donor_zone][(slice(None),) + tuple(table.sources)])
    return out


@dataclass(frozen=True)
class CollinearCandidates:
    """T 形共线处一条被让掉的棱上的两个候选。"""

    interface: Interface       # 让掉这条棱的段
    written: Any               # 拥有者写进这些槽的值，从交换后的接收块上读
    candidate: Any             # 本段的供体在这些槽上的值，从交换前的供体上读


def audit_collinear(tables, before, after) -> tuple[CollinearCandidates, ...]:
    """导出相接处的候选：让掉棱或角点的段本来会写入的值，和拥有者实际写入的值。

    第一步时两个候选逐位相同（各块从同一份源数据切出）；各自推进之后由验收按
    容差核对。运行路径不开审计。
    """
    out = []
    for table in tables:
        if table.yielded_slots.shape[1] == 0:
            continue
        it = table.interface
        out.append(CollinearCandidates(
            interface=it,
            written=after[it.zone][(slice(None),) + tuple(table.yielded_slots)],
            candidate=before[it.donor_zone][(slice(None),) + tuple(table.yielded_sources)]))
    return tuple(out)


# ---- 编排：逐 RK 阶段做准备、交换、求导、组合 ----

@dataclass(frozen=True)
class StepTrace:
    """一个整步里各阶段的量，验收和诊断用。

    ``final`` 是整步后准备并交换过的状态，也是下一步的起点。``du`` 是三个阶段的
    右端项（粘性打开时含粘性项），``prepared`` 是三个阶段和整步末尾的四次准备，都是
    {块名: 数组}。``q1``、``q2`` 是准备之前的阶段态。``audits`` 是四次准备各次的共线
    候选，没开审计时每项都是空元组。

    ``viscous`` 是粘性打开时三个阶段的粘性分项 {块名: 数组}，关闭时是空元组。
    ``flux`` 是审计打开且粘性打开时三个阶段交换过的 ``Ev``，否则是空元组。
    """

    final: dict
    du: tuple
    prepared: tuple
    q1: dict
    q2: dict
    audits: tuple
    viscous: tuple = ()
    flux: tuple = ()


class BlockKernels:
    """各块 jit 好的物理准备、无粘 RHS、粘性的两段，以及整轮的同步交换。

    无粘 RHS 直接调 :func:`inviscid_rhs`，实参与单块 ``main.build_rhs`` 逐项对应：

    ===============  ===============================  ==================================
    实参              单块 main.build_rhs               多块
    ===============  ===============================  ==================================
    cons             右端项函数的入参 q                 准备且交换过的块状态
    grid             desc.grid                        block.grid
    gamma            case["gamma"]                    built.gamma
    bcs              execution.bcs                    block.execution.bcs（占位值）
    axes             desc.axes                        (0, 1, 2)
    dx               desc.dxs                         (1.0, 1.0, 1.0)
    scheme           args.scheme                      "WENO7"
    weno_tv          WENO7-SYMBO 且开关打开时给出      None
    metrics          desc.metrics                     block.metrics
    dif_bound        execution.band                   block.execution.band
    dif_axis_map     desc.dif_axis_map                (0, 1, 2)
    farfield         execution.farfield               block.execution.farfield
    faces            execution.faces                  block.faces
    flux_scale       flux_scales_for(grid, dxs, ...)   配置给的三元组，按 i、j、k 排列
    ===============  ===============================  ==================================

    ``vis_order`` 不是 ``None`` 时粘性打开，每块多两个函数，阶数是 ``vis_order``
    （CD6 是 6，CD8 是 8）。``flux`` 由状态算粘性通量 ``Ev``，直接调
    :func:`viscous_fluxes`，实参与单块 ``main.build_rhs`` 调用 ``viscous_rhs`` 时逐项对应：

    ====================  ===============================  ==================================
    实参                  单块 main.build_rhs               多块
    ====================  ===============================  ==================================
    cons                  右端项函数里准备过的 q            准备且交换过的块状态
    grid                  desc.grid                        block.grid
    params                case["params"]                   block.execution.phys
    bcs                   execution.bcs                    block.execution.bcs（占位值）
    axes                  desc.axes                        (0, 1, 2)
    dx                    desc.dxs                         block.dxs，即 (1.0, 1.0, 1.0)
    order                 vis_order                        ``vis_order``
    metrics               desc.metrics                     block.metrics
    dif_bound             execution.band                   block.execution.band
    dif_axis_map          desc.dif_axis_map                (0, 1, 2)
    boundary_execution    execution                        block.execution
    farfield              execution.farfield               block.execution.farfield
    faces                 execution.faces                  block.faces；没有面时是空元组
    zero_state_ghosts     False                            False
    fill_state_halos      False                            False
    ====================  ===============================  ==================================

    ``divergence`` 对交换过的 ``Ev`` 求导，直接调 :func:`viscous_divergence`；
    ``grid``、``axes``、``dx``、``order``、``metrics``、``dif_bound``、``dif_axis_map``
    与上表相同。

    ``swap`` 搬的是含鬼胞数组的下标，最前面一根轴整段一起搬：守恒量是 5 个分量，
    ``Ev`` 是 12 个平面，都用它。

    多块里恒为空的项：体积力、槽道源项。没有物理面的块没有 ``prepare``。
    """

    def __init__(self, built: BlockSet, flux_scale, vis_order=None, plain=None):
        self.blocks = built.blocks
        self.names = [block.name for block in built.blocks]
        self.tables = built.tables
        self.vis_order = vis_order
        if plain is None:
            plain = _plain_kernels(built, flux_scale)
        # 与粘性阶数无关的函数用同一批：compile_blocks 在同一组参考尺度下已经编过时就传进来
        self.prepare, self.rhs = plain.prepare, plain.rhs
        self.q1, self.q2, self.q3, self.swap = plain.q1, plain.q2, plain.q3, plain.swap
        # 粘性关闭时不编这两个函数，求解函数与无粘时相同
        self.flux = ([flux_kernel(block, vis_order) for block in built.blocks]
                     if vis_order is not None else [])
        self.divergence = ([divergence_kernel(block, vis_order) for block in built.blocks]
                           if vis_order is not None else [])


def flux_kernel(block, order):
    """一块“由状态算 Ev”的 jit 函数，实参见 :class:`BlockKernels` 说明里的第二张表。

    测试里只要换粘性这两段时直接用它，不必为了改动粘性把无粘的部分整套重编。
    """
    execution = block.execution
    return jax.jit(lambda q: viscous_fluxes(
        q, block.grid, execution.phys, bcs=execution.bcs, axes=block.axes,
        dx=block.dxs, order=order, metrics=block.metrics,
        dif_bound=execution.band, zero_state_ghosts=False,
        dif_axis_map=(0, 1, 2), boundary_execution=execution,
        farfield=execution.farfield, faces=block.faces, fill_state_halos=False))


def divergence_kernel(block, order):
    """一块“对 Ev 求导”的 jit 函数，实参见 :class:`BlockKernels` 的说明。"""
    execution = block.execution
    return jax.jit(lambda ev: viscous_divergence(
        ev, block.grid, axes=block.axes, dx=block.dxs, order=order,
        metrics=block.metrics, dif_bound=execution.band, dif_axis_map=(0, 1, 2)))


def _plain_kernels(built: BlockSet, flux_scale):
    """与粘性无关的那部分求解函数：各块的物理准备、无粘 RHS、RK 组合和整轮的守恒量交换。"""
    gamma = built.gamma

    def rhs_of(block):
        return jax.jit(lambda q: inviscid_rhs(
            q, block.grid, gamma=gamma, bcs=block.execution.bcs,
            axes=block.axes, dx=block.dxs, scheme="WENO7", weno_tv=None,
            metrics=block.metrics, dif_bound=block.execution.band,
            zero_state_ghosts=False, dif_axis_map=(0, 1, 2),
            farfield=block.execution.farfield, faces=block.faces,
            fill_state_halos=False, flux_scale=flux_scale))

    prepare = [jax.jit(block.execution.prepare) if block.faces else None
               for block in built.blocks]
    rhs = [rhs_of(block) for block in built.blocks]
    q1, q2, q3 = jax.jit(rk3_q1), jax.jit(rk3_q2), jax.jit(rk3_q3)
    # 整轮交换编成一个函数：接口多、块小时，逐表在 jit 外做会被宿主开销压倒
    where = {block.name: n for n, block in enumerate(built.blocks)}
    plan = [(where[t.interface.zone], where[t.interface.donor_zone],
             jnp.asarray(t.slots), jnp.asarray(t.sources))
            for t in built.tables]

    def swap(*states):                         # 读的都是交换前的快照
        out = list(states)
        for recv, give, slots, sources in plan:
            out[recv] = out[recv].at[(slice(None),) + tuple(slots)].set(
                states[give][(slice(None),) + tuple(sources)])
        return tuple(out)

    return SimpleNamespace(prepare=prepare, rhs=rhs, q1=q1, q2=q2, q3=q3,
                           swap=jax.jit(swap))


def compile_blocks(built: BlockSet, flux_scale, vis_order=None) -> BlockKernels:
    """给一套块编求解函数；同一套块、同一组参考尺度、同一个粘性导数的阶数只编一次。

    ``vis_order`` 是粘性导数的阶数，CD6 是 6、CD8 是 8（表见 ``CD_ORDER``）；
    ``None`` 关闭粘性，编出来的求解函数与无粘时相同。与粘性无关的函数（物理准备、无粘
    RHS、RK 组合、守恒量交换）在同一组参考尺度下只编一次，几种阶数共用同一批。
    """
    key = (tuple(float(value) for value in flux_scale), vis_order)
    if key not in built.kernels:
        plain = next((kernels for (scale, _), kernels in built.kernels.items()
                      if scale == key[0]), None)
        built.kernels[key] = BlockKernels(built, key[0], key[1], plain)
    return built.kernels[key]


def solve_step(kernels: BlockKernels, cons: dict, dt, order=None,
               audit=False) -> StepTrace:
    """推进一个整步（R30），返回这一步各个阶段的量。

    每个阶段按固定顺序：各块从自己的原始阶段态出发完成物理准备；全块都准备好后，
    从这份状态同步交换守恒量鬼胞；各块用它算无粘右端项；最后用原始阶段态和整步
    起点做 RK 组合。整步组合完成后全块再做一次带交换的准备，交给下一步。

    ``cons`` 是 {块名: 状态}。``order`` 是交换表的遍历顺序，缺省按拓扑里的顺序，
    结果与它无关。``audit`` 打开时导出每次准备的共线候选；粘性打开时另外导出
    三个阶段交换过的 ``Ev``。

    粘性打开时（``kernels.vis_order`` 不是 ``None``），算完无粘右端项以后，各块再用
    同一份状态算自己的 ``Ev``，全块同步交换一次 ``Ev``，各块对换过的 ``Ev`` 求导，
    累加进右端项。
    """
    names = kernels.names
    tables = kernels.tables if order is None else [
        kernels.tables[index] for index in order]
    count = len(names)

    def swap_all(pieces):
        """全块同步交换一次，守恒量和 ``Ev`` 都走这里，读的都是交换前的快照。jit
        路径整轮编成一个函数；急切路径按 ``tables`` 的顺序。"""
        if order is None:
            return list(kernels.swap(*pieces))
        swapped = exchange(tables, dict(zip(names, pieces)))
        return [swapped[name] for name in names]

    def get_ready(stage):
        """物理准备，然后同步交换；返回 (准备好的各块, 准备并交换过的各块)。"""
        ready = [prepare(state) if prepare is not None else state
                 for prepare, state in zip(kernels.prepare, stage)]
        return ready, swap_all(ready)

    def get_audit(ready, state):
        if not audit:
            return ()
        return audit_collinear(tables, dict(zip(names, ready)), dict(zip(names, state)))

    raw = [cons[name] for name in names]
    force, viscous, fluxes, prepared, audits = [], [], [], [], []
    stage, first, second = raw, None, None
    for index in range(3):
        ready, state = get_ready(stage)
        du = [rhs(q) for rhs, q in zip(kernels.rhs, state)]
        if kernels.vis_order is not None:
            own = [flux(q) for flux, q in zip(kernels.flux, state)]
            moved = swap_all(own)
            part = [divergence(ev) for divergence, ev in zip(kernels.divergence, moved)]
            du = [a + b for a, b in zip(du, part)]
            viscous.append(dict(zip(names, part)))
            if audit:
                fluxes.append(dict(zip(names, moved)))
        force.append(dict(zip(names, du)))
        prepared.append(dict(zip(names, state)))
        audits.append(get_audit(ready, state))
        if index == 0:
            first = [kernels.q1(raw[n], dt, du[n]) for n in range(count)]
            stage = first
        elif index == 1:
            second = [kernels.q2(raw[n], dt, first[n], du[n]) for n in range(count)]
            stage = second
    third = [kernels.q3(raw[n], dt, second[n], force[2][name])
             for n, name in enumerate(names)]
    ready, state = get_ready(third)
    prepared.append(dict(zip(names, state)))
    audits.append(get_audit(ready, state))
    return StepTrace(final=dict(zip(names, state)), du=tuple(force),
                     prepared=tuple(prepared), q1=dict(zip(names, first)),
                     q2=dict(zip(names, second)), audits=tuple(audits),
                     viscous=tuple(viscous), flux=tuple(fluxes))


def solve(kernels: BlockKernels, dt, steps: int, initial=None, order=None,
          audit=False) -> list:
    """串行推进 ``steps`` 个整步，返回每一步的量。``initial`` 缺省是各块的默认初场。"""
    cons = initial if initial is not None else {
        block.name: block.cons0 for block in kernels.blocks}
    traces = []
    for _ in range(steps):
        trace = solve_step(kernels, cons, dt, order=order, audit=audit)
        traces.append(trace)
        cons = trace.final
    return traces


def _sidecar(mesh):
    """网格旁边的旁记 JSON。没有旁记文件时返回 None；内容不是对象时返回空字典。

    旁记文件在而内容读不出来是 ``SetupError``：悄悄当作没有，各块就按自己的轴长选
    坐标鬼层的规则，短块物理面旁的坐标会变。
    """
    path = Path(mesh).with_suffix(".json")
    try:
        note = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:      # 语法坏了、读不了、嵌套太深：一律算旁记读不出来
        raise SetupError(f"多块网格的旁记 {path} 读不出来：{exc}") from exc
    return note if isinstance(note, dict) else {}


def _block_list(blocks, note) -> dict:
    """检查点的块清单：每块的名字和尺寸。旁记记着等价单块的全局顶点数和各块的起止
    下标、并且与网格对得上时，把它们也记进去。

    对得上是指块名按顺序与网格的 zone 一致，每一块 ``stop − start`` 等于该块的尺寸。
    旁记里没有这些项，或者尺寸对不上时，只记名字和尺寸，不提示；旁记在、块名却对
    不上时另外打印一句提示。
    """
    sizes = [[block.grid.ni, block.grid.nj, block.grid.nk] for block in blocks]
    plain = [{"name": block.name, "size": size} for block, size in zip(blocks, sizes)]
    entries = note.get("blocks") if note else None
    names = ([entry.get("name") for entry in entries]
             if isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries)
             else [])
    if names != [block.name for block in blocks]:
        if note:
            print("提示：多块网格的旁记里的块名与网格的 zone 顺序对不上，"
                  "检查点不记各块的位置")
        return {"blocks": plain}
    if ("global_vertices" in note and len(entries) == len(blocks)
            and all(_window(entry, size) for entry, size in zip(entries, sizes))):
        return {"global_vertices": list(note["global_vertices"]),
                "blocks": [dict(entry, size=size) for entry, size in zip(entries, sizes)]}
    return {"blocks": plain}


def _window(entry, size) -> bool:
    """旁记里这一块的起止下标是三个数，并且 ``stop − start`` 等于 ``size``。"""
    start, stop = entry.get("start"), entry.get("stop")
    if not (isinstance(start, list) and isinstance(stop, list)
            and len(start) == len(stop) == 3):
        return False
    return [b - a for a, b in zip(start, stop)] == size


def _global_sizes(mesh):
    """网格旁边的旁记 JSON 记着等价单块的全局顶点数时取它；没有旁记文件、或旁记里
    没记这一项时返回 None。

    旁记文件在而内容读不出来是 ``SetupError``，见 :func:`_sidecar`。
    """
    note = _sidecar(mesh)
    if note is None or "global_vertices" not in note:
        return None                               # 旁记里没记这一项
    whole = note["global_vertices"]
    reason = _why_not_sizes(whole)
    if reason is not None:
        raise SetupError(
            f"多块网格的旁记 {Path(mesh).with_suffix('.json')} 里 "
            f"global_vertices={whole!r} 不对：{reason}")
    return tuple(int(value) for value in whole)


def _why_not_sizes(whole) -> str | None:
    """``global_vertices`` 不是等价单块的全局顶点数 (Ni, Nj, Nk) 时返回原因，合格
    返回 None。合格的写法是三个 1 到 2³¹−1 的整数（CGNS 的下标范围是 32 位整数）；
    整数值的小数可以，字符串、布尔值都不算数字。"""
    if isinstance(whole, (str, bytes)) or not isinstance(whole, (list, tuple)):
        return "要写成含三个正整数的列表"
    if len(whole) != 3:
        return f"要有三个数，得到 {len(whole)} 个"
    for value in whole:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return f"{value!r} 不是数字"
        if isinstance(value, numbers.Integral):    # 整数单独判：超大的转不成浮点数
            integral = value
        elif np.isfinite(value) and value == int(value):
            integral = int(value)
        else:
            return f"{value!r} 不是整数"
        if not 1 <= integral <= MAX_VERTICES:
            return f"{value!r} 不在 1 到 {MAX_VERTICES} 之间"
    return None


def _restart(args, built, params):
    """读多块的检查点，核对网格，给出续算的起点、时间和步号。

    每块以自己的 ``cons0`` 为底，只把内区换成检查点的原始变量换算出的守恒量。报错分
    两类，与单块相同。``ConfigError``：这一次的输出会覆盖这张检查点、网格对不上。
    ``SetupError``：检查点读不了、给反了。返回 ({块名: 起点}, t, step, 启动时要提示的事)。
    """
    try:
        meta, blocks, fields = io.load_checkpoint(args.restart, params)
        step, t = io.restart_clock(meta)
    except io.CheckpointError as exc:
        raise SetupError(f"续算的检查点不可用：{exc}") from exc
    if not blocks:
        raise SetupError("续算的检查点不可用：这是单块的检查点，多块运行只能从多块的检查点续算")
    prefix = getattr(args, "out_prefix", None) or args.case
    clash = io.clobber_step(args.restart, args.out_dir, prefix, step)
    if clash is not None:
        raise runtime_config.ConfigError(
            f"续算的检查点 {args.restart} 会被这次运行在第 {clash} 步写出的文件覆盖"
            "（文件名里的步号比它记的步号大）；换一个输出目录，或把检查点挪走")
    zones = [(block.name, (block.grid.ni, block.grid.nj, block.grid.nk)) for block in built.blocks]
    problems, unchecked = io.check_mesh(meta, args.case, zones, blocks["blocks"])
    if problems:
        raise runtime_config.ConfigError(
            "续算的检查点与这一次的网格对不上：" + "；".join(problems))
    start, changed, dtype = {}, [], None
    for block, field in zip(built.blocks, fields):
        dtype = np.asarray(block.cons0).dtype
        changed.append(field.dtype != dtype)
        own = np.array(block.cons0)
        own[block.grid.inner()] = np.asarray(io.conservative_of(field.astype(dtype), params))
        start[block.name] = own
    messages = []
    if unchecked:
        messages.append("检查点里没有记这些项，没有核对：" + "、".join(unchecked))
    if any(changed):
        messages.append(f"精度变了：检查点是 {field.dtype}，这一次是 {dtype}，续出来的值会跟着变")
    return start, t, step, messages


def run(args, setup: MultiSetup) -> None:
    """多块运行的主循环：逐块串行推进，每个保存步存一个检查点（R30、R33、KTD10）。

    步长、终止条件和保存时机与单块的 ``main.run`` 相同。每个显示步对每块各打印
    一行诊断。检查点里每块一个数组，带一份块清单，``_meta`` 不写 ``N``。粘性打开时
    启动信息多一行：粘性导数的阶数和 Re、Ama、参考温度、Pr；参考温度取缺省值时的
    提示与单块相同。给了 ``--restart`` 就从那张检查点接着算。
    """
    whole = _global_sizes(args.mesh)
    try:
        built = build_blocks(setup.topology, args, whole)
        # plate 不强制粘性，生效的粘性就是 args.viscous；导数的阶数与单块同一张表
        vis_order = CD_ORDER[args.vis_order] if args.viscous else None
        kernels = compile_blocks(built, setup.flux_scale, vis_order)
    except mesh_cgns.CgnsError as exc:       # 交换表建完后的自检
        raise SetupError(f"多块网格没有通过校验：{exc}") from exc
    except Exception as exc:
        raise SetupError(f"多块网格装配失败：{exc}") from exc
    params = physical_params(args)
    prefix = getattr(args, "out_prefix", None) or args.case
    listing = _block_list(built.blocks, _sidecar(args.mesh))
    print(f"case={args.case} blocks={len(built.blocks)} scheme={args.scheme} "
          f"viscous={vis_order is not None} precision={config.precision_name()} "
          f"flux_scale={tuple(setup.flux_scale)} dt-mode=fixed({args.dt}) "
          f"out-prefix={prefix} "
          f"config={getattr(args, 'config_path', None) or 'cli'}")
    if vis_order is not None:
        print(f"vis-order={args.vis_order} Re={params.Re} Ama={params.Ama} "
              f"Ref_T={params.Ref_T} Pr={params.Pr}")
        if args.ref_t is None:
            print(REF_T_WARNING)

    cons, t, step = {block.name: block.cons0 for block in built.blocks}, 0.0, 0
    if getattr(args, "restart", None):            # 给了空字符串也算没给，从头算
        cons, t, step, messages = _restart(args, built, params)
        print(f"restart: file={args.restart} step={step} t={t}")
        for message in messages:
            print(f"提示：{message}")
    t0 = time.time()
    # 进循环以前先用起点状态空跑一步、结果丢掉：所有编译都在这一步完成，编译出的错
    # 和第一步跑出来的错分得开。solve_step 是纯函数，不改变 cons。
    try:
        solve_step(kernels, cons, args.dt)
    except Exception as exc:
        raise RuntimeExecutionError(
            "运行期错误（RHS 预热）："
            f"case={args.case} step=0 t={t} dt={args.dt} "
            f"blocks={len(built.blocks)} "
            f"原始异常={type(exc).__name__}: {exc}") from exc
    while t < args.end_time and step < args.end_step:
        dt = min(args.dt, args.end_time - t)
        try:
            cons = solve_step(kernels, cons, dt).final
        except Exception as exc:
            raise RuntimeExecutionError(
                "运行期错误（时间推进）："
                f"case={args.case} step={step + 1} t={t} dt={dt} "
                f"blocks={len(built.blocks)} "
                f"原始异常={type(exc).__name__}: {exc}") from exc
        t += dt
        step += 1
        if step % args.kstep_show == 0:
            for block in built.blocks:
                print(f"{block.name}: " + diagnostic_line(
                    cons[block.name], block.grid, t, step, time.time() - t0))
        if io.time_to_save(step, args.kstep_save, t >= args.end_time):
            path = f"{args.out_dir}/{io.checkpoint_name(prefix, step)}"
            try:
                io.save_checkpoint(
                    path,
                    {f"b{index}": io.primitive_field(cons[block.name], block.grid, params)
                     for index, block in enumerate(built.blocks)},
                    meta=io.checkpoint_meta(
                        args, params, t, step, dt, viscous=bool(args.viscous),
                        mesh={"format": "cgns", "file": args.mesh}),
                    blocks=listing)
            except Exception as exc:
                raise RuntimeExecutionError(
                    "运行期错误（存盘）："
                    f"case={args.case} step={step} t={t} dt={dt} "
                    f"blocks={len(built.blocks)} "
                    f"输出路径={path} 原始异常={type(exc).__name__}: {exc}") from exc
    print(f"done: t={t:.4f}, steps={step}, wall={time.time() - t0:.2f}s")


# ---- 入口判定与放行校验 ----

# 多块模式只放行缺省值、显式给出就报错的参数；出处由 runtime_config.where 反查
_NOT_GIVEN = (
    "weno_tv",
    "boundary_outlet_ghost",
    "ni", "nj", "nk",
    "tw",
    "inflow_profile",
    "plate_x0",
    "plate_x1",
    "plate_h",
    "plate_dif_bound",
)


@dataclass(frozen=True)
class MultiSetup:
    """多块路径的入口结果：校验过的拓扑和放行的配置。"""

    topology: Topology
    flux_scale: tuple[float, float, float]


def _given(args, name) -> bool:
    """这个参数是用户显式给出的（JSON 或命令行），不是取了缺省值。"""
    return getattr(args, "config_sources", {}).get(name, "default") != "default"


def _check_release(args) -> None:
    """放行校验：生效值不在放行清单里、或显式给出了不参与的参数，都报错。"""
    sources = getattr(args, "config_sources", None)

    def stop(name: str, text: str) -> None:
        raise SetupError(f"多块模式不放行：{runtime_config.blame(sources, name, text)}")

    if getattr(args, "case", None) != "plate":
        stop("case", f"算例是 {getattr(args, 'case', None)!r}，只放行 plate")
    if getattr(args, "scheme", None) != "WENO7":
        stop("scheme", f"格式是 {getattr(args, 'scheme', None)!r}，只放行 WENO7")
    if getattr(args, "fp32", False):
        stop("fp32", "精度是 FP32，只放行 FP64")
    dt = getattr(args, "dt", None)
    if dt is None or not np.isfinite(dt) or dt <= 0.0:
        stop("dt", f"步长 dt={dt!r}，需要有限正数的固定步长")
    if getattr(args, "specificboundary", None):
        stop("specificboundary",
             f"带了特殊边界脚本 {args.specificboundary!r}（specificboundary），多块暂不支持")
    for name in _NOT_GIVEN:
        if _given(args, name):
            stop(name, "显式给出了这个参数；多块模式只放行它的缺省值，取值与缺省相同时也算给出")


def _check_flux_scale(args, meta) -> tuple[float, float, float]:
    """参考尺度要给出、是有限正数，并且不小于任何一块按自身尺寸算出的分量。"""
    sources = getattr(args, "config_sources", None)

    def stop(text: str) -> None:
        raise runtime_config.ConfigError(
            runtime_config.blame(sources, "flux_scale", text))

    scale = getattr(args, "flux_scale", None)
    if scale is None:
        stop("多块运行必须给出三轴参考尺度：取等价单块的全局顶点数里"
             "其余两轴 (N-1) 的乘积，100×40×40 对应 1521 3861 3861")
    scale = tuple(float(value) for value in scale)
    bad = [f"{value:g}" for value in scale if not np.isfinite(value) or value <= 0.0]
    if bad:
        stop(f"每个分量都要是有限正数，得到 {bad}")
    bounds = [scales_from_sizes(zone.vertex_size) for zone in meta]
    for axis, name in enumerate(AXIS_NAMES):
        widest = int(np.argmax([bound[axis] for bound in bounds]))
        if scale[axis] < bounds[widest][axis]:
            stop(f"在 {name} 向的分量 {scale[axis]:g} 小于按块 "
                 f"{meta[widest].name} 自身尺寸算出的 {bounds[widest][axis]:g}；"
                 "参考尺度取等价单块的全局顶点数，不能按某一块的尺寸填")
    return scale


def _check_config(args, meta) -> tuple[float, float, float]:
    """配置层面的要求：输出间隔为正，边条状态齐全，壁温为正，参考尺度有着落。"""
    sources = getattr(args, "config_sources", None)
    for name in ("kstep_show", "kstep_save"):
        if getattr(args, name, 0) <= 0:
            raise runtime_config.ConfigError(runtime_config.blame(
                sources, name, f"输出间隔 {name} 必须是正数，得到 {getattr(args, name)}"))
    labels = (("boundary_inflow", "inflow"),
              ("boundary_farfield", "farfield"),
              ("boundary_wall_temperature", "wall_temperature"))
    missing = [f"boundary.{label}" for key, label in labels
               if getattr(args, key, None) is None]
    if missing:
        raise runtime_config.ConfigError(
            f"plate 的多块运行缺少 boundary 参数 {missing}；入口、远场和壁温由 "
            "JSON 给出，命令行没有对应的选项")
    if args.boundary_wall_temperature <= 0:
        raise runtime_config.ConfigError(runtime_config.blame(
            sources, "boundary_wall_temperature",
            "plate 的多块运行要求等温壁的 boundary.wall_temperature 大于 0，得到 "
            f"{args.boundary_wall_temperature:g}"))
    return _check_flux_scale(args, meta)


def enter(args) -> MultiSetup | None:
    """入口判定与放行校验。返回 None 表示走单块路径。

    ``main.run`` 在开头调用它。有网格、没有 ``--cgns-zone``、文件里不止一个 zone
    时走多块路径。判定只用 zone 元数据：文件打不开、只有一个 zone 时交给单块路径，
    它的报错与没有多块路径时相同；不止一个 zone 而其中有读不出来的，按多块网格
    的校验报错。放行校验在读取坐标之前完成。
    """
    meta = None
    if (getattr(args, "mesh", None)
            and getattr(args, "cgns_zone", None) is None):
        try:
            meta = mesh_cgns.zone_metadata(args.mesh)
        except Exception:
            # 探测只为选路径。探不出来（打不开、没有 Base 或 Zone，或别的残缺）就交回
            # 单块路径，由它报错：探测这一步不许改变单块路径的报错。
            meta = None
    if meta is None or len(meta) < 2:
        if getattr(args, "flux_scale", None) is not None:
            raise runtime_config.ConfigError(runtime_config.blame(
                getattr(args, "config_sources", None), "flux_scale",
                "只在多块网格上使用：内置网格、单 zone 的文件和显式选了 --cgns-zone "
                "的运行不要给出"))
        return None
    problems = [zone.problem for zone in meta if zone.problem is not None]
    if problems:
        raise SetupError(f"多块网格没有通过校验：{problems[0]}")
    _check_release(args)
    scale = _check_config(args, meta)
    init.configure_precision(args)
    try:
        topology = build_topology(mesh_cgns.read_cgns_zones(args.mesh))
    except mesh_cgns.CgnsError as exc:
        raise SetupError(f"多块网格没有通过校验：{exc}") from exc
    except Exception as exc:
        # 读的是用户的文件：节点残缺的方式枚举不完，读取器抛出的任何异常都是网格
        # 没通过校验。消息里带着原始异常的类型和文字。
        raise SetupError(
            f"多块网格没有通过校验：{type(exc).__name__}: {exc}") from exc
    print(topology.report)
    return MultiSetup(topology=topology, flux_scale=scale)
