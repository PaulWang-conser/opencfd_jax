"""第六步的拓扑校验与接口报告（U8）。

锁五件事，按校验的先后：
  1. 连接：只认同向、低侧接高侧、切向点数分配相同、两条方向记录互逆的连接，两个
     范围各自的书写方向不算；``Transform`` 先于范围的各项报错；自连接只认相对整面；
     ``Periodic_t`` 的字段要齐全，周期接缝的判据只用整面上的对称量，换 zone 的先后、
     记录怎么倒写，结论都不变；
  2. 边条：必须是落在某个侧面上的面片，类型在可执行子集内；
  3. 覆盖表：每个面元恰好被一个来源覆盖，带面积的重叠和缺口都报错；
  4. 面的归类：物理面、接口面、块内周期面，混着的报错；
  5. 尺寸：物理面所在的轴至少 7 个顶点，普通切口的供体在法向至少 5 个。
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import mesh_cgns, multiblock
from opencfd_jax.src.mesh_cgns import CgnsError

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

from Grid.make_cgns_fixture import ConnSpec, build_multizone  # noqa: E402
import step6_support as support  # noqa: E402
from step6_support import (OUTER, WHOLE, block, cut_pair,  # noqa: E402
                           outer_bcs, seam_blocks, t_blocks, two_blocks)

CFL3D = REPO / "Grid" / "samples" / "multi_hdf5.cgns"
PERIODIC = {"RotationCenter": (0.0, 0.0, 0.0), "RotationAngle": (0.0, 0.0, 0.0)}


def _topology(tmp_path, zones):
    path = tmp_path / "mesh.cgns"
    build_multizone(path, zones)
    return multiblock.build_topology(mesh_cgns.read_cgns_zones(path))


def _rejected(tmp_path, zones):
    """校验必须报 ``CgnsError``，返回消息。"""
    with pytest.raises(CgnsError) as info:
        _topology(tmp_path, zones)
    return str(info.value)


_with_record = support.with_record

def _without(zones, zone, bcs=(), conns=()):
    """去掉 ``zone`` 里的若干边条和方向记录。"""
    return [replace(spec,
                    bcs=tuple(b for b in spec.bcs if b[0] not in bcs),
                    connections=tuple(c for c in spec.connections
                                      if c.name not in conns))
            if spec.name == zone else spec for spec in zones]


def _with_bcs(zones, zone, *added):
    """给 ``zone`` 加几条边条。"""
    return [replace(spec, bcs=tuple(spec.bcs) + added) if spec.name == zone else spec
            for spec in zones]


# ---- 通过的网格与报告 ----

def test_two_block_cut_passes_and_reports(tmp_path):
    top = _topology(tmp_path, two_blocks())       # 记录不带 Transform 节点
    assert len(top.interfaces) == 1
    it, = top.interfaces
    assert (it.kind, it.zone, it.side, it.donor_zone, it.donor_side) == (
        multiblock.CUT, "Zone 1", "i+", "Zone 2", "i-")
    assert (it.point_range, it.point_range_donor) == (
        ((7, 0, 0), (7, 6, 6)), ((0, 0, 0), (0, 6, 6)))
    assert (it.points, it.max_gap, it.translation) == (49, 0.0, None)
    assert it.names == ("Zone 1 / cut", "Zone 2 / cut")
    assert it.tolerance == multiblock.GEOMETRY_RTOL * 18.0     # 两侧坐标最大绝对值是 18
    kinds = [(top.faces[("Zone 1", side)].kind,
              top.faces[("Zone 1", side)].bc_type)
             for side in ("i-", "i+", "j-", "j+", "k-", "k+")]
    assert kinds == [("physical", "BCInflow"), ("interface", None),
                     ("physical", "BCWall"), ("physical", "BCFarfield"),
                     ("periodic", "BCSymmetryPolar"), ("periodic", "BCSymmetryPolar")]
    report = top.report
    for text in ("接口对 1 个（普通切口 1 个，周期接缝 0 个）",
                 "Zone 1 / cut ↔ Zone 2 / cut",
                 "Zone 1 的 i+ 面 ↔ Zone 2 的 i- 面",
                 "(7, 0, 0)..(7, 6, 6) ↔ (0, 0, 0)..(0, 6, 6)",
                 "49 个对应点", "最大坐标差 0.0e+00",
                 f"几何容差 {it.tolerance:.1e}",
                 "块内周期：Zone 1 的 k 轴", "块内周期：Zone 2 的 k 轴"):
        assert text in report


def test_seam_blocks_pass_and_report_translation(tmp_path):
    top = _topology(tmp_path, seam_blocks())
    cut, seam = top.interfaces
    assert (cut.kind, cut.zone, cut.side) == (multiblock.CUT, "Zone 1", "i+")
    assert (seam.kind, seam.zone, seam.side, seam.donor_zone, seam.donor_side) == (
        multiblock.SEAM, "Zone 1", "i-", "Zone 2", "i+")
    np.testing.assert_array_equal(seam.translation, [14.0, 0.0, 0.0])
    assert seam.max_gap == 0.0
    assert top.faces[("Zone 2", "i+")].kind == "interface"
    for text in ("周期接缝 1 个", "平移（供体减接收）(14.0, 0.0, 0.0)", "最大偏离 0.0e+00"):
        assert text in top.report


def test_t_mesh_passes_with_collinear_segments(tmp_path):
    top = _topology(tmp_path, t_blocks())
    assert [(it.names[0], it.side, it.donor_zone, it.donor_side)
            for it in top.interfaces] == [
        ("Zone 1 / cut lo", "i+", "Zone 2", "i-"),
        ("Zone 1 / cut hi", "i+", "Zone 3", "i-"),
        ("Zone 2 / cut mid", "j+", "Zone 3", "j-")]
    assert [(it.points, it.max_gap) for it in top.interfaces] == [
        (49, 0.0), (49, 0.0), (56, 0.0)]
    assert "接口对 3 个（普通切口 3 个" in top.report
    assert top.faces[("Zone 1", "i+")].interfaces == top.interfaces[:2]


# ---- 连接：Transform ----

def test_transform_other_than_identity_is_unsupported(tmp_path):
    message = _rejected(tmp_path, _with_record(
        two_blocks(), "Zone 1", "cut", transform=(1, -2, 3)))
    assert "Zone 1 / cut" in message and "Transform=[1, -2, 3]" in message
    assert "第六步暂不支持" in message


def test_zero_normal_component_of_transform_is_called_unsupported(tmp_path):
    message = _rejected(tmp_path, _with_record(
        two_blocks(), "Zone 1", "cut", transform=(0, 2, 3)))
    assert "第六步暂不支持" in message and "格式" not in message


def test_axis_swapped_transform_is_rejected_by_name(tmp_path):
    zones = _with_record(two_blocks(), "Zone 1", "cut", transform=(2, 1, 3),
                         point_range=OUTER,
                         point_range_donor=((1, 1, 1), (7, 1, 7)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "Transform=[2, 1, 3]" in message


def test_transform_is_reported_before_an_out_of_range_point_range(tmp_path):
    """同一条连接的 Transform 和接收范围都有错时，先报 Transform（方案 KTD3）。"""
    zones = _with_record(two_blocks(), "Zone 1", "cut",
                         point_range=((9, 1, 1), (9, 7, 7)), transform=(1, -2, 3))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "Transform=[1, -2, 3]" in message


def test_receiving_range_beyond_the_zone_is_rejected(tmp_path):
    zones = _with_record(two_blocks(), "Zone 1", "cut", point_range=((9, 1, 1), (9, 7, 7)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "超出 i 顶点数 8" in message


def test_transform_is_reported_before_a_non_positive_index(tmp_path):
    """下标非正也属于范围的问题，同样排在 Transform 之后（方案 KTD3）。"""
    zones = _with_record(two_blocks(), "Zone 1", "cut",
                         point_range=((0, 1, 1), (8, 7, 7)), transform=(1, -2, 3))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "Transform=[1, -2, 3]" in message


def test_non_positive_index_is_rejected_in_the_multiblock_path(tmp_path):
    zones = _with_record(two_blocks(), "Zone 1", "cut", point_range=((0, 1, 1), (8, 7, 7)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "超出 i 顶点数 8" in message


@pytest.mark.skipif(not CFL3D.exists(), reason="缺少 CFL3D 样例")
def test_cfl3d_sample_is_rejected_at_its_first_bad_connection():
    with pytest.raises(CgnsError) as info:
        multiblock.build_topology(mesh_cgns.read_cgns_zones(CFL3D))
    message = str(info.value)
    assert "Zone   1 / 1to1InterfaceA   1" in message
    assert "Transform=[1, -2, 3]" in message


# ---- 连接：范围、倒写、互逆 ----

def test_reversed_writing_must_be_on_both_sides(tmp_path):
    zones = two_blocks()
    back = (((8, 7, 7), (8, 1, 1)), ((1, 7, 7), (1, 1, 1)))
    half = (((8, 7, 1), (8, 1, 7)), ((1, 7, 1), (1, 1, 7)))
    both = _with_record(_with_record(
        zones, "Zone 1", "cut", point_range=back[0], point_range_donor=back[1]),
        "Zone 2", "cut", point_range=back[1], point_range_donor=back[0])
    assert len(_topology(tmp_path, both).interfaces) == 1      # 两侧一同倒写
    both_half = _with_record(_with_record(
        zones, "Zone 1", "cut", point_range=half[0], point_range_donor=half[1]),
        "Zone 2", "cut", point_range=half[1], point_range_donor=half[0])
    assert len(_topology(tmp_path, both_half).interfaces) == 1
    one = _with_record(zones, "Zone 1", "cut",
                       point_range=back[0], point_range_donor=WHOLE)
    message = _rejected(tmp_path, one)
    assert "Zone 1 / cut" in message and "只倒写了一侧" in message


def test_reverse_record_may_also_be_written_backwards_on_its_own(tmp_path):
    """只把反向记录的两个范围一同倒写：它的两个有向增量仍然相等，逐点映射与正向互逆。"""
    zones = _with_record(two_blocks(), "Zone 2", "cut",
                         point_range=((1, 7, 7), (1, 1, 1)),
                         point_range_donor=((8, 7, 7), (8, 1, 1)))
    top = _topology(tmp_path, zones)
    assert [(it.zone, it.side, it.donor_zone, it.donor_side)
            for it in top.interfaces] == [("Zone 1", "i+", "Zone 2", "i-")]
    straight = multiblock.build_exchange_tables(_topology(tmp_path, two_blocks()))
    for mine, theirs in zip(multiblock.build_exchange_tables(top), straight):
        np.testing.assert_array_equal(mine.slots, theirs.slots)
        np.testing.assert_array_equal(mine.sources, theirs.sources)


def test_reported_point_count_ignores_how_the_ranges_are_written(tmp_path):
    back = (((8, 7, 7), (8, 1, 1)), ((1, 7, 7), (1, 1, 1)))      # 三条轴都倒写
    zones = _with_record(_with_record(
        two_blocks(), "Zone 1", "cut", point_range=back[0], point_range_donor=back[1]),
        "Zone 2", "cut", point_range=back[1], point_range_donor=back[0])
    top = _topology(tmp_path, zones)
    assert top.interfaces[0].points == 49 and "49 个对应点" in top.report


def test_different_tangential_allocation_is_rejected(tmp_path):
    zones = _with_record(two_blocks(), "Zone 1", "cut",
                         point_range=((8, 1, 1), (8, 4, 7)),
                         point_range_donor=((1, 1, 1), (1, 7, 4)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "点数分配必须相同" in message


def test_same_sense_sides_are_rejected(tmp_path):
    zones = _with_record(two_blocks(), "Zone 2", "cut",
                         point_range=WHOLE, point_range_donor=WHOLE)
    message = _rejected(tmp_path, zones)
    assert "Zone 2 / cut" in message and "i- 面接给 Zone 1 的 i- 面" in message


def test_interior_plane_connection_is_rejected(tmp_path):
    zones = _with_record(two_blocks(), "Zone 1", "cut",
                         point_range=((4, 1, 1), (4, 7, 7)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "不在 Zone 1 的表面上" in message


def test_donor_range_beyond_donor_size_is_rejected(tmp_path):
    zones = _with_record(two_blocks(), "Zone 1", "cut",
                         point_range_donor=((1, 1, 1), (1, 9, 7)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "超出 j 顶点数 7" in message


def test_unknown_donor_is_rejected(tmp_path):
    message = _rejected(tmp_path, _with_record(
        two_blocks(), "Zone 1", "cut", donor="Zone 9"))
    assert "Zone 1 / cut" in message and "供体 zone 'Zone 9' 不存在" in message


def test_missing_reverse_record_is_rejected(tmp_path):
    message = _rejected(tmp_path, _without(two_blocks(), "Zone 2", conns=["cut"]))
    assert "Zone 1 / cut" in message and "互逆" in message
    assert "Zone 2 的 i- 面接回 Zone 1 的 i+ 面" in message


def test_non_reciprocal_reverse_record_is_rejected(tmp_path):
    smaller = (((1, 1, 1), (1, 6, 6)), ((8, 1, 1), (8, 6, 6)))
    zones = _with_record(two_blocks(), "Zone 2", "cut",
                         point_range=smaller[0], point_range_donor=smaller[1])
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "Zone 2 / cut" in message and "不互逆" in message


def test_zone_names_that_coincide_after_stripping_are_rejected(tmp_path):
    left, right = two_blocks()
    left = replace(left, connections=(
        replace(left.connections[0], donor="Zone 1 "),))
    right = replace(right, name="Zone 1 ", connections=(
        replace(right.connections[0], donor="Zone 1"),))
    assert "zone 名 'Zone 1' 出现了不止一次" in _rejected(tmp_path, [left, right])


# ---- 自连接与块内周期 ----

def _self_periodic(zones):
    """把各块 k± 的 BCSymmetryPolar 换成相对整面的 1to1 自连接。"""
    out = []
    for spec in zones:
        ni, nj, nk = spec.coords["Axx"].shape[::-1]
        low, high = ((1, 1, 1), (ni, nj, 1)), ((1, 1, nk), (ni, nj, nk))
        out.append(replace(
            spec, bcs=tuple(b for b in spec.bcs if b[1] != "BCSymmetryPolar"),
            connections=spec.connections + (
                ConnSpec("1to1Klo", spec.name, low, high),
                ConnSpec("1to1Khi", spec.name, high, low))))
    return out


def test_whole_face_self_connection_is_in_block_periodic(tmp_path):
    top = _topology(tmp_path, _self_periodic(two_blocks()))
    assert len(top.interfaces) == 1                # 自连接不算接口对
    face = top.faces[("Zone 1", "k-")]
    assert face.kind == "periodic"
    assert face.sources == ("Zone 1 / 1to1Klo", "Zone 1 / 1to1Khi")
    assert top.report.count("块内周期：") == 2       # 表头里那句“块内周期轴”不带冒号
    assert "Zone 1 / 1to1Klo ↔ Zone 1 / 1to1Khi" in top.report


def test_self_connection_on_one_side_or_partial_face_is_rejected(tmp_path):
    left, right = _without(two_blocks(), "Zone 1", bcs=["Klo", "Khi"])
    low, high = ((1, 1, 1), (8, 7, 1)), ((1, 1, 7), (8, 7, 7))
    same = [replace(left, connections=(ConnSpec("self", "Zone 1", low, low),)),
            right]
    message = _rejected(tmp_path, same)
    assert "Zone 1 / self" in message and "k- 面接给 Zone 1 的 k- 面" in message
    part_low, part_high = ((1, 1, 1), (8, 4, 1)), ((1, 1, 7), (8, 4, 7))
    partial = [replace(left, connections=(
        ConnSpec("self k-", "Zone 1", part_low, part_high),
        ConnSpec("self k+", "Zone 1", part_high, part_low))), right]
    message = _rejected(tmp_path, partial)
    assert "Zone 1 / self" in message and "相对的两个整面" in message


def test_self_connection_mixed_with_polar_pair_keeps_single_block_error(tmp_path):
    zones = _self_periodic(two_blocks())           # Polar 边条和自连接都留着
    polar = two_blocks()
    zones = [replace(z, bcs=polar[0].bcs) if index == 0 else z
             for index, z in enumerate(zones)]
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 k 轴周期两侧来源不同" in message
    assert "不与 1to1 自连接混配" in message


def test_periodic_on_self_connection_or_across_blocks_is_rejected(tmp_path):
    left, right = _without(two_blocks(), "Zone 1", bcs=["Klo", "Khi"])
    low, high = ((1, 1, 1), (8, 7, 1)), ((1, 1, 7), (8, 7, 7))
    periodic = dict(PERIODIC, Translation=(0.0, 0.0, 21.0))
    with_periodic = [replace(
        left, connections=left.connections + (
            ConnSpec("self k-", "Zone 1", low, high, periodic=periodic),
            ConnSpec("self k+", "Zone 1", high, low, periodic=dict(
                periodic, Translation=(0.0, 0.0, -21.0))))), right]
    message = _rejected(tmp_path, with_periodic)
    assert "Zone 1 / self" in message and "同块自连接却带 Periodic_t" in message
    # 不同块上的成对周期边条：左块 i−、右块 i+ 各写一个 BCSymmetryPolar
    zones = _without(two_blocks(), "Zone 1", bcs=["Ilo"])
    zones = _without(zones, "Zone 2", bcs=["Ihi"])
    zones = _with_bcs(zones, "Zone 1",
                      ("Ilo", "BCSymmetryPolar", ((1, 1, 1), (1, 7, 7))))
    zones = _with_bcs(zones, "Zone 2",
                      ("Ihi", "BCSymmetryPolar", ((8, 1, 1), (8, 7, 7))))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 i- 面是 BCSymmetryPolar" in message
    assert "跨块的周期要用块间 1to1 连接加 Periodic_t 表达" in message


# ---- Periodic_t 的字段与平移量 ----

def _seam(zones, zone, **periodic):
    return _with_record(zones, zone, "seam", periodic=periodic)


@pytest.mark.parametrize("periodic, pattern", [
    (dict(PERIODIC, Translation=(14.0, 0.0, 0.0)), None),
    ({"RotationAngle": (0.0, 0.0, 0.0), "Translation": (14.0, 0.0, 0.0)},
     "缺少 RotationCenter"),
    ({"RotationCenter": (0.0, 0.0), "RotationAngle": (0.0, 0.0, 0.0),
      "Translation": (14.0, 0.0, 0.0)}, "RotationCenter 长度是 (2,)"),
    (dict(PERIODIC, RotationAngle=(np.nan, 0.0, 0.0),
          Translation=(14.0, 0.0, 0.0)), "RotationAngle 含非有限值"),
    (dict(PERIODIC, RotationAngle=(0.1, 0.0, 0.0),
          Translation=(14.0, 0.0, 0.0)), "带旋转的周期"),
    (dict(PERIODIC, Translation=(-14.0, 0.0, 0.0)), "实测的平移量"),
    (dict(PERIODIC, Translation=(np.inf, 0.0, 0.0)), "Translation 含非有限值"),
], ids=["齐全", "缺旋转中心", "长度不对", "非有限", "旋转角非零", "符号写反", "平移非有限"])
def test_periodic_fields_are_checked(tmp_path, periodic, pattern):
    zones = _seam(seam_blocks(), "Zone 1", **periodic)
    if pattern is None:
        assert len(_topology(tmp_path, zones).interfaces) == 2
        return
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / seam" in message and pattern in message
    if pattern == "实测的平移量":
        assert "Translation=[-14.0, 0.0, 0.0]" in message
        assert "[14.0, 0.0, 0.0]" in message


def test_near_zero_translation_is_rejected_with_a_hint(tmp_path):
    zones = two_blocks()
    flat = dict(PERIODIC, Translation=(0.0, 0.0, 0.0))
    zones = _with_record(zones, "Zone 1", "cut", periodic=flat)
    zones = _with_record(zones, "Zone 2", "cut", periodic=flat)
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "去掉 Periodic_t" in message


def test_missing_translation_falls_back_to_constant_vector_check(tmp_path):
    zones = _seam(seam_blocks(), "Zone 1", **PERIODIC)
    zones = _seam(zones, "Zone 2", **PERIODIC)
    top = _topology(tmp_path, zones)               # 平移在整个面上是常向量
    np.testing.assert_array_equal(top.interfaces[1].translation, [14.0, 0.0, 0.0])
    right = zones[1]                               # 右块的 y 随 i 增长，平移不再是常向量
    i = right.coords["Axx"] - 7.0
    zones[1] = replace(right, coords=dict(
        right.coords, Ayy=right.coords["Ayy"] * (1.0 + 0.25 * i / 7.0)))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / seam" in message and "不是常向量" in message


def _tilted_seam(forward_offset, backward_offset=None):
    """seam_blocks 的副本：右块 i+ 面的 x 坐标沿 j 线性偏 1.5e-5（容差是 1.8e-5）；
    两条记录的 ``Translation`` 在首点实测值 ±14 的基础上再各偏一个量。"""
    zones = seam_blocks()
    right = zones[1]
    x = np.array(right.coords["Axx"])
    x[:, :, 7] += 1.5e-5 * np.arange(7) / 6.0
    zones[1] = replace(right, coords=dict(right.coords, Axx=x))
    if backward_offset is None:
        backward_offset = forward_offset
    return _seam(_seam(zones, "Zone 1",
                       **dict(PERIODIC, Translation=(14.0 + forward_offset, 0.0, 0.0))),
                 "Zone 2",
                 **dict(PERIODIC, Translation=(-14.0 - backward_offset, 0.0, 0.0)))


def test_given_translation_is_checked_at_every_vertex(tmp_path):
    """文件给了 Translation：每个顶点按它平移后的残差都要在容差内，不只是首点。"""
    top = _topology(tmp_path, _tilted_seam(0.0))       # 声明等于首点的实测值
    seam = top.interfaces[1]
    assert seam.max_gap == pytest.approx(0.75e-5)      # 实测位移离中点最远 0.75e-5
    assert "最大偏离 7.5e-06" in top.report
    # 往反方向偏 1.5e-5：与首点实测的差还在容差内，按它平移的残差却是 3.0e-5
    message = _rejected(tmp_path, _tilted_seam(-1.5e-5))
    assert "Zone 1 / seam" in message and "Translation=[13.999985, 0.0, 0.0]" in message
    assert "最大残差 3.000e-05" in message
    message = _rejected(tmp_path, _tilted_seam(0.0, -1.5e-5))    # 只有反向记录偏了
    assert "Zone 2 / seam" in message and "最大残差 3.000e-05" in message


def _offset_cut(offset, forward, backward):
    """two_blocks 的副本：切口两侧的 x 相差 ``offset``，两条记录都带 Periodic_t，
    ``Translation`` 分别写成 ``forward``、``backward``（None 表示不写）。"""
    zones = two_blocks()
    right = zones[1]
    x = np.array(right.coords["Axx"])
    x[:, :, 0] += offset
    zones[1] = replace(right, coords=dict(right.coords, Axx=x))
    for zone, translation in (("Zone 1", forward), ("Zone 2", backward)):
        if translation is not None:
            zones = _with_record(zones, zone, "cut",
                                 periodic=dict(PERIODIC, Translation=translation))
    return zones


def test_seam_verdict_does_not_depend_on_zone_order(tmp_path):
    """实测位移 1.4e-5、容差 1.8e-5，两条记录分别声明 2e-5 和 0：各自与实测的残差
    都合格，但两条声明彼此矛盾。正序、逆序都要拒绝，理由相同。"""
    zones = _offset_cut(1.4e-5, (2e-5, 0.0, 0.0), (0.0, 0.0, 0.0))
    for order in (zones, zones[::-1]):
        message = _rejected(tmp_path, order)
        assert "Zone 1 / cut" in message and "Zone 2 / cut" in message
        assert "Translation 不一致" in message


def test_near_zero_seam_is_judged_by_the_measured_shift_not_the_declaration(tmp_path):
    """位移只有 1.4e-5（不超过容差）的接缝，两条声明彼此一致也放行不了：坐标其实重合。"""
    zones = _offset_cut(1.4e-5, (2e-5, 0.0, 0.0), (-2e-5, 0.0, 0.0))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "平移量约为零" in message


def test_seam_whose_shift_overflows_is_rejected(tmp_path):
    zones = seam_blocks()
    moved = []
    for zone, column, value in zip(zones, (0, 7), (-1.5e308, 1.5e308)):
        coords = dict(zone.coords, Axx=np.array(zone.coords["Axx"]))
        coords["Axx"][:, :, column] = value          # 只动周期接缝那一面的 x
        moved.append(replace(zone, coords=coords))
    zones = _seam(_seam(moved, "Zone 1", **PERIODIC), "Zone 2", **PERIODIC)
    with np.errstate(over="ignore", invalid="ignore"):       # 故意让坐标差溢出
        message = _rejected(tmp_path, zones)
    assert "Zone 1 / seam" in message and "坐标差溢出" in message


def _varying_seam(low, high, forward=None, backward=None):
    """two_blocks 的副本：把中间的切口当成周期接缝，实测位移沿 j 从 ``low`` 线性变到
    ``high``。Zone 2 的记录倒写，所以两条记录的首点是不同的物理顶点。两条记录的
    ``Translation`` 是 ``forward``、``backward``，None 表示不写。"""
    zones = two_blocks()
    right = zones[1]
    x = np.array(right.coords["Axx"])
    x[:, :, 0] += low + (high - low) * np.arange(7)[:, None] / 6.0
    zones[1] = replace(right, coords=dict(right.coords, Axx=x))
    zones = _with_record(zones, "Zone 1", "cut", periodic=_periodic_with(forward))
    return _with_record(zones, "Zone 2", "cut", periodic=_periodic_with(backward),
                        point_range=((1, 7, 7), (1, 1, 1)),
                        point_range_donor=((8, 7, 7), (8, 1, 1)))


def _periodic_with(translation):
    periodic = dict(PERIODIC)
    if translation is not None:
        periodic["Translation"] = translation
    return periodic


def test_seam_verdict_is_the_same_whatever_the_zone_order(tmp_path):
    """判据只用整面上的对称量：实测位移的中点和半个跨度。换 zone 的先后、记录怎么
    倒写，都不变（容差 1.8e-5）。"""
    wide = _varying_seam(1.4e-5, 2.4e-5)        # 半个跨度 0.5e-5 合格，中点 1.9e-5 不算近零
    for order in (wide, wide[::-1]):
        seam, = _topology(tmp_path, order).interfaces
        assert seam.max_gap == pytest.approx(0.5e-5, rel=1e-6)
    narrow = _varying_seam(1.0e-5, 1.6e-5)      # 中点 1.3e-5，算近零
    for order in (narrow, narrow[::-1]):
        message = _rejected(tmp_path, order)
        assert "平移量约为零" in message


def test_seam_report_does_not_depend_on_which_declaration_comes_first(tmp_path):
    """两条声明都合格但略有差异：报告的最大偏离是实测的半个跨度，不随声明变。"""
    zones = _varying_seam(1.4e-5, 2.4e-5, forward=(1.9e-5, 0.0, 0.0),
                          backward=(-2.3e-5, 0.0, 0.0))
    for order, declared in ((zones, 1.9e-5), (zones[::-1], -2.3e-5)):
        seam, = _topology(tmp_path, order).interfaces
        assert seam.max_gap == pytest.approx(0.5e-5, rel=1e-6)
        assert seam.translation[0] == pytest.approx(declared)   # 平移量跟着前一条记录的方向


def test_seam_with_a_huge_constant_shift_stays_finite(tmp_path):
    """实测位移恒为 1e308 时中点也要算得出来：“最小值加最大值”会溢出，所以先减后除。
    报告里的平移量用 Python 的 round，它在这么大的数上也不溢出（Astra 第 20 轮）。"""
    def seam_zones(forward, backward):
        """两条记录都带 Periodic_t；``Translation`` 是 ``forward``、``backward``，None 不写。"""
        zones = _with_record(_offset_cut(1e308, None, None), "Zone 1", "cut",
                             periodic=_periodic_with(forward))
        return _with_record(zones, "Zone 2", "cut", periodic=_periodic_with(backward))

    huge = (1e308, 0.0, 0.0)
    for forward, backward in ((None, None), (huge, None), (None, (-1e308, 0.0, 0.0))):
        top = _topology(tmp_path, seam_zones(forward, backward))
        seam, = top.interfaces
        assert (seam.translation.tolist(), seam.max_gap) == ([1e308, 0.0, 0.0], 0.0)
        assert "平移（供体减接收）(1e+308, 0.0, 0.0)" in top.report


SEAM_REASONS = ("平移量约为零", "不是常向量", "不一致", "不符", "坐标差溢出", "非有限值")


def _exact_seam(values):
    """two_blocks 的副本，把切口当周期接缝。接收面（Zone 1 的 i+）的 x 置 0，供体面
    （Zone 2 的 i−）的 x 置成 ``values`` (k, j)，于是实测位移逐位等于 ``values``，
    可以把判据逼到容差的边界上。Zone 2 的记录倒写。"""
    left, right = two_blocks()
    x_left, x_right = np.array(left.coords["Axx"]), np.array(right.coords["Axx"])
    x_left[:, :, 7] = 0.0
    x_right[:, :, 0] = values
    zones = [replace(left, coords=dict(left.coords, Axx=x_left)),
             replace(right, coords=dict(right.coords, Axx=x_right))]
    zones = _with_record(zones, "Zone 1", "cut", periodic=dict(PERIODIC))
    return _with_record(zones, "Zone 2", "cut",
                        point_range=((1, 7, 7), (1, 1, 1)),
                        point_range_donor=((8, 7, 7), (8, 1, 1)),
                        periodic=dict(PERIODIC))


def _seam_verdict(tmp_path, zones):
    """接缝的结论：("放行", 平移量, 最大偏离) 或 ("拒绝", 原因)。"""
    try:
        seam, = _topology(tmp_path, zones).interfaces
    except CgnsError as exc:
        return ("拒绝",) + (next(word for word in SEAM_REASONS if word in str(exc)),)
    return "放行", seam.translation.copy(), seam.max_gap


@pytest.mark.parametrize("low, high", [
    (5.2e-6, 3.08e-5),        # 中点恰好等于容差 1.8e-5（Astra 第 21 轮的反例）
    (1.0e-5, 2.6e-5),         # 中点也恰好等于容差
    (0.0, 3.6e-5),            # 同上，跨度到容差的边界
    (1.79e-5, 1.81e-5),       # 中点在容差上，位移几乎是常量
    (1.4e-5, 2.4e-5),         # 中点 1.9e-5，比容差大，放行
    (1.0e-5, 1.6e-5),         # 中点 1.3e-5，近零
], ids=["恰好等于容差", "恰好等于容差 2", "跨度到边界", "容差上的常量", "略大于容差", "近零"])
def test_seam_verdict_at_the_tolerance_edge_does_not_depend_on_zone_order(
        tmp_path, low, high):
    """中点恰好压在容差上时，换 zone 的先后结论也要相同。中点按“最小值的一半加最大值的
    一半”算：除以 2 和取负在浮点里都是精确的，加法对正负号对称，所以换序后中点恰好只差
    符号。按“最小值加半个跨度”算时两者会差最后一位，一边等于容差、另一边略大，
    结论就变了（Astra 第 21 轮）。"""
    values = np.full((7, 7), low)
    values[3, 3] = high
    zones = _exact_seam(values)
    forward, backward = _seam_verdict(tmp_path, zones), _seam_verdict(tmp_path, zones[::-1])
    if forward[0] == "放行":
        assert backward[0] == "放行"
        assert np.array_equal(backward[1], -forward[1])      # 平移量恰好只差符号
        assert backward[2] == forward[2]
    else:
        assert backward == forward


def test_cut_coordinate_mismatch_is_rejected_with_max_gap(tmp_path):
    zones = two_blocks()
    zones[1] = replace(zones[1], coords=dict(
        zones[1].coords, Axx=zones[1].coords["Axx"] + 0.1))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "最大坐标差 1.000e-01" in message


def test_non_finite_coordinate_on_a_cut_is_rejected(tmp_path):
    zones = two_blocks()
    coords = {key: np.array(value) for key, value in zones[1].coords.items()}
    coords["Ayy"][3, 3, 0] = np.nan                  # Zone 2 的 i- 面（切口）上一个顶点
    zones[1] = replace(zones[1], coords=coords)
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / cut" in message and "坐标含非有限值" in message


# ---- 边条、覆盖表、面的归类 ----

def test_area_overlaps_are_rejected(tmp_path):
    zones = _with_bcs(two_blocks(), "Zone 1", ("Ihi", "BCWall", OUTER))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 i+ 面带面积的重叠" in message
    assert "边条 Ihi（BCWall）" in message and "接口段 Zone 1 / cut" in message
    zones = _with_bcs(two_blocks(), "Zone 1",
                      ("Jlo again", "BCWall", ((1, 1, 1), (4, 1, 7))))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 j- 面带面积的重叠" in message
    assert "边条 Jlo again（BCWall）" in message and "边条 Jlo（BCWall）" in message
    zones = two_blocks()
    extra = cut_pair(zones[0], zones[1], OUTER, WHOLE, name="dup")
    zones[1] = replace(zones[1], connections=zones[1].connections + (extra[1],))
    message = _rejected(tmp_path, zones)              # 供体那边把反向记录写了两遍
    assert "Zone 1 / cut 在 Zone 2 里有 2 条互逆的方向记录" in message
    zones = two_blocks()
    zones[0] = replace(zones[0], connections=zones[0].connections + (extra[0],))
    message = _rejected(tmp_path, zones)              # 接收这边写了两遍，反向记录只有一条
    assert "Zone 1 / dup 与 Zone 2 / cut 重复描述同一个接口" in message
    zones = two_blocks()
    lower = (((8, 1, 1), (8, 5, 7)), ((1, 1, 1), (1, 5, 7)))
    zones = _with_record(zones, "Zone 1", "cut",
                         point_range=lower[0], point_range_donor=lower[1])
    zones = _with_record(zones, "Zone 2", "cut",
                         point_range=lower[1], point_range_donor=lower[0])
    overlap = cut_pair(zones[0], zones[1], ((8, 4, 1), (8, 7, 7)),
                       ((1, 4, 1), (1, 7, 7)), name="overlap")
    zones[0] = replace(zones[0], connections=zones[0].connections + (overlap[0],))
    zones[1] = replace(zones[1], connections=zones[1].connections + (overlap[1],))
    message = _rejected(tmp_path, zones)              # 两段在 j=4 那一带上重叠
    assert "Zone 1 的 i+ 面带面积的重叠" in message
    assert "接口段 Zone 1 / overlap" in message and "接口段 Zone 1 / cut" in message


def test_gap_is_rejected_naming_face_and_range(tmp_path):
    message = _rejected(tmp_path, _without(two_blocks(), "Zone 1", bcs=["Jhi"]))
    assert "Zone 1 的 j+ 面有缺口" in message
    assert "i=0..7，k=0..6" in message


def test_bc_and_connection_on_one_face_is_rejected_at_classification(tmp_path):
    lower = (((8, 1, 1), (8, 4, 7)), ((1, 1, 1), (1, 4, 7)))
    zones = _with_record(two_blocks(), "Zone 1", "cut",
                         point_range=lower[0], point_range_donor=lower[1])
    zones = _with_record(zones, "Zone 2", "cut",
                         point_range=lower[1], point_range_donor=lower[0])
    zones = _with_bcs(zones, "Zone 1", ("Ihi upper", "BCWall", ((8, 4, 1), (8, 7, 7))))
    zones = _with_bcs(zones, "Zone 2", ("Ilo upper", "BCOutflow", ((1, 4, 1), (1, 7, 7))))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 i+ 面上边条段与接口段并存" in message
    assert "Ihi upper" in message and "Zone 1 / cut" in message


def test_split_bc_faces_merge_or_reject(tmp_path):
    zones = _without(two_blocks(), "Zone 1", bcs=["Jlo"])
    zones = _with_bcs(zones, "Zone 1",
                      ("Jlo a", "BCWall", ((1, 1, 1), (4, 1, 7))),
                      ("Jlo b", "BCWall", ((4, 1, 1), (8, 1, 7))))
    face = _topology(tmp_path, zones).faces[("Zone 1", "j-")]
    assert (face.kind, face.bc_type, face.sources) == (
        "physical", "BCWall", ("Jlo a", "Jlo b"))
    zones = _with_bcs(_without(two_blocks(), "Zone 1", bcs=["Jlo"]), "Zone 1",
                      ("Jlo a", "BCWall", ((1, 1, 1), (4, 1, 7))),
                      ("Jlo b", "BCFarfield", ((4, 1, 1), (8, 1, 7))))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 j- 面上边条段的类型不同" in message
    assert "BCWall Jlo a" in message and "BCFarfield Jlo b" in message


def test_cut_and_seam_on_one_face_is_rejected(tmp_path):
    left = block("Zone 1", 8, 13, 7)
    low = block("Zone 2", 8, 7, 7, origin=(7.0, 0.0, 0.0))
    far = block("Zone 3", 8, 7, 7, origin=(7.0, 36.0, 0.0))
    cut_low, from_low = cut_pair(left, low, ((8, 1, 1), (8, 7, 7)), WHOLE,
                                 name="cut lo")
    upper = ((8, 7, 1), (8, 13, 7))
    zones = [
        replace(left, bcs=outer_bcs(8, 13, 7, ("Ilo", "BCInflow")), connections=(
            cut_low,
            ConnSpec("seam hi", "Zone 3", upper, WHOLE,
                     periodic=dict(PERIODIC, Translation=(0.0, 24.0, 0.0))))),
        replace(low, bcs=outer_bcs(8, 7, 7, high=("Ihi", "BCOutflow")),
                connections=(from_low,)),
        replace(far, bcs=outer_bcs(8, 7, 7, high=("Ihi", "BCOutflow")),
                connections=(
                    ConnSpec("seam hi", "Zone 1", WHOLE, upper,
                             periodic=dict(PERIODIC, Translation=(0.0, -24.0, 0.0))),)),
    ]
    message = _rejected(tmp_path, zones)
    assert "Zone 1 的 i+ 面上普通接口段与周期接口段并存" in message
    assert "Zone 1 / cut lo" in message and "Zone 1 / seam hi" in message


@pytest.mark.parametrize("rng", [((1, 1, 1), (3, 3, 3)), ((1, 1, 1), (3, 1, 1)),
                                 ((1, 1, 1), (1, 1, 1))],
                         ids=["体型", "线状", "点状"])
def test_bc_patch_that_is_not_a_face_is_rejected(tmp_path, rng):
    zones = _with_bcs(two_blocks(), "Zone 1", ("odd", "BCWall", rng))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / odd" in message and "不是面片" in message


def test_unsupported_bc_type_is_rejected(tmp_path):
    zones = _with_bcs(two_blocks(), "Zone 1",
                      ("Kmid", "BCSymmetryPlane", ((1, 1, 1), (8, 7, 1))))
    message = _rejected(tmp_path, zones)
    assert "Zone 1 / Kmid" in message and "BCSymmetryPlane" in message and "执行子集" in message


# ---- 尺寸 ----

def test_short_axis_with_physical_or_periodic_face_is_rejected(tmp_path):
    def box(top_type):
        return replace(block("Zone 1", 8, 7, 6), bcs=(
            ("Ilo", "BCInflow", ((1, 1, 1), (1, 7, 6))),
            ("Ihi", "BCOutflow", ((8, 1, 1), (8, 7, 6))),
            ("Jlo", "BCWall", ((1, 1, 1), (8, 1, 6))),
            ("Jhi", "BCFarfield", ((1, 7, 1), (8, 7, 6))),
            ("Klo", top_type, ((1, 1, 1), (8, 7, 1))),
            ("Khi", top_type, ((1, 1, 6), (8, 7, 6)))))

    message = _rejected(tmp_path, [box("BCFarfield")])
    assert "Zone 1 的 k- 面是物理面，k 向只有 6 个顶点，至少 7 个" in message
    message = _rejected(tmp_path, [box("BCSymmetryPolar")])
    assert "Zone 1 的 k- 面是块内周期面，k 向只有 6 个顶点" in message


def test_cut_with_short_donor_is_rejected(tmp_path):
    small = replace(
        block("Zone 2", 4, 7, 7, origin=(7.0, 0.0, 0.0)),
        bcs=outer_bcs(4, 7, 7, high=("Ihi", "BCOutflow")),
        connections=(ConnSpec("cut", "Zone 1", WHOLE, OUTER),))
    message = _rejected(tmp_path, [two_blocks()[0], small])
    assert "Zone 1 的 i+ 面通过 Zone 1 / cut 接 Zone 2" in message
    assert "供体在 i 向只有 4 个顶点，至少 5 个" in message


# ---- 名字里的连续空格 ----

def test_consecutive_spaces_in_names_survive(tmp_path):
    left, right = two_blocks()
    left = replace(left, name="Zone   1", connections=(
        replace(left.connections[0], name="1to1 cut   1", donor="Zone   2"),))
    right = replace(right, name="Zone   2", connections=(
        replace(right.connections[0], name="1to1 cut   1", donor="Zone   1"),))
    top = _topology(tmp_path, [left, right])
    assert top.interfaces[0].names == ("Zone   1 / 1to1 cut   1",
                                       "Zone   2 / 1to1 cut   1")
    assert "Zone   1 / 1to1 cut   1 ↔ Zone   2 / 1to1 cut   1" in top.report
    message = _rejected(tmp_path, _with_record(
        [left, right], "Zone   1", "1to1 cut   1", transform=(1, -2, 3)))
    assert "Zone   1 / 1to1 cut   1" in message
