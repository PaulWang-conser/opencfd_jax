"""第六步的守恒量交换（U9）。

锁五件事：
  1. 常数场的接口鬼胞从供体的内区取来，表以外的鬼胞不动；线性场的界面导数与单块一致；
  2. 四种取法逐层核对：普通切口接低侧供体、接高侧供体，周期接缝接低侧供体、接高侧
     供体；部分面的两段各取到自己的供体；两侧一同倒写的连接与正写相同；
  3. 接口段的遍历顺序不影响结果，逐位相同；
  4. T 形共线处只有拥有者写入；审计导出两个候选，第一步时逐位相同；
  5. 交换不改动任何内区的值，棱角处的鬼胞也不动；
  6. 每个接收面上应收的鬼胞槽恰好被一张表写到：T 形、阶梯（长棱对两条短棱）、
     T 形处一格宽的条带都如此，结果与表的先后无关。

状态数组的轴序是 (变量, k, j, i)，下面的下标都按这个顺序写。
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import mesh, mesh_cgns, multiblock  # noqa: E402
from opencfd_jax.src.mesh_cgns import CgnsError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
from Grid.make_cgns_fixture import build_multizone  # noqa: E402
from step6_support import (seam_blocks, stair_blocks, strip_blocks, t_blocks,  # noqa: E402
                           two_blocks)

COORDS = support.COORDS
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")


def _topology(tmp_path, zones):
    """写出一个合成网格并校验拓扑。"""
    path = tmp_path / "mesh.cgns"
    build_multizone(path, zones)
    return multiblock.build_topology(mesh_cgns.read_cgns_zones(path))


def _build(tmp_path, zones):
    """写出一个合成网格并装配成块。"""
    return multiblock.build_blocks(
        _topology(tmp_path, zones), support.plate_args(support.mesh_path("x2")))


_with_record = support.with_record

def _code(index, i, j, k):
    """沿 i、j、k 各不相同、块与块之间也不同的编码；都是 float64 里精确的整数。"""
    return 1_000_000.0 * index + 10_000.0 * i + 100.0 * j + k


def _states(built, code=_code):
    """各块的状态：内区是 code(块号, i, j, k)，鬼胞先留成 -1；五个分量依次加 0..4。"""
    states = {}
    for index, block in enumerate(built.blocks):
        k, j, i = np.meshgrid(*(np.arange(n) for n in block.grid.n_inner),
                              indexing="ij")
        own = np.full(block.grid.shape, -1.0)
        own[block.grid.inner()] = code(index, i, j, k) + np.arange(5.0)[:, None, None, None]
        states[block.name] = jnp.asarray(own)
    return states


def _swap(built, states, tables=None):
    """交换一次，取第 0 个分量，转成 numpy。"""
    return {name: np.asarray(value)[0] for name, value in multiblock.exchange(
        built.tables if tables is None else tables, states).items()}


_same = support.same_bits


# ---- 1. 常数场与线性场 ----

def test_constant_field_fills_ghost_cells_from_the_donor(tmp_path):
    built = _build(tmp_path, two_blocks())
    states = {}
    for block in built.blocks:                    # 内区是常数，鬼胞先留成 -1
        own = np.full(block.grid.shape, -1.0)
        own[block.grid.inner()] = 3.5
        states[block.name] = jnp.asarray(own)
    left_block = built["Zone 1"]
    received = np.zeros(left_block.grid.shape[1:], dtype=bool)       # Zone 1 的接收槽
    for table in built.tables:
        if table.interface.zone == "Zone 1":
            received[tuple(table.slots)] = True
    ghost = np.ones(left_block.grid.shape[1:], dtype=bool)
    ghost[left_block.grid.inner()[1:]] = False
    every = (slice(None),)                                          # 五个变量都要看
    skipped = np.asarray(multiblock.exchange((), states)["Zone 1"])   # 对照：不交换
    assert np.all(skipped[every + np.nonzero(received)] == -1.0)      # 接收槽上看得到 -1
    after = multiblock.exchange(built.tables, states)
    for block in built.blocks:
        value = np.asarray(after[block.name])
        for table in built.tables:
            if table.interface.zone == block.name:
                assert np.all(value[every + tuple(table.slots)] == 3.5)   # 取自供体的内区
    left = np.asarray(after["Zone 1"])
    assert np.all(left[every + np.nonzero(ghost & ~received)] == -1.0)   # 表以外的鬼胞不碰


def test_linear_field_keeps_the_single_block_derivative_across_a_cut():
    built = support.blocks("y2")                    # y 向拉伸，在 j=20 切开
    state = support.reference()
    slopes = (0.3, -1.7, 0.9)
    states = {}
    for block in built.blocks:
        own = np.full(block.grid.shape, np.nan)
        value = sum(a * np.asarray(block.coords[key])[block.grid.inner()[1:]]
                    for a, key in zip(slopes, COORDS))
        own[block.grid.inner()] = np.broadcast_to(value, (5,) + value.shape)
        states[block.name] = jnp.asarray(own)
    after = multiblock.exchange(built.tables, states)
    for block in built.blocks:
        for table in built.tables:
            if table.interface.zone == block.name:
                want = sum(a * np.asarray(block.coords[key])[tuple(table.slots)]
                           for a, key in zip(slopes, COORDS))
                _same(np.asarray(after[block.name])[(0,) + tuple(table.slots)], want)
    # 界面导数：下块在切口旁三层的 ∂f/∂j，与单块同一全局点上的一致
    padded = {key: mesh.pad_coords(value, periodic=(False, False, True))
              for key, value in state.desc.coords.items()}
    whole = sum(a * padded[key] for a, key in zip(slopes, COORDS))
    single = np.asarray(mesh.cd_along_axis(jnp.asarray(whole), 1, 1.0, 6))
    lower = np.asarray(after["blk-i0j0k0"])[0]
    block = np.asarray(mesh.cd_along_axis(jnp.asarray(lower), 1, 1.0, 6))
    window = (slice(4 + 10, 4 + 30), slice(4 + 18, 4 + 21), slice(4 + 10, 4 + 90))
    np.testing.assert_allclose(block[window], single[window], rtol=1e-12, atol=1e-12)


# ---- 2. 四种取法、部分面、两侧倒写 ----

def test_four_mappings_on_cuts_and_seams(tmp_path):
    built = _build(tmp_path, seam_blocks())        # 两块 8×7×7：中间切口，外侧周期接缝
    assert [block.name for block in built.blocks] == ["Zone 1", "Zone 2"]
    after = _swap(built, _states(built))
    one, two = after["Zone 1"], after["Zone 2"]    # 内区 k、j 是 4..10，i 是 4..11
    k, j = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    for r in range(1, 5):
        # 普通切口：接收块的低侧面接供体的高侧面，取 n-1-r
        assert np.array_equal(two[4:11, 4:11, 4 - r], _code(0, 7 - r, j, k)), r
        # 普通切口：接收块的高侧面接供体的低侧面，取 r
        assert np.array_equal(one[4:11, 4:11, 11 + r], _code(1, r, j, k)), r
        # 周期接缝：接收块的低侧面接供体的高侧面，取 n-r
        assert np.array_equal(one[4:11, 4:11, 4 - r], _code(1, 8 - r, j, k)), r
        # 周期接缝：接收块的高侧面接供体的低侧面，取 r-1
        assert np.array_equal(two[4:11, 4:11, 11 + r], _code(0, r - 1, j, k)), r


def test_partial_face_segments_read_their_own_donor(tmp_path):
    built = _build(tmp_path, t_blocks())
    assert [block.name for block in built.blocks] == ["Zone 1", "Zone 2", "Zone 3"]
    after = _swap(built, _states(built))
    kk = np.arange(7)[:, None]                     # 沿 k，放在第 0 轴
    for r in range(1, 5):
        # 左块（j 是 4..16）的 i+ 鬼层：j=0..6 接 Zone 2，j=7..12 接 Zone 3。
        # Zone 3 的全局 j 是 6..12，它的局部 j 比左块小 6。
        ghost = after["Zone 1"][4:11, 4:17, 11 + r]
        assert np.array_equal(ghost[:, :7], _code(1, r, np.arange(7)[None, :], kk))
        assert np.array_equal(ghost[:, 7:], _code(2, r, np.arange(1, 7)[None, :], kk))
        # 反方向：两个右块的 i- 接左块的 i+，是接供体的高侧面，取 n-1-r
        assert np.array_equal(after["Zone 2"][4:11, 4:11, 4 - r],
                              _code(0, 7 - r, np.arange(7)[None, :], kk))
        assert np.array_equal(after["Zone 3"][4:11, 4:11, 4 - r],
                              _code(0, 7 - r, 6 + np.arange(7)[None, :], kk))
        # 右下块的 j+ 接右上块的 j-：普通切口，接的是低侧面，取 r
        assert np.array_equal(after["Zone 2"][4:11, 10 + r, 4:12],
                              _code(2, np.arange(8)[None, :], r, kk))
        assert np.array_equal(after["Zone 3"][4:11, 4 - r, 4:12],
                              _code(1, np.arange(8)[None, :], 6 - r, kk))


def test_reversed_records_exchange_like_forward_ones(tmp_path):
    forward = two_blocks()
    back = (((8, 7, 7), (8, 1, 1)), ((1, 7, 7), (1, 1, 1)))
    flipped = _with_record(_with_record(
        forward, "Zone 1", "cut", point_range=back[0], point_range_donor=back[1]),
        "Zone 2", "cut", point_range=back[1], point_range_donor=back[0])
    states = _states(_build(tmp_path, forward))
    straight = _swap(_build(tmp_path, forward), states)
    turned = _swap(_build(tmp_path, flipped), states)
    for name in straight:
        _same(straight[name], turned[name])


# ---- 3. 遍历顺序 ----

def test_traversal_order_does_not_change_the_result(tmp_path):
    built = _build(tmp_path, t_blocks())
    states = _states(built)                # 共线处两个供体的副本故意不同
    straight = _swap(built, states)
    assert all(np.any(value != -1.0) for value in straight.values())
    tables = list(built.tables)
    for order in (tables[::-1], tables[2:] + tables[:2], tables[1::2] + tables[0::2]):
        shuffled = _swap(built, states, order)
        for name in straight:
            _same(shuffled[name], straight[name])


# ---- 4. T 形共线处 ----

def test_t_junction_line_has_one_writer_and_audit_shows_both(tmp_path):
    built = _build(tmp_path, t_blocks())
    states = _states(built)
    kk = np.arange(7)
    for order in (built.tables, built.tables[::-1]):
        left = np.asarray(multiblock.exchange(order, states)["Zone 1"])[0]
        for r in range(1, 5):              # j=6（数组下标 10）这条棱由下段拥有
            assert np.array_equal(left[4:11, 10, 11 + r], _code(1, r, 6, kk))
    after = multiblock.exchange(built.tables, states)
    audit, = multiblock.audit_collinear(built.tables, states, after)
    assert (audit.interface.zone, audit.interface.donor_zone) == ("Zone 1", "Zone 3")
    layers = np.arange(1, 5)[None, :]      # 交换表的点按 (k, 鬼层号) 铺开
    _same(np.asarray(audit.written)[0].reshape(7, 4), _code(1, layers, 6, kk[:, None]))
    _same(np.asarray(audit.candidate)[0].reshape(7, 4), _code(2, layers, 0, kk[:, None]))
    assert not np.array_equal(np.asarray(audit.written), np.asarray(audit.candidate))
    # 第一步：共线的那一层在两个右块里是同一份，两个候选逐位相同
    def copied(index, i, j, k):            # 按全局位置编码：Zone 3 的 j=0 就是 j=6
        return 10_000.0 * i + 100.0 * (j + (6 if index == 2 else 0)) + k

    first = _states(built, copied)
    after = multiblock.exchange(built.tables, first)
    audit, = multiblock.audit_collinear(built.tables, first, after)
    _same(audit.written, audit.candidate)


# ---- 5. 内区不动 ----

def test_exchange_never_changes_interior_values(tmp_path):
    built = _build(tmp_path, t_blocks())
    states = _states(built)
    after = multiblock.exchange(built.tables, states)
    for name, value in after.items():
        inner = built[name].grid.inner()
        _same(np.asarray(value)[inner], np.asarray(states[name])[inner])
    built = _build(tmp_path, two_blocks())
    zone = _swap(built, _states(built))["Zone 1"]
    assert np.all(zone[4:11, 4:11, 12:16] != -1.0)     # i+ 面内区的四层都换到了
    assert np.all(zone[0:4, :, 12:16] == -1.0)         # k 是鬼胞的棱上不换
    assert np.all(zone[:, 0:4, 12:16] == -1.0)         # j 是鬼胞的棱上不换
    assert np.all(zone[4:11, 4:11, 0:4] == -1.0)       # i- 是入口面，鬼层不走交换表


# ---- 6. 每个接收槽恰好一个写入者 ----

def _assert_one_writer(top, built):
    """独立核对交换表。依据只有拓扑（有哪些接口、哪些接收面）和块的尺寸（应收哪些
    槽），不从现有的表反推：
      1. 拓扑里每个跨块接口的两侧都是接收面，都恰好有表；表不写到别的面上；
      2. 每个接收面上应收的鬼胞槽恰好被一张表写到。应收的槽按定义摆出来：接口面
         的内区顶点，往面外数四层。
    """
    sizes = {zone.name: zone.vertex_size for zone in top.zones}
    faces = set()
    for pair in top.interfaces:
        faces |= {(pair.zone, pair.side), (pair.donor_zone, pair.donor_side)}
    written = {}
    for table in built.tables:
        written.setdefault((table.interface.zone, table.interface.side),
                           []).append(table.slots)
    assert faces, "拓扑里没有跨块接口，这个核对会空转"
    assert faces == set(written), (sorted(faces - set(written)), sorted(set(written) - faces))
    for (name, side), parts in written.items():
        normal = "ijk".index(side[0])
        index = [np.arange(multiblock.LAP, multiblock.LAP + n) for n in sizes[name]]
        index[normal] = (np.arange(multiblock.LAP) if side[1] == "-" else
                         np.arange(multiblock.LAP + sizes[name][normal],
                                   multiblock.LAP + sizes[name][normal] + multiblock.LAP))
        wanted = np.stack(np.meshgrid(*(index[a] for a in (2, 1, 0)),
                                      indexing="ij")).reshape(3, -1)
        slots = np.concatenate(parts, axis=1)
        assert slots.shape[1] == wanted.shape[1], (name, side, slots.shape[1], wanted.shape[1])
        assert np.array_equal(np.unique(slots, axis=1), np.unique(wanted, axis=1)), (name, side)


MESHES = [two_blocks(), seam_blocks(), t_blocks(), stair_blocks(), strip_blocks()]
MESH_NAMES = ["两块切口", "周期接缝", "T 形", "阶梯", "一格宽的条带"]


def _built(tmp_path, zones):
    """写出合成网格，返回 (校验过的拓扑, 装配好的块)。"""
    top = _topology(tmp_path, zones)
    return top, multiblock.build_blocks(top, support.plate_args(support.mesh_path("x2")))


@pytest.mark.parametrize("zones", MESHES, ids=MESH_NAMES)
def test_every_ghost_slot_beyond_an_interface_has_exactly_one_writer(tmp_path, zones):
    _assert_one_writer(*_built(tmp_path, zones))


def test_table_self_check_compares_the_slots_and_the_faces(tmp_path):
    """自检要有判别力：写重了、写到了别的槽、整个面没有表、表挂到了别的面上，都要报。"""
    top = _topology(tmp_path, two_blocks())
    first, second = multiblock.build_exchange_tables(top)
    multiblock._check_single_writer(top, (first, second))            # 原样通过
    with pytest.raises(CgnsError, match=r"Zone 1 的 i\+ 面上应收 196 个鬼胞槽，交换表写到 392 个"):
        multiblock._check_single_writer(top, (first, first, second))  # 写重了
    moved = first.slots.copy()
    moved[:, 0] = 0                                                  # 个数不变，槽不对
    with pytest.raises(CgnsError, match=r"Zone 1 的 i\+ 面上应收 196 个鬼胞槽"):
        multiblock._check_single_writer(top, (replace(first, slots=moved), second))
    with pytest.raises(CgnsError, match=r"Zone 2 的 i- 面上应收 196 个鬼胞槽，交换表写到 0 个"):
        multiblock._check_single_writer(top, (first,))               # 少了一个方向
    stray = replace(first, interface=replace(first.interface, side="i-"))
    with pytest.raises(CgnsError, match="不是接口面"):               # 挂到了入口面上
        multiblock._check_single_writer(top, (first, second, stray))


def test_independent_check_notices_tables_that_are_missing_or_stray(tmp_path):
    """独立核对本身也要有判别力，依据是拓扑而不是现有的表：少一个方向、整个接口的
    来去两张表都没了、多出一张挂在别的面上的表，都要报出来，而且报的是那个面。
    挂错的面要用物理面（Zone 1 的 j- 是壁面）：挂到另一个接口面上时，这条断言会
    因为那个面上多写了槽才通过，验不到它自己。"""
    top, built = _built(tmp_path, seam_blocks())     # 两个接口：切口、周期接缝
    _assert_one_writer(top, built)
    tables = built.tables
    with pytest.raises(AssertionError, match=r"\[\('Zone 2', 'i-'\)\], \[\]\)"):
        _assert_one_writer(top, replace(built, tables=(tables[0], *tables[2:])))  # 少了切口的回程
    with pytest.raises(AssertionError,
                       match=r"\[\('Zone 1', 'i\+'\), \('Zone 2', 'i-'\)\], \[\]\)"):
        _assert_one_writer(top, replace(built, tables=tables[2:]))            # 少了整个切口
    stray = replace(tables[0], interface=replace(tables[0].interface, side="j-"))
    with pytest.raises(AssertionError, match=r"\(\[\], \[\('Zone 1', 'j-'\)\]\)"):
        _assert_one_writer(top, replace(built, tables=tables + (stray,)))


def test_staircase_junction_does_not_depend_on_table_order(tmp_path):
    built = _build(tmp_path, stair_blocks())
    states = _states(built)                # 几段的供体副本故意不同
    straight = _swap(built, states)
    tables = list(built.tables)
    for order in (tables[::-1], tables[2:] + tables[:2], tables[1::2] + tables[0::2]):
        shuffled = _swap(built, states, order)
        for name in straight:
            _same(shuffled[name], straight[name])


def test_staircase_edge_and_corner_go_to_the_segment_with_the_smaller_start(tmp_path):
    """Zone 1 的 k+ 面上，i=6 这条棱和角点 (6, 6) 归长段（接 Zone 2）：它的 i 轴起点
    是 0，两个短段的是 6。Zone 2 自己的 i+ 面上是个 T 形，j=6 那条棱归下段。"""
    built = _build(tmp_path, stair_blocks())
    after = _swap(built, _states(built))
    for r in range(1, 5):                  # 普通切口：高侧面接供体的低侧面，取 r
        # Zone 1 的内区 k 是 4..10，k+ 鬼层从 11 起；i=6 在数组下标 10
        assert np.array_equal(after["Zone 1"][10 + r, 4:17, 10],
                              _code(1, 6, np.arange(13), r))
        # Zone 2 只有 7 个顶点：内区 i 是 4..10，i+ 鬼层从 11 起；j=6 在数组下标 10
        assert np.array_equal(after["Zone 2"][4:11, 10, 10 + r],
                              _code(2, r, 6, np.arange(7)))


def test_one_cell_strip_at_a_t_junction_still_gets_its_donor(tmp_path):
    """T 形处一格宽的条带。Zone 1 的 i+ 面最上面一行（j=12）接 Zone 3，其余接 Zone 2；
    Zone 3 的 i− 面最下面一行（它的 j=1）接 Zone 1，其余（它的 j=2 起）接 Zone 4。
    Zone 3 的 j 比全局小 11，Zone 4 的小 12，供体的 j 下标要换算。"""
    built = _build(tmp_path, strip_blocks())
    after = _swap(built, _states(built))
    kk = np.arange(7)
    for r in range(1, 5):
        # 接收块的高侧面接供体的低侧面，取 r；Zone 1 内区 j 是 4..16，j=12 在下标 16
        assert np.array_equal(after["Zone 1"][4:11, 16, 11 + r], _code(2, r, 1, kk))
        assert np.array_equal(after["Zone 1"][4:11, 15, 11 + r], _code(1, r, 11, kk))
        # 接收块的低侧面接供体的高侧面，取 n-1-r；Zone 3 内区 j 是 4..11
        assert np.array_equal(after["Zone 3"][4:11, 5, 4 - r], _code(0, 7 - r, 12, kk))
        assert np.array_equal(after["Zone 3"][4:11, 6, 4 - r], _code(3, 7 - r, 1, kk))
