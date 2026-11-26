"""第六步的块装配（U3）。

锁五件事：
  1. 各块内区的十个度量与等价单块在同一全局点的度量逐位相同；
  2. 接口旁的鬼胞：普通切口旁的坐标是供体从共享层向内的四层、度量是供体的内区
     度量；周期接缝旁的坐标直接环绕、度量是本端最近内点的值；都与单块逐位相同；
  3. 物理面旁的坐标和度量鬼胞与单块逐位相同，规则按等价单块的轴长选；
  4. 每块的 faces 只含物理面和块内周期面，执行器带着壁温、入口和远场；块的
     ``prepare`` 作用在单块状态的对应区域上，结果与单块逐位相同；
  5. 交换表的不变量：接收槽全在鬼胞，供体源全在内区；T 形共线的那条棱只有
     一个写入者。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import boundary, mesh, mesh_cgns, multiblock  # noqa: E402
from opencfd_jax.src.axes import SIDES, side_axis  # noqa: E402
from opencfd_jax.src.mesh import METRIC_KEYS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
from Grid.split_cgns import periodic_source  # noqa: E402
from opencfd_jax.src.boundary import LAP  # noqa: E402

COORDS = support.COORDS
PLATE_PRESETS = [preset for preset in support.PRESETS if preset != "per3"]
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")


def _inner(field_grid):
    """三维场（没有变量轴）的内区切片。"""
    return field_grid.inner()[1:]


def _window(note, name):
    """块 ``name`` 的范围在等价单块的含鬼胞数组里的切片，轴序 (k, j, i)。"""
    entry = next(b for b in note["blocks"] if b["name"] == name)
    return support.block_window(entry, LAP)


_same = support.same_bits


def _ghost_slices(ghost_grid, side, whole=None, window=None):
    """一个面旁边四层鬼胞的切片。

    给块的网格时返回块数组里的切片（切向只取内区）；再给全局顶点数和块的全局
    范围时，第二个返回值是单块数组里同一批位置的切片。
    """
    axis, low = side_axis(side), side.endswith("-")
    size = (ghost_grid.ni, ghost_grid.nj, ghost_grid.nk)[axis]
    mine = [slice(LAP, LAP + n) for n in ghost_grid.n_inner]
    mine[2 - axis] = slice(0, LAP) if low else slice(LAP + size, 2 * LAP + size)
    if window is None:
        return tuple(mine), None
    theirs = list(window)
    theirs[2 - axis] = (slice(0, LAP) if low
                        else slice(LAP + whole[axis], 2 * LAP + whole[axis]))
    return tuple(mine), tuple(theirs)


@pytest.fixture(scope="module")
def single():
    """等价的单块：描述和补了鬼胞的坐标。"""
    state = support.reference()
    padded = {key: mesh.pad_coords(value, periodic=(False, False, True))
              for key, value in state.desc.coords.items()}
    return state, padded


def _source_coords(preset):
    if preset == "per3":
        return periodic_source().coords
    return mesh_cgns.read_cgns_zone(support.SOURCE, validate=False).coords


# ---- 1. 内区的度量 ----

@pytest.mark.parametrize("preset", support.PRESETS)
def test_block_interior_metrics_match_the_single_block_bitwise(preset, single):
    reference = support.periodic_reference() if preset == "per3" else single[0].desc
    note = support.sidecar(preset)
    for block in support.blocks(preset).blocks:
        for key in METRIC_KEYS:
            _same(block.metrics[key][_inner(block.grid)],
                  reference.metrics[key][_window(note, block.name)],
                  f"{block.name} {key}")


# ---- 2. 接口旁的鬼胞 ----

def test_cut_ghost_metrics_come_from_the_donor_interior():
    built = support.blocks("y2")                    # y 向拉伸，在 j=20 切开
    table, = [table for table in built.tables
              if table.interface.kind == multiblock.CUT
              and table.interface.zone == "blk-i0j0k0"]
    lower, upper = built["blk-i0j0k0"], built["blk-i0j1k0"]
    for key in METRIC_KEYS:
        _same(np.asarray(lower.metrics[key])[tuple(table.slots)],
              np.asarray(upper.metrics[key])[tuple(table.sources)], key)
    # 换成本端最近内点的值会错：供体那四层的度量和接收块自己的末层不同
    last = np.asarray(lower.metrics["Ajac"])[_inner(lower.grid)][..., -1, :]  # (k, i)
    ghost = np.asarray(lower.metrics["Ajac"])[tuple(table.slots)].reshape(40, 4, 100)
    assert not np.array_equal(ghost, np.broadcast_to(last[:, None, :], ghost.shape))


def test_seam_ghosts_match_the_single_periodic_end(single):
    desc, padded = single[0].desc, single[1]
    low, high = (support.blocks("z2")[name] for name in ("blk-i0j0k0", "blk-i0j0k1"))
    inside = (slice(4, 44), slice(4, 104))
    for key in COORDS:                              # 坐标直接环绕
        _same(low.coords[key][(slice(0, 4),) + inside], padded[key][(slice(0, 4),) + inside], key)
        _same(high.coords[key][(slice(24, 28),) + inside], padded[key][(slice(44, 48),) + inside], key)
    for key in METRIC_KEYS:                         # 度量复制本端最近内点的值
        own = np.asarray(low.metrics[key])
        _same(own[(slice(0, 4),) + inside], np.broadcast_to(own[(4,) + inside], (4, 40, 100)), key)
        _same(own[(slice(0, 4),) + inside], np.asarray(desc.metrics[key])[(slice(0, 4),) + inside], key)
        own = np.asarray(high.metrics[key])
        _same(own[(slice(24, 28),) + inside], np.broadcast_to(own[(23,) + inside], (4, 40, 100)), key)
        _same(own[(slice(24, 28),) + inside], np.asarray(desc.metrics[key])[(slice(44, 48),) + inside], key)


@pytest.mark.parametrize("preset", support.PRESETS)
def test_interface_ghost_coordinates_follow_the_mapping(preset):
    """普通切口旁是供体从共享层向内的四层，周期接缝旁是从端点向内的四层。

    期望值直接取自源网格在同一全局下标上的坐标。
    """
    built, note = support.blocks(preset), support.sidecar(preset)
    source = _source_coords(preset)
    start = {b["name"]: b["start"] for b in note["blocks"]}
    stop = {b["name"]: b["stop"] for b in note["blocks"]}
    whole = note["global_vertices"]
    for table in built.tables:
        it = table.interface
        axis, low = side_axis(it.side), it.side.endswith("-")
        k, j, i = table.slots
        local = [i - LAP, j - LAP, k - LAP]          # 按计算轴 (i, j, k) 的块内下标
        size = stop[it.zone][axis] - start[it.zone][axis]
        r = (LAP - table.slots[2 - axis]) if low else (table.slots[2 - axis] - LAP - size + 1)
        if it.kind == multiblock.SEAM:
            depth = whole[axis] - r if low else r - 1
        else:
            depth = start[it.zone][axis] - r if low else stop[it.zone][axis] - 1 + r
        whole_index = [local[a] + start[it.zone][a] for a in range(3)]
        whole_index[axis] = depth                    # 法向直接给全局下标
        gk, gj, gi = whole_index[2], whole_index[1], whole_index[0]
        for key in COORDS:
            _same(np.asarray(built[it.zone].coords[key])[tuple(table.slots)],
                  source[key][gk, gj, gi], f"{it.names[0]} {key}")


# ---- 3. 物理面旁的鬼胞 ----

@pytest.mark.parametrize("preset", PLATE_PRESETS)
def test_face_ghosts_of_blocks_match_the_single_block(preset, single):
    desc, padded = single[0].desc, single[1]
    note = support.sidecar(preset)
    whole = note["global_vertices"]
    for block in support.blocks(preset).blocks:
        for face in block.faces:
            mine, theirs = _ghost_slices(block.grid, face.side, whole,
                                         _window(note, block.name))
            for key in COORDS:
                _same(block.coords[key][mine], padded[key][theirs], f"{block.name} {face.side} {key}")
            for key in METRIC_KEYS:
                _same(block.metrics[key][mine], np.asarray(desc.metrics[key])[theirs],
                      f"{block.name} {face.side} {key}")


def test_short_block_pads_physical_faces_by_the_global_axis_length():
    left = support.blocks("x2i6")["blk-i0j0k0"]     # i 向只有 7 个顶点，i- 是入口
    source = mesh_cgns.read_cgns_zone(support.SOURCE, validate=False).coords
    for key in COORDS:
        first, inner = source[key][..., 0], source[key][..., 1:5]
        mirror = 2.0 * first[..., None] - inner[..., ::-1]
        _same(left.coords[key][4:44, 4:44, 0:4], mirror, key)
    # 按本块轴长（7 < 8）会退化成边缘复制；x 坐标上两种规则的结果不同
    first = source["Axx"][..., 0]
    assert not np.array_equal(left.coords["Axx"][4:44, 4:44, 0:4],
                              np.broadcast_to(first[..., None], (40, 40, 4)))


@pytest.mark.parametrize("preset", support.PRESETS)
def test_all_coordinates_and_metrics_are_finite(preset):
    for block in support.blocks(preset).blocks:
        for key in COORDS:
            assert np.all(np.isfinite(block.coords[key]))
        for key in METRIC_KEYS:
            assert np.all(np.isfinite(block.metrics[key]))


# ---- 4. 面的子集与边界执行器 ----

@pytest.mark.parametrize("preset", support.PRESETS)
def test_block_faces_hold_only_physical_and_in_block_periodic_sides(preset):
    top = support.topology(preset)
    for block in support.blocks(preset).blocks:
        kept = [side for side in SIDES
                if top.faces[(block.name, side)].kind in (multiblock.PHYSICAL,
                                                          multiblock.PERIODIC_FACE)]
        assert [face.side for face in block.faces] == kept
        for face in block.faces:
            assert face.executor == "halo"
            info = top.faces[(block.name, face.side)]
            want = boundary.PERIODIC if info.kind == multiblock.PERIODIC_FACE else \
                mesh_cgns.bc_kind(info.bc_type)
            assert face.kind == want


def test_center_block_of_the_periodic_mesh_has_no_faces():
    center = support.blocks("per3")["blk-i1j1k1"]
    assert center.faces == ()
    assert center.execution.mode == "generic" and center.execution.band == (0,) * 6


@pytest.mark.parametrize("preset", PLATE_PRESETS)
def test_plate_blocks_carry_wall_band_inflow_and_farfield(preset):
    for block in support.blocks(preset).blocks:
        kinds = [face.kind for face in block.faces]
        wall = boundary.WALL in kinds
        execution = block.execution
        assert execution.band == (0, 0, int(wall), 0, 0, 0)
        assert execution.phys is not None
        assert execution.inflow is not None and execution.farfield is not None
        for face in block.faces:
            if face.kind == boundary.WALL:
                assert face.thermal == "isothermal" and face.params == {"Tw": 1.0}


def _smooth_state(desc, params):
    """单块内区上的一个光滑状态：三个方向都变，不满足边界条件。"""
    x, y, z = desc.coords["Axx"], desc.coords["Ayy"], desc.coords["Azz"]
    rho = 1.0 + 0.10 * np.cos(x) * np.cos(0.5 * y)
    u = 1.0 + 0.05 * np.sin(0.7 * x + 0.3 * z)
    v = 0.03 * np.sin(y) * np.cos(0.4 * x)
    w = 0.02 * np.cos(0.5 * y + 0.2 * z)
    T = 1.0 + 0.05 * np.cos(y) * np.sin(0.5 * x)
    energy = params.Cv * T + 0.5 * (u * u + v * v + w * w)
    q = np.zeros(desc.grid.shape)
    q[desc.grid.inner()] = np.stack([rho, rho * u, rho * v, rho * w, rho * energy])
    return q


@pytest.mark.parametrize("preset", PLATE_PRESETS)
def test_block_prepare_matches_the_single_block_on_the_same_region(preset, single):
    state, _ = single
    desc = state.desc
    q = _smooth_state(desc, state.case["params"])
    prepared = np.asarray(state.execution.prepare(jnp.asarray(q)))
    note = support.sidecar(preset)
    whole = note["global_vertices"]
    for block in support.blocks(preset).blocks:
        window = _window(note, block.name)
        own = np.zeros(block.grid.shape)
        own[block.grid.inner()] = q[(slice(None),) + window]
        out = own if not block.faces else np.asarray(block.execution.prepare(jnp.asarray(own)))
        _same(out[block.grid.inner()], prepared[(slice(None),) + window], f"{block.name} 内区")
        for face in block.faces:
            mine, theirs = _ghost_slices(block.grid, face.side, whole, window)
            _same(out[(slice(None),) + mine], prepared[(slice(None),) + theirs],
                  f"{block.name} {face.side} 鬼层")


@pytest.mark.parametrize("preset", PLATE_PRESETS)
def test_default_initial_state_fills_the_interior_with_the_inlet_state(preset, single):
    state, _ = single
    note = support.sidecar(preset)
    for block in support.blocks(preset).blocks:
        _same(np.asarray(block.cons0)[block.grid.inner()],
              np.asarray(state.case["cons0"])[(slice(None),) + _window(note, block.name)],
              block.name)
        ghosts = np.asarray(block.cons0).copy()
        ghosts[block.grid.inner()] = 0.0
        assert not np.any(ghosts), block.name


# ---- 5. 交换表 ----

@pytest.mark.parametrize("preset", support.PRESETS)
def test_exchange_tables_put_slots_in_ghosts_and_sources_in_the_interior(preset):
    built = support.blocks(preset)
    for table in built.tables:
        it = table.interface
        axis = side_axis(it.side)
        for name, index, inside in ((it.zone, table.slots, False),
                                    (it.donor_zone, table.sources, True)):
            size = (built[name].grid.ni, built[name].grid.nj, built[name].grid.nk)
            depth = index[2 - axis] - LAP
            if inside:
                assert np.all((depth >= 0) & (depth < size[axis]))
            else:
                assert np.all((depth < 0) | (depth >= size[axis]))
            for other in range(3):
                if other != axis:
                    lateral = index[2 - other] - LAP
                    assert np.all((lateral >= 0) & (lateral < size[other]))


def test_t_junction_line_is_written_by_the_lower_segment_only():
    built = support.blocks("t3")
    by_donor = {table.interface.donor_zone: table for table in built.tables
                if table.interface.zone == "blk-left"}
    lower, upper = by_donor["blk-rlo"], by_donor["blk-rhi"]
    # 左块的 j 是 0..39（数组下标 4..43）；两段在 j=20（数组下标 24）上共线
    assert lower.slots[1].min() == 4 and lower.slots[1].max() == 24
    assert upper.slots[1].min() == 25 and upper.slots[1].max() == 43
    every = np.concatenate([lower.slots, upper.slots], axis=1)
    assert every.shape[1] == 40 * 40 * 4                 # 40 条 j 线 × 40 条 k 线 × 4 层
    assert len({tuple(column) for column in every.T}) == every.shape[1]   # 没有重复的槽
