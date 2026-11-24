"""第六步的验收网格与辅助模块（U2）。

锁四件事：
  1. 七个预设都能被本项目读取器读入并通过拓扑校验，接口对的个数与计划一致；
  2. 各块的顶点数符合切口约定；按旁记把各块拼回去，与源网格逐位相同；
  3. 外表面上的边条类型与源网格一致；周期接缝的平移是 (0, 0, 3)；
  4. 同一个预设生成两次读回的坐标逐位相同；输出放到源网格上被拒绝；辅助模块
     对同一个预设只生成一次；预设从命令行跑得通。
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import mesh_cgns
from opencfd_jax.src.axes import AXIS_NAMES
from opencfd_jax.src.multiblock import CUT, SEAM

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
from Grid.split_cgns import PRESETS, write_preset  # noqa: E402

SOURCE = support.SOURCE
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="缺少源网格")
FROM_PLATE = [preset for preset in PRESETS if preset != "per3"]
INTERFACES = {"x2": (1, 1, 0), "y2": (1, 1, 0), "z2": (2, 1, 1),
              "xyz8": (16, 12, 4), "t3": (3, 3, 0), "x2i6": (1, 1, 0),
              "per3": (81, 54, 27)}
DOCUMENTED = {
    "x2": {"blk-i0j0k0": (51, 40, 40), "blk-i1j0k0": (50, 40, 40)},
    "y2": {"blk-i0j0k0": (100, 21, 40), "blk-i0j1k0": (100, 20, 40)},
    "z2": {"blk-i0j0k0": (100, 40, 21), "blk-i0j0k1": (100, 40, 20)},
    "t3": {"blk-left": (51, 40, 40), "blk-rlo": (50, 21, 40),
           "blk-rhi": (50, 20, 40)},
    "x2i6": {"blk-i0j0k0": (7, 40, 40), "blk-i1j0k0": (94, 40, 40)},
    "per3": {"blk-i0j0k0": (9, 9, 9), "blk-i2j2k2": (9, 9, 9)},
}


@pytest.fixture(scope="module")
def source():
    return mesh_cgns.read_cgns_zone(SOURCE, validate=False)


def _face_codes(zone):
    """源网格每个侧面上的边条类型，按面元存成编号；返回 (类型表, {侧面: 编号数组})。"""
    types, codes = [], {}
    size = zone.vertex_size
    for bc in zone.bcs:
        lo, hi = bc.point_range
        axis, *tangential = mesh_cgns.face_axes(lo, hi)
        side = mesh_cgns.side_of_face(axis, lo, size)
        code = codes.setdefault(
            side, np.full((size[tangential[1]] - 1, size[tangential[0]] - 1), -1))
        if bc.bc_type not in types:
            types.append(bc.bc_type)
        code[tuple(slice(min(lo[a], hi[a]), max(lo[a], hi[a]))
                   for a in (tangential[1], tangential[0]))] = types.index(bc.bc_type)
    return types, codes


# ---- 1. 读得进来、过得了校验 ----

@pytest.mark.parametrize("preset", PRESETS)
def test_preset_passes_topology_with_the_planned_interfaces(preset):
    top = support.topology(preset)
    total, cuts, seams = INTERFACES[preset]
    kinds = [it.kind for it in top.interfaces]
    assert (len(kinds), kinds.count(CUT), kinds.count(SEAM)) == (total, cuts, seams)
    assert f"接口对 {total} 个（普通切口 {cuts} 个，周期接缝 {seams} 个）" in top.report


# ---- 2. 尺寸、拼回 ----

@pytest.mark.parametrize("preset", PRESETS)
def test_block_sizes_follow_the_cut_convention(preset):
    note = support.sidecar(preset)
    expected = {}
    for block in note["blocks"]:
        start, stop = block["start"], block["stop"]
        expected[block["name"]] = tuple(stop[a] - start[a] for a in range(3))
    zones = mesh_cgns.read_cgns_zones(support.mesh_path(preset))
    assert {zone.name: zone.vertex_size for zone in zones} == expected
    for name, size in DOCUMENTED.get(preset, {}).items():
        assert expected[name] == size
    if preset == "xyz8":
        assert {v[0] for v in expected.values()} == {51, 50}
        assert {v[1] for v in expected.values()} == {21, 20}
        assert {v[2] for v in expected.values()} == {21, 20}


@pytest.mark.parametrize("preset", PRESETS)
def test_sidecar_ranges_tile_the_domain(preset):
    note = support.sidecar(preset)
    nk, nj, ni = note["global_vertices"][::-1]
    cells = np.zeros((nk - 1, nj - 1, ni - 1), dtype=np.int64)
    for block in note["blocks"]:
        k0, j0, i0 = block["start"][::-1]
        k1, j1, i1 = block["stop"][::-1]
        assert 0 <= i0 < i1 <= ni and 0 <= j0 < j1 <= nj and 0 <= k0 < k1 <= nk
        cells[k0:k1 - 1, j0:j1 - 1, i0:i1 - 1] += 1
    assert np.all(cells == 1)                     # 每个面元恰好属于一个块


@pytest.mark.parametrize("preset", PRESETS)
def test_reassembled_coordinates_match_the_source(preset, source):
    zones = mesh_cgns.read_cgns_zones(support.mesh_path(preset))
    note = support.sidecar(preset)
    if preset == "per3":
        k, j, i = np.meshgrid(*(np.arange(n) for n in (25, 25, 25)), indexing="ij")
        whole = {"Axx": 0.5 * i, "Ayy": 0.7 * j, "Azz": 1.1 * k}
    else:
        whole = source.coords
    for key, value in whole.items():
        back = support.reassemble(note, {z.name: z.coords[key] for z in zones})
        assert back.tobytes() == value.tobytes(), key


# ---- 3. 边条与周期接缝 ----

@pytest.mark.parametrize("preset", FROM_PLATE)
def test_outer_bc_types_match_the_source(preset, source):
    types, codes = _face_codes(source)
    start = {b["name"]: b["start"] for b in support.sidecar(preset)["blocks"]}
    for zone in mesh_cgns.read_cgns_zones(support.mesh_path(preset)):
        assert zone.bcs, f"{zone.name} 没有边条"
        for bc in zone.bcs:
            lo, hi = bc.point_range
            axis = next(a for a in range(3) if lo[a] == hi[a])
            side = AXIS_NAMES[axis] + ("-" if lo[axis] == 0 else "+")
            tangential = [a for a in range(3) if a != axis]
            window = tuple(slice(start[zone.name][a] + min(lo[a], hi[a]),
                                 start[zone.name][a] + max(lo[a], hi[a]))
                           for a in (tangential[1], tangential[0]))
            assert np.all(codes[side][window] == types.index(bc.bc_type)), (
                f"{zone.name} / {bc.name}")


def test_x2_puts_inflow_on_the_left_and_outflow_on_the_right():
    zones = mesh_cgns.read_cgns_zones(support.mesh_path("x2"))
    assert {z.name: {b.bc_type for b in z.bcs} for z in zones} == {
        "blk-i0j0k0": {"BCInflow", "BCWall", "BCFarfield", "BCSymmetryPolar"},
        "blk-i1j0k0": {"BCOutflow", "BCWall", "BCFarfield", "BCSymmetryPolar"}}


@pytest.mark.parametrize("preset", ["z2", "xyz8"])
def test_periodic_seams_translate_by_one_period(preset):
    seams = [it for it in support.topology(preset).interfaces if it.kind == SEAM]
    assert len(seams) == INTERFACES[preset][2]
    for seam in seams:
        assert np.abs(seam.translation - [0.0, 0.0, 3.0]).max() < 1e-12


def test_xyz8_report_separates_cut_and_seam_of_the_same_pair():
    top = support.topology("xyz8")
    pair = {"blk-i0j0k0", "blk-i0j0k1"}
    between = [it for it in top.interfaces if {it.zone, it.donor_zone} == pair]
    assert sorted(it.kind for it in between) == sorted([CUT, SEAM])
    for it in between:
        line, = [line for line in top.report.splitlines() if it.names[0] in line]
        assert f"范围 {it.point_range[0]}..{it.point_range[1]}" in line
        if it.kind == SEAM:
            assert "平移" in line and "最大偏离" in line


# ---- 4. 生成的确定性 ----

def test_generating_twice_gives_identical_coordinates(tmp_path):
    first = mesh_cgns.read_cgns_zones(write_preset("y2", tmp_path / "a"))
    second = mesh_cgns.read_cgns_zones(write_preset("y2", tmp_path / "b"))
    assert [z.name for z in first] == [z.name for z in second]
    for a, b in zip(first, second):
        for key in a.coords:
            assert a.coords[key].tobytes() == b.coords[key].tobytes()


def test_output_on_the_source_is_rejected():
    before = SOURCE.read_bytes()
    with pytest.raises(SystemExit, match="源网格"):
        write_preset(SOURCE.stem, SOURCE.parent)      # 输出路径正好是源网格
    assert SOURCE.read_bytes() == before


def test_support_generates_each_preset_once(monkeypatch):
    import Grid.split_cgns as split
    support._MESHES.pop("y2", None)
    path = support.mesh_path("y2")

    def refuse(*args, **kwargs):
        raise AssertionError("同一个预设不应再生成一次")

    monkeypatch.setattr(split, "write_preset", refuse)
    assert support.mesh_path("y2") == path
    assert path.parent == support.MESH_DIR / "y2"


def test_presets_run_from_the_command_line(tmp_path):
    def run(*args):
        done = subprocess.run([sys.executable, "Grid/split_cgns.py", *args],
                              cwd=REPO, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return done

    run("--out-dir", str(tmp_path))
    for preset in PRESETS:
        assert (tmp_path / f"{preset}.cgns").is_file()
        assert (tmp_path / f"{preset}.json").is_file()
    run("--preset", "t3", "--out-dir", str(tmp_path / "one"))
    assert [z.name for z in mesh_cgns.read_cgns_zones(tmp_path / "one" / "t3.cgns")] == [
        "blk-left", "blk-rlo", "blk-rhi"]
