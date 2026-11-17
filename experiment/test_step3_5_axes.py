"""Step 3.5 计算方向命名与兼容迁移的验收测试。

覆盖：i/j/k 与旧 x/y/z 的词表映射、Grid 新旧构造入口、FaceBoundary/
BlockMesh 的规范面名与旧名查找、boundary 填充别名、`infile.grid_dims_ijk`
及 CLI `--ni/--nj/--nk` 入口。
"""

import argparse
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from opencfd_jax.src import axes, boundary, fields, infile, mesh_data, runtime_config, run
from opencfd_jax.src.fields import Grid


def _base_case(**overrides):
    """构造 from_case 所需的最小通用算例。"""
    grid = overrides.pop("grid", Grid(ni=4, nj=4, nk=4))
    case = dict(grid=grid, dxs=(0.1, 0.1, 0.1), axes=(0, 1, 2),
                bcs=(boundary.PERIODIC,) * 3, dif_bound=None,
                dif_axis_map=(0, 1, 2))
    case.update(overrides)
    return case


def _resolve(argv):
    """按 run.main 的顺序执行默认值 + 显式 CLI 合并。"""
    defaults = run.build_parser().parse_args([])
    parser = run.build_parser(explicit_only=True)
    config_path, explicit = run.resolve_cli(
        vars(parser.parse_args(argv)), parser)
    assert config_path is None
    return runtime_config.resolve_args(explicit, None, defaults)


# ---------------------------------------------------------------- 词表与 Grid

def test_axis_vocabulary_and_mappings():
    assert axes.AXIS_NAMES == ("i", "j", "k")
    assert axes.SIDES == ("i-", "i+", "j-", "j+", "k-", "k+")
    assert axes.LEGACY_AXIS_NAMES == ("x", "y", "z")
    assert axes.LEGACY_SIDES == ("x-", "x+", "y-", "y+", "z-", "z+")
    assert axes.normalize_axis(0) == 0
    assert axes.normalize_axis("i") == 0
    assert axes.normalize_axis("x") == 0
    assert axes.normalize_axis("j") == 1
    assert axes.normalize_side("i-") == "i-"
    assert axes.normalize_side("x-") == "i-"
    assert axes.normalize_side("y+") == "j+"
    assert axes.legacy_side("i-") == "x-"
    assert axes.legacy_side("k+") == "z+"
    assert axes.side_axis("j-") == 1
    assert axes.side_sign("i-") == 0
    assert axes.side_sign("k+") == 1
    assert axes.array_axis("i") == 2
    assert axes.array_axis("j") == 1
    assert axes.array_axis("k") == 0
    with pytest.raises(ValueError):
        axes.normalize_side("q+")


def test_grid_new_and_legacy_construction():
    new = Grid(ni=8, nj=7, nk=6)
    old = Grid(8, 7, 6)
    legacy_kw = Grid(nx=8, ny=7, nz=6)
    for grid in (new, old, legacy_kw):
        assert (grid.ni, grid.nj, grid.nk) == (8, 7, 6)
        assert (grid.nx, grid.ny, grid.nz) == (8, 7, 6)
        assert grid.n_inner == (6, 7, 8)
        assert grid.shape == (5, 14, 15, 16)
        assert grid.alloc().shape == grid.shape


def test_grid_rejects_conflict_and_missing_dimensions():
    same = Grid(ni=8, nj=7, nk=6, nx=8, ny=7, nz=6)
    assert (same.ni, same.nj, same.nk) == (8, 7, 6)
    with pytest.raises(ValueError, match="冲突"):
        Grid(ni=8, nj=7, nk=6, nx=9)
    with pytest.raises(ValueError, match="缺少网格尺寸"):
        Grid()
    with pytest.raises(AttributeError):
        Grid(ni=8, nj=7, nk=6).nx = 9


# ---------------------------------------------------------- 面名与填充入口

def test_face_boundary_normalizes_legacy_side():
    face = mesh_data.FaceBoundary("x-", 0, boundary.PERIODIC, "halo", False)
    assert face.side == "i-"
    assert face.legacy_side == "x-"
    with pytest.raises(ValueError, match="不一致"):
        mesh_data.FaceBoundary("x-", 1, boundary.PERIODIC, "halo", False)


def test_blockmesh_faces_are_canonical_and_legacy_lookup_works():
    desc = mesh_data.from_case(_base_case())
    assert tuple(f.side for f in desc.faces) == axes.SIDES
    assert desc.face("i-") is desc.face("x-")
    assert desc.face("k+") is desc.face("z+")
    assert desc.inner_shape == (4, 4, 4)


def test_legacy_only_grid_is_accepted_and_conflicting_views_rejected():
    class LegacyGrid:
        nx, ny, nz = 4, 4, 4

        def inner(self):
            return (slice(None), slice(4, 8), slice(4, 8), slice(4, 8))

        def alloc(self):
            return None

    desc = mesh_data.from_case(_base_case(grid=LegacyGrid()))
    assert desc.inner_shape == (4, 4, 4)

    class ConflictingGrid(LegacyGrid):
        ni, nj, nk = 5, 4, 4

    with pytest.raises(ValueError, match="冲突"):
        mesh_data.from_case(_base_case(grid=ConflictingGrid()))


def test_boundary_canonical_aliases_are_same_objects():
    assert boundary.fill_halos_i is boundary.fill_halos_x
    assert boundary.fill_halos_j is boundary.fill_halos_y
    assert boundary.fill_halos_k is boundary.fill_halos_z
    assert boundary.fill_wall_j is boundary.fill_wall_y
    assert boundary.fill_ev_wall_j is boundary.fill_ev_wall_y
    assert boundary.fill_ev_plate_j is boundary.fill_ev_plate_y
    assert fields.fill_halos_i is boundary.fill_halos_i
    assert fields.fill_halos_x is boundary.fill_halos_x
    assert fields.fill_wall_y is boundary.fill_wall_y


def test_infile_grid_dims_ijk_matches_legacy():
    path = _REPO / "opencfd_jax" / "experiment" / "cases" / "channel_small.in"
    inp = infile.read_ocfd_in(str(path))
    legacy = infile.grid_dims(inp)
    canonical = infile.grid_dims_ijk(inp)
    assert canonical == legacy
    assert legacy[:3] == (24, 25, 16)


@pytest.mark.skipif(
    not (_REPO / "channel_case_iiA" / "quick" / "OCFD3d-Mesh.dat").exists(),
    reason="缺少 channel_case_iiA/quick/OCFD3d-Mesh.dat")
def test_mesh_reader_new_and_old_dimensions_agree():
    from opencfd_jax.src import mesh

    path = _REPO / "channel_case_iiA" / "quick" / "OCFD3d-Mesh.dat"
    old = mesh.read_ocfd_mesh(str(path), 96, 97, 64)
    legacy_kw = mesh.read_ocfd_mesh(str(path), nx=96, ny=97, nz=64)
    new = mesh.read_ocfd_mesh(str(path), ni=96, nj=97, nk=64)
    for key in old:
        assert old[key].shape == (64, 97, 96)
        assert old[key].dtype == new[key].dtype
        assert (old[key] == legacy_kw[key]).all()
        assert (old[key] == new[key]).all()


@pytest.mark.skipif(
    not (_REPO / "channel_case_iiA" / "quick" / "opencfd.dat").exists(),
    reason="缺少 channel_case_iiA/quick/opencfd.dat")
def test_field_reader_new_and_old_dimensions_agree():
    from opencfd_jax.src import mesh

    path = _REPO / "channel_case_iiA" / "quick" / "opencfd.dat"
    old = mesh.read_ocfd_field(str(path), 96, 97, 64)
    new = mesh.read_ocfd_field(str(path), ni=96, nj=97, nk=64)
    assert len(old) == len(new) == 5
    for a, b in zip(old, new):
        assert a.shape == (64, 97, 96)
        assert (a == b).all()


# --------------------------------------------------------------------- CLI

def test_cli_new_dimension_names_fill_canonical_values():
    args = _resolve(["--ni", "8", "--nj", "7", "--nk", "6"])
    assert (args.ni, args.nj, args.nk) == (8, 7, 6)
    assert args.config_sources["ni"] == "cli"


def test_cli_old_dimension_flags_are_gone():
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["--N", "8"])
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["--ny", "7"])
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["--nz", "6"])
