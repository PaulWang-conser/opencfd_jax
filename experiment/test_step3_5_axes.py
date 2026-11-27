"""Step 3.5 计算方向命名的验收测试。

覆盖：i/j/k 词表、Grid 的 canonical 构造、FaceBoundary/BlockMesh 的规范
面名、boundary/fields 的 canonical 填充入口、`infile.grid_dims_ijk` 及
CLI `--ni/--nj/--nk` 入口。旧 x/y/z 面名、nx/ny/nz 尺寸名和旧填充别名
均视为已移除，测试只断言它们不可用。
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from opencfd_jax.src import axes, boundary, fields, infile, mesh_data, runtime_config, main
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
    """按 main.main 的顺序执行默认值 + 显式 CLI 合并。"""
    defaults = main.build_parser().parse_args([])
    parser = main.build_parser(explicit_only=True)
    config_path, explicit = main.resolve_cli(vars(parser.parse_args(argv)))
    assert config_path is None
    return runtime_config.resolve_args(explicit, None, defaults)


# ---------------------------------------------------------------- 词表与 Grid

def test_axis_vocabulary_and_mappings():
    assert axes.AXIS_NAMES == ("i", "j", "k")
    assert axes.DIMENSION_NAMES == ("ni", "nj", "nk")
    assert axes.SIDES == ("i-", "i+", "j-", "j+", "k-", "k+")
    assert axes.normalize_axis(0) == 0
    assert axes.normalize_axis("i") == 0
    assert axes.normalize_axis("j") == 1
    assert axes.normalize_side("i-") == "i-"
    assert axes.normalize_side("j+") == "j+"
    assert axes.side_axis("j-") == 1
    assert axes.side_sign("i-") == 0
    assert axes.side_sign("k+") == 1
    assert axes.array_axis("i") == 2
    assert axes.array_axis("j") == 1
    assert axes.array_axis("k") == 0
    with pytest.raises(ValueError):
        axes.normalize_axis("x")
    with pytest.raises(ValueError):
        axes.normalize_side("x-")
    assert not hasattr(axes, "LEGACY_SIDES")
    assert not hasattr(axes, "legacy_side")


def test_grid_canonical_construction():
    new = Grid(ni=8, nj=7, nk=6)
    positional = Grid(8, 7, 6)
    for grid in (new, positional):
        assert (grid.ni, grid.nj, grid.nk) == (8, 7, 6)
        assert grid.n_inner == (6, 7, 8)
        assert grid.shape == (5, 14, 15, 16)
        assert grid.alloc().shape == grid.shape
        assert not hasattr(grid, "nx")
        assert not hasattr(grid, "ny")
        assert not hasattr(grid, "nz")


def test_grid_rejects_missing_or_legacy_dimensions():
    with pytest.raises(ValueError, match="缺少网格尺寸"):
        Grid()
    with pytest.raises(TypeError):
        Grid(ni=8, nj=7, nk=6, nx=9)
    with pytest.raises(ValueError, match="正整数"):
        Grid(ni=8, nj=7, nk=0)


# ---------------------------------------------------------- 面名与填充入口

def test_face_boundary_rejects_legacy_side():
    face = mesh_data.FaceBoundary("i-", 0, boundary.PERIODIC, "halo", False)
    assert face.side == "i-"
    assert not hasattr(face, "legacy_side")
    with pytest.raises(ValueError):
        mesh_data.FaceBoundary("x-", 0, boundary.PERIODIC, "halo", False)


def test_blockmesh_faces_are_canonical_only():
    desc = mesh_data.from_case(_base_case())
    assert tuple(f.side for f in desc.faces) == axes.SIDES
    assert desc.face("i-") is desc.face("i-")
    with pytest.raises(ValueError):
        desc.face("x-")
    assert desc.inner_shape == (4, 4, 4)


def test_from_case_requires_canonical_grid():
    class LegacyOnlyGrid:
        nx, ny, nz = 4, 4, 4

        def inner(self):
            return (slice(None), slice(4, 8), slice(4, 8), slice(4, 8))

        def alloc(self):
            return None

    with pytest.raises(ValueError, match="ni/nj/nk"):
        mesh_data.from_case(_base_case(grid=LegacyOnlyGrid()))


def test_boundary_and_fields_legacy_aliases_are_gone():
    assert fields.fill_halos_i is boundary.fill_halos_i
    assert fields.fill_halos_j is boundary.fill_halos_j
    assert fields.fill_halos_k is boundary.fill_halos_k
    assert fields.fill_wall_j is boundary.fill_wall_j
    assert fields.zero_nonperiodic_ghosts is boundary.zero_nonperiodic_ghosts
    assert fields.array_axis is not None
    for name in ("fill_halos_x", "fill_halos_y", "fill_halos_z",
                 "fill_wall_y", "fill_ev_wall_y", "fill_ev_plate_y",
                 "_fill_ev_wall_y", "_fill_ev_plate_y", "_fill_axis"):
        assert not hasattr(boundary, name), name
    for name in ("fill_halos_x", "fill_halos_y", "fill_halos_z",
                 "fill_wall_y", "spatial_array_axis"):
        assert not hasattr(fields, name), name


def test_infile_grid_dims_ijk_is_only_dimension_api():
    path = _REPO / "opencfd_jax" / "experiment" / "cases" / "channel_small.in"
    inp = infile.read_ocfd_in(str(path))
    dims = infile.grid_dims_ijk(inp)
    assert dims[:3] == (24, 25, 16)
    assert not hasattr(infile, "grid_dims")


@pytest.mark.skipif(
    not (_REPO / "channel_case_iiA" / "quick" / "OCFD3d-Mesh.dat").exists(),
    reason="缺少 channel_case_iiA/quick/OCFD3d-Mesh.dat")
def test_mesh_reader_canonical_dimensions_only():
    from opencfd_jax.src import mesh

    path = _REPO / "channel_case_iiA" / "quick" / "OCFD3d-Mesh.dat"
    positional = mesh.read_ocfd_mesh(str(path), 96, 97, 64)
    keyword = mesh.read_ocfd_mesh(str(path), ni=96, nj=97, nk=64)
    for key in positional:
        assert positional[key].shape == (64, 97, 96)
        assert keyword[key].dtype == positional[key].dtype
        assert (positional[key] == keyword[key]).all()
    with pytest.raises(TypeError):
        mesh.read_ocfd_mesh(str(path), nx=96, ny=97, nz=64)


@pytest.mark.skipif(
    not (_REPO / "channel_case_iiA" / "quick" / "opencfd.dat").exists(),
    reason="缺少 channel_case_iiA/quick/opencfd.dat")
def test_field_reader_canonical_dimensions_only():
    from opencfd_jax.src import mesh

    path = _REPO / "channel_case_iiA" / "quick" / "opencfd.dat"
    positional = mesh.read_ocfd_field(str(path), 96, 97, 64)
    keyword = mesh.read_ocfd_field(str(path), ni=96, nj=97, nk=64)
    assert len(positional) == len(keyword) == 5
    for a, b in zip(positional, keyword):
        assert a.shape == (64, 97, 96)
        assert (a == b).all()
    with pytest.raises(TypeError):
        mesh.read_ocfd_field(str(path), nx=96, ny=97, nz=64)


# --------------------------------------------------------------------- CLI

def test_cli_new_dimension_names_fill_canonical_values():
    args = _resolve(["--ni", "8", "--nj", "7", "--nk", "6"])
    assert (args.ni, args.nj, args.nk) == (8, 7, 6)
    assert args.config_sources["ni"] == "cli"


def test_cli_old_dimension_flags_are_gone():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--N", "8"])
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--ny", "7"])
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--nz", "6"])
