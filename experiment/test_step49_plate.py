"""真实 CGNS 平板的 Family 边条与周期配对。"""
from pathlib import Path
import sys
import json

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opencfd_jax.src import mesh_cgns, runtime_config, cases, main  # noqa: E402

SAMPLE = ROOT / "opencfd_jax/Grid/testcgnstest.cgns"


def test_plate_family_boundaries_are_resolved():
    zone = mesh_cgns.read_cgns_zone(SAMPLE)
    block = mesh_cgns.to_block_mesh(zone)
    assert (zone.ni, zone.nj, zone.nk) == (100, 40, 40)
    assert tuple(f.kind for f in block.faces) == (
        "inflow", "outflow", "wall", "farfield", "periodic", "periodic")


def test_single_polar_periodic_side_is_rejected():
    zone = mesh_cgns.read_cgns_zone(SAMPLE, validate=False)
    from dataclasses import replace
    modified = replace(zone, bcs=tuple(
        replace(bc, bc_type="BCOutflow") if bc.bc_type == "BCSymmetryPolar"
        and mesh_cgns._side_of(*bc.point_range, zone.vertex_size) == "k+" else bc
        for bc in zone.bcs))
    with pytest.raises(mesh_cgns.CgnsError, match="周期"):
        mesh_cgns.to_block_mesh(modified)


def test_symmetry_plane_is_not_periodic():
    from dataclasses import replace
    zone = mesh_cgns.read_cgns_zone(SAMPLE, validate=False)
    zone = replace(zone, bcs=tuple(replace(bc, bc_type="BCSymmetryPlane")
                   if bc.bc_type == "BCSymmetryPolar" else bc for bc in zone.bcs))
    with pytest.raises(mesh_cgns.CgnsError, match="BCSymmetryPlane"):
        mesh_cgns.to_block_mesh(zone)


def test_missing_family_reports_patch(tmp_path):
    import h5py
    import shutil
    import numpy as np
    path = tmp_path / "missing.cgns"
    shutil.copyfile(SAMPLE, path)
    with h5py.File(path, "r+") as f:
        node = f["Base/blk-1/ZoneBC/dom-2/FamilyName/ data"]
        node[...] = np.frombuffer(b"none", dtype=np.int8)
    with pytest.raises(mesh_cgns.CgnsError, match="dom-2.*Family"):
        mesh_cgns.read_cgns_zone(path)


def test_plate_cgns_json_and_boundary_states(tmp_path):
    state = {"rho": 1.0, "u": 1.0, "v": 0.0, "w": 0.0, "T": 1.0}
    payload = {"case": {"name": "plate"},
               "mesh": {"format": "cgns", "file": str(SAMPLE)},
               "boundary": {"inflow": state, "farfield": dict(state, u=2.0), "wall_temperature": 1.0},
               "time": {"dt": 1e-4}}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf8")
    cfg = runtime_config.load_config(path)
    defaults = main.build_parser().parse_args([])
    args = runtime_config.resolve_args({}, cfg, defaults)
    case = cases.build_case(args)
    assert (case["grid"].ni, case["grid"].nj, case["grid"].nk) == (100, 40, 40)
    assert tuple(face.kind for face in case["faces"]) == (
        "inflow", "outflow", "wall", "farfield", "periodic", "periodic")
    assert float(case["farfield"][1]) == 2.0
    assert float(case["inflow"][1, 0, 0]) == 1.0


def test_cgns_plate_missing_boundary_data_fails(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"case": {"name": "plate"},
        "mesh": {"format": "cgns", "file": str(SAMPLE)}, "time": {"dt": 1e-4}}))
    with pytest.raises(runtime_config.ConfigError, match="boundary"):
        runtime_config.load_config(path)


def test_cgns_plate_outlet_uses_ocfd_ghosts():
    import numpy as np
    import jax.numpy as jnp
    from opencfd_jax.src.boundary import LAP, build_execution, fill_ev_faces

    cfg = runtime_config.load_config(ROOT / "opencfd_jax/experiment/cases/plate_cgns_step49.json")
    args = runtime_config.resolve_args({}, cfg, main.build_parser().parse_args([]))
    case = cases.build_case(args)
    execution = build_execution(case)
    grid = case["grid"]
    q = execution.prepare(case["cons0"])
    inner = (slice(LAP, LAP + grid.nk), slice(LAP + 2, LAP + grid.nj - 2))
    assert np.all(np.asarray(q[:, inner[0], inner[1], LAP + grid.ni:]) == 0)
    assert np.all(np.asarray(q[0, inner[0], inner[1], LAP + grid.ni - 1]) == 1)
    ev = fill_ev_faces(jnp.ones((4,) + grid.shape[1:]), execution.faces, grid)
    assert np.all(np.asarray(ev[:, inner[0], inner[1], LAP + grid.ni:]) == 0)
    assert np.all(np.asarray(ev[:, inner[0], inner[1], LAP + grid.ni - 1]) == 1)


def test_viscous_derivative_reads_zero_outlet_temperature(monkeypatch):
    import numpy as np
    from opencfd_jax.src import viscous
    from opencfd_jax.src.boundary import LAP, fill_faces
    from opencfd_jax.src.fields import Grid
    from opencfd_jax.src.mesh_data import FaceBoundary

    grid = Grid(ni=8, nj=8, nk=8)
    params = viscous.PhysParams()
    q = grid.alloc().at[0].set(1).at[1].set(1).at[4].set(params.Cv + 0.5)
    faces = (FaceBoundary("i+", 0, "outflow", "halo", False),)
    q = q.at[:, :, :, LAP + grid.ni:].set(0)
    seen = []
    original = viscous.cd_along_axis

    def capture(g, *args, **kwargs):
        seen.append(np.asarray(g))
        return original(g, *args, **kwargs)

    monkeypatch.setattr(viscous, "cd_along_axis", capture)
    viscous.viscous_rhs(q, grid, params, ("outflow",) * 3, axes=(0,),
                        dx=(1, 1, 1), faces=faces, fill_state_halos=False)
    temperature = seen[9]
    np.testing.assert_array_equal(temperature[LAP, LAP, LAP + grid.ni:], 0)
    assert temperature[LAP, LAP, LAP + grid.ni - 1] == pytest.approx(1)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", ["-", "+"])
def test_generic_outflow_preserves_state_ghosts_and_clears_ev(axis, side):
    import numpy as np
    import jax.numpy as jnp
    from opencfd_jax.src.boundary import fill_faces, fill_ev_faces
    from opencfd_jax.src.fields import Grid
    from opencfd_jax.src.mesh_data import FaceBoundary

    grid = Grid(ni=8, nj=8, nk=8)
    face = FaceBoundary("ijk"[axis] + side, axis, "outflow", "halo", False)
    q = jnp.arange(np.prod(grid.shape), dtype=jnp.float64).reshape(grid.shape)
    state = np.asarray(fill_faces(q, (face,), grid))
    np.testing.assert_array_equal(state, np.asarray(q))
    ev = np.asarray(fill_ev_faces(q[:4], (face,), grid))
    where = [slice(None)] * 4
    where[3 - axis] = slice(None, 4) if side == "-" else slice(-4, None)
    assert np.all(ev[tuple(where)] == 0)


def test_ocfd_conversion_with_distinct_axes_and_velocities(tmp_path, monkeypatch):
    """用独立构造的原始文件核验转置和物理速度分量映射。"""
    import importlib.util
    import struct
    import numpy as np

    converter = ROOT / "opencfd_jax/experiment/baseline/ocfd_plate/to_npz.py"
    spec = importlib.util.spec_from_file_location("step49_converter", converter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ni, nj, nk = 9, 8, 7
    j, k, i = np.indices((nj, nk, ni))
    rho = 1 + .001 * (100*j + 10*k + i)
    u, v, w = 2 + .01*i, 3 + .01*k, 4 + .01*j
    temperature = 1 + .001*j
    source = tmp_path / "ocfd.dat"
    with source.open("wb") as f:
        f.write(struct.pack("<iidi", 12, 1, 0.0001, 12))
        for field in (rho, u, v, w, temperature):
            for plane in field:
                payload = np.asarray(plane, dtype="<f8").tobytes()
                f.write(struct.pack("<i", len(payload)))
                f.write(payload)
                f.write(struct.pack("<i", len(payload)))
    dest = tmp_path / "result.npz"
    monkeypatch.setattr(sys, "argv", [str(converter), "--file", str(source),
                        "--out", str(dest), "--ni", str(ni), "--nj", str(nj),
                        "--nk", str(nk), "--ama", "0.1"])
    module.main()
    with np.load(dest) as result:
        # 直接按 JAX (k,j,i) 下标写期望值，不使用转换器的转置过程。
        for kk, jj, ii in ((0, 0, 0), (6, 7, 8), (4, 2, 5)):
            r = 1 + .001*(100*jj + 10*kk + ii)
            uu, vv, ww = 2 + .01*ii, 4 + .01*jj, 3 + .01*kk
            T = 1 + .001*jj
            expected = (r, r*uu, r*vv, r*ww,
                        r*(T/(1.4*.4*.1*.1) + .5*(uu**2+vv**2+ww**2)))
            for field, value in enumerate(expected):
                assert result[f"cons_{field}"][kk, jj, ii] == pytest.approx(value)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_periodic_state_and_ev_preserve_inner_endpoints(axis):
    import numpy as np
    import jax.numpy as jnp
    from opencfd_jax.src.boundary import fill_faces, fill_ev_faces
    from opencfd_jax.src.fields import Grid
    from opencfd_jax.src.mesh_data import FaceBoundary

    grid = Grid(ni=9, nj=8, nk=7)
    n = (grid.ni, grid.nj, grid.nk)[axis]
    faces = tuple(FaceBoundary("ijk"[axis]+s, axis, "periodic", "halo", False)
                  for s in ("-", "+"))
    a = jnp.arange(np.prod(grid.shape), dtype=jnp.float64).reshape(grid.shape)
    for source, fill in ((a, fill_faces), (a[:4], fill_ev_faces)):
        src = np.moveaxis(np.asarray(source), 3-axis, -1)
        got = np.moveaxis(np.asarray(fill(source, faces, grid)), 3-axis, -1)
        np.testing.assert_array_equal(got[..., 4:4+n], src[..., 4:4+n])
        np.testing.assert_array_equal(got[..., :4], src[..., n:n+4])
        np.testing.assert_array_equal(got[..., 4+n:], src[..., 4:8])


@pytest.mark.parametrize("axis", ["i", "j", "k"])
def test_polar_pair_supported_on_each_axis(axis):
    from dataclasses import replace
    zone = mesh_cgns.read_cgns_zone(SAMPLE, validate=False)
    faces = []
    for bc in zone.bcs:
        side = mesh_cgns._side_of(*bc.point_range, zone.vertex_size)
        kind = "BCSymmetryPolar" if side[0] == axis else "BCOutflow"
        faces.append(replace(bc, bc_type=kind))
    block = mesh_cgns.to_block_mesh(replace(zone, bcs=tuple(faces)))
    assert block.face(axis + "-").kind == "periodic"
    assert block.face(axis + "+").kind == "periodic"


def test_cross_axis_polar_and_duplicate_or_partial_patch_rejected():
    from dataclasses import replace
    zone = mesh_cgns.read_cgns_zone(SAMPLE, validate=False)
    faces = []
    for bc in zone.bcs:
        side = mesh_cgns._side_of(*bc.point_range, zone.vertex_size)
        faces.append(replace(bc, bc_type="BCSymmetryPolar"
                             if side in ("i-", "k+") else "BCOutflow"))
    with pytest.raises(mesh_cgns.CgnsError, match="周期"):
        mesh_cgns.to_block_mesh(replace(zone, bcs=tuple(faces)))
    with pytest.raises(mesh_cgns.CgnsError, match="同时由"):
        mesh_cgns.to_block_mesh(replace(zone, bcs=zone.bcs + (zone.bcs[0],)))
    bc = zone.bcs[0]
    lo, hi = bc.point_range
    axis = next(i for i in range(3) if hi[i] > lo[i])
    shortened = list(hi)
    shortened[axis] -= 1
    partial = replace(bc, point_range=(lo, tuple(shortened)))
    with pytest.raises(mesh_cgns.CgnsError, match="不是整面"):
        mesh_cgns.to_block_mesh(replace(zone, bcs=(partial,) + zone.bcs[1:]))


def test_rhs_report_records_nonfinite_and_global_position():
    import importlib.util
    import numpy as np
    path = ROOT / "opencfd_jax/experiment/step49/compare_rhs.py"
    spec = importlib.util.spec_from_file_location("step49_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = np.ones((1, 6, 7, 9))
    candidate = ref.copy()
    candidate[0, 2, 3, 8] = 4
    candidate[0, 1, 2, 7] = np.nan
    report = module.compare_regions(ref, candidate, ("rho",),
                                    {"outlet": (slice(None), slice(None), slice(-4, None))})
    result = report["rho"]["outlet"]
    assert result["nonfinite_jax"] == 1
    assert result["max_abs"] == 3
    assert result["max_abs_index"] == [2, 3, 8]


def test_prepare_reference_reproduces_existing_mesh_and_initial_field(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    path = ROOT / "opencfd_jax/experiment/step49/prepare_reference.py"
    spec = importlib.util.spec_from_file_location("step49_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "input"
    monkeypatch.setattr(sys, "argv", [str(path), "--mesh", str(SAMPLE), "--output", str(output)])
    module.main()
    for name, expected in (
        ("OCFD3d-Mesh.dat", "32eea1ea886fdf42cdbb186a6463f6e18b5841575cf7145d12287701232bfce8"),
        ("opencfd.dat", "444b86a3f2df7683558062cb3abad3212aca03cabf221674098ee970ebe14cae"),
    ):
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        module.main()


def test_cgns_plate_farfield_uses_reference_band_and_interior_window():
    import numpy as np
    from opencfd_jax.src.boundary import LAP, build_execution, write_faces

    cfg = runtime_config.load_config(ROOT / "opencfd_jax/experiment/cases/plate_cgns_step49.json")
    args = runtime_config.resolve_args({}, cfg, main.build_parser().parse_args([]))
    case = cases.build_case(args)
    execution = build_execution(case)
    grid = case["grid"]
    assert execution.band[3] == 0
    q = grid.alloc().at[0].set(2.0).at[4].set(10.0)
    far = next(f for f in execution.faces if f.kind == "farfield")
    out = write_faces(q, (far,), grid, farfield=case["farfield"])
    np.testing.assert_array_equal(np.asarray(out[:, LAP:LAP+grid.nk, LAP+grid.nj-1,
                                              LAP+grid.ni:]),
                                  np.asarray(q[:, LAP:LAP+grid.nk, LAP+grid.nj-1,
                                               LAP+grid.ni:]))
    assert float(out[0, LAP, LAP+grid.nj-1, LAP]) == 1.0


def test_explicit_face_band_controls_execution():
    from opencfd_jax.src.boundary import dif_bound_from_faces
    from opencfd_jax.src.mesh_data import FaceBoundary

    face = FaceBoundary("j+", 1, "farfield", "halo", False)
    assert dif_bound_from_faces((face,)) == (0, 0, 0, 0, 0, 0)
    face = FaceBoundary("j+", 1, "farfield", "halo", True)
    assert dif_bound_from_faces((face,)) == (0, 0, 0, 1, 0, 0)


def test_viscous_derivative_reads_zero_farfield_temperature(monkeypatch):
    import numpy as np
    from opencfd_jax.src import viscous
    from opencfd_jax.src.fields import Grid
    from opencfd_jax.src.mesh_data import FaceBoundary

    grid = Grid(ni=8, nj=8, nk=8)
    params = viscous.PhysParams()
    q = grid.alloc().at[0].set(1).at[1].set(1).at[4].set(params.Cv + .5)
    q = q.at[:, :, 4 + grid.nj:, :].set(0)
    faces = (FaceBoundary("j+", 1, "farfield", "halo", False),)
    seen = []
    original = viscous.cd_along_axis

    def capture(g, *args, **kwargs):
        seen.append(np.asarray(g))
        return original(g, *args, **kwargs)

    monkeypatch.setattr(viscous, "cd_along_axis", capture)
    viscous.viscous_rhs(q, grid, params, ("outflow",) * 3, axes=(1,),
                        dx=(1, 1, 1), faces=faces, fill_state_halos=False)
    np.testing.assert_array_equal(seen[10][4, 4+grid.nj:, 4], 0)
    assert seen[10][4, 4+grid.nj-1, 4] == pytest.approx(1)


def test_inflow_ev_ghost_is_zero_without_changing_state():
    import numpy as np
    import jax.numpy as jnp
    from opencfd_jax.src.boundary import fill_faces, fill_ev_faces
    from opencfd_jax.src.fields import Grid
    from opencfd_jax.src.mesh_data import FaceBoundary

    grid = Grid(ni=8, nj=8, nk=8)
    face = FaceBoundary("i-", 0, "inflow", "halo", False)
    state = jnp.ones(grid.shape)
    ev = jnp.ones((4,) + grid.shape[1:])
    np.testing.assert_array_equal(np.asarray(fill_faces(state, (face,), grid)), 1)
    result = fill_ev_faces(ev, (face,), grid)
    np.testing.assert_array_equal(np.asarray(result[..., :4]), 0)
    np.testing.assert_array_equal(np.asarray(result[..., 4:]), 1)
