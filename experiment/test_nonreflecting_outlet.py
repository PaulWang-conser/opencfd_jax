"""无反射出口（原码 NON_REFLETION 口径）的行为测试。

原码做法（OCFD_Schemes.cu:293-381 的 put_du_p/m_kernel，任务区间在
正分裂趟向低侧多探一格、负分裂趟向高侧多探一格，见 OCFD_Schemes_Choose.cu:99、437）：
  - 某轴低侧开了无反射：第一个内点沿该轴的**正分裂通量差分**不累加（加 0）；
  - 某轴高侧开了无反射：最后一个内点沿该轴的**负分裂通量差分**不累加。
横向覆盖全部内点；只作用于无粘通量，鬼层、黏性项、边界带都不变。
所以无反射面在其它方面与出流面相同：状态鬼层缺省保持原值、黏性通量 Ev
鬼层清零、不开边界带。

CGNS 边条按项目约定分配：``BCOutflowSupersonic`` → ``nonreflecting``
（SIDS 字面是"全部外推"，与此不同，见 mesh_cgns._bc_kind）。

锁五件事：
  1. 注册格式：nonref 只保留边界内点的那一半分裂通量差；其余点两半分别差分
     再相加（按原码分别形成 P/M 面差），与合并差分只差舍入；不传 nonref 时走
     原路径（逐位相同）；
  2. UP7 同上；
  3. 无反射面的鬼层、Ev、边界带与出流面相同；
  4. inviscid_rhs 接线：i+、i-、k+ 三个方向上，无反射与出流的无粘 RHS
     只在该轴的边界内点列上有实质差别，差值正好是被丢掉的那一半；其余点
     只差舍入；
  5. CGNS 映射与端到端一步可跑；
  6. 无反射对照所用网格的来源可复建：派生脚本的缺省参数加源网格，只改出口
     边条的类型，坐标相同；脚本失败时不破坏已有的产物。
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import boundary, init, main, mesh_cgns, runtime_config, schemes, solver  # noqa: E402
from opencfd_jax.src.fields import LAP, Grid  # noqa: E402
from opencfd_jax.src.mesh_data import FaceBoundary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "Grid" / "testcgnstest.cgns"
NR = "nonreflecting"
DX = 0.1


def _split_fluxes(n, seed):
    rng = np.random.default_rng(seed)
    fp = jnp.asarray(0.5 + rng.random((5, n + 2 * LAP)))
    fm = jnp.asarray(0.1 + rng.random((5, n + 2 * LAP)))
    return fp, fm


def test_registered_scheme_keeps_one_split_at_boundary_cell():
    weno7 = schemes.SOLVE_SCHEMES["WENO7"]
    fp, fm = _split_fluxes(16, 0)
    P = schemes._weno7_face_P(fp)
    M = schemes._weno7_face_M(fm)
    full = np.asarray(schemes._du_from_faces(P + M))
    # 不传 / 传全 False：原路径，逐位相同
    np.testing.assert_array_equal(np.asarray(weno7(fp, fm, 1.0)), full)
    np.testing.assert_array_equal(
        np.asarray(weno7(fp, fm, 1.0, nonref=(False, False))), full)
    # 高侧：末内点只留正分裂通量差
    hi = np.asarray(weno7(fp, fm, 1.0, nonref=(False, True)))
    np.testing.assert_array_equal(
        hi[..., -1], np.asarray(schemes._du_from_faces(P)[..., -1]))
    np.testing.assert_allclose(hi[..., :-1], full[..., :-1], rtol=1e-12,
                               atol=1e-14 * np.abs(full).max())
    # 低侧：首内点只留负分裂通量差
    lo = np.asarray(weno7(fp, fm, 1.0, nonref=(True, False)))
    np.testing.assert_array_equal(
        lo[..., 0], np.asarray(schemes._du_from_faces(M)[..., 0]))
    np.testing.assert_allclose(lo[..., 1:], full[..., 1:], rtol=1e-12,
                               atol=1e-14 * np.abs(full).max())


def test_up7_keeps_one_split_at_boundary_cell():
    fp, fm = _split_fluxes(16, 1)
    P = schemes._face_flux_along_last_axis(fp, schemes.UP7_P_COEFFS, schemes.UP7_P_OFF_LO)
    M = schemes._face_flux_along_last_axis(fm, schemes.UP7_M_COEFFS, schemes.UP7_M_OFF_LO)
    full = np.asarray(schemes._du_from_faces(P + M))
    np.testing.assert_array_equal(
        np.asarray(schemes.up7_rhs_along_last_axis(fp, fm, 1.0)), full)
    hi = np.asarray(schemes.up7_rhs_along_last_axis(fp, fm, 1.0, nonref=(False, True)))
    np.testing.assert_array_equal(
        hi[..., -1], np.asarray(schemes._du_from_faces(P)[..., -1]))
    lo = np.asarray(schemes.up7_rhs_along_last_axis(fp, fm, 1.0, nonref=(True, False)))
    np.testing.assert_array_equal(
        lo[..., 0], np.asarray(schemes._du_from_faces(M)[..., 0]))


def test_nonreflecting_face_matches_outflow_for_ghosts_and_band():
    grid = Grid(10, 8, 6)
    nr = FaceBoundary("i+", 0, NR, "halo", False)
    of = FaceBoundary("i+", 0, "outflow", "halo", False)
    assert boundary.halo_mode(NR) == boundary.halo_mode("outflow") == boundary.HALO_KEEP
    assert boundary.dif_band(NR) is False and boundary.dif_band("outflow") is False
    f = jnp.full(grid.shape, 3.0)
    # 状态鬼层：两者都保持原值；显式 halo=copy 的覆盖仍然生效
    np.testing.assert_array_equal(np.asarray(boundary.fill_faces(f, (nr,), grid)),
                                  np.asarray(boundary.fill_faces(f, (of,), grid)))
    # copy 覆盖：给鬼层一个与内点不同的错值，四层都应等于边界点那一层
    q = jnp.asarray(np.asarray(f).copy())
    q = q.at[..., 4 + grid.ni:].set(7.0)
    cp = np.asarray(boundary.fill_faces(q, (replace(nr, params={"halo": "copy"}),), grid))
    f3 = np.asarray(f)
    for layer in range(1, 5):       # 鬼层 1～4，逐层等于边界点那一层
        np.testing.assert_array_equal(
            cp[..., 4 + grid.ni - 1 + layer], f3[..., 4 + grid.ni - 1])
    # Ev 鬼层：两者都清零
    for face in (nr, of):
        ev = np.asarray(boundary.fill_ev_faces(f, (face,), grid))
        assert np.all(ev[:, 4:4 + grid.nk, 4:4 + grid.nj, 4 + grid.ni:] == 0.0)
        assert np.all(ev[:, 4:4 + grid.nk, 4:4 + grid.nj, 4:4 + grid.ni] == 3.0)


def _faces_with(side, kind):
    spec = {"i-": (0, "inflow"), "i+": (0, "outflow"), "j-": (1, "outflow"),
            "j+": (1, "outflow"), "k-": (2, "periodic"), "k+": (2, "periodic")}
    return tuple(FaceBoundary(s, axis, kind if s == side else default, "halo", False)
                 for s, (axis, default) in spec.items())


def _state(grid, seed):
    rng = np.random.default_rng(seed)
    cv = 1.0 / (1.4 * 0.4 * 0.04)
    rho = 1.0 + 0.1 * rng.random(grid.shape[1:])
    u, v, w = (0.3 * rng.random(grid.shape[1:]) for _ in range(3))
    temperature = 1.0 + 0.1 * rng.random(grid.shape[1:])
    return jnp.asarray(np.stack([rho, rho * u, rho * v, rho * w,
                                 rho * (cv * temperature + 0.5 * (u * u + v * v + w * w))]))


@pytest.mark.parametrize("side,axis,arr_axis", [("i+", 0, 3), ("i-", 0, 3), ("k+", 2, 1)])
def test_inviscid_rhs_drops_one_split_only_at_boundary_cell(side, axis, arr_axis):
    grid = Grid(16, 10, 8)
    cons = _state(grid, 5)
    kw = dict(gamma=1.4, bcs=("outflow", "outflow", "periodic"), axes=(0, 1, 2),
              dx=(DX, DX, DX), scheme="WENO7", fill_state_halos=True)
    base = np.asarray(solver.inviscid_rhs(cons, grid, faces=_faces_with(side, "outflow"), **kw))
    nr = np.asarray(solver.inviscid_rhs(cons, grid, faces=_faces_with(side, NR), **kw))
    diff = nr - base
    on = np.zeros(grid.shape[1:], dtype=bool)
    if side == "i+":
        on[:, :, LAP + grid.ni - 1] = True
    elif side == "i-":
        on[:, :, LAP] = True
    else:
        on[LAP + grid.nk - 1] = True
    scale = np.abs(base).max()
    assert np.any(diff[:, on] != 0.0), "边界内点列上应有变化"
    np.testing.assert_allclose(nr[:, ~on], base[:, ~on], rtol=1e-12, atol=1e-12 * scale)
    # 变化量 = 被丢掉的那一半（低侧丢正分裂、高侧丢负分裂）
    prepared = boundary.fill_faces(cons, _faces_with(side, NR), grid)
    fp, fm = schemes.steger_warming(jnp.moveaxis(prepared, arr_axis, -1), 1.4, 0.0, axis)
    dropped_faces = (schemes._weno7_face_P(fp) if side.endswith("-")
                     else schemes._weno7_face_M(fm))
    dropped = np.asarray(schemes._du_from_faces(dropped_faces)) / DX
    expected = -dropped[..., 0 if side.endswith("-") else -1]
    got = diff[:, on].reshape(expected.shape)
    assert got.shape == expected.shape
    # 只比较横向内点（被清零的鬼层行两侧都是 0，比较无意义）
    inner = (slice(LAP, LAP + grid.nk), slice(LAP, LAP + grid.nj))
    inner_k = (slice(LAP, LAP + grid.nj), slice(LAP, LAP + grid.ni))
    sel = inner if axis == 0 else inner_k
    np.testing.assert_allclose(got[(slice(None),) + sel], expected[(slice(None),) + sel],
                               rtol=1e-9, atol=1e-12 * scale)


def test_symbo_rejects_nonreflecting_on_solved_axis():
    grid = Grid(12, 8, 6)
    cons = _state(grid, 6)
    with pytest.raises(ValueError, match="SYMBO"):
        solver.inviscid_rhs(cons, grid, bcs=("outflow", "outflow", "periodic"),
                            axes=(0,), dx=(DX, DX, DX), scheme="WENO7-SYMBO",
                            faces=_faces_with("i+", NR), fill_state_halos=True)


def _cgns_with_bc(src, dst, bc_name, bc_type):
    import h5py
    with h5py.File(src, "r") as fsrc, h5py.File(dst, "w") as fdst:
        fsrc.copy("Base", fdst)
        zone = next(obj for obj in fdst["Base"].values()
                    if obj.attrs.get("label") == np.bytes_("Zone_t"))
        node = zone["ZoneBC"][bc_name]
        del node[" data"]
        node.create_dataset(" data", data=np.frombuffer(bc_type.encode("ascii"), dtype=np.int8))


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少 testcgnstest.cgns")
def test_cgns_outflow_supersonic_maps_to_nonreflecting(tmp_path):
    assert mesh_cgns._bc_kind("BCOutflowSupersonic") == NR
    assert mesh_cgns._bc_kind("BCOutflowSubsonic") == "outflow"   # 原有映射不变
    path = tmp_path / "nr.cgns"
    _cgns_with_bc(SAMPLE, path, "dom-5", "BCOutflowSupersonic")
    zone = mesh_cgns.read_cgns_zone(str(path))            # validate=True
    kinds = tuple(f.kind for f in mesh_cgns.to_block_mesh(zone).faces)
    assert kinds == ("inflow", NR, "wall", "farfield", "periodic", "periodic")


@pytest.mark.skipif(not (REPO / "Grid" / "testcgnstest_ystr.cgns").exists(),
                    reason="缺少 testcgnstest_ystr.cgns")
def test_nonreflecting_grid_is_derived_from_recorded_source(tmp_path):
    """无反射对照所用网格的来源：派生脚本的缺省参数加源网格。

    只改 dom-5 的类型，坐标与源网格相同，其余五面不变；源文件不可被覆盖。
    产物不记哈希：来源由入库的源网格、脚本、配置和已验证的重跑确定。
    失败时不得破坏已有的产物（Astra 第十五轮）：边条名敲错，输出文件应原样保留。
    """
    import subprocess
    script = REPO / "Grid" / "override_cgns_bc.py"
    src = REPO / "Grid" / "testcgnstest_ystr.cgns"
    out = tmp_path / "nr.cgns"
    subprocess.run([sys.executable, str(script), "--out", str(out)],
                   check=True, capture_output=True)
    a = mesh_cgns.read_cgns_zone(str(src))
    b = mesh_cgns.read_cgns_zone(str(out))
    assert all(np.array_equal(a.coords[k], b.coords[k]) for k in a.coords)
    assert tuple(f.kind for f in mesh_cgns.to_block_mesh(a).faces) == (
        "inflow", "outflow", "wall", "farfield", "periodic", "periodic")
    assert tuple(f.kind for f in mesh_cgns.to_block_mesh(b).faces) == (
        "inflow", NR, "wall", "farfield", "periodic", "periodic")
    # 边条名敲错：报错退出，已有产物不变，临时文件不留
    good = out.read_bytes()
    bad = subprocess.run([sys.executable, str(script), "--out", str(out), "--bc", "dom-99"],
                         capture_output=True)
    assert bad.returncode != 0 and out.read_bytes() == good
    assert not out.with_name(out.name + ".tmp").exists()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, str(script), "--out", str(src)],
                       check=True, capture_output=True)


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少 testcgnstest.cgns")
def test_plate_with_nonreflecting_outlet_runs_one_step(tmp_path):
    path = tmp_path / "nr.cgns"
    _cgns_with_bc(SAMPLE, path, "dom-5", "BCOutflowSupersonic")
    text = (REPO / "experiment" / "cases" / "plate_cgns_step49.json").read_text(encoding="utf-8")
    cfg = json.loads(text)
    cfg["mesh"]["file"] = path.as_posix()
    cfg["time"] = {"dt": 1e-4, "end_time": 1e-4, "end_step": 1}
    tmp_json = REPO / "experiment" / "cases" / "_tmp_nonreflecting_test.json"
    try:
        tmp_json.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        loaded = runtime_config.load_config(str(tmp_json))
    finally:
        tmp_json.unlink(missing_ok=True)
    args = runtime_config.resolve_args({}, loaded, main.build_parser().parse_args([]))
    state = init.prepare(args)
    outlet = state.execution.faces[1]
    assert outlet.kind == NR and outlet.side == "i+"
    assert state.execution.band == (0, 0, 1, 0, 0, 0)
    rhs = main.build_rhs(state.case, args, execution=state.execution)
    q1 = state.execution.prepare(solver.rk3_step(state.cons0, 1e-4, rhs))
    out = np.asarray(q1)
    assert np.isfinite(out).all()
    inner = state.desc.grid.inner()
    assert np.all(np.asarray(q1[inner])[0] > 0)
