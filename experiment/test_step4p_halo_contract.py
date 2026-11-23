"""Step 4.p：鬼层/边界点/边界带的分工，以及 INFLOW 剖面文件。

本轮口径（用户拍板）：
  * 鬼层由 `fill_faces` 单独负责：周期面复制对侧，INFLOW 保持，其余清零；
  * 常规边界只写**边界点**（内区第一层）：INFLOW 例外，写边界点 + 四层鬼层；
    FARFIELD 只写单点；WALL/OUTFLOW 不赋值（由算例回调写壁面）；
  * `DIF_BOUND` 由 kind 派生：非周期且非 INFLOW 才开单侧带，特殊回调继承；
  * 入口剖面走 OCFD 1D 文件格式（`y rho u v T`，`w≡0`），与格点逐点对应、
    不插值；不给文件就回落自由来流常数。
"""

import argparse

import numpy as np
import pytest

from opencfd_jax.src import cases, inflow
from opencfd_jax.src.boundary import (FARFIELD, INFLOW, LAP, OUTFLOW,
                                      PERIODIC, WALL, build_execution,
                                      dif_band, fill_faces, halo_mode,
                                      write_faces)
from opencfd_jax.src.errors import SetupError
from opencfd_jax.src.fields import Grid
from opencfd_jax.src.mesh_data import FaceBoundary

NJ, NK = 6, 5          # 鬼层是 4 层，内区至少 4 层周期复制才无歧义


def _grid():
    return Grid(ni=5, nj=NJ, nk=NK)


def _faces(i_minus=INFLOW, i_plus=OUTFLOW, j_minus=WALL, j_plus=FARFIELD,
           k_kind=PERIODIC):
    return (
        FaceBoundary("i-", 0, i_minus, "halo", False),
        FaceBoundary("i+", 0, i_plus, "halo", False),
        FaceBoundary("j-", 1, j_minus, "halo", False),
        FaceBoundary("j+", 1, j_plus, "halo", False),
        FaceBoundary("k-", 2, k_kind, "halo", False),
        FaceBoundary("k+", 2, k_kind, "halo", False),
    )


def test_band_is_derived_from_kind_not_from_case():
    """DIF_BOUND 由 kind 派生：壁面与远场开带，其余不开。

    壁面/远场的鬼层是 0（边界点由 `write_faces` 单独赋值），中心模板跨边界
    会读到 0，必须换单侧；周期（鬼层是对侧内区）、入口（鬼层是剖面）、
    出口（鬼层是边界点延拓）鬼层合法，开带只会降精度。
    """
    assert dif_band(PERIODIC) is False
    assert dif_band(INFLOW) is False
    assert dif_band(OUTFLOW) is False
    assert dif_band(WALL) is True
    assert dif_band(WALL, executor="callback") is True
    assert dif_band(FARFIELD) is True


def test_halo_mode_splits_periodic_keep_copy_zero():
    assert halo_mode(PERIODIC) == "periodic"
    assert halo_mode(INFLOW) == "keep"
    # 出口保留已有鬼层；壁面/远场鬼层清零、边界点单独赋值。
    assert halo_mode(OUTFLOW) == "keep"
    assert halo_mode(WALL) == "zero"
    assert halo_mode(FARFIELD) == "zero"


def test_nonperiodic_ghosts_are_zero_and_periodic_ghosts_copied():
    grid = _grid()
    nk, nj, ni = grid.nk, grid.nj, grid.ni
    q = grid.alloc().at[:, LAP:LAP + nk, LAP:LAP + nj, LAP:LAP + ni].set(3.0)
    # j- 用 WALL+callback（壁面点由回调写，鬼层归零），避免走通用壁面镜像。
    faces = (FaceBoundary("i-", 0, INFLOW, "halo", False),
             FaceBoundary("i+", 0, OUTFLOW, "halo", False),
             FaceBoundary("j-", 1, WALL, "callback", False),
             FaceBoundary("j+", 1, OUTFLOW, "halo", False),
             FaceBoundary("k-", 2, PERIODIC, "halo", False),
             FaceBoundary("k+", 2, PERIODIC, "halo", False))
    q = fill_faces(q, faces, grid)
    # 周期 k：鬼层复制对侧内区（取一条确定在内区的 j 剖面）
    np.testing.assert_array_equal(
        np.asarray(q[:, :LAP, LAP, LAP:LAP + ni]),
        np.broadcast_to(np.asarray(q[:, LAP:LAP + 1, LAP, LAP:LAP + ni]),
                        (5, LAP, ni)))
    # 非周期：i/j 鬼层清零（k 鬼层是周期复制，不在清零之列）
    assert np.all(np.asarray(q[:, LAP:LAP + nk, :, :LAP]) == 0.0)
    assert np.all(np.asarray(q[:, LAP:LAP + nk, :LAP, :]) == 0.0)


def test_generic_wall_zeroes_halos_like_callback_wall():
    """通用 WALL 与回调 WALL 一样把鬼层清零：壁面点由 write_faces/回调单独赋值。"""
    grid = _grid()
    nk, nj = grid.nk, grid.nj
    q = grid.alloc().at[:, LAP:LAP + nk, LAP:LAP + nj, :].set(3.0)
    q = fill_faces(q, _faces(j_minus=WALL), grid)
    assert np.all(np.asarray(q[:, LAP:LAP + nk, :LAP, LAP:LAP + grid.ni]) == 0.0)


def test_inflow_keeps_its_ghosts_until_it_writes_them():
    grid = _grid()
    nk, nj = grid.nk, grid.nj
    profile = np.zeros((5, nk, nj))
    profile[0] = 1.0                     # rho 剖面
    q = grid.alloc().at[:, LAP:LAP + nk, :, :].set(7.0)
    faces = _faces(j_plus=OUTFLOW)
    q = fill_faces(q, faces, grid)
    # INFLOW 不参与清零：入口鬼层保留原值，交给 write_faces 覆写
    assert np.all(np.asarray(q[:, LAP:LAP + nk, LAP:LAP + nj, :LAP]) == 7.0)

    q = write_faces(q, faces, grid, inflow=profile)
    expect = np.broadcast_to(profile.reshape(5, nk, nj, 1),
                             (5, nk, nj, LAP + 1))
    np.testing.assert_array_equal(
        np.asarray(q[:, LAP:LAP + nk, LAP:LAP + nj, :LAP + 1]), expect)


def test_farfield_zeroes_ghosts_and_outflow_keeps_existing_state():
    """远场清零鬼层并写边界点；出口保留初始鬼层。"""
    grid = _grid()
    nk, ni = grid.nk, grid.ni
    far = np.array([1.0, 1.0, 0.0, 0.0, 179.07142857142858])
    initial = grid.alloc().at[:, LAP:LAP + nk, LAP + 1:LAP + grid.nj - 1,
                              LAP + ni:].set(9.0)
    q = write_faces(fill_faces(initial, _faces(), grid), _faces(), grid,
                    farfield=far)
    q = fill_faces(q, _faces(), grid)        # 与 prepare 一致：写点后再填
    nj_in = LAP + grid.nj
    np.testing.assert_array_equal(
        np.asarray(q[:, LAP:LAP + nk, nj_in - 1, LAP:LAP + ni]),
        np.broadcast_to(far.reshape(5, 1, 1), (5, nk, ni)))
    assert np.all(np.asarray(q[:, LAP:LAP + nk, nj_in:, LAP:LAP + ni]) == 0.0)
    np.testing.assert_array_equal(
        np.asarray(q[:, LAP:LAP + nk, LAP + 1:LAP + grid.nj - 1, LAP + ni:]), 9.0)


def _ns(**kw):
    base = dict(case="plate", ni=5, nj=NJ, nk=NK, dt=1e-3, end_time=1e-3,
                end_step=1, Re=1000.0, Ama=0.1, ref_t=288.15, pr=0.7,
                p_grad=None, forcing_work=False, plate_x0=0.1, plate_x1=1.1,
                plate_h=0.12, plate_dif_bound="0 0 0 0 1 0",
                config_sources={}, mesh=None, mesh_format="builtin",
                cgns_zone=None, inflow_profile=None, tw=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_plate_band_derived_and_inflow_constant(tmp_path):
    case = cases.build_case(_ns())
    execution = build_execution(case)
    # 壁面与远场开带；i+ 出口保留鬼层、不开带；周期不开。
    assert execution.band == (0, 0, 0, 0, 1, 0)  # 显式口径生效（2026-11-23）
    assert np.asarray(execution.inflow)[0, 0, 0] == 1.0


def test_plate_reads_ocfd_1d_profile(tmp_path):
    path = tmp_path / "flow1d-inlet.dat"
    rows = ["y rho u v T"]
    for j in range(NJ):
        rows.append(f"{j * 0.02:.6f} 1.0 1.0 0.0 1.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    prof = inflow.read_inflow_profile(path, NJ, cases.build_case(_ns())["params"].Cv)
    assert prof.shape == (5, NJ)
    assert np.all(prof[3] == 0.0)                  # 原码 w 固定 0
    cv = 178.57142857142858
    np.testing.assert_allclose(prof[4], 1.0 * cv + 0.5, rtol=1e-15)

    case = cases.build_case(_ns(inflow_profile=str(path)))
    assert tuple(case["inflow"].shape) == (5, NK, NJ)
    np.testing.assert_allclose(np.asarray(case["inflow"])[1], 1.0)


def test_profile_point_count_must_match_grid(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text("y rho u v T\n0 1 1 0 1\n", encoding="utf-8")
    with pytest.raises(SetupError, match="逐点对应"):
        cases.build_case(_ns(nj=2, inflow_profile=str(path)))


def test_profile_file_missing_is_setup_error(tmp_path):
    with pytest.raises(SetupError, match="不存在"):
        cases.build_case(_ns(inflow_profile=str(tmp_path / "nope.dat")))
