"""Step 4.n：通用 FARFIELD 与 CGNS BCFarfield 接入测试。"""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from opencfd_jax.src.boundary import (FARFIELD, LAP, PERIODIC, WALL,
                                      BoundaryExecution, build_execution,
                                      fill_ev_axis, fill_faces, fill_halos,
                                      zero_nonperiodic_ghosts)
from opencfd_jax.src.cases import build_case
from opencfd_jax.src.fields import Grid
from opencfd_jax.src.mesh_cgns import CgnsBC, read_cgns_zone, to_block_mesh
from opencfd_jax.src.mesh_data import FaceBoundary

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "Grid" / "fixtures" / "poiseuille_box.cgns"
FAR = jnp.array([1.0, 0.1, 0.2, 0.3, 2.5])


def _make_farfield_fixture(tmp_path, sides=("Jlo", "Jhi")):
    """复制 Step 4 fixture，把指定 j 侧改成 BCFarfield。"""
    dst = tmp_path / "poiseuille_farfield.cgns"
    shutil.copyfile(FIXTURE, dst)
    with h5py.File(dst, "r+") as f:
        bcnode = f["Base"]["Zone 1"]["ZoneBC"]
        for name in sides:
            node = bcnode[name]
            del node[" data"]
            node.create_dataset(
                " data", data=np.frombuffer(b"BCFarfield", dtype=np.int8))
    return dst


def test_fill_halos_farfield_writes_boundary_and_ghost_layers():
    grid = Grid(ni=4, nj=5, nk=6)
    q = grid.alloc()
    q = fill_halos(q, (PERIODIC, FARFIELD, PERIODIC),
                   grid.ni, grid.nj, grid.nk, farfield=FAR)
    layer = FAR.astype(q.dtype).reshape(5, 1, 1, 1)
    assert jnp.all(q[:, :, :LAP + 1, :] == layer)
    assert jnp.all(q[:, :, LAP + grid.nj - 1:, :] == layer)


def test_zero_state_ghosts_clears_farfield_ghosts():
    grid = Grid(ni=4, nj=5, nk=6)
    q = fill_halos(grid.alloc(), (PERIODIC, FARFIELD, PERIODIC),
                   grid.ni, grid.nj, grid.nk, farfield=FAR)
    q = zero_nonperiodic_ghosts(q, (PERIODIC, FARFIELD, PERIODIC))
    assert jnp.all(q[:, :, :LAP, :] == 0.0)
    assert jnp.all(q[:, :, LAP + grid.nj:, :] == 0.0)
    # 清零只覆盖鬼层，远场边界第一层内点保留。
    assert jnp.all(q[:, :, LAP:LAP + 1, :] == FAR.astype(q.dtype).reshape(5, 1, 1, 1))


def test_fill_ev_farfield_uses_zero_gradient_ghosts():
    grid = Grid(ni=4, nj=5, nk=6)
    ev = jnp.arange(4 * (grid.nk + 2 * LAP) * (grid.nj + 2 * LAP) *
                    (grid.ni + 2 * LAP), dtype=jnp.float64).reshape(
                        4, grid.nk + 2 * LAP, grid.nj + 2 * LAP,
                        grid.ni + 2 * LAP)
    out = fill_ev_axis(ev, 1, FARFIELD, grid.nj, is_ev=True)
    assert jnp.all(out[:, :, :LAP, :] == ev[:, :, LAP:LAP + 1, :])
    assert jnp.all(out[:, :, LAP + grid.nj:, :] ==
                   ev[:, :, LAP + grid.nj - 1:LAP + grid.nj, :])


def test_generic_execution_requires_valid_farfield_state():
    grid = Grid(ni=4, nj=5, nk=6)
    with pytest.raises(ValueError, match="farfield"):
        build_execution(dict(grid=grid, bcs=(PERIODIC, FARFIELD, PERIODIC),
                             dxs=(1.0, 1.0, 1.0), axes=(0, 1)))
    for raw in (jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])):
        with pytest.raises(ValueError, match="farfield"):
            build_execution(dict(grid=grid, bcs=(PERIODIC, FARFIELD, PERIODIC),
                                 dxs=(1.0, 1.0, 1.0), axes=(0, 1),
                                 farfield=raw))


def test_cgns_bcfarfield_maps_to_generic_farfield():
    zone = read_cgns_zone(FIXTURE)
    bcs = tuple(
        CgnsBC(name=b.name,
               bc_type="BCFarfield" if b.name in ("Jlo", "Jhi") else b.bc_type,
               point_range=b.point_range)
        for b in zone.bcs)
    far_zone = replace(zone, bcs=bcs)
    block = to_block_mesh(far_zone, axes=(0, 1))
    assert block.bcs == (PERIODIC, FARFIELD, PERIODIC)
    assert [f.kind for f in block.faces] == [
        PERIODIC, PERIODIC, FARFIELD, FARFIELD, PERIODIC, PERIODIC]
    execution = build_execution(
        dict(grid=block.grid, bcs=block.bcs, dxs=block.dxs, axes=block.axes,
             farfield=FAR), block)
    assert execution.mode == "generic"
    assert execution.farfield is not None
    assert tuple(execution.farfield.shape) == (5,)


def test_single_sided_cgns_farfield_does_not_require_opposite_face():
    """CGNS 只有 j+ 是 BCFarfield；j- 仍是壁面，不能被轴级 bcs 带成远场。"""
    zone = read_cgns_zone(FIXTURE)
    bcs = tuple(
        CgnsBC(name=b.name,
               bc_type="BCFarfield" if b.name == "Jhi" else b.bc_type,
               point_range=b.point_range)
        for b in zone.bcs)
    far_zone = replace(zone, bcs=bcs)
    block = to_block_mesh(far_zone, axes=(0, 1))
    assert block.bcs == (PERIODIC, WALL, PERIODIC)
    kinds = [f.kind for f in block.faces]
    assert kinds == [PERIODIC, PERIODIC, WALL, FARFIELD, PERIODIC, PERIODIC]
    execution = build_execution(
        dict(grid=block.grid, bcs=block.bcs, dxs=block.dxs,
             axes=block.axes, farfield=FAR), block)
    q = block.grid.alloc().at[:, :, LAP, :].set(10.0)
    q = execution.prepare(q)
    layer = FAR.astype(q.dtype).reshape(5, 1, 1, 1)
    ni, nj, nk = block.grid.ni, block.grid.nj, block.grid.nk
    # FARFIELD 写内区最后一层内点，鬼层清零（开带，模板不读鬼胞）。
    assert jnp.all(q[:, LAP:LAP + nk, LAP + nj - 1:LAP + nj, LAP:LAP + ni] == layer)
    assert jnp.all(q[:, LAP:LAP + nk, LAP + nj:, LAP:LAP + ni] == 0.0)
    # j- 边界内点没有被远场覆盖，说明只填了 j+ 这一面。
    assert jnp.all(q[:, LAP:LAP + nk, LAP, LAP:LAP + ni] == 10.0)


def test_default_cgns_farfield_is_plus_x_free_stream(tmp_path):
    mesh = _make_farfield_fixture(tmp_path, sides=("Jhi",))
    args = argparse.Namespace(
        case="poiseuille", mesh_format="cgns", mesh=str(mesh), cgns_zone=None,
        config_sources={}, ni=None, nj=None, nk=None, p_grad=1.0,
        forcing_work=True, Re=1000.0, Ama=0.1, ref_t=288.15, pr=0.7)
    case = build_case(args)
    expected = jnp.array([1.0, 1.0, 0.0, 0.0,
                          1.0 * case["params"].Cv + 0.5])
    np.testing.assert_allclose(np.asarray(case["farfield"]),
                               np.asarray(expected), rtol=1e-14, atol=0.0)


def test_prepare_runs_special_boundary_after_general_ghosts():
    grid = Grid(4, 5, 6)
    faces = (
        FaceBoundary("i-", 0, PERIODIC, "halo", False),
        FaceBoundary("i+", 0, PERIODIC, "halo", False),
        FaceBoundary("j-", 1, WALL, "halo", False),
        FaceBoundary("j+", 1, FARFIELD, "halo", False),
        FaceBoundary("k-", 2, PERIODIC, "halo", False),
        FaceBoundary("k+", 2, PERIODIC, "halo", False),
    )
    seen = []

    def special(q):
        seen.append(float(np.asarray(q[0, 0, LAP + grid.nj, 0])))
        return q.at[..., LAP + grid.nj:, :].set(99.0)

    execution = BoundaryExecution("custom", (PERIODIC, WALL, PERIODIC),
                                  special, faces, grid=grid, farfield=FAR)
    q = execution.prepare(grid.alloc())
    assert seen == [0.0]              # 回调执行时远场鬼层是 0（边界点已单独赋值）
    assert jnp.all(q[..., LAP + grid.nj:, :] == 99.0)
    ni, nj, nk = grid.ni, grid.nj, grid.nk
    assert jnp.all(q[:, LAP:LAP + nk, LAP + nj - 1, LAP:LAP + ni]
                   == FAR.reshape(5, 1, 1))


def test_callback_face_gets_general_outflow_before_special():
    """callback 面也按 kind 先清零普通鬼层，再由特殊回调覆盖。"""
    grid = Grid(4, 5, 6)
    faces = (
        FaceBoundary("i-", 0, "outflow", "callback", False),
        FaceBoundary("i+", 0, "outflow", "halo", False),
        FaceBoundary("j-", 1, PERIODIC, "halo", False),
        FaceBoundary("j+", 1, PERIODIC, "halo", False),
        FaceBoundary("k-", 2, PERIODIC, "halo", False),
        FaceBoundary("k+", 2, PERIODIC, "halo", False),
    )
    seen = []

    def special(q):
        seen.append(float(np.asarray(q[0, 0, 0, LAP - 1])))
        return q.at[..., :LAP].set(88.0)

    execution = BoundaryExecution("custom", ("outflow", PERIODIC, PERIODIC),
                                  special, faces, grid=grid)
    q = grid.alloc().at[:, :, :, LAP].set(7.0)
    q = execution.prepare(q)
    assert seen == [0.0]             # OUTFLOW 保留初始鬼层，随后由回调覆盖
    assert jnp.all(q[..., LAP] == 7.0)
    assert jnp.all(q[..., :LAP] == 88.0)


def test_poiseuille_cgns_bcfarfield_builds_a_farfield_state(tmp_path):
    mesh = _make_farfield_fixture(tmp_path)
    args = argparse.Namespace(
        case="poiseuille", mesh_format="cgns", mesh=str(mesh), cgns_zone=None,
        config_sources={}, ni=None, nj=None, nk=None, p_grad=1.0,
        forcing_work=True, Re=1000.0, Ama=0.1, ref_t=288.15, pr=0.7)
    case = build_case(args)
    assert case["bcs"] == (PERIODIC, FARFIELD, PERIODIC)
    assert case["farfield"] is not None
    assert tuple(case["farfield"].shape) == (5,)
    execution = build_execution(case)
    assert execution.farfield is not None
    assert tuple(execution.farfield.shape) == (5,)
