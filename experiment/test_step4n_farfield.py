"""Step 4.n：通用 FARFIELD 与 CGNS BCFarfield 接入测试。"""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from opencfd_jax.src.boundary import (FARFIELD, LAP, PERIODIC, fill_ev_axis,
                                      fill_halos, build_execution)
from opencfd_jax.src.cases import build_case
from opencfd_jax.src.fields import Grid
from opencfd_jax.src.mesh_cgns import CgnsBC, read_cgns_zone, to_block_mesh

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "Grid" / "fixtures" / "poiseuille_box.cgns"
FAR = jnp.array([1.0, 0.1, 0.2, 0.3, 2.5])


def _make_farfield_fixture(tmp_path):
    """复制 Step 4 fixture，把 j 双壁改成 BCFarfield。"""
    dst = tmp_path / "poiseuille_farfield.cgns"
    shutil.copyfile(FIXTURE, dst)
    with h5py.File(dst, "r+") as f:
        bcnode = f["Base"]["Zone 1"]["ZoneBC"]
        for name in ("Jlo", "Jhi"):
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
