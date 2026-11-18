"""Step 4 CGNS -> BlockMesh 适配与曲线度量的验收测试。"""

from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src.boundary import PERIODIC, WALL
from opencfd_jax.src.axes import SIDES
from opencfd_jax.src.mesh import METRIC_KEYS
from opencfd_jax.src.mesh_cgns import (CgnsError, build_cgns_block,
                                      read_cgns_zone, to_block_mesh)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "Grid" / "fixtures" / "poiseuille_box.cgns"


def test_to_block_mesh_uses_vertex_sizes_and_periodic_wall_bcs():
    zone = read_cgns_zone(FIXTURE)
    block = to_block_mesh(zone, axes=(0, 1))
    assert (block.grid.ni, block.grid.nj, block.grid.nk) == (16, 16, 8)
    assert block.dxs == (1.0, 1.0, 1.0)
    assert block.axes == (0, 1)
    assert block.bcs == (PERIODIC, WALL, PERIODIC)
    assert tuple(f.side for f in block.faces) == SIDES
    assert [f.kind for f in block.faces] == [
        PERIODIC, PERIODIC, WALL, WALL, PERIODIC, PERIODIC]
    assert all(f.executor == "halo" for f in block.faces)
    assert block.metrics is None
    assert block.coords is zone.coords


def test_to_block_mesh_rejects_invalid_axes():
    zone = read_cgns_zone(FIXTURE)
    with pytest.raises(ValueError, match="axes"):
        to_block_mesh(zone, axes=(0, 0))
    with pytest.raises(ValueError, match="axes"):
        to_block_mesh(zone, axes=())


def test_build_cgns_block_metrics_match_uniform_spacing():
    zone = read_cgns_zone(FIXTURE)
    block = build_cgns_block(zone, axes=(0, 1))
    assert block.has_metrics
    assert set(METRIC_KEYS).issubset(block.metrics)
    dx = 2.0 * np.pi / zone.ni
    dy = 1.0 / zone.nj
    dz = 2.0 * np.pi / zone.ni
    assert block.metrics["Ajac"].shape == (zone.nk + 8, zone.nj + 8,
                                          zone.ni + 8)
    # jac_bound=(1,1,1) 的固定内点，均匀网格应与解析值一致。
    assert block.metrics["Ajac"][4, 4, 4] == pytest.approx(
        1.0 / (dx * dy * dz), rel=2e-3)


def test_non_executable_zone_is_rejected_by_adapter():
    zone = read_cgns_zone(FIXTURE)
    object.__setattr__(zone, "bcs", ())  # dataclass frozen，仅本测试模拟坏输入
    with pytest.raises(CgnsError):
        to_block_mesh(zone)
