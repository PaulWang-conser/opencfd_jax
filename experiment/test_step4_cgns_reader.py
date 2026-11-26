"""Step 4 单块 CGNS/HDF5 reader 的验收测试。

只覆盖当前执行子集：三维 Structured、Vertex 坐标、整面 BC、
同块 identity 1to1；真实多 zone 样例只做 ``validate=False`` 的结构读取。
"""

from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src.mesh_cgns import CgnsError, read_cgns_zone

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "Grid" / "fixtures" / "poiseuille_box.cgns"
SAMPLE = REPO / "Grid" / "samples" / "multi_hdf5.cgns"


def test_fixture_vertex_and_cell_sizes():
    zone = read_cgns_zone(FIXTURE)
    assert zone.name == "Zone 1"
    assert zone.vertex_size == (16, 16, 8)
    assert zone.cell_size == (15, 15, 7)


def test_fixture_coords_are_vertex_centered_ijk():
    zone = read_cgns_zone(FIXTURE)
    dx = 2.0 * np.pi / zone.ni
    dz = 2.0 * np.pi / zone.ni
    dy = 1.0 / zone.nj
    assert zone.coords["Axx"].shape == (zone.nk, zone.nj, zone.ni)
    assert zone.coords["Axx"][0, 0, 0] == pytest.approx(0.5 * dx)
    assert zone.coords["Axx"][0, 0, -1] == pytest.approx((zone.ni - 0.5) * dx)
    assert zone.coords["Ayy"][0, 0, 0] == pytest.approx(0.5 * dy)
    assert zone.coords["Azz"][0, 0, 0] == pytest.approx(0.0)
    assert zone.coords["Azz"][-1, 0, 0] == pytest.approx((zone.nk - 1) * dz)


def test_fixture_bcs_and_identity_periodic_connections():
    zone = read_cgns_zone(FIXTURE)
    assert {b.bc_type for b in zone.bcs} == {"BCWall"}
    assert {b.name for b in zone.bcs} == {"Jhi", "Jlo"}
    assert len(zone.connections) == 4
    for conn in zone.connections:
        assert conn.donor_zone == zone.name
        assert conn.transform == (1, 2, 3)


def test_zone_selection_and_normalized_name():
    zone = read_cgns_zone(FIXTURE, zone="  Zone   1 ")
    assert zone.name == "Zone 1"
    with pytest.raises(CgnsError, match="找不到唯一 zone"):
        read_cgns_zone(FIXTURE, zone="missing")


def test_missing_file_is_reported_as_cgns_error(tmp_path):
    with pytest.raises(CgnsError, match="无法打开"):
        read_cgns_zone(tmp_path / "missing.cgns")


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少 Grid/samples/multi_hdf5.cgns")
def test_real_sample_raw_read_keeps_all_bc_types():
    zone = read_cgns_zone(SAMPLE, zone="Zone 1", validate=False)
    assert zone.vertex_size == (2, 123, 25)
    assert zone.cell_size == (1, 122, 24)
    types = {b.name: b.bc_type for b in zone.bcs}
    assert types["Jlo_Seg   1"] == "BCFarfield"
    assert types["Ilo_Seg   1"] == "BCSymmetryPlane"
    assert any(set(c.point_range_donor) == {(0, 122, 0), (1, 98, 0)}
               for c in zone.connections)


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少 Grid/samples/multi_hdf5.cgns")
def test_real_sample_strict_mode_rejects_non_executable_bc():
    # symmetry plane 不再视作周期；该样例在第一个此类边条处被拒绝。
    with pytest.raises(CgnsError, match="BCSymmetryPlane"):
        read_cgns_zone(SAMPLE, zone="Zone 1", validate=True)


# ---- 2026-11-23 忠实性复审补的负例（路线图 Step 4 拒绝清单） ----

def _fixture_copy(tmp_path, ni=8, nj=8, nk=4):
    """在临时目录生成标准 fixture 的副本，供负例篡改。"""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from Grid.make_cgns_fixture import build_fixture
    path = tmp_path / "neg.cgns"
    build_fixture(path, ni=ni, nj=nj, nk=nk)
    return path


def test_rejects_periodic_property_under_1to1(tmp_path):
    """GridConnectivityProperty/Periodic 层级不许漏检（曾只查直接子节点）。"""
    import h5py
    path = _fixture_copy(tmp_path)
    with h5py.File(path, "r+") as f:
        node = f["Base/Zone 1/ZoneGridConnectivity/1to1Ilo"]
        prop = node.create_group("GridConnectivityProperty")
        prop.attrs["label"] = np.bytes_("GridConnectivityProperty_t")
        per = prop.create_group("Periodic")
        per.attrs["label"] = np.bytes_("Periodic_t")
    with pytest.raises(CgnsError, match="Periodic_t"):
        read_cgns_zone(path)


def test_rejects_mixed_polar_bc_and_conn_periodic_pair(tmp_path):
    """一面 BCSymmetryPolar、对面 identity 1to1 的混配必须被拒绝。"""
    import h5py
    path = _fixture_copy(tmp_path)
    with h5py.File(path, "r+") as f:
        zone = f["Base/Zone 1"]
        del zone["ZoneGridConnectivity/1to1Ihi"]
        node = zone["ZoneBC"].create_group("IhiPolar")
        node.attrs["label"] = np.bytes_("BC_t")
        node.create_dataset(
            " data", data=np.frombuffer(b"BCSymmetryPolar", dtype=np.int8))
        rng = node.create_group("PointRange")
        rng.attrs["label"] = np.bytes_("IndexRange_t")
        rng.create_dataset(
            " data", data=np.asarray([[8, 1, 1], [8, 8, 4]], dtype=np.int32))
    with pytest.raises(CgnsError, match="来源不同"):
        read_cgns_zone(path)
