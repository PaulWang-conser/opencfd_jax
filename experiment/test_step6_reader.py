"""第六步的多 zone 读取与写入（U1）。

锁四件事：
  1. 单 zone 文件用 ``read_cgns_zones`` 读，结果与 ``read_cgns_zone(validate=False)``
     逐个相同：名字、尺寸、坐标、边条、连接；
  2. 多 zone 文件按文件顺序返回，名字里的连续空格原样保留；``Periodic_t`` 的三个
     数组读进连接记录，缺的记为不存在；没有 ``Transform`` 节点时按 identity；
  3. ``GridConnectivity_t``、带正厚度的 ``Rind``、多个 Base、非 Structured 的 zone
     各自被拒绝并点名；``zone_metadata`` 不读坐标，也不挑文件结构；
  4. ``Grid/make_cgns_fixture.py`` 的多 zone 写入函数写出的文件读回后，坐标、
     边条和连接与写入的逐位相同。
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import mesh_cgns
from opencfd_jax.src.mesh_cgns import CgnsError

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

from functools import partial

from Grid.make_cgns_fixture import _group, build_multizone  # noqa: E402
from step6_support import ConnSpec, block  # noqa: E402

PLATE = REPO / "Grid" / "testcgnstest.cgns"
POISEUILLE = REPO / "Grid" / "fixtures" / "poiseuille_box.cgns"
CFL3D = REPO / "Grid" / "samples" / "multi_hdf5.cgns"


def _two_blocks():
    """左右相邻的两块：左块 6×5×4，右块 5×4×3，中间是一条普通切口。"""
    left = block("Zone 1", 6, 5, 4, bcs=[("Jlo", "BCWall", ((1, 1, 1), (6, 1, 4)))],
                  connections=[ConnSpec("cut", "Zone 2", ((6, 1, 1), (6, 5, 4)),
                                        ((1, 1, 1), (1, 4, 3)))])
    right = block("Zone 2", 5, 4, 3, origin=(5.0, 0.0, 0.0),
                   connections=[ConnSpec("cut", "Zone 1", ((1, 1, 1), (1, 4, 3)),
                                         ((6, 1, 1), (6, 5, 4)))])
    return [left, right]


def _assert_same_zone(a, b):
    assert (a.name, a.vertex_size, a.cell_size) == (b.name, b.vertex_size, b.cell_size)
    assert tuple(a.coords) == tuple(b.coords)
    for key in a.coords:
        assert a.coords[key].dtype == b.coords[key].dtype
        np.testing.assert_array_equal(a.coords[key], b.coords[key])
    assert a.bcs == b.bcs
    assert len(a.connections) == len(b.connections)
    for ca, cb in zip(a.connections, b.connections):
        assert (ca.name, ca.donor_zone, ca.point_range, ca.point_range_donor,
                ca.transform) == (cb.name, cb.donor_zone, cb.point_range,
                                  cb.point_range_donor, cb.transform)
        assert ca.periodic == cb.periodic


# ---- 1. 单 zone 文件读得一样 ----

@pytest.mark.parametrize("path", [
    pytest.param(PLATE, id="plate"),
    pytest.param(POISEUILLE, id="poiseuille"),
    pytest.param(CFL3D, id="cfl3d",
                 marks=pytest.mark.skipif(not CFL3D.exists(),
                                          reason="缺少 CFL3D 样例")),
])
def test_zones_match_the_single_reader(path):
    zones = mesh_cgns.read_cgns_zones(path)
    assert zones
    for zone in zones:
        _assert_same_zone(
            zone, mesh_cgns.read_cgns_zone(path, zone=zone.name, validate=False))


# ---- 2. 多 zone 文件的读取 ----

def test_zones_come_in_file_order_with_original_names(tmp_path):
    path = tmp_path / "two.cgns"
    build_multizone(path, [block("Zone  2", 6, 5, 4), block("Zone 1", 5, 4, 3)])
    zones = mesh_cgns.read_cgns_zones(path)
    assert [zone.name for zone in zones] == ["Zone  2", "Zone 1"]
    assert [zone.coords["Axx"].shape for zone in zones] == [(4, 5, 6), (3, 4, 5)]


def test_periodic_arrays_are_read_and_missing_ones_recorded(tmp_path):
    ranges = (((6, 1, 1), (6, 4, 3)), ((1, 1, 1), (1, 4, 3)))
    full = {"RotationCenter": (0.25, 0.5, 0.75),
            "RotationAngle": (0.0, 0.0, 0.0),
            "Translation": (0.0, 0.0, 3.0)}
    conns = [ConnSpec("full", "Zone 2", *ranges, periodic=full),
             ConnSpec("part", "Zone 2", *ranges,
                      periodic={"Translation": (0.0, 0.0, 3.0)}),
             ConnSpec("bare", "Zone 2", *ranges, periodic={}),
             ConnSpec("plain", "Zone 2", *ranges)]
    path = tmp_path / "periodic.cgns"
    build_multizone(path, [
        block("Zone 1", 6, 4, 3, connections=conns), block("Zone 2", 5, 4, 3)])
    full_conn, part, bare, plain = mesh_cgns.read_cgns_zones(path)[0].connections
    assert full_conn.periodic is not None
    np.testing.assert_array_equal(full_conn.periodic.rotation_center, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(full_conn.periodic.rotation_angle, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(full_conn.periodic.translation, [0.0, 0.0, 3.0])
    assert part.periodic is not None
    np.testing.assert_array_equal(part.periodic.translation, [0.0, 0.0, 3.0])
    assert part.periodic.rotation_center is None
    assert part.periodic.rotation_angle is None
    assert bare.periodic is not None
    assert bare.periodic == mesh_cgns.CgnsPeriodic(None, None, None)
    assert plain.periodic is None


def test_connection_without_transform_node_reads_identity(tmp_path):
    ranges = (((6, 1, 1), (6, 4, 3)), ((1, 1, 1), (1, 4, 3)))
    path = tmp_path / "transform.cgns"
    build_multizone(path, [
        block("Zone 1", 6, 4, 3, connections=[
            ConnSpec("bare", "Zone 2", *ranges),
            ConnSpec("identity", "Zone 2", *ranges, transform=(1, 2, 3)),
            ConnSpec("swapped", "Zone 2", *ranges, transform=(2, 1, 3))]),
        block("Zone 2", 5, 4, 3)])
    bare, identity, swapped = mesh_cgns.read_cgns_zones(path)[0].connections
    assert (bare.transform, identity.transform) == ((1, 2, 3), (1, 2, 3))
    assert swapped.transform == (2, 1, 3)


# ---- 3. 拒绝与元数据 ----

_add_node = partial(_group, track_times=False)      # 建组、贴 label，可选写 data


def _zone_groups(f, base="Base"):
    import h5py

    return [obj for obj in f[base].values()
            if isinstance(obj, h5py.Group)
            and obj.attrs.get("label") == b"Zone_t"]


def _damaged(tmp_path, mutate):
    """写一个合法的两块文件，按 ``mutate`` 改坏一处，返回路径。"""
    import h5py

    path = tmp_path / "damaged.cgns"
    build_multizone(path, _two_blocks())
    with h5py.File(path, "r+") as f:
        mutate(f)
    return path


def test_general_connectivity_is_rejected_by_name(tmp_path):
    path = _damaged(tmp_path, lambda f: _add_node(
        f["Base/Zone 1/ZoneGridConnectivity"], "abutting", "GridConnectivity_t"))
    with pytest.raises(CgnsError, match="Zone 1/abutting.*一般连接"):
        mesh_cgns.read_cgns_zones(path)


def test_positive_rind_is_rejected_by_name(tmp_path):
    def mutate(f):
        _add_node(f["Base/Zone 2/GridCoordinates"], "Rind", "Rind_t",
                  [1, 0, 0, 1, 0, 0])

    path = _damaged(tmp_path, mutate)
    with pytest.raises(CgnsError, match="Zone 2/GridCoordinates.*Rind"):
        mesh_cgns.read_cgns_zones(path)


def test_several_bases_are_rejected_by_name(tmp_path):
    path = _damaged(tmp_path, lambda f: _add_node(
        f, "Base 2", "CGNSBase_t", [3, 3]))
    with pytest.raises(CgnsError, match="2 个 CGNSBase_t.*Base 2"):
        mesh_cgns.read_cgns_zones(path)


def test_single_reader_keeps_the_order_it_had_before_step6(tmp_path):
    """单块读取 ``read_cgns_zone`` 逐个 Base 先验维数、再列出它的 zone。两个 Base 各坏
    一处时，先报排在前面那个 Base 的错。这是第六步之前就有的次序，这里锁的是原有行为，
    不是期望行为。对照的依据见下一个测试的说明。"""
    import h5py

    path = tmp_path / "two_broken_bases.cgns"
    shutil.copyfile(PLATE, path)
    with h5py.File(path, "r+") as f:
        f["Base/nowhere"] = h5py.SoftLink("/no/such/node")     # Base A：列 zone 时出错
        _add_node(f, "Base B", "CGNSBase_t")                    # Base B：没有维数数据
    with pytest.raises(AttributeError):
        mesh_cgns.read_cgns_zone(path)


def test_single_reader_keeps_its_raw_errors_on_damaged_files(tmp_path):
    """残缺的文件在单块读取里抛的是原始异常，不是 ``CgnsError``。这是第六步之前就有的
    行为，这里锁的是它；要让这些报错变成带名字的 ``CgnsError``，得作为对单块路径的
    有意改动另行决定。

    对照的依据（2026-11-24）：把提交前的 ``src/mesh_cgns.py`` 单独载入，对 64 个正常和
    残缺的文件，按每个文件适用的方式各读若干次（缺省、``validate=False``；每个 zone 名
    两种，带不带可执行校验；再选一个不存在的 zone），共 422 次；每一次抛出的异常类型
    和消息、或者读出的内容，都与现在的 ``read_cgns_zone`` 逐项相同。"""
    import h5py

    def base_without_data(f):
        del f["Base"][" data"]

    def base_with_dangling_link(f):
        f["Base/nowhere"] = h5py.SoftLink("/no/such/node")

    def sizes_are_text(f):
        node = _zone_groups(f)[0]
        del node[" data"]
        node.create_dataset(" data", data=np.full((3, 3), b"x", "S3"))

    for damage, error in ((base_without_data, KeyError),
                          (base_with_dangling_link, AttributeError),
                          (sizes_are_text, ValueError)):
        path = tmp_path / "damaged.cgns"
        shutil.copyfile(PLATE, path)
        with h5py.File(path, "r+") as f:
            damage(f)
        with pytest.raises(error):
            mesh_cgns.read_cgns_zone(path)


def test_both_readers_raise_only_cgns_errors_on_damaged_files(tmp_path):
    """两个公开读取器只抛 ``CgnsError``。残缺的方式枚举不完，这里列的是实际被找到过
    会漏出裸异常的四种：Base 缺维数数据、根下有悬空链接、标成 Base 的节点是标量
    数据集、zone 名含非 UTF-8 的字节。``read_cgns_zones`` 对四种都报错；``zone_metadata``
    不查 Base 的维数、跳过悬空的链接，所以前两种它不报错，只是不许抛别的异常。"""
    import h5py

    def without_base_data(f):
        del f["Base"][" data"]

    def with_root_link(f):
        f["nowhere"] = h5py.SoftLink("/no/such/node")

    def with_scalar_base(f):
        node = f.create_dataset("Extra", data=3)
        node.attrs["label"] = np.bytes_("CGNSBase_t")

    def with_odd_zone_name(f):
        f["Base"].move("Zone 2", b"zone\xff\xfe")

    for mutate in (without_base_data, with_root_link, with_scalar_base,
                   with_odd_zone_name):
        path = _damaged(tmp_path, mutate)
        with pytest.raises(CgnsError):
            mesh_cgns.read_cgns_zones(path)
        try:
            mesh_cgns.zone_metadata(path)
        except CgnsError:
            pass


def test_unstructured_zone_is_rejected_by_name(tmp_path):
    def mutate(f):
        node = f["Base/Zone 2/ZoneType"]
        del node[" data"]
        node.create_dataset(
            " data", data=np.frombuffer(b"Unstructured", dtype=np.int8))

    path = _damaged(tmp_path, mutate)
    with pytest.raises(CgnsError, match="Zone 2.*Unstructured"):
        mesh_cgns.read_cgns_zones(path)


def test_zone_metadata_skips_coordinates_and_file_structure(tmp_path):
    import h5py

    path = tmp_path / "no_x.cgns"
    shutil.copyfile(PLATE, path)
    with h5py.File(path, "r+") as f:
        del _zone_groups(f)[0]["GridCoordinates/CoordinateX"]
    meta, = mesh_cgns.zone_metadata(path)
    assert (meta.name, meta.zone_type, meta.vertex_size) == ("blk-1", "Structured",
                                                             (100, 40, 40))
    assert meta.cell_size == (99, 39, 39)
    with pytest.raises(CgnsError, match="CoordinateX"):
        mesh_cgns.read_cgns_zones(path)


def test_zone_metadata_reports_every_zone_in_file_order(tmp_path):
    """按文件顺序列出每个 zone。内容有问题的 zone 不报错：原因记进 ``problem``，
    读不出的尺寸是 None；原因与读取器的报错是同一句话。"""
    import h5py

    path = tmp_path / "meta.cgns"
    build_multizone(path, [block("Zone  2", 6, 5, 4), block("Zone 1", 5, 4, 3),
                           block("Zone 3", 5, 4, 3)])
    with h5py.File(path, "r+") as f:
        node = f["Base/Zone 1/ZoneType"]
        del node[" data"]
        node.create_dataset(
            " data", data=np.frombuffer(b"Unstructured", dtype=np.int8))
        third = f["Base/Zone 3"]
        del third[" data"]
        third.create_dataset(" data", data=np.array([[5], [4], [0]], dtype=np.int32))
        _add_node(f, "Base 2", "CGNSBase_t", [3, 3])
    good, unstructured, misshapen = mesh_cgns.zone_metadata(path)
    assert (good.name, good.zone_type, good.vertex_size, good.cell_size, good.problem) == (
        "Zone  2", "Structured", (6, 5, 4), (5, 4, 3), None)
    assert (unstructured.name, unstructured.zone_type, unstructured.vertex_size) == (
        "Zone 1", "Unstructured", None)
    assert unstructured.problem == "Zone 1 的 ZoneType='Unstructured'；只支持 Structured"
    assert (misshapen.name, misshapen.vertex_size, misshapen.problem) == (
        "Zone 3", None, "Zone 3 的 Zone_t data 需要 (3,3)，得到 (3, 1)")
    with pytest.raises(CgnsError) as info:        # 读取器对同一个 zone 的报错
        mesh_cgns.read_cgns_zone(path, zone="Zone 1")
    assert str(info.value) == unstructured.problem


# ---- 4. 写入器的往返 ----

def _zero_based(rng):
    """把写入时用的 1-based 闭区间换成读取器返回的 0-based 闭区间。"""
    return (tuple(x - 1 for x in rng[0]), tuple(x - 1 for x in rng[1]))


def test_writer_round_trip_is_bitwise(tmp_path):
    conns = [ConnSpec("cut", "Zone 2", ((6, 1, 1), (6, 5, 4)),
                      ((1, 1, 1), (1, 4, 3))),
             ConnSpec("seam", "Zone 2", ((1, 1, 4), (6, 5, 4)),
                      ((1, 1, 1), (5, 4, 1)),
                      periodic={"RotationAngle": (0.0, 0.0, 0.0),
                                "Translation": (0.0, 0.0, 3.0)})]
    zones = [
        block("Zone 1", 6, 5, 4, bcs=[
            ("Jlo", "BCWall", ((1, 1, 1), (6, 1, 4))),
            ("Ilo", "BCInflow", ((1, 1, 1), (1, 5, 4)))], connections=conns),
        block("Zone 2", 5, 4, 3, origin=(5.0, 0.0, 0.0), bcs=[
            ("Jhi", "BCFarfield", ((1, 4, 1), (5, 4, 3)))]),
    ]
    path = tmp_path / "round.cgns"
    build_multizone(path, zones)
    read = mesh_cgns.read_cgns_zones(path)
    assert [zone.name for zone in read] == ["Zone 1", "Zone 2"]
    assert [(z.vertex_size, z.cell_size) for z in read] == [
        ((6, 5, 4), (5, 4, 3)), ((5, 4, 3), (4, 3, 2))]
    for spec, zone in zip(zones, read):
        for key, value in spec.coords.items():
            assert zone.coords[key].dtype == np.float64
            np.testing.assert_array_equal(zone.coords[key], value)
        assert [(b.name, b.bc_type, b.point_range) for b in zone.bcs] == [
            (name, bc_type, _zero_based(rng)) for name, bc_type, rng in spec.bcs]
        assert [(c.name, c.donor_zone, c.point_range, c.point_range_donor,
                 c.transform) for c in zone.connections] == [
            (c.name, c.donor, _zero_based(c.point_range),
             _zero_based(c.point_range_donor), (1, 2, 3))
            for c in spec.connections]
    seam = read[0].connections[1].periodic
    np.testing.assert_array_equal(seam.translation, [0.0, 0.0, 3.0])
    np.testing.assert_array_equal(seam.rotation_angle, [0.0, 0.0, 0.0])
    assert seam.rotation_center is None
    assert read[0].connections[0].periodic is None
