"""生成 Step 4 单块 HDF5/CGNS fixture。

用法（仓库根目录）：:

    python Grid/make_cgns_fixture.py
    python Grid/make_cgns_fixture.py --ni 16 --nj 16 --nk 4

官方 pyCGNS 在本机 Python 3.13 上构建失败；这里按
``Grid/samples/multi_hdf5.cgns`` 的 label/节点布局用 h5py 写最小 HDF5-CGNS。
该文件用于仓库内 reader/adapter 契约测试，不宣称是 SIDS 校验器认可的产物。

多 zone 的文件由 ``build_multizone`` 写，供 ``Grid/split_cgns.py`` 和第六步的
测试调用。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "fixtures" / "poiseuille_box.cgns"


def _string_bytes(text: str) -> np.ndarray:
    """CGNS/HDF5 字符串按 int8 存。"""
    return np.frombuffer(text.encode("ascii"), dtype=np.int8)


def _group(parent, name, label, data=None, dtype=None, track_times=True,
           track_order=False):
    """建一个带 ``label`` 属性的组，可选写 `` data`` dataset。

    ``track_order=True`` 让这个组记住子节点的创建顺序，读回时按这个顺序遍历。
    """
    group = parent.create_group(name, track_order=track_order)
    group.attrs["label"] = np.bytes_(label)
    if data is not None:
        group.create_dataset(" data", data=np.asarray(data, dtype=dtype),
                             track_times=track_times)
    return group


def _text_group(parent, name, label, text, track_times=True):
    """建一个字符串节点。"""
    return _group(parent, name, label, _string_bytes(text), dtype=np.int8,
                  track_times=track_times)


def _range_group(parent, name, lo, hi, track_times=True):
    """建 ``IndexRange_t``；输入是 1-based 闭区间。"""
    return _group(parent, name, "IndexRange_t",
                  np.asarray([lo, hi], dtype=np.int32), dtype=np.int32,
                  track_times=track_times)


def build_fixture(path: Path, ni: int = 16, nj: int = 16, nk: int = 8) -> None:
    """写一个 i/k 周期、j 双壁的均匀长方体 CGNS。"""
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if value < 2:
            raise ValueError(f"{name} 必须 >= 2，得到 {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dx = 2.0 * math.pi / ni
    dy = 1.0 / nj
    dz = 2.0 * math.pi / ni

    with h5py.File(path, "w") as f:
        f.attrs["label"] = np.bytes_("Root Node of HDF5 File")
        _group(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t",
               np.asarray([3.21], dtype=np.float32), dtype=np.float32)
        base = _group(f, "Base", "CGNSBase_t",
                      np.asarray([3, 3], dtype=np.int32), dtype=np.int32)
        zone = _group(base, "Zone 1", "Zone_t",
                      np.asarray([[ni, nj, nk],
                                  [ni - 1, nj - 1, nk - 1],
                                  [0, 0, 0]], dtype=np.int32),
                      dtype=np.int32)
        _text_group(zone, "ZoneType", "ZoneType_t", "Structured")

        coords = _group(zone, "GridCoordinates", "GridCoordinates_t")
        i = np.arange(ni, dtype=np.float64)
        j = np.arange(nj, dtype=np.float64)
        k = np.arange(nk, dtype=np.float64)
        x = np.broadcast_to((i + 0.5) * dx, (nk, nj, ni))
        y = np.broadcast_to(((j + 0.5) * dy)[:, None], (nk, nj, ni))
        z = np.broadcast_to((k * dz)[:, None, None], (nk, nj, ni))
        for name, arr in (("CoordinateX", x), ("CoordinateY", y),
                          ("CoordinateZ", z)):
            _group(coords, name, "DataArray_t", arr, dtype=np.float64)

        bc = _group(zone, "ZoneBC", "ZoneBC_t")
        for name, bc_type, lo, hi in (
            ("Jlo", "BCWall", (1, 1, 1), (ni, 1, nk)),
            ("Jhi", "BCWall", (1, nj, 1), (ni, nj, nk)),
        ):
            node = _text_group(bc, name, "BC_t", bc_type)
            _range_group(node, "PointRange", lo, hi)

        conn = _group(zone, "ZoneGridConnectivity", "ZoneGridConnectivity_t")
        pairs = (
            ("1to1Ilo", (1, 1, 1), (1, nj, nk), (ni, 1, 1), (ni, nj, nk)),
            ("1to1Ihi", (ni, 1, 1), (ni, nj, nk), (1, 1, 1), (1, nj, nk)),
            ("1to1Klo", (1, 1, 1), (ni, nj, 1), (1, 1, nk), (ni, nj, nk)),
            ("1to1Khi", (1, 1, nk), (ni, nj, nk), (1, 1, 1), (ni, nj, 1)),
        )
        for name, lo, hi, dlo, dhi in pairs:
            node = _text_group(conn, name, "GridConnectivity1to1_t", "Zone 1")
            _group(node, "Transform", '"int[IndexDimension]"',
                   np.asarray([1, 2, 3], dtype=np.int32), dtype=np.int32)
            _range_group(node, "PointRange", lo, hi)
            _range_group(node, "PointRangeDonor", dlo, dhi)


@dataclass
class ConnSpec:
    """一条 ``GridConnectivity1to1_t`` 的写入内容；范围是 1-based 闭区间。"""

    name: str
    donor: str
    point_range: tuple
    point_range_donor: tuple
    transform: tuple | None = None    # None 时不写 Transform 节点
    periodic: dict | None = None      # 键取 RotationCenter/RotationAngle/Translation，
                                      # 没有的键不写；空字典写出不带数组的 Periodic_t


@dataclass
class ZoneSpec:
    """一个 Structured zone 的写入内容；``coords`` 三个数组的形状是 (nk, nj, ni)。"""

    name: str
    coords: dict
    bcs: tuple = ()                   # 每项 (名字, BCType, (lo, hi))
    connections: tuple = ()           # ConnSpec


def build_multizone(path: Path, zones) -> None:
    """写一个 Base、多个 Structured zone 的 CGNS。

    边条只写 BCType 本身，不写 Family；连接是 ``GridConnectivity1to1_t``，可以带
    ``Periodic_t``。数据集不记时间戳。各组节点按创建顺序遍历，读回的顺序与写入
    的顺序相同。
    """
    # 三个写入函数各绑一次“不记时间戳”，调用处不用逐个重复
    put = partial(_group, track_times=False)
    text = partial(_text_group, track_times=False)
    span = partial(_range_group, track_times=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["label"] = np.bytes_("Root Node of HDF5 File")
        put(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t",
            np.asarray([3.21], dtype=np.float32), dtype=np.float32)
        base = _group(f, "Base", "CGNSBase_t",
                      np.asarray([3, 3], dtype=np.int32), dtype=np.int32,
                      track_times=False, track_order=True)
        for spec in zones:
            ni, nj, nk = spec.coords["Axx"].shape[::-1]
            zone = put(base, spec.name, "Zone_t",
                       np.asarray([[ni, nj, nk], [ni - 1, nj - 1, nk - 1],
                                   [0, 0, 0]], dtype=np.int32), dtype=np.int32)
            text(zone, "ZoneType", "ZoneType_t", "Structured")
            coords = _group(zone, "GridCoordinates", "GridCoordinates_t")
            for name, key in (("CoordinateX", "Axx"), ("CoordinateY", "Ayy"),
                              ("CoordinateZ", "Azz")):
                put(coords, name, "DataArray_t", spec.coords[key], dtype=np.float64)
            if spec.bcs:
                bc = _group(zone, "ZoneBC", "ZoneBC_t", track_order=True)
                for name, bc_type, (lo, hi) in spec.bcs:
                    span(text(bc, name, "BC_t", bc_type), "PointRange", lo, hi)
            if spec.connections:
                conn = _group(zone, "ZoneGridConnectivity",
                              "ZoneGridConnectivity_t", track_order=True)
                for c in spec.connections:
                    node = text(conn, c.name, "GridConnectivity1to1_t", c.donor)
                    if c.transform is not None:
                        put(node, "Transform", '"int[IndexDimension]"',
                            np.asarray(c.transform, dtype=np.int32),
                            dtype=np.int32)
                    span(node, "PointRange", *c.point_range)
                    span(node, "PointRangeDonor", *c.point_range_donor)
                    if c.periodic is not None:
                        prop = _group(node, "GridConnectivityProperty",
                                      "GridConnectivityProperty_t")
                        per = _group(prop, "Periodic", "Periodic_t")
                        for name, value in c.periodic.items():
                            put(per, name, "DataArray_t", value, dtype=np.float64)


def main(argv=None):
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--ni", type=int, default=16)
    parser.add_argument("--nj", type=int, default=16)
    parser.add_argument("--nk", type=int, default=8)
    args = parser.parse_args(argv)
    build_fixture(args.out, ni=args.ni, nj=args.nj, nk=args.nk)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
