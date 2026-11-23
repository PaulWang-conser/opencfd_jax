"""复制一份 CGNS/HDF5 网格，只把指定边条的 ``BCType`` 换成给定值，其余原样保留。

用来从已有网格派生"只改一条边条"的变体，并让变体可以复建。无反射对照用的
``testcgnstest_ystr_nr.cgns``（出口边条 ``dom-5`` 改为 ``BCOutflowSupersonic``）
就是这样派生的；派生文件不入库，来源由"源文件 + 本脚本 + 参数 + 产物的逻辑内容哈希"确定。

用法（任意目录，缺省路径相对本脚本所在目录）：

    python Grid/override_cgns_bc.py
    python Grid/override_cgns_bc.py --bc dom-5 --type BCOutflowSupersonic

缺省参数即无反射对照实验所用的那一组。结束时打印源文件的 SHA-256 与产物的逻辑
内容哈希。产物的文件字节哈希不能用：改一个节点时 HDF5 对象头里的时间字段会随
生成时间改变，隔一秒再生成字节就不同，而读取器消费的是逻辑内容。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def _string_bytes(text: str) -> np.ndarray:
    """CGNS/HDF5 字符串按 int8 存。"""
    return np.frombuffer(text.encode("ascii"), dtype=np.int8)


def _label(node) -> str | None:
    v = node.attrs.get("label")
    return v.decode() if isinstance(v, bytes) else v


def override(src: Path, out: Path, bc: str, bc_type: str) -> str:
    """复制 ``src`` 到 ``out``，把名为 ``bc`` 的 ``BC_t`` 节点的类型改成 ``bc_type``。

    直接改写节点自身的类型。节点原为 ``FamilySpecified`` 时，遗留的 ``FamilyName``
    子节点不再被读取器使用（读取器以节点自身的类型为准）。返回原来的类型。
    """
    if src.resolve() == out.resolve():
        raise SystemExit("--out 不能与 --src 相同：派生网格不得覆盖源文件")
    shutil.copyfile(src, out)
    with h5py.File(out, "r+") as f:
        nodes = [zone["ZoneBC"][bc]
                 for base in f.values() if isinstance(base, h5py.Group)
                 for zone in base.values()
                 if isinstance(zone, h5py.Group) and _label(zone) == "Zone_t"
                 and "ZoneBC" in zone and bc in zone["ZoneBC"]]
        if len(nodes) != 1:
            raise SystemExit(f"名为 {bc!r} 的 BC_t 节点应恰有 1 个，找到 {len(nodes)} 个")
        node = nodes[0]
        old = np.asarray(node[" data"][()]).tobytes().decode("ascii")
        del node[" data"]
        node.create_dataset(" data", data=_string_bytes(bc_type), track_times=False)
    return old


def _feed(h, *chunks):
    """按长度前缀依次喂入；字段之间因此没有歧义边界。"""
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "big"))
        h.update(chunk)


def _feed_value(h, value):
    """喂入一个属性值或数据集内容：类型、形状和内容，不经过打印表示。"""
    a = np.asarray(value)
    if a.dtype.kind == "O":                      # 逐元素编码，不含对象指针
        _feed(h, b"O", repr(a.shape).encode("ascii"))
        for x in a.ravel().tolist():
            if isinstance(x, bytes):
                _feed(h, b"b", x)
            elif isinstance(x, str):
                _feed(h, b"s", x.encode("utf-8"))
            else:
                raise TypeError(f"逻辑内容哈希不支持的元素类型 {type(x).__name__}")
    elif a.dtype.kind == "U":                    # 定长字符串：按 UTF-8，与字节序无关
        _feed(h, b"U", repr(a.shape).encode("ascii"))
        for x in a.ravel().tolist():
            _feed(h, x.encode("utf-8"))
    elif a.dtype.kind in "biufcS":               # 数值与定长字节串
        _feed(h, a.dtype.str.encode("ascii"), repr(a.shape).encode("ascii"),
              np.ascontiguousarray(a).tobytes())
    else:
        raise TypeError(f"逻辑内容哈希不支持 dtype={a.dtype.str}")


def logical_digest(path: Path) -> str:
    """CGNS 节点树的逻辑内容哈希（算法 v2）。

    从根节点起按原始名字排序遍历，喂入节点种类、名字、全部属性（含根的）、
    数据集的类型、形状和内容。字段带长度前缀，值按类型和原始字节编码。不含
    HDF5 对象头里的时间字段，所以同一份逻辑内容总是得到同一个哈希。
    """
    h = hashlib.sha256(b"cgns-logical-digest-v2")

    def feed(name, node):
        h.update(b"G" if isinstance(node, h5py.Group) else b"D")
        _feed(h, name.encode("utf-8"))
        for k in sorted(node.attrs):
            _feed(h, k.encode("utf-8"))
            _feed_value(h, node.attrs[k])
        if isinstance(node, h5py.Dataset):
            _feed_value(h, node[()])
        else:
            for key in sorted(node):
                feed(key, node[key])

    with h5py.File(path, "r") as f:
        feed("", f)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="只改一条边条的 BCType，派生新的 CGNS/HDF5 网格")
    ap.add_argument("--src", default=str(HERE / "testcgnstest_ystr.cgns"))
    ap.add_argument("--out", default=str(HERE / "testcgnstest_ystr_nr.cgns"))
    ap.add_argument("--bc", default="dom-5", help="要改的 BC_t 节点名（ZoneBC 下，原样匹配）")
    ap.add_argument("--type", default="BCOutflowSupersonic", help="新的 BCType")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    old = override(src, out, args.bc, args.type)
    print(f"{out.name}: {args.bc} 的 BCType {old!r} -> {args.type!r}")
    print(f"  源文件     SHA-256 {hashlib.sha256(src.read_bytes()).hexdigest()}")
    print(f"  产物逻辑内容 SHA-256 {logical_digest(out)}（算法 v2）")


if __name__ == "__main__":
    main()
