"""复制一份 CGNS/HDF5 网格，只把指定边条的 ``BCType`` 换成给定值，其余原样保留。

用来从已有网格派生"只改一条边条"的变体，并让变体可以复建。无反射对照用的
``testcgnstest_ystr_nr.cgns``（出口边条 ``dom-5`` 改为 ``BCOutflowSupersonic``）
就是这样派生的；派生文件不入库，来源由"源文件 + 本脚本 + 参数 + 产物的逻辑内容哈希"确定。

用法（任意目录，缺省路径相对本脚本所在目录）：

    python Grid/override_cgns_bc.py
    python Grid/override_cgns_bc.py --bc dom-5 --type BCOutflowSupersonic

缺省参数即无反射对照实验所用的那一组。结束时打印源文件的 SHA-256 与产物的逻辑
内容哈希。产物的文件字节哈希不能用：改一个节点时 HDF5 会把父组的修改时间写进
对象头，隔一秒再生成字节就不同，而读取器消费的是逻辑内容。
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


def logical_digest(path: Path) -> str:
    """CGNS 节点树的逻辑内容哈希。

    按名字排序遍历全部节点，依次喂入节点名、属性、数据集的类型、形状和内容。
    不含 HDF5 对象头里的时间戳，所以同一份逻辑内容总是得到同一个哈希。
    """
    h = hashlib.sha256()

    def feed(node, prefix):
        for key in sorted(node):
            child = node[key]
            name = f"{prefix}/{key.strip()}"
            h.update(name.encode("utf-8"))
            for k, v in sorted(child.attrs.items()):
                h.update(f"{name}.{k}={v!r}".encode("utf-8"))
            if isinstance(child, h5py.Dataset):
                h.update(f"{name}:{child.dtype.str}:{child.shape}:".encode("utf-8"))
                h.update(np.ascontiguousarray(child[()]).tobytes())
            else:
                feed(child, name)

    with h5py.File(path, "r") as f:
        feed(f, "")
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
    print(f"  产物逻辑内容 SHA-256 {logical_digest(out)}")


if __name__ == "__main__":
    main()
