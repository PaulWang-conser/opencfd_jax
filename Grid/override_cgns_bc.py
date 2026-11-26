"""复制一份 CGNS/HDF5 网格，只把指定边条的 ``BCType`` 换成给定值，其余原样保留。

用来从已有网格派生"只改一条边条"的变体，并让变体可以复建。无反射对照用的
``testcgnstest_ystr_nr.cgns``（出口边条 ``dom-5`` 改为 ``BCOutflowSupersonic``）
就是这样派生的；派生文件不入库。

来源靠四样入库的东西确定，不给产物记哈希：源网格、本脚本、配置
（``experiment/cases/e1_outlet_nonreflecting.json``）、按此重跑与归档快照逐位
相同。派生网格是否正确由读取器核对：坐标与源网格相同，六侧类型符合预期。
（产物的文件字节哈希不稳定：改一个节点时 HDF5 对象头里的时间字段会随生成
时间改变，所以不用它。）

用法（任意目录，缺省路径相对本脚本所在目录）：

    python Grid/override_cgns_bc.py
    python Grid/override_cgns_bc.py --bc dom-5 --type BCOutflowSupersonic

缺省参数即无反射对照实验所用的那一组。结束时打印源文件的 SHA-256。
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


def _find_bc(root, bc: str) -> list:
    """全文件里名为 ``bc`` 的 ``BC_t`` 节点。"""
    return [zone["ZoneBC"][bc]
            for base in root.values() if isinstance(base, h5py.Group)
            for zone in base.values()
            if isinstance(zone, h5py.Group) and _label(zone) == "Zone_t"
            and "ZoneBC" in zone and bc in zone["ZoneBC"]]


def override(src: Path, out: Path, bc: str, bc_type: str) -> str:
    """复制 ``src`` 到 ``out``，把名为 ``bc`` 的 ``BC_t`` 节点的类型改成 ``bc_type``。

    直接改写节点自身的类型。节点原为 ``FamilySpecified`` 时，遗留的 ``FamilyName``
    子节点不再被读取器使用（读取器以节点自身的类型为准）。返回原来的类型。

    先在源文件里确认边条存在、新类型可写，然后才动输出文件；写的时候先写
    同目录的临时文件，成功后才替换。所以任何一步失败都不会破坏 ``out`` 里
    已有的产物。
    """
    if src.resolve() == out.resolve():
        raise SystemExit("--out 不能与 --src 相同：派生网格不得覆盖源文件")
    try:
        _string_bytes(bc_type)
    except UnicodeEncodeError:
        raise SystemExit(f"新的 BCType 只能用 ASCII 字符：{bc_type!r}") from None
    with h5py.File(src, "r") as f:
        nodes = _find_bc(f, bc)
        if len(nodes) != 1:
            raise SystemExit(f"名为 {bc!r} 的 BC_t 节点应恰有 1 个，找到 {len(nodes)} 个")
        old = np.asarray(nodes[0][" data"][()]).tobytes().decode("ascii")
    tmp = out.with_name(out.name + ".tmp")
    try:
        shutil.copyfile(src, tmp)
        with h5py.File(tmp, "r+") as f:
            node = _find_bc(f, bc)[0]
            del node[" data"]
            node.create_dataset(" data", data=_string_bytes(bc_type), track_times=False)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return old


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
    print(f"  源文件 SHA-256 {hashlib.sha256(src.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
