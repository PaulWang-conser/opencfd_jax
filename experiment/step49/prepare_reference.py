"""为单块 CGNS plate 生成 OCFD IBC=124 单步参考输入。"""
import argparse
import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from opencfd_jax.src.mesh_cgns import read_cgns_zone, to_block_mesh  # noqa: E402


def write_field(stream, values):
    for plane in values:
        payload = np.ascontiguousarray(plane, dtype="<f8").tobytes()
        stream.write(struct.pack("<i", len(payload)))
        stream.write(payload)
        stream.write(struct.pack("<i", len(payload)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default=str(ROOT / "opencfd_jax/Grid/testcgnstest.cgns"))
    parser.add_argument("--output", default=None, help="输出目录；不指定时创建唯一目录")
    args = parser.parse_args()
    source = Path(args.mesh)
    zone = read_cgns_zone(source)
    kinds = tuple(face.kind for face in to_block_mesh(zone).faces)
    if kinds != ("inflow", "outflow", "wall", "farfield", "periodic", "periodic"):
        raise ValueError(f"参考输入仅支持 i 入口/出口、j 壁面/远场、k 周期，得到 {kinds}")
    if args.output is None:
        output = Path(tempfile.mkdtemp(prefix="step49_generated_reference_",
                                       dir=ROOT / "opencfd_jax/experiment"))
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
    # 原版 IBC=124 的壁面在第三计算方向；坐标与物理分量同时交换。
    coords = [np.transpose(zone.coords[k], (1, 0, 2)) for k in ("Axx", "Azz", "Ayy")]
    with (output / "OCFD3d-Mesh.dat").open("wb") as f:
        for coord in coords:
            write_field(f, coord)
    ones = np.ones((zone.nj, zone.nk, zone.ni), dtype="<f8")
    zeros = np.zeros_like(ones)
    with (output / "opencfd.dat").open("wb") as f:
        f.write(struct.pack("<iidi", 12, 0, 0., 12))
        for value in (ones, ones, zeros, zeros, ones):
            write_field(f, value)
    text = f"""# Step 4.9：原版 IBC=124、均匀单位来流，原版 y 对应 JAX k。
GRID_3D = {zone.ni} {zone.nk} {zone.nj}
PARALLEL_3D = 1 1 1
STREAM = 0
CHARTERIC = 0
TEST = 0
IPERIODIC = 0 1 0
JAC_BOUND = 1 1 1
DIF_BOUND = 0 0 0 0 1 0
NON_REFLETION = 0 0 0 0 0 0
SCHEME_INVIS = WENO7
SCHEME_VIS = CD8
RE = 1000
AMA = 0.1
GAMMA = 1.4
PR = 0.7
T_REF = 288.15
EPSL_SW = 0.0
DT = 0.0001
END_TIME = 0.0001
KSTEP_SHOW = 1
KSTEP_SAVE = 1
INIT_STAT = 1
IBC = 124
BC_NPARA = 0 0 0 0 5
BC_RPARA = 0 1 0 0 0 0 0
"""
    (output / "opencfd-scu.in").write_text(text, encoding="utf8", newline="\n")
    paths = (source, output / "OCFD3d-Mesh.dat", output / "opencfd.dat", output / "opencfd-scu.in")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output / "inputs.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf8")
    print(f"参考输入已生成：{output}")


if __name__ == "__main__":
    main()
