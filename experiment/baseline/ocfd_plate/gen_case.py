"""生成 OCFD 侧 plate_1step 准例的输入文件（网格 + 初值 + .in）。

口径与 experiment/baseline/plate_1step 完全对齐（JAX 侧 argv 见 manifest.json）：
  --case plate --N 64 --ny 64 --nz 4  →  ni=64, nj=64, nk=4
  x0=0.05, x1=0.25 (main.py:109-110), H=0.12 (main.py:111)
  dx=(x1-x0)/ni, xc=x0+(i+0.5)dx
  dy=H/(nj-1),   yc=j*dy            （壁面在 j=0 的格心）
  k 方向周期，间距 dx、nk=4 点      （cases.py: dxs=(dx,dy,dx)）
  Re=1000, Ama=0.1, Ref_T=288.15, Pr=0.7, gamma=1.4, dt=1e-4
  初值 rho=u=T=1, v=w=0

OCFD 索引约定：nx=ni(流向)、ny=nk(展向, 周期)、nz=nj(壁法向)。
所以 GRID_3D = 64 4 64；Ayy 是展向、Azz 是壁法向。

写出：
  OCFD3d-Mesh.dat  —— 3 个场 X/Y/Z，每场 nz 条记录（int32 + ni*nj f8 + int32）
  opencfd.dat      —— 20 字节头 + 5 个场 rho,U,V,W,T（同上记录格式）
  opencfd-scu.in   —— IBC=124 / DIF_BOUND 只开 z- / IPERIODIC 只开 y
"""
import argparse
import math
import os
import struct

import numpy as np

# ---------------- 与 JAX plate_1step 对齐的常数 ----------------
NI = 64          # 流向 i
NJ = 64          # 壁法向 j
NK = 4           # 展向 k（周期）
X0 = 0.05
X1 = 0.25
H = 0.12
RE = 1000.0
AMA = 0.1
T_REF = 288.15
PR = 0.7
GAMMA = 1.4
DT = 1e-4
END_TIME = 1e-4
KSTEP = 1


def write_field(fh, arr):
    """arr: (nz, ny, nx) float64；每个 k 面一条记录。"""
    nz, ny, nx = arr.shape
    size = nx * ny * 8
    for k in range(nz):
        plane = np.ascontiguousarray(arr[k], dtype="<f8")
        fh.write(struct.pack("<i", size))
        fh.write(plane.tobytes())
        fh.write(struct.pack("<i", size))


def build_coords():
    dx = (X1 - X0) / NI
    dy = H / (NJ - 1)
    dz = dx                      # 展向间距取 dx（cases.py 的 dxs=(dx,dy,dx)）
    x = X0 + (np.arange(NI) + 0.5) * dx
    y = np.arange(NK) * dz
    z = np.arange(NJ) * dy
    nx, ny, nz = NI, NK, NJ
    X = np.broadcast_to(x[None, None, :], (nz, ny, nx)).copy()
    Y = np.broadcast_to(y[None, :, None], (nz, ny, nx)).copy()
    Z = np.broadcast_to(z[:, None, None], (nz, ny, nx)).copy()
    return X, Y, Z, dict(dx=dx, dy=dy, dz=dz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)

    X, Y, Z, h = build_coords()
    nx, ny, nz = NI, NK, NJ

    # ---- 网格 ----
    with open(os.path.join(out, "OCFD3d-Mesh.dat"), "wb") as fh:
        for arr in (X, Y, Z):
            write_field(fh, arr)

    # ---- 初值（原始变量）----
    ones = np.ones((nz, ny, nx))
    zeros = np.zeros((nz, ny, nx))
    with open(os.path.join(out, "opencfd.dat"), "wb") as fh:
        fh.write(struct.pack("<i", 12))
        fh.write(struct.pack("<i", 0))
        fh.write(struct.pack("<d", 0.0))
        fh.write(struct.pack("<i", 12))
        for arr in (ones.copy(), ones.copy(), zeros.copy(), zeros.copy(), ones.copy()):
            write_field(fh, arr)

    # ---- .in ----
    txt = f"""# OpenCFD-SCU 输入 —— plate_1step 准例（与 experiment/baseline/plate_1step 同网格同边条）
# 生成者：experiment/baseline/ocfd_plate/gen_case.py
# OCFD 索引：x=流向(JAX i, 64)  y=展向周期(JAX k, 4)  z=壁法向(JAX j, 64)
GRID_3D = {nx}   {ny}   {nz}
PARALLEL_3D = 1   1   1
STREAM = 0
CHARTERIC = 0
TEST = 0

# x 非周期、y 展向周期、z 非周期
IPERIODIC = 0   1   0
JAC_BOUND = 1   1   1
# DIF_BOUND = x- x+ y- y+ z- z+ ；只有 z-（壁面）开粘性带
DIF_BOUND = 0   0   0   0   1   0
NON_REFLETION = 0   0   0   0   0   0

SCHEME_INVIS = WENO7
SCHEME_VIS = CD8

RE = {RE}
AMA = {AMA}
GAMMA = {GAMMA}
PR = {PR}
T_REF = {T_REF}
EPSL_SW = 0.0

DT = {DT}
END_TIME = {END_TIME}
KSTEP_SHOW = 1
KSTEP_SAVE = {KSTEP}
INIT_STAT = 1

IBC = 124
# IF_SYMMETRY  IF_WITHLEADING  IFLAG_UPPERBOUNDARY  MZMAX  MTMAX
# MZMAX 必须为 0：非 0 分支不填 pu_dist_upper（未初始化显存），顶部 u 会变成
# 1 + 垃圾；为 0 时 u_dist = EPSL_UPPER*rand = 0，顶部退化为常数 (1,1,0,0,1)。
BC_NPARA = 0   0   0   0   5
# AoA  Tw  EPSL_WALL  EPSL_UPPER  BETA  WALL_DIS_BEGIN  WALL_DIS_END
BC_RPARA = 0.0   1.0   0.0   0.0   0.0   0.0   0.0
"""
    with open(os.path.join(out, "opencfd-scu.in"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)

    nx_, ny_, nz_ = NI, NK, NJ
    n = nx_ * ny_
    print(f"grid  nx={nx_} ny={ny_} nz={nz_}")
    print(f"dx={h['dx']:.10g} dy={h['dy']:.10g} dz={h['dz']:.10g}")
    print(f"x=[{X[0,0,0]:.10g},{X[0,0,-1]:.10g}] y=[{Y[0,0,0]:.10g},{Y[0,-1,0]:.10g}] "
          f"z=[{Z[0,0,0]:.10g},{Z[-1,0,0]:.10g}]")
    for name in ("OCFD3d-Mesh.dat", "opencfd.dat"):
        p = os.path.join(out, name)
        print(f"{name}: {os.path.getsize(p)} B")
    print(f"expect mesh {3*nz_*(8+n*8)} B, init {20+5*nz_*(8+n*8)} B")


if __name__ == "__main__":
    main()
