"""诊断用：生成几个变体网格，定位 OCFD 报 Negative Jac 的原因。

变体：
  base    —— 与 gen_case.py 完全一致
  znorm   —— z 归一化到 [0,1]（dz=1/(nz-1)），只改坐标，不改 .in
  ny8     —— 展向 8 点（跨度 8*dz），其余同 base
  xnorm   —— x/y/z 全部归一化到 [0,1]
"""
import os
import struct
import sys

import numpy as np

NI, NJ, NK = 64, 64, 4
X0, X1, H = 0.05, 0.25, 0.12


def write_field(fh, arr):
    nz, ny, nx = arr.shape
    size = nx * ny * 8
    for k in range(nz):
        plane = np.ascontiguousarray(arr[k], dtype="<f8")
        fh.write(struct.pack("<i", size))
        fh.write(plane.tobytes())
        fh.write(struct.pack("<i", size))


def build(ni, nj, nk, x, y, z):
    X = np.broadcast_to(x[None, None, :], (nj, nk, ni)).copy()
    Y = np.broadcast_to(y[None, :, None], (nj, nk, ni)).copy()
    Z = np.broadcast_to(z[:, None, None], (nj, nk, ni)).copy()
    return X, Y, Z


def emit(outdir, ni, nj, nk, x, y, z, init0=True):
    os.makedirs(outdir, exist_ok=True)
    X, Y, Z = build(ni, nj, nk, x, y, z)
    with open(os.path.join(outdir, "OCFD3d-Mesh.dat"), "wb") as fh:
        for arr in (X, Y, Z):
            write_field(fh, arr)
    ones = np.ones((nj, nk, ni))
    zeros = np.zeros((nj, nk, ni))
    with open(os.path.join(outdir, "opencfd.dat"), "wb") as fh:
        fh.write(struct.pack("<i", 12))
        fh.write(struct.pack("<i", 0))
        fh.write(struct.pack("<d", 0.0))
        fh.write(struct.pack("<i", 12))
        for arr in (ones.copy(), ones.copy(), zeros.copy(), zeros.copy(), ones.copy()):
            write_field(fh, arr)
    tmpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "opencfd-scu.in"), encoding="utf-8").read()
    txt = (tmpl.replace("GRID_3D = 64   4   64", f"GRID_3D = {ni}   {nk}   {nj}")
               .replace("INIT_STAT = 1", "INIT_STAT = %d" % (0 if init0 else 1)))
    with open(os.path.join(outdir, "opencfd-scu.in"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"{outdir}: ni={ni} nk={nk} nj={nj} "
          f"x=[{x[0]:.4g},{x[-1]:.4g}] y=[{y[0]:.4g},{y[-1]:.4g}] z=[{z[0]:.4g},{z[-1]:.4g}]")


here = os.path.dirname(os.path.abspath(__file__))
dx = (X1 - X0) / NI
dy = H / (NJ - 1)
dz = dx

xc = X0 + (np.arange(NI) + 0.5) * dx
yc = np.arange(NK) * dz
zc = np.arange(NJ) * dy

which = sys.argv[1]
if which == "znorm":
    emit(os.path.join(here, "_diag_znorm"), NI, NJ, NK, xc, yc, np.arange(NJ) / (NJ - 1))
elif which == "ny8":
    emit(os.path.join(here, "_diag_ny8"), NI, NJ, 8,
         xc, np.arange(8) * dz, zc, init0=False)
elif which == "base":
    emit(os.path.join(here, "_diag_base"), NI, NJ, NK, xc, yc, zc)
elif which == "znorm_shift":
    emit(os.path.join(here, "_diag_zs"), NI, NJ, NK,
         xc, yc, 0.5 + 0.2 * (np.arange(NJ) / (NJ - 1)))
elif which == "xyz01":
    emit(os.path.join(here, "_diag_xyz01"), NI, NJ, NK,
         np.arange(NI) / (NI - 1), np.arange(NK) / (NK - 1),
         np.arange(NJ) / (NJ - 1), init0=False)
elif which == "phys4":
    s = 4.0
    emit(os.path.join(here, "_diag_phys4"), NI, NJ, NK,
         s * xc, s * yc, s * zc, init0=False)
else:
    raise SystemExit("unknown variant")
