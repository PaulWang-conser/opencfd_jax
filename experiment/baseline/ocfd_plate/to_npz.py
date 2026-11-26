"""把 OCFD 原码输出的 OCFD%08d.dat 转成与 JAX 基线同构的 npz。

原码输出的是**原始变量** rho,U,V,W,T（内区、(nz,ny,nx)）；JAX 基线存的是
**守恒量** cons_0..4，形状 (nk,nj,ni)。本脚本做两件事：
  1) (nz,ny,nx)=(nj,nk,ni) → (nk,nj,ni) 的轴置换；
  2) 原始量 → 守恒量：cons = [rho, rho*u, rho*v, rho*w, rho*(Cv*T + 0.5|U|^2)]，
     Cv = 1/(gamma*(gamma-1)*Ama^2)，与 src/cases.py 的 PhysParams 同式。
"""
import argparse
import json
import os
import struct

import numpy as np


def read_field(path, ni, nj, nk, nvars=5):
    payload = 8 * ni * nj
    rec = 8 + payload
    buf = np.fromfile(path, dtype=np.uint8)
    head = buf[:20].copy().view("<i4")
    if head[0] != 12 or head[-1] != 12:
        raise ValueError("not an OCFD field file")
    istep = int(head[1])
    tt = float(buf[8:16].copy().view("<f8")[0])
    out = []
    for f in range(nvars):
        blk = buf[20 + f * nk * rec:20 + (f + 1) * nk * rec].reshape(nk, rec)
        marks = blk[:, :4].copy().view("<i4").ravel()
        if not np.all(marks == payload):
            raise ValueError(f"bad record marker in field {f}")
        out.append(blk[:, 4:4 + payload].copy().view("<f8").reshape(nk, nj, ni))
    return out, istep, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ni", type=int, default=64)
    ap.add_argument("--nj", type=int, default=64)
    ap.add_argument("--nk", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--ama", type=float, default=0.1)
    ap.add_argument("--re", type=float, default=1000.0)
    ap.add_argument("--ref-t", type=float, default=288.15)
    ap.add_argument("--pr", type=float, default=0.7)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--scheme", default="WENO7")
    ap.add_argument("--vis-order", default="CD8")
    args = ap.parse_args()

    # 原码 (nx,ny,nz) = (JAX ni, JAX nk, JAX nj)；返回 (nz,ny,nx) = (nj,nk,ni)
    fields, istep, tt = read_field(args.file, args.ni, args.nk, args.nj)
    rho, u, v, w, T = fields
    Cv = 1.0 / (args.gamma * (args.gamma - 1.0) * args.ama * args.ama)
    k2 = 0.5 * (u * u + v * v + w * w)
    # 分量对应：JAX (u,v,w)=(流向, 壁法向, 展向)；原码 (U,V,W)=(流向, 展向, 壁法向)
    # 所以 JAX 的 v <- 原码 W，JAX 的 w <- 原码 V。
    cons = [rho, rho * u, rho * w, rho * v, rho * (Cv * T + k2)]
    # (nj,nk,ni) → (nk,nj,ni)
    cons = [np.transpose(c, (1, 0, 2)) for c in cons]

    meta = dict(t=tt, step=istep, dt=args.dt, case="plate",
                scheme=args.scheme, gamma=args.gamma, Re=args.re,
                Ama=args.ama, Ref_T=args.ref_t, Pr=args.pr,
                N=args.ni, source="OpenCFD-SCU", Cv=Cv)
    np.savez(args.out, _meta=json.dumps(meta),
             **{f"cons_{i}": np.asarray(c, dtype=np.float64) for i, c in enumerate(cons)})
    for i, c in enumerate(cons):
        print(f"cons_{i}: min={c.min():.17g} max={c.max():.17g} mean={c.mean():.17g}")


if __name__ == "__main__":
    main()
