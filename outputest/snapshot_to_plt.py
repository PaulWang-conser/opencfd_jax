#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三维快照 (.npz: 5 个守恒量) -> Tecplot 二进制 .plt（v112）

自包含实现：不依赖 Tecplot 程序（不用 preplot / tecplot.exe），也不依赖其它脚本。

守恒量 cons_0..4 = rho, rho*u, rho*v, rho*w, rhoE，数组形状 (nz, ny, nx)，x 变化最快。
派生量：
    u,v,w = rho{u,v,w}/rho
    p     = (gamma-1) * (rhoE - 0.5*rho*(u²+v²+w²))
    T     = p * gamma * Mr² / rho
    Ma    = Mr * |V| / sqrt(T)
坐标：优先读 OCFD3d-Mesh.dat（record = int32 标记 + nx*ny 个值 + int32 标记，x 最快），
      找不到时按 case iiA 解析网格重建（Lx=4π, Lz=4π/3, tanh 拉伸 β=1.7）。

二进制结构（Tecplot Data 格式说明文档第 2 章，v112）：
    HEADER: "#!TDV112" | INT32 1 | INT32 FileType | TITLE | INT32 NumVars | 变量名
            | 每个 zone: FLOAT32 299.0 + 名字 + ParentZone(-1) + StrandID + SolutionTime
              + 颜色(-1) + ZoneType(0=ORDERED) + SpecifyVarLocation(0) + 2 个邻接标志
              + IMax/JMax/KMax + AUXDATA 对 ... + INT32 0
            | FLOAT32 357.0 (EOHMARKER)
    DATA  : 每个 zone: FLOAT32 299.0 + INT32*NV 数据类型 + passive 标志 + sharing 标志
            + 共享 connectivity(-1) + FLOAT64*2*NV min/max + 数据（BLOCK，i 最快）
注意：字符串是「逐字符 INT32 + 0 结尾」，不写长度计数字段；ORDERED zone 不写 NumElements。

用法：
    python snapshot_to_plt.py channel_case_iiA\\jax_from_initial_60000\\channel_step038000.npz
    python snapshot_to_plt.py xxx.npz --report-only        # 只看统计，不写文件
    python snapshot_to_plt.py xxx.npz --precision float32  # 体积减半
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

GAMMA = 1.40
M_R = 0.0955
PRANDTL = 0.70
RE_TAU = 190.7
TW = 1.0
BETA = 1.7
LY = 2.0

VERSION = 112
ZONE_MARKER = 299.0
EOH_MARKER = 357.0


# ------------------------------------------------------------------ 写 v112
def _w_i32(buf, v):
    buf.extend(struct.pack("<i", int(v)))


def _w_f32(buf, v):
    buf.extend(struct.pack("<f", float(v)))


def _w_f64(buf, v):
    buf.extend(struct.pack("<d", float(v)))


def _w_str(buf, s):
    buf.extend(np.array([ord(c) for c in str(s)] + [0], dtype="<i4").tobytes())


def write_plt(title, variables, zones, path, precision="float64"):
    """zones: [{name, i, j, k, solution_time, strand, aux[(name,value)], data}]，
    data 为 (nvar, i*j*k) 的 i-最快 展平数组（或可 reshape 的等长一维数组）。"""
    nvar = len(variables)
    fmt = 1 if precision == "float32" else 2
    np_dt = np.dtype("<f4" if fmt == 1 else "<f8")

    buf = bytearray()
    t0 = time.time()

    # HEADER
    buf.extend(("#!TDV%03d" % VERSION).encode("ascii"))
    _w_i32(buf, 1)                  # 字节序标志
    _w_i32(buf, 0)                  # FileType = FULL
    _w_str(buf, title)
    _w_i32(buf, nvar)
    for v in variables:
        _w_str(buf, v)

    for z in zones:
        npts = z["i"] * z["j"] * z["k"]
        _w_f32(buf, ZONE_MARKER)
        _w_str(buf, z["name"])
        _w_i32(buf, -1)                      # ParentZone（已废弃）
        _w_i32(buf, z.get("strand", -1))      # StrandID，-1 = static
        _w_f64(buf, z.get("solution_time", 0.0))
        _w_i32(buf, -1)                      # 默认颜色（已废弃）
        _w_i32(buf, 0)                       # ZoneType = ORDERED
        _w_i32(buf, 0)                       # SpecifyVarLocation = 0（全 NODAL）
        _w_i32(buf, 0)                       # raw 1-to-1 face neighbors
        _w_i32(buf, 0)                       # 额外 user-defined 邻接条数
        _w_i32(buf, z["i"])
        _w_i32(buf, z["j"])
        _w_i32(buf, z["k"])
        for nm, val in z.get("aux", []):
            _w_i32(buf, 1)
            _w_str(buf, nm)
            _w_i32(buf, 0)                   # 值格式：0 = 字符串
            _w_str(buf, val)
        _w_i32(buf, 0)                       # aux 结束
        z["_npts"] = npts

    _w_f32(buf, EOH_MARKER)

    # DATA
    for z in zones:
        npts = z["_npts"]
        a = np.array(z["data"], dtype=np_dt, order="C", copy=True).reshape(nvar, npts)
        _w_f32(buf, ZONE_MARKER)
        for _ in range(nvar):
            _w_i32(buf, fmt)
        _w_i32(buf, 0)                       # 无 passive 变量
        _w_i32(buf, 0)                       # 无变量共享
        _w_i32(buf, -1)                      # 不共享 connectivity
        for k in range(nvar):
            col = a[k].astype(np.float64)
            _w_f64(buf, col.min())
            _w_f64(buf, col.max())
        buf.extend(a.tobytes(order="C"))

    with open(path, "wb") as fh:
        fh.write(buf)
    return {"bytes": len(buf), "precision": precision, "seconds": time.time() - t0}


# ------------------------------------------------------------------ 读回自检
class _R:
    def __init__(self, data):
        self.d, self.p = data, 8

    def i32(self):
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def f64(self):
        v = struct.unpack_from("<d", self.d, self.p)[0]
        self.p += 8
        return v

    def s(self):
        out = []
        while True:
            c = self.i32()
            if c == 0:
                return "".join(out)
            out.append(chr(c))

    def peek(self):
        return struct.unpack_from("<f", self.d, self.p)[0]


def read_plt(path):
    """只解析本脚本写出的那种结构（ORDERED、无共享、无 geometry/text）。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"#!TDV112":
        raise ValueError("魔数不是 #!TDV112：%r" % data[:8])
    r = _R(data)
    if r.i32() != 1:
        raise ValueError("字节序标志不是 1")
    filetype = r.i32()
    title = r.s()
    nvar = r.i32()
    variables = [r.s() for _ in range(nvar)]

    zones = []
    while r.peek() == ZONE_MARKER:
        r.f32()
        z = {"name": r.s(), "parent": r.i32(), "strand": r.i32(),
             "solution_time": r.f64(), "color": r.i32(), "zonetype": r.i32(),
             "specify_loc": r.i32()}
        if z["specify_loc"] == 1:
            z["varloc"] = [r.i32() for _ in range(nvar)]
        z["raw_neighbors"] = r.i32()
        if r.i32() != 0:                      # 额外邻接条数
            raise NotImplementedError("未实现 misc face neighbor")
        if z["zonetype"] != 0:
            raise NotImplementedError("只实现了 ORDERED zone")
        z["i"], z["j"], z["k"] = r.i32(), r.i32(), r.i32()
        z["aux"] = []
        while r.i32() == 1:
            nm = r.s()
            assert r.i32() == 0
            z["aux"].append((nm, r.s()))
        zones.append(z)

    if r.peek() == 799.0:                     # dataset aux（本脚本不写，容错跳过）
        r.f32()
        raise NotImplementedError("dataset aux 未实现")
    if abs(r.peek() - EOH_MARKER) < 1e-6:
        r.f32()
    else:
        raise ValueError("头段末尾没有 EOHMARKER(357.0)，读到 %.3f" % r.peek())

    for z in zones:
        if abs(r.f32() - ZONE_MARKER) > 1e-6:
            raise ValueError("data 段 zone marker 不对")
        fmts = [r.i32() for _ in range(nvar)]
        if r.i32() != 0:
            raise NotImplementedError("passive 变量未实现")
        if r.i32() != 0:
            raise NotImplementedError("变量共享未实现")
        z["connectivity_shared"] = r.i32()
        z["minmax"] = [(r.f64(), r.f64()) for _ in range(nvar)]
        npts = z["i"] * z["j"] * z["k"]
        dt = np.dtype("<f4" if fmts[0] == 1 else "<f8")
        arr = np.empty((nvar, npts), dtype=dt)
        for v in range(nvar):
            arr[v] = np.frombuffer(data, dtype=dt, count=npts, offset=r.p)
            r.p += npts * dt.itemsize
        z["data"] = arr
        z["formats"] = fmts

    return {"magic": data[:8], "filetype": filetype, "title": title,
            "variables": variables, "zones": zones,
            "file_bytes": len(data), "trailing": len(data) - r.p}


# ------------------------------------------------------------------ 快照/网格
def load_snapshot(path):
    z = np.load(path, allow_pickle=True)
    keys = [k for k in z.files if k.startswith("cons_")]
    if len(keys) != 5:
        raise ValueError("期望 cons_0..4 共 5 个守恒量，实际 %s" % z.files)
    cons = np.stack([z[k] for k in keys], axis=0)          # (5, nz, ny, nx)
    meta = {}
    if "_meta" in z.files:
        try:
            meta = json.loads(z["_meta"].tobytes().decode("utf-8"))
        except Exception:
            pass
    return cons, meta


def read_mesh_records(path, nx, ny, nz):
    """OCFD3d-Mesh.dat -> (x,y,z)，每个 (nz,ny,nx)，x 变化最快"""
    n = nx * ny
    size = os.path.getsize(path)
    dsz = 8 if size == 3 * nz * (8 + 8 * n) else 4
    if size != 3 * nz * (8 + dsz * n):
        raise ValueError("mesh 大小与 %d×%d×%d 不符：%d 字节（推断 dsz=%d）"
                         % (nx, ny, nz, size, dsz))
    dt = "<f8" if dsz == 8 else "<f4"
    rec = 8 + dsz * n
    out = np.empty((3, nz, ny, nx), dtype=np.float64)
    with open(path, "rb") as fh:
        for f in range(3):
            fh.seek(f * nz * rec)
            for k in range(nz):
                fh.seek(4, 1)
                out[f, k] = np.frombuffer(fh.read(dsz * n), dtype=dt, count=n).reshape(ny, nx)
                fh.seek(4, 1)
    return out[0], out[1], out[2]


def analytic_mesh(nx, ny, nz, lx=4 * math.pi, lz=4 * math.pi / 3):
    xi = np.linspace(0.0, 1.0, ny)
    ys = (LY / 2.0) * (1.0 - np.tanh(BETA * (1.0 - 2.0 * xi)) / math.tanh(BETA)) - 1.0
    xs = np.arange(nx) * (lx / nx)
    zs = np.arange(nz) * (lz / nz)
    shape = (nz, ny, nx)
    return (np.broadcast_to(xs[None, None, :], shape).copy(),
            np.broadcast_to(ys[None, :, None], shape).copy(),
            np.broadcast_to(zs[:, None, None], shape).copy())


def find_mesh(snap, meta):
    cand = os.path.join(os.path.dirname(snap), "OCFD3d-Mesh.dat")
    if os.path.exists(cand) and os.path.getsize(cand) > 1000:
        return cand
    if meta.get("source"):
        run_root = os.path.dirname(os.path.dirname(snap))
        cand = os.path.join(run_root, os.path.dirname(meta["source"]), "OCFD3d-Mesh.dat")
        if os.path.exists(cand) and os.path.getsize(cand) > 1000:
            return cand
    return None


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="三维快照 .npz -> Tecplot 二进制 .plt")
    ap.add_argument("snapshot")
    ap.add_argument("--mesh", default=None, help="OCFD3d-Mesh.dat（缺省自动查找）")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--precision", choices=["float32", "float64"], default="float64")
    ap.add_argument("--title", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    snap = os.path.abspath(args.snapshot)
    t0 = time.time()
    cons, meta = load_snapshot(snap)
    nz, ny, nx = cons.shape[1:]
    step = int(meta.get("step", 0))
    npts = nx * ny * nz

    print("快照      : %s" % os.path.basename(snap))
    print("  网格    : nx=%d ny=%d nz=%d  点数=%d（源数据 float64）" % (nx, ny, nz, npts))
    print("  meta    : step=%s  t=%s  dt=%s  scheme=%s  source=%s"
          % (meta.get("step"), meta.get("t"), meta.get("dt"),
             meta.get("scheme"), meta.get("source")))
    print("  读入    : %.1f s, %.1f MB" % (time.time() - t0, cons.nbytes / 1e6))

    mesh = args.mesh or find_mesh(snap, meta)
    if mesh:
        x, y, z = read_mesh_records(mesh, nx, ny, nz)
        print("坐标      : %s" % mesh)
    else:
        x, y, z = analytic_mesh(nx, ny, nz)
        print("坐标      : 解析网格（Lx=4π, Lz=4π/3, tanh β=1.7）")
    print("          x∈[%.4f, %.4f]  y∈[%.4f, %.4f]  z∈[%.4f, %.4f]"
          % (x.min(), x.max(), y.min(), y.max(), z.min(), z.max()))

    # 派生量
    rho = cons[0]
    u = cons[1] / rho
    v = cons[2] / rho
    w = cons[3] / rho
    q2 = u * u + v * v + w * w
    p = (GAMMA - 1.0) * (cons[4] - 0.5 * rho * q2)
    T = p * GAMMA * M_R ** 2 / rho
    Ma = M_R * np.sqrt(q2) / np.sqrt(T)

    print("\n派生量范围（可与 run.log 对照）:")
    for nm, a in (("rho", rho), ("u", u), ("v", v), ("w", w), ("T", T), ("p", p), ("Ma", Ma)):
        print("  %-4s min=% .8e  max=% .8e" % (nm, a.min(), a.max()))

    if args.report_only:
        print("\n(--report-only: 未写文件)")
        return

    variables = ["x", "y", "z", "rho", "u", "v", "w", "T", "p", "Ma"]
    arrays = [x, y, z, rho, u, v, w, T, p, Ma]
    data = np.concatenate([np.asarray(a, dtype=np.float64).ravel(order="C") for a in arrays])

    title = args.title or ("OCFD-JAX Case iiA  Istep=%d" % step)
    aux = [("Re_tau", "%.1f" % RE_TAU), ("M_r", "%.4f" % M_R),
           ("gamma", "%.2f" % GAMMA), ("Pr", "%.2f" % PRANDTL),
           ("Tw", "%.1f" % TW), ("bodyforce", "rho*1.0"),
           ("note", "Case iiA, Lusher & Coleman 2023"),
           ("time", "%g" % meta.get("t", 0.0)),
           ("scheme", str(meta.get("scheme", "") or "-")),
           ("source", "jax_from_initial_60000")]
    zones = [{"name": "Istep %d" % step, "i": nx, "j": ny, "k": nz,
              "solution_time": float(step), "strand": -1, "aux": aux, "data": data}]

    dst = os.path.abspath(args.output) if args.output else os.path.splitext(snap)[0] + ".plt"
    wp = write_plt(title, variables, zones, dst, args.precision)
    print("\n写出      : %s (%.1f MB, %s, %.1f s)"
          % (dst, wp["bytes"] / 1e6, wp["precision"], wp["seconds"]))
    print("            数据 %d 变量 × %d 点 × %d 字节 = %.1f MB"
          % (len(variables), npts, 4 if wp["precision"] == "float32" else 8,
             len(variables) * npts * (4 if wp["precision"] == "float32" else 8) / 1e6))

    res = read_plt(dst)
    z0 = res["zones"][0]
    ok = (z0["i"] == nx and z0["j"] == ny and z0["k"] == nz and z0["name"] == "Istep %d" % step
          and res["variables"] == variables and res["trailing"] == 0)
    same = np.array_equal(z0["data"],
                          np.array(data, dtype=z0["data"].dtype).reshape(len(variables), npts))
    print("读回自检  : 结构=%s  尾随字节=%d  数据逐位一致=%s  |Δ|max=%.3e"
          % ("OK" if ok else "FAIL", res["trailing"], same,
             np.abs(z0["data"].astype(np.float64) - data.reshape(len(variables), npts)).max()))
    print("总耗时    : %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
