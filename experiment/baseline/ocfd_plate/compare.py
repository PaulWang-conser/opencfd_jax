"""OCFD 准例 vs JAX 基线：单步逐点对比报表。

用法：
  python compare.py --ocfd ocfd_plate_step000001.npz \
                    --jax  ../plate_1step/plate_step000001.npz
输出：终端报表 + compare_report.json
"""
import argparse
import json

import numpy as np

NAMES = ["rho", "rho_u", "rho_v", "rho_w", "E"]


def load(path):
    d = np.load(path, allow_pickle=True)
    meta = d["_meta"]
    try:
        meta = meta.item()
    except Exception:
        pass
    if isinstance(meta, np.ndarray):
        meta = "".join(chr(int(c)) for c in meta.ravel()).strip()
    if isinstance(meta, bytes):
        meta = meta.decode()
    if isinstance(meta, str):
        meta = json.loads(meta)
    return [np.asarray(d[f"cons_{i}"], dtype=float) for i in range(5)], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocfd", required=True)
    ap.add_argument("--jax", required=True)
    ap.add_argument("--out", default="compare_report.json")
    args = ap.parse_args()

    o, mo = load(args.ocfd)
    j, mj = load(args.jax)
    assert o[0].shape == j[0].shape, (o[0].shape, j[0].shape)
    # 原码坐标系：0=展向 k，1=壁法向 j，2=流向 i
    ref = np.abs(j[0]).max()
    rep = {"ocfd_meta": mo, "jax_meta": mj, "shape": list(o[0].shape), "fields": {}}
    print(f"shape {o[0].shape}  (k=span, j=wall-normal, i=streamwise)")
    print(f"{'field':6s} {'max|ocfd|':>12s} {'max|jax|':>12s} {'max|d|':>12s} "
          f"{'rel':>10s} {'argmax (k,j,i)':>16s}")
    for n, a, b in zip(NAMES, o, j):
        d = a - b
        ad = np.abs(d)
        idx = np.unravel_index(np.argmax(ad), ad.shape)
        rel = ad.max() / max(1e-300, np.abs(b).max())
        print(f"{n:6s} {np.abs(a).max():12.5e} {np.abs(b).max():12.5e} "
              f"{ad.max():12.5e} {rel:10.3e} {str(idx):>16s}")
        rep["fields"][n] = dict(max_ocfd=float(np.abs(a).max()),
                                max_jax=float(np.abs(b).max()),
                                max_abs_diff=float(ad.max()),
                                rel=float(rel), argmax=list(map(int, idx)))

    # 分块统计：把 i 分成 入口带 / 内区 / 出口带，看误差集中在哪
    ni = o[0].shape[2]
    bands = {"inlet i<4": slice(0, 4), "inner 4<=i<ni-4": slice(4, ni - 4),
             "outlet i>=ni-4": slice(ni - 4, ni)}
    print("\n按 i 分带的 max|d(rho_u)| / max|d(E)|：")
    for tag, s in bands.items():
        du = np.abs(o[1][:, :, s] - j[1][:, :, s]).max()
        de = np.abs(o[4][:, :, s] - j[4][:, :, s]).max()
        dr = np.abs(o[0][:, :, s] - j[0][:, :, s]).max()
        print(f"  {tag:20s} rho={dr:.3e}  rho_u={du:.3e}  E={de:.3e}")
        rep.setdefault("bands", {})[tag] = dict(rho=float(dr), rho_u=float(du), E=float(de))

    # 壁面行 / 顶行
    print("\n壁面行 j=0 与顶行 j=nj-1 的 max|d|：")
    for tag, jj in (("wall j=0", 0), ("top j=nj-1", o[0].shape[1] - 1)):
        du = np.abs(o[1][:, jj, :] - j[1][:, jj, :]).max()
        dr = np.abs(o[0][:, jj, :] - j[0][:, jj, :]).max()
        print(f"  {tag:12s} rho={dr:.3e}  rho_u={du:.3e}")
        rep.setdefault("rows", {})[tag] = dict(rho=float(dr), rho_u=float(du))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
