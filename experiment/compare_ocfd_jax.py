"""原码 OCFD vs JAX 的单步数值对拍（WENO7 + CD8，channel，96×97×64）。

流程（只比**一步**）：
  1. 解析 `dump/` 下 main_dump 的文本场：cons_*_step000000 / step000001、
     du_krk1..3_step000001（内区，5 分量）；
  2. 用 JAX 侧的 `build_case` + `build_rhs`（与 run.py 完全同一条链）以原码
     step0 的守恒量为初值推进一步，取三个子步的 du 与末态；
  3. 逐场给 max|diff| / rms / 相对量 / 位置，并按 1e-12 判 PASS。

用法（在 D:\\Desktop\\ocfd2jax 下）：

    python opencfd_jax/experiment/compare_ocfd_jax.py

缺省目录就是 experiment/ocfd/channel_weno7（与原码 dump 输出同处）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

CONS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def parse_text_dump(path: Path, shape) -> np.ndarray:
    """把 main_dump 的 'i=..,j=..,k=..[,n=..]\\t<值>' 文本读成数组。

    行序是 n（若有）最慢、k 次之、j 次之、i 最快（与原码内区写出一致），
    所以直接 reshape 即可：cons_* 是 (nz,ny,nx)，du_krk* 是 (5,nz,ny,nx)。
    """
    raw = path.read_bytes()
    # 每行 "标签\t数值"：把 tab 换成换行后，奇数位就是数值
    tokens = raw.replace(b"\t", b"\n").split(b"\n")
    vals = np.array([float(t) for t in tokens[1::2]], dtype=np.float64)
    if vals.size != int(np.prod(shape)):
        raise ValueError(f"{path.name}: 读到 {vals.size} 个值，"
                         f"与 shape {shape} 不符（dump 被截断或网格不一致）")
    return vals.reshape(shape)


def load_or_parse(dump_dir: Path, shape, cache: Path) -> dict:
    """解析全部 dump；已有 cache.npz 时直接复用（省 10-20 s）。"""
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    t0 = __import__("time").time()
    out = {}
    for tag in CONS:
        for step in (0, 1):
            key = f"cons_{tag}_step{step:06d}"
            out[key] = parse_text_dump(dump_dir / f"{key}.dat", shape)
    for krk in (1, 2, 3):
        key = f"du_krk{krk}_step000001"
        out[key] = parse_text_dump(dump_dir / f"{key}.dat", (5, *shape))
    print(f"[parse] {len(out)} 个场，{__import__('time').time() - t0:.1f}s -> {cache.name}")
    np.savez_compressed(cache, **out)
    return out


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    """单场误差统计：max/rms/相对量/位置。"""
    d = np.abs(a - b)
    pos = [int(x) for x in np.unravel_index(int(np.argmax(d)), d.shape)]
    return {"max": float(d.max()),
            "rms": float(np.sqrt((d ** 2).mean())),
            "scale": float(np.abs(a).max()),
            "rel": float(d.max() / max(np.abs(a).max(), 1e-300)),
            "at": pos}


def main():
    p = argparse.ArgumentParser(description="OCFD vs JAX 单步对拍（WENO7+CD8）")
    p.add_argument("--dir", default=str(HERE / "ocfd" / "channel_weno7"),
                   help="含 opencfd-scu.in / dump/ 的原码运行目录")
    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--scheme", default="WENO7")
    p.add_argument("--vis-order", default="CD8")
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--baseline",
                   default=str(HERE / "baseline" / "channel_1step" / "channel_step000001.npz"),
                   help="JAX 第0步基准 npz（只用来对照 OCFD 的末态）")
    args = p.parse_args()

    run_dir = Path(args.dir)
    dump_dir = run_dir / "dump"
    shape = (64, 97, 96)
    fields = load_or_parse(dump_dir, shape, dump_dir / "parsed.npz")

    # ---------- JAX 侧：与原版对拍用同一条 RHS 链 ----------
    from opencfd_jax.src import config
    config.enable_x64()
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from opencfd_jax.src.cases import build_case
    from opencfd_jax.src.run import build_rhs
    from opencfd_jax.src.solver import rk3_step_stages

    ns = SimpleNamespace(
        case="channel", ni=96, nj=97, nk=64, Re=190.7, Ama=0.0955,
        ref_t=300.0, pr=0.7, p_grad=0.1, scheme=args.scheme,
        vis_order=args.vis_order, weno_tv=False, viscous=True,
        forcing_work="off", infile=str(run_dir / "opencfd-scu.in"),
        field=str(run_dir / "opencfd.dat"),
        mesh=str(run_dir / "OCFD3d-Mesh.dat"))
    case = build_case(ns)
    grid = case["grid"]
    rhs = build_rhs(case, ns)
    bc = case.get("bc_apply")

    # 初值一致性：JAX 从 opencfd.dat 造的 cons0 vs 原码 init 后的 step0
    cons_jax0 = np.asarray(case["cons0"][grid.inner()])
    cons0_diff = {tag: stats(fields[f"cons_{tag}_step000000"], cons_jax0[v])
                  for v, tag in enumerate(CONS)}

    # 以原码 step0 为共同初值：只比一步，杜绝初值差异
    q = grid.alloc()
    for v, tag in enumerate(CONS):
        q = q.at[(v,) + grid.inner()[1:]].set(jnp.asarray(fields[f"cons_{tag}_step000000"]))
    q1, dus = rk3_step_stages(q, args.dt, rhs, bc_apply=bc)

    rows = []
    print(f"\n初值（原码 step0 vs JAX 由 opencfd.dat 构造）")
    print(f"{'场':<8s}{'max|diff|':>12s}{'rms':>12s}{'相对':>10s}")
    for tag in CONS:
        s = cons0_diff[tag]
        print(f"{tag:<8s}{s['max']:>12.3e}{s['rms']:>12.3e}{s['rel']:>10.3e}")

    print(f"\n一步：三个 RK 子步的 du，与末态 cons（同类相减，容差 {args.tol:g}）")
    print(f"{'场':<22s}{'|OCFD|max':>12s}{'max|diff|':>13s}{'rms':>12s}"
          f"{'相对':>10s}   位置")
    ok = True
    for krk, du in enumerate(dus, start=1):
        du_np = np.asarray(du[grid.inner()])
        for v, tag in enumerate(CONS):
            # 注意：du 的 dump 一个文件里有 5 个分量，缓存的键存的是整体 (5,nz,ny,nx)，
            # 这里必须显式取 [v]，否则会按 5 分量广播、把 5 个分量混在一起比。
            s = stats(fields[f"du_krk{krk}_step000001"][v], du_np[v])
            if s["max"] > args.tol:
                ok = False
            rows.append({"kind": f"du_krk{krk}", "tag": tag, **s})
            print(f"du{krk} {tag:<17s}{s['scale']:>12.3e}{s['max']:>13.3e}"
                  f"{s['rms']:>12.3e}{s['rel']:>10.3e}   {s['at']}")
    q1_np = np.asarray(q1[grid.inner()])
    for v, tag in enumerate(CONS):
        s = stats(fields[f"cons_{tag}_step000001"], q1_np[v])
        if s["max"] > args.tol:
            ok = False
        rows.append({"kind": "cons_step1", "tag": tag, **s})
        print(f"Q1   {tag:<17s}{s['scale']:>12.3e}{s['max']:>13.3e}"
              f"{s['rms']:>12.3e}{s['rel']:>10.3e}   {s['at']}")

    # ---------- 分区：差值是不是只集中在贴壁那几行 ----------
    # j 是壁法向；内区行 12..84 之外算"贴壁带"（WENO7 模板 4 点 + CD8 模板 4 点，
    # 再加上边界带/度量，12 行足够把"只受壁面影响"的部分隔离出来）。
    NB = 12
    kw = 0.0
    print(f"\n分区（du krk=1 与 Q1；贴壁带 = j<{NB} 或 j>{shape[1]-NB-1}）")
    print(f"{'场':<8s}{'du1 贴壁带':>13s}{'du1 内区':>12s}"
          f"{'Q1 贴壁带':>12s}{'Q1 内区':>11s}")
    band_stats = {}
    for v, tag in enumerate(CONS):
        d1 = np.abs(fields[f"du_krk{1}_step000001"][v] - np.asarray(dus[0][grid.inner()])[v])
        dq = np.abs(fields[f"cons_{tag}_step000001"] - q1_np[v])
        p1, pq = d1.max(axis=(0, 2)), dq.max(axis=(0, 2))
        b1 = max(float(p1[:NB].max()), float(p1[-NB:].max()))
        i1 = float(p1[NB:-NB].max())
        bq = max(float(pq[:NB].max()), float(pq[-NB:].max()))
        iq = float(pq[NB:-NB].max())
        band_stats[tag] = {"du1_band": b1, "du1_inner": i1,
                           "q1_band": bq, "q1_inner": iq,
                           "du1_row_profile": [float(x) for x in p1]}
        kw = max(kw, i1)
        print(f"{tag:<8s}{b1:>13.3e}{i1:>12.3e}{bq:>12.3e}{iq:>11.3e}")
    print(f"内区最坏（du1，j={NB}..{shape[1]-NB-1}）= {kw:.3e}")

    # 与第 0 步基准（JAX 自己跑出来的 channel_1step）对照原码末态
    bp = Path(args.baseline)
    base_rows = []
    if bp.exists():
        z = np.load(bp)
        print(f"\n原码末态 vs JAX 第0步基准（{bp.name}）")
        print(f"{'场':<8s}{'|JAX|max':>12s}{'max|diff|':>13s}{'相对':>10s}")
        for v, tag in enumerate(CONS):
            s = stats(fields[f"cons_{tag}_step000001"], np.asarray(z[f"cons_{v}"]))
            base_rows.append({"tag": tag, **s})
            print(f"{tag:<8s}{np.abs(np.asarray(z[f'cons_{v}'])).max():>12.3e}"
                  f"{s['max']:>13.3e}{s['rel']:>10.3e}")

    worst = max(r["max"] for r in rows)
    verdict = "PASS" if (ok and worst <= args.tol) else "FAIL"
    print(f"\n最坏 = {worst:.3e}（判据 {args.tol:g}）判定：{verdict}")

    out = {"dir": str(run_dir), "scheme": args.scheme, "vis_order": args.vis_order,
           "dt": args.dt, "tol": args.tol, "verdict": verdict, "worst": worst,
           "init": cons0_diff, "rows": rows, "vs_baseline": base_rows,
           "bands": band_stats}
    (run_dir / "compare_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {run_dir / 'compare_report.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
