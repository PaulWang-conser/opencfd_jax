"""第 0 步逐点比对：候选目录 vs 基准目录。

用法：

    python opencfd_jax/experiment/check_step0.py
    python opencfd_jax/experiment/check_step0.py --candidate experiment/candidate \\
        --tol 1e-12

判定口径：
  - 缺省 ``--tol 0``：要求逐位一致（同机同 JAX 版本下，纯重构不应改变任何一位）；
  - 若因运算顺序确实改变（例如 sum 顺序、kernel 融合）导致末位抖动，
    可放宽到 ``--tol 1e-12``（与方案里"原版对拍判据"同一量级），
    但必须在改动说明里写清是哪个文件、为什么。
  - ``_meta`` 里的 t / step / dt 要求严格一致：dt 序列变了就说明 CFL 口径变了。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from step0_cases import CASES, by_id        # noqa: E402


def compare_arrays(a: np.ndarray, b: np.ndarray, tol: float) -> dict:
    """比较形状、类型及有限数值；零容差要求逐位相同。"""
    if a.shape != b.shape:
        return {"ok": False, "reason": f"shape {a.shape} vs {b.shape}"}
    if a.dtype != b.dtype:
        return {"ok": False, "reason": f"dtype {a.dtype} vs {b.dtype}"}
    for label, value in (("基准", a), ("候选", b)):
        bad = ~np.isfinite(value)
        if np.any(bad):
            at = tuple(np.argwhere(bad)[0])
            return {"ok": False, "reason": f"{label}非有限值 {value[at]}，位置 {at}"}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).ravel()
    if d.size == 0:
        return {"ok": True, "max": 0.0, "rms": 0.0, "n_bad": 0, "at": None,
                "nan": 0}
    imax = int(np.argmax(d))
    n_bad = int(np.count_nonzero(d > tol))
    if tol == 0 and a.tobytes(order="C") != b.tobytes(order="C"):
        n_bad = max(n_bad, 1)
    return {
        "ok": n_bad == 0,
        "max": float(d.max()),
        "rms": float(np.sqrt((d ** 2).mean())),
        "n_bad": n_bad,
        "at": [int(x) for x in np.unravel_index(imax, a.shape)],
        "nan": 0,
    }


def compare_case(base_dir: Path, cand_dir: Path, tol: float) -> dict:
    """比较一个用例目录：文件集合、_meta、逐场数值。"""
    base_json = base_dir / "case.json"
    if not base_json.exists():
        return {"id": base_dir.name, "status": "NO_BASELINE"}
    if not cand_dir.exists():
        return {"id": base_dir.name, "status": "MISSING"}

    try:
        base = json.loads(base_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"id": base_dir.name, "status": "FAIL", "problems": [f"基准清单错误: {exc}"]}
    rows, problems = [], []
    base_files = {f["name"]: f for f in base["files"]}
    if not base_files:
        problems.append("基准文件集合为空")
    cand_files = sorted(p.name for p in cand_dir.glob("*.npz"))
    only_base = sorted(set(base_files) - set(cand_files))
    only_cand = sorted(set(cand_files) - set(base_files))
    if only_base:
        problems.append(f"候选缺少文件 {only_base}")
    if only_cand:
        problems.append(f"候选多出文件 {only_cand}")

    for name in sorted(set(base_files) & set(cand_files)):
        try:
            with (base_dir / name).open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != base_files[name].get("sha256"):
                problems.append(f"{name}: 基准 SHA-256 不匹配")
            with np.load(base_dir / name, allow_pickle=False) as A, np.load(cand_dir / name, allow_pickle=False) as B:
                if set(A.files) != set(B.files):
                    problems.append(f"{name}: 字段不一致，缺少 {sorted(set(A.files)-set(B.files))}，多出 {sorted(set(B.files)-set(A.files))}")
                if "_meta" not in A.files or "_meta" not in B.files:
                    problems.append(f"{name}: 缺少 _meta")
                else:
                    ma = json.loads(A["_meta"].tobytes().decode())
                    mb = json.loads(B["_meta"].tobytes().decode())
                    for key in ("t", "step", "dt"):
                        if key not in ma or key not in mb or ma[key] != mb[key]:
                            problems.append(f"{name}: _meta.{key} 缺失或不一致")
                for key in sorted((set(A.files) & set(B.files)) - {"_meta"}):
                    res = compare_arrays(A[key], B[key], tol)
                    rows.append({"file": name, "var": key, **res})
                    if not res.get("ok"):
                        problems.append(f"{name}:{key} {res.get('reason', '数值不一致')} max={res.get('max')} at={res.get('at')}")
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            problems.append(f"{name}: 读取或解析失败: {exc}")

    worst = max((r.get("max", 0.0) for r in rows), default=0.0)
    status = "PASS" if (not problems) else "FAIL"
    return {"id": base_dir.name, "status": status, "worst": worst,
            "files": len(rows), "problems": problems,
            "rows": rows, "wall_seconds": base.get("wall_seconds"),
            "command": base.get("command")}


def main():
    p = argparse.ArgumentParser(description="第 0 步逐点比对")
    p.add_argument("--baseline", default=str(HERE / "baseline"))
    p.add_argument("--candidate", default=str(HERE / "candidate"))
    p.add_argument("--tol", type=float, default=0.0,
                   help="绝对误差阈值；0 = 逐位一致")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--verbose", action="store_true", help="打印每个场的误差")
    args = p.parse_args()
    if not np.isfinite(args.tol) or args.tol < 0:
        p.error("--tol 必须是非负有限数")

    base_root = Path(args.baseline)
    cand_root = Path(args.candidate)
    if not (base_root / "manifest.json").exists():
        print(f"没有基准：{base_root}（先跑 run_step0.py --phase baseline）")
        return 2
    ids = args.only if args.only else [c["id"] for c in CASES]
    results = [compare_case(base_root / i, cand_root / i, args.tol) for i in ids]

    print(f"基准 {base_root}\n候选 {cand_root}   判据 max|diff| <= {args.tol:g}\n")
    print(f"{'用例':<24s} {'判定':<6s} {'最坏误差':>12s}  {'场数':>4s}  备注")
    print("-" * 78)
    for r in results:
        worst = f"{r['worst']:.3e}" if "worst" in r else "—"
        note = "" if not r.get("problems") else r["problems"][0]
        print(f"{r['id']:<24s} {r['status']:<6s} {worst:>12s}  {r.get('files', 0):>4d}  {note}")
    if args.verbose:
        for r in results:
            for row in r.get("rows", []):
                if not row.get("ok") or row.get("max", 0) > 0:
                    print(f"  {r['id']} {row['file']} {row['var']}: "
                          f"max={row.get('max')} rms={row.get('rms')} "
                          f"n={row.get('n_bad')} at {row.get('at')}")

    report = {"baseline": str(base_root), "candidate": str(cand_root),
              "tol": args.tol, "results": results}
    out = cand_root / "check_report.json"
    if cand_root.exists():
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n报告 -> {out}")
    failed = [r["id"] for r in results if r["status"] != "PASS"]
    print(f"\n总判定：{'PASS' if not failed else 'FAIL ' + str(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
