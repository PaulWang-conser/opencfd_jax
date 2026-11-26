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
  - 基准是第八步以前的格式（cons_0..4）、候选是第八步起的格式（b0）时，基准的守恒量
    先用同一份换算换成原始量再比，物性取基准自己 ``_meta`` 里的 gamma、Ama。
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent.parent))      # opencfd_jax.src.io，换算旧格式的基准时用
from step0_cases import CASES, by_id        # noqa: E402

_STEP_RE = re.compile(
    r"(?:_step(?P<legacy>\d{6})|-istep-(?P<istep>\d{8,}))$")
_OLD_KEYS = {f"cons_{i}" for i in range(5)}


def step_of(name: str):
    """从旧 ``*_stepNNNNNN.npz`` 或新 ``*-istep-NNNNNNNN.npz`` 提取步号。"""
    match = _STEP_RE.search(Path(name).stem)
    if not match:
        return None
    value = match.group("legacy") or match.group("istep")
    return int(value)


def pair_files(base_files, cand_names):
    """把基准文件与候选文件一对一配对。

    先按文件名精确匹配（兼容不带步号的历史用例），剩下的一侧再按旧
    ``_stepNNNNNN`` 或新 ``-istep-NNNNNNNN`` 的步号配对。这样网格文件名
    变化或输出命名迁移时，基准文件本身不需要改名，检查器仍能逐位比较。
    """
    pairs = []
    problems = []
    cand_set = set(cand_names)
    for name in base_files:
        if name in cand_set:
            pairs.append((name, name))
    base_rest = [n for n in base_files if n not in cand_set]
    cand_rest = [n for n in cand_names if n not in base_files]

    cand_by_step = {}
    for name in cand_rest:
        step = step_of(name)
        if step is not None:
            cand_by_step.setdefault(step, []).append(name)

    used = set()
    for name in base_rest:
        step = step_of(name)
        if step is None:
            problems.append(
                f"基准 {name} 在候选中没有同名文件，且文件名不含步号，无法映射")
            continue
        options = cand_by_step.get(step, [])
        if not options:
            problems.append(f"基准 {name}（step={step}）在候选中缺少对应文件")
        elif len(options) > 1:
            problems.append(
                f"基准 {name}（step={step}）对应多个候选文件 {options}")
        else:
            pairs.append((name, options[0]))
            used.add(options[0])

    leftovers = [n for n in cand_rest if n not in used]
    if leftovers:
        problems.append(f"候选多出无法对应到基准的文件 {sorted(leftovers)}")
    return pairs, problems


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


def _as_candidate_layout(base, candidate) -> dict:
    """基准文件读成 {名字: 数组}；基准是第八步以前的格式（cons_0..4）而候选是第八步
    起的格式（b0）时，守恒量先换成原始量。"""
    fields = {name: base[name] for name in base.files}
    if set(fields) - {"_meta"} == _OLD_KEYS and "b0" in candidate.files:
        from opencfd_jax.src import io
        meta = json.loads(fields["_meta"].tobytes().decode())
        cons = np.stack([fields[f"cons_{i}"] for i in range(5)])
        fields = {"_meta": fields["_meta"],
                  "b0": io.primitive_of(cons, io.phys_from_meta(meta))}
    return fields


def compare_case(base_dir: Path, cand_dir: Path, tol: float) -> dict:
    """比较一个用例目录：文件集合、_meta、逐场数值。"""
    base_json = base_dir / "case.json"
    if not base_json.exists():
        return {"id": base_dir.name, "status": "NO_BASELINE"}
    if not cand_dir.exists():
        return {"id": base_dir.name, "status": "MISSING"}

    try:
        base = json.loads(base_json.read_text(encoding="utf-8"))
        base_files = {f["name"]: f for f in base["files"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"id": base_dir.name, "status": "FAIL", "problems": [f"基准清单错误: {exc}"]}
    rows, problems = [], []
    if not base_files:
        problems.append("基准文件集合为空")
    cand_names = sorted(p.name for p in cand_dir.glob("*.npz"))
    pairs, pair_problems = pair_files(base_files, cand_names)
    problems.extend(pair_problems)

    for base_name, cand_name in pairs:
        try:
            with (base_dir / base_name).open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != base_files[base_name].get("sha256"):
                problems.append(f"{base_name}: 基准 SHA-256 不匹配")
            with np.load(base_dir / base_name, allow_pickle=False) as A, \
                    np.load(cand_dir / cand_name, allow_pickle=False) as B:
                A = _as_candidate_layout(A, B)
                names, other = set(A), set(B.files)
                if not names - {"_meta"}:
                    problems.append(f"{base_name}: 没有数值字段")
                if names != other:
                    problems.append(
                        f"基准 {base_name} / 候选 {cand_name}: 字段不一致，"
                        f"缺少 {sorted(names - other)}，"
                        f"多出 {sorted(other - names)}")
                if "_meta" not in names or "_meta" not in other:
                    problems.append(f"基准 {base_name} / 候选 {cand_name}: 缺少 _meta")
                else:
                    ma = json.loads(A["_meta"].tobytes().decode())
                    mb = json.loads(B["_meta"].tobytes().decode())
                    for key in ("t", "step", "dt"):
                        if key not in ma or key not in mb or ma[key] != mb[key]:
                            problems.append(
                                f"基准 {base_name} / 候选 {cand_name}: "
                                f"_meta.{key} 缺失或不一致")
                    expected = step_of(base_name)
                    if expected is not None and mb.get("step") != expected:
                        problems.append(
                            f"候选 {cand_name}: _meta.step={mb.get('step')} "
                            f"与文件名步号 {expected} 不一致")
                for key in sorted((names & other) - {"_meta"}):
                    res = compare_arrays(A[key], B[key], tol)
                    rows.append({"file": cand_name, "baseline": base_name,
                                 "var": key, **res})
                    if not res.get("ok"):
                        problems.append(
                            f"基准 {base_name} / 候选 {cand_name}:{key} "
                            f"{res.get('reason', '数值不一致')} "
                            f"max={res.get('max')} at={res.get('at')}")
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            problems.append(f"基准 {base_name} / 候选 {cand_name}: "
                            f"读取或解析失败: {exc}")

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
