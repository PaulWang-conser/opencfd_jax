"""第 0 步基准生成器：按 ``step0_cases.py`` 逐条跑真实入口命令，存数据 + 哈希。

用法（在 D:\\Desktop\\ocfd2jax 下执行，或直接 python 本文件）：

    # 生成/重写基准
    python opencfd_jax/experiment/run_step0.py --phase baseline

    # 重构后跑候选并输出到另一个目录
    python opencfd_jax/experiment/run_step0.py --phase candidate

    # 只跑某几条
    python opencfd_jax/experiment/run_step0.py --phase candidate --only sod_1step tg_10step

每个用例的产物：
    <out>/<id>/run.log              计算日志（含 ana 诊断行）
    <out>/<id>/*.npz                run.py 自己存的守恒量（内区 5 分量 + _meta）
    <out>/<id>/case.json            命令行、墙钟、逐文件 sha256 与逐场统计
<out>/manifest.json                 全局清单：版本、git、逐用例摘要

设计要点：这里**不复制 run.py 的参数解析**，一律走子进程
``python -m opencfd_jax.src.run``，所以基准与真实入口永远同源；
以后 run.py 改错参数名，这一步会直接报非零退出码。
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

sys.path.insert(0, str(HERE))
from step0_cases import CASES, by_id, config_path   # noqa: E402


def sha256_file(path: Path) -> str:
    """按 1 MiB 分块算文件 sha256（22 MB 的 npz 也只需几十毫秒）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def npz_summary(path: Path) -> dict:
    """读一个 npz，返回 {变量名: 形状/统计} 与 _meta。"""
    data = np.load(path)
    fields, meta = {}, None
    for key in data.files:
        arr = data[key]
        if key == "_meta":
            meta = json.loads(arr.tobytes().decode("utf-8"))
            continue
        a = np.asarray(arr)
        with np.errstate(invalid="ignore"):
            fields[key] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "min": float(a.min()) if a.size else None,
                "max": float(a.max()) if a.size else None,
                "mean": float(a.mean()) if a.size else None,
                "nan": int(np.count_nonzero(~np.isfinite(a))),
            }
    return {"fields": fields, "meta": meta}


def git_info() -> dict:
    """收集 git 提交号与工作区脏文件数；没有 git 时返回空字典。"""
    repo = PROJECT / "opencfd_jax"
    info = {}
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True)
        info["commit"] = head.stdout.strip()
        br = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, check=True)
        info["branch"] = br.stdout.strip()
        st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True, check=True)
        info["dirty_files"] = len([l for l in st.stdout.splitlines() if l.strip()])
    except Exception as exc:                     # git 不在 PATH / 非仓库：不阻塞基准
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def env_info() -> dict:
    """记录会改变数值结果的解释器与库版本（换机器时靠它判断基准是否可比）。"""
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__}
    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
    except Exception as exc:
        info["jax_error"] = f"{type(exc).__name__}: {exc}"
    return info


def safe_rmtree(path: Path, root: Path):
    """只允许删除 root 之下的目录，避免误删用户自己的产物。"""
    path, root = path.resolve(), root.resolve()
    if path == root or root not in path.parents:
        raise RuntimeError(f"拒绝删除 {path}：不在 {root} 之内")
    if path.exists():
        shutil.rmtree(path)


def run_one(case: dict, out_root: Path, python: str, arg_source: str = "cli") -> dict:
    """跑一个用例：清目录 → 子进程 → 收 npz → 写 case.json。

    ``arg_source="json"`` 时改用用例的 ``input.json``（只把 JSON 无可表达的
    网格维数留在命令行），用于验证 JSON 与旧 CLI 的数值等价。
    """
    case_out = out_root / case["id"]
    safe_rmtree(case_out, out_root)
    case_out.mkdir(parents=True, exist_ok=True)

    if arg_source == "json":
        run_argv = ["--config", str(config_path(case)), *case.get("json_argv", [])]
    else:
        run_argv = list(case["argv"])
    cmd = [python, "-m", "opencfd_jax.src.run", *run_argv,
           "--out-dir", str(case_out)]
    print(f"\n[{case['id']}] {case['title']}  ({arg_source})")
    print("  " + " ".join(cmd[1:]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    wall = time.time() - t0
    log = (proc.stdout or "") + (proc.stderr or "")
    (case_out / "run.log").write_text(log, encoding="utf-8")

    if proc.returncode != 0:
        print(f"  !!! 退出码 {proc.returncode}，见 {case_out / 'run.log'}")
    files = []
    for npz in sorted(case_out.glob("*.npz")):
        summary = npz_summary(npz)
        files.append({"name": npz.name, "bytes": npz.stat().st_size,
                      "sha256": sha256_file(npz), **summary})
    record = {
        "id": case["id"], "title": case["title"], "cover": case["cover"],
        "note": case["note"], "argv": run_argv, "arg_source": arg_source,
        "command": " ".join(cmd), "wall_seconds": round(wall, 2),
        "returncode": proc.returncode, "files": files,
        "log_tail": log.strip().splitlines()[-6:],
    }
    (case_out / "case.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {len(files)} 个 npz, {wall:.1f}s, rc={proc.returncode}")
    return record


def write_markdown(manifest: dict, path: Path):
    """把基准摘要写成一张表，便于直接贴进改动说明。"""
    lines = [
        "# 第 0 步基准（自动生成，请勿手改）",
        "",
        f"- 生成时间：{manifest['created']}",
        f"- 解释器：Python {manifest['env']['python']} / numpy {manifest['env']['numpy']}"
        f" / jax {manifest['env'].get('jax', '?')} ({manifest['env'].get('jax_backend', '?')})",
        f"- git：{manifest['git'].get('commit', '?')[:12]}"
        f"（{manifest['git'].get('branch', '?')}，脏文件 "
        f"{manifest['git'].get('dirty_files', '?')}）",
        "",
        "| 用例 | 说明 | npz 数 | 体积 | rho 范围 | 耗时 |",
        "|---|---|---|---|---|---|",
    ]
    for rec in manifest["cases"]:
        n_files = len(rec["files"])
        mb = sum(f["bytes"] for f in rec["files"]) / 1e6
        rho_lo = rho_hi = None
        for f in rec["files"]:
            s = f.get("fields", {}).get("cons_0")
            if s:
                lo, hi = s["min"], s["max"]
                rho_lo = lo if rho_lo is None else min(rho_lo, lo)
                rho_hi = hi if rho_hi is None else max(rho_hi, hi)
        rng = ("—" if rho_lo is None else f"{rho_lo:.6g} ~ {rho_hi:.6g}")
        lines.append(f"| `{rec['id']}` | {rec['title']} | {n_files} | {mb:.1f} MB "
                     f"| {rng} | {rec['wall_seconds']:.1f}s |")
    lines += ["", "判定：重跑后用 `check_step0.py` 与本目录逐点比对，"
                  "缺省要求 max|diff| = 0（逐位一致）。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="第 0 步基准生成/候选运行")
    p.add_argument("--phase", default="baseline",
                   help="子目录名：baseline（基准）或 candidate（重构后候选）")
    p.add_argument("--out", default=None, help="输出根目录，缺省 experiment/<phase>")
    p.add_argument("--only", nargs="*", default=None, help="只跑这些 id")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--arg-source", choices=["cli", "json"], default="cli",
                   help="走旧命令行（cli）还是用例 JSON（json）；基准始终用 cli")
    p.add_argument("--list", action="store_true", help="只打印用例清单")
    args = p.parse_args()

    if args.list:
        for c in CASES:
            print(f"{c['id']:<24s} {c['title']}\n{'':<24s} 覆盖: {c['cover']}")
        return 0

    if args.phase == "baseline" and args.arg_source == "json":
        p.error("baseline 只能用 --arg-source cli 生成；JSON 只用于候选验证")

    out_root = Path(args.out) if args.out else HERE / args.phase
    out_root.mkdir(parents=True, exist_ok=True)
    cases = [by_id(i) for i in args.only] if args.only else CASES

    records = []
    for case in cases:
        records.append(run_one(case, out_root, args.python, args.arg_source))

    manifest_path = out_root / "manifest.json"
    old = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged = {c["id"]: c for c in old.get("cases", [])}
    merged.update({c["id"]: c for c in records})
    manifest = {
        "schema": 1,
        "phase": args.phase,
        "arg_source": args.arg_source,
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "env": env_info(),
        "git": git_info(),
        "scheme": {"inviscid": "WENO7", "viscous": "CD8",
                   "precision": "float64", "filter": "off"},
        "cases": [merged[c["id"]] for c in CASES if c["id"] in merged],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    write_markdown(manifest, out_root / "STEP0.md")
    bad = [r["id"] for r in records if r["returncode"] != 0]
    print(f"\n清单 -> {manifest_path}")
    if bad:
        print(f"!!! 退出码非零的用例: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
