"""对拍工具：原版 dump 文本解析 + JAX 对照跑 + 逐场误差报告。

子命令：
  compare.py dat2npz <原版dump目录> [--out dump-npz]
      原版 dump（patches/main_dump.c 的 %.17e 文本）→ npz（统一对拍格式）。
  compare.py jax-run --case sod --ni 128 --steps 3 --dt 0.001 [--fp64] [--out dump-jax]
      JAX 侧对照跑：读原版 step0 守恒量作初值，跑同 dt 步数，逐 RK 子步存 du、
      每整步存 cons（npz）。
  compare.py diff <原版npz目录> <jax npz目录>
      逐场逐步误差报告（RMS/max + 最大误差位置）。

判定标准（方案 9 节）：FP64 下 du/cons 逐点 < 1e-12（运算顺序不同的
舍入级差异）；超差 → 报告场名/步号/位置，进入定位流程。

口径（2026-11-14 修订）：
- 原版 dump 是**守恒量**（pf_d）：rho, rho*u, rho*v, rho*w, rho*E，标签
  rho/rhou/rhov/rhow/rhoE。旧版 dump 的是原始变量（rho,u,v,w,T），与这里的
  守恒量口径错位，已随 patches/main_dump.c 一并修正。
- 原版 dump 是**内区**（pf_d/pdu_d 都是 new_cudaSoA(nx,ny,nz)，无鬼胞），
  所以这里不做 LAP 裁剪；JAX 侧也存 grid.inner()，两侧同为 (nz,ny,nx)。
- 原版 du 的 5 个分量与 JAX 的 cons 分量同序（守恒量 RHS）。
- 粘性项：原码 du_comput 恒算无粘 + 粘性（Stream_MODE 只影响 stream 调度），
  所以这里也无条件加 viscous_rhs；--viscous 开关保留只为兼容旧命令，不起作用。
- μ 的参数：Ref_T 是**有量纲参考温度(K)**（原码 T_REF），Pr 是原码 PR。
  两侧移植的是同一套无量纲关系（Tsb=110.4/Ref_T、amu_C0=(1+110.4/Ref_T)/Re），
  所以 μ(T=1)=1/Re 与 Ref_T 无关，但 T≠1 时 Sutherland 曲线随 Ref_T 变；
  对拍必须 --ref-t/--pr 与 in 文件同值，否则 du 的粘性部分必然差。
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from . import boundary
from .mesh_data import from_case

# 原版 dump 的守恒量标签，顺序 = pf 的第 0..4 分量
CONS_TAGS = ("rho", "rhou", "rhov", "rhow", "rhoE")

LINE_ME1 = re.compile(
    r"i=\s*(-?\d+),j=\s*(-?\d+),k=\s*(-?\d+)\s+([+-]?[\d.eE+-]+)\s*$")
LINE_ME2 = re.compile(
    r"i=\s*(-?\d+),j=\s*(-?\d+),k=\s*(-?\d+),n=\s*(\d+)\s+([+-]?[\d.eE+-]+)\s*$")


def parse_me1(path):
    """单场文本 → (nk, nj, ni) 数组（内区）。索引为物理 1-based。"""
    vals = {}
    i_max = j_max = k_max = -10 ** 9
    with open(path) as fh:
        for line in fh:
            m = LINE_ME1.match(line.strip())
            if not m:
                continue
            i, j, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
            vals[(i, j, k)] = float(m.group(4))
            i_max, j_max, k_max = max(i_max, i), max(j_max, j), max(k_max, k)
    nx, ny, nz = i_max, j_max, k_max
    out = np.empty((nz, ny, nx))
    for (i, j, k), v in vals.items():
        out[k - 1, j - 1, i - 1] = v
    return out, (nx, ny, nz)


def parse_me2(path, n_var=5):
    """5 分量文本 → (n_var, nk, nj, ni) 数组（内区）。"""
    vals = {}
    i_max = j_max = k_max = -10 ** 9
    with open(path) as fh:
        for line in fh:
            m = LINE_ME2.match(line.strip())
            if not m:
                continue
            i, j, k, n = (int(m.group(1)), int(m.group(2)),
                          int(m.group(3)), int(m.group(4)))
            vals[(i, j, k, n)] = float(m.group(5))
            i_max, j_max, k_max = max(i_max, i), max(j_max, j), max(k_max, k)
    nx, ny, nz = i_max, j_max, k_max
    out = np.empty((n_var, nz, ny, nx))
    for (i, j, k, n), v in vals.items():
        out[n - 1, k - 1, j - 1, i - 1] = v
    return out, (nx, ny, nz)


def _save_with_meta(path, field, meta):
    """把单个对拍场及其网格/步段元信息写入同一个 NPZ 文件。"""
    np.savez(path, field=field,
             meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


def cmd_dat2npz(args):
    """把原版逐场文本 dump 批量转换为带元信息的 NPZ，供后续差分比较。"""
    out = args.out
    os.makedirs(out, exist_ok=True)
    n_cons = 0
    n_du = 0
    for tag in CONS_TAGS:
        for path in sorted(glob.glob(os.path.join(args.dump_dir,
                                                 f"cons_{tag}_step*.dat"))):
            step = int(re.search(r"step(\d+)", os.path.basename(path)).group(1))
            arr, (nx, ny, nz) = parse_me1(path)
            _save_with_meta(os.path.join(out, f"cons_{tag}_step{step:06d}.npz"),
                            arr, {"nx": nx, "ny": ny, "nz": nz,
                                  "tag": tag, "layout": "inner"})
            n_cons += 1
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "du_krk*_step*.dat"))):
        m = re.search(r"krk(\d)_step(\d+)", os.path.basename(path))
        krk, step = int(m.group(1)), int(m.group(2))
        arr, (nx, ny, nz) = parse_me2(path)
        _save_with_meta(os.path.join(out, f"du_krk{krk}_step{step:06d}.npz"),
                        arr, {"nx": nx, "ny": ny, "nz": nz, "krk": krk,
                              "layout": "inner"})
        n_du += 1
    if n_cons == 0 and n_du == 0:
        raise SystemExit(f"no dump found in {args.dump_dir} "
                         f"(looked for cons_{{{','.join(CONS_TAGS)}}}_step*.dat "
                         f"and du_krk*_step*.dat)")
    print(f"converted -> {out}  (cons files: {n_cons}, du files: {n_du})")


def cmd_jax_run(args):
    """从原版 step0 初值出发执行 JAX 对拍，并逐 RK 子步保存 du、逐步保存守恒量。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from . import config
    config.enable_x64()
    import jax
    import jax.numpy as jnp
    from .cases import apply_forcing, build_case
    from .solver import inviscid_rhs, rk3_step_stages
    from .viscous import viscous_rhs

    args_case = argparse.Namespace(case=args.case, ni=args.ni, nj=args.nj,
                                   Re=args.Re, Ama=args.Ama,
                                   ref_t=getattr(args, "ref_t", None),
                                   pr=getattr(args, "pr", 0.7),
                                   nk=getattr(args, "nk", 4),
                                   p_grad=args.p_grad, scheme=args.scheme,
                                   viscous=args.viscous,
                                   forcing_work=getattr(args, "forcing_work", "off"))
    case = build_case(args_case)
    vis_order = {"CD6": 6, "CD8": 8}[getattr(args, "vis_order", "CD6")]
    # 原码差分是 -Ajac*(F_r - F_l)/hx（OCFD_Schemes.cu:306），其中
    # hx = 1/(NX-1)（OCFD_init.cu:29-31，不是 1/NX）。JAX 算例缺省 dx=1/N，
    # 对拍必须显式覆盖，否则 du 就差 N/(N-1)（N=128 时 0.8%，必然超差）。
    dx_over = (getattr(args, "dx_x", None), getattr(args, "dx_y", None),
               getattr(args, "dx_z", None))
    dxs = tuple(d if o is None else o for d, o in zip(case["dxs"], dx_over))
    case["dxs"] = dxs
    # 对拍入口先覆盖步长，再构造描述，保证描述与后续 RHS 用同一组步长。
    desc = from_case(case)
    execution = boundary.build_execution(case, desc)
    grid = desc.grid
    dxs = desc.dxs
    bcs = execution.bcs
    params = case["params"]
    gamma = case["gamma"]

    # 初值 = 原版 step0 的守恒量（main_dump 现在 dump 的是 pf：rho, rho*u..., rhoE）
    cons = grid.alloc()
    for v, tag in enumerate(CONS_TAGS):
        path = os.path.join(args.ref_dir, f"cons_{tag}_step000000.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing reference init: {path}（先跑 compare.py dat2npz）")
        arr = np.load(path)["field"]
        if arr.shape != (grid.nk, grid.nj, grid.ni):
            raise ValueError(
                f"ref {tag} shape {arr.shape} != grid (nk,nj,ni)="
                f"{(grid.nk, grid.nj, grid.ni)}——原版 in 文件 GRID_3D 与 JAX 侧"
                f"网格不一致（用 --ni/--nj/--nk 对齐 in 文件的三维）")
        cons = cons.at[(v,) + grid.inner()[1:]].set(jnp.asarray(arr))

    metrics = desc.metrics

    def rhs(q):
        """按原版 du_comput 的口径组合无粘、粘性和体积力右端项。"""
        # 原码 du_comput 恒算无粘 + 粘性，这里不加 --viscous 条件
        q = execution.prepare(q)
        du = inviscid_rhs(q, grid, gamma=gamma, bcs=bcs,
                          axes=desc.axes, dx=dxs, scheme=args.scheme,
                          metrics=metrics, farfield=execution.farfield,
                          faces=execution.faces, fill_state_halos=False)
        du = du + viscous_rhs(q, grid, params, bcs, axes=desc.axes,
                              dx=dxs, order=vis_order, metrics=metrics,
                              farfield=execution.farfield,
                              faces=execution.faces, fill_state_halos=False)
        du = du + apply_forcing(case, q)
        return du

    rhs_j = jax.jit(rhs)
    rhs_j(cons)

    os.makedirs(args.out, exist_ok=True)
    q = cons
    dt = args.dt
    for step in range(1, args.steps + 1):
        # 与 solver.rk3_step 同一套三系数：du_krk1 = L(Q^n)、
        # du_krk2 = L(Q^*)、du_krk3 = L(Q^{**})，对应原版每个 RK 子步的 pdu
        q, dus = rk3_step_stages(q, dt, rhs_j)
        for krk, du in enumerate(dus, start=1):
            np.savez(os.path.join(args.out, f"jax_du_krk{krk}_step{step:06d}.npz"),
                     field=np.asarray(du[grid.inner()]))
        np.savez(os.path.join(args.out, f"jax_cons_step{step:06d}.npz"),
                 **{f"cons_{v}": np.asarray(q[grid.inner()][v]) for v in range(5)})
        print(f"jax step {step} done")
    print(f"jax-run -> {args.out}  (steps={args.steps}, dt={dt})")


def cmd_diff(args):
    """逐文件比较原版与 JAX 的 du/cons，报告最大误差、RMS、位置及总体判定。"""
    worst = 0.0
    worst_name = ""
    checked = 0
    for jpath in sorted(glob.glob(os.path.join(args.jax_dir, "jax_du_*.npz"))):
        m = re.search(r"jax_du_krk(\d)_step(\d+)", os.path.basename(jpath))
        krk, step = int(m.group(1)), int(m.group(2))
        ref_path = os.path.join(args.ref_dir, f"du_krk{krk}_step{step:06d}.npz")
        if not os.path.exists(ref_path):
            continue
        du_ref = np.load(ref_path)["field"]
        du_jax = np.load(jpath)["field"]
        if du_ref.shape != du_jax.shape:
            print(f"du krk{krk} step{step:<4d} SHAPE MISMATCH "
                  f"ref{du_ref.shape} vs jax{du_jax.shape} (skipped)")
            continue
        d = np.abs(du_ref - du_jax)
        pos = np.unravel_index(int(np.argmax(d)), d.shape)
        checked += 1
        print(f"du krk{krk} step{step:<4d} max|diff| = {d.max():.3e}  "
              f"rms = {np.sqrt((d ** 2).mean()):.3e}  at {pos}")
        if d.max() > worst:
            worst, worst_name = d.max(), f"du krk{krk} step{step}"
    for jpath in sorted(glob.glob(os.path.join(args.jax_dir, "jax_cons_*.npz"))):
        m = re.search(r"jax_cons_step(\d+)", os.path.basename(jpath))
        step = int(m.group(1))
        for v, tag in enumerate(CONS_TAGS):
            ref_path = os.path.join(args.ref_dir, f"cons_{tag}_step{step:06d}.npz")
            if not os.path.exists(ref_path):
                continue
            ref = np.load(ref_path)["field"]
            jax_ = np.load(jpath)[f"cons_{v}"]
            if ref.shape != jax_.shape:
                print(f"cons {tag} step{step:<4d} SHAPE MISMATCH "
                      f"ref{ref.shape} vs jax{jax_.shape} (skipped)")
                continue
            d = np.abs(ref - jax_)
            pos = np.unravel_index(int(np.argmax(d)), d.shape)
            checked += 1
            print(f"cons {tag} step{step:<4d} max|diff| = {d.max():.3e}  "
                  f"rms = {np.sqrt((d ** 2).mean()):.3e}  at {pos}")
            if d.max() > worst:
                worst, worst_name = d.max(), f"cons {tag} step{step}"
    if checked == 0:
        print("\n没有可比对的场：检查 ref/jax 目录的文件名与步号是否对齐")
        return 1
    verdict = ("PASS" if worst < 1e-12 else
               "FAIL（进入定位流程：换格式二分 / 查该场对应环节）")
    print(f"\ncompared {checked} fields, worst = {worst:.3e} @ {worst_name}\n"
          f"判定：{verdict}")
    return 0 if worst < 1e-12 else 1


def main():
    """建立 dat2npz、jax-run、diff 三个子命令并分发执行。"""
    p = argparse.ArgumentParser(description="opencfd 对拍工具")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("dat2npz")
    sp.add_argument("dump_dir", help="原版 dump/ 目录（main_dump 的文本）")
    sp.add_argument("--out", default="dump-npz")
    sp2 = sub.add_parser("jax-run")
    sp2.add_argument("--case", default="sod",
                     choices=["sod", "tg", "poiseuille", "channel"])
    sp2.add_argument("--in", dest="infile", default=None,
                     help="原码 in 文件（case=channel 必需）")
    sp2.add_argument("--field", default=None,
                     help="原码场文件（opencfd.dat / OCFD000xx.dat），channel 初场")
    sp2.add_argument("--mesh", default=None,
                     help="原码 OCFD3d-Mesh.dat；给出则按 JAC_BOUND=1 接入度量")
    sp2.add_argument("--ni", type=int, default=128)
    sp2.add_argument("--nj", type=int, default=8)
    sp2.add_argument("--nk", type=int, default=4,
                     help="k 向网格数，必须等于 in 文件 GRID_3D 的第三个数")
    sp2.add_argument("--steps", type=int, default=3)
    sp2.add_argument("--dt", type=float, default=0.001)
    sp2.add_argument("--Re", type=float, default=1000.0)
    sp2.add_argument("--Ama", type=float, default=0.1)
    sp2.add_argument("--ref-t", type=float, default=None,
                     help="有量纲参考温度(K)，必须与原码 in 文件的 T_REF 同值"
                          "（缺省 288.15 K；Ref_T 只影响 μ 的温度曲线，"
                          "μ(T=1)=1/Re 与它无关）")
    sp2.add_argument("--pr", type=float, default=0.7,
                     help="Prandtl 数，必须与原码 PR 同值")
    sp2.add_argument("--vis-order", default="CD6", choices=["CD6", "CD8"],
                     help="粘性导数格式，必须与原码 SCHEME_VIS 同值")
    sp2.add_argument("--forcing-work", default="off", choices=["on", "off"],
                     help="体积力能量源 rho*u*G。缺省 off = 与原码 channel 模块"
                          "（只做 du[1] += F*rho）逐项一致")
    for _a in "xyz":
        sp2.add_argument(f"--dx-{_a}", type=float, default=None,
                         help=f"覆盖 {_a} 轴物理步长。原码 hx=1/(NX-1)"
                              f"（OCFD_init.cu:29-31）；JAX 算例缺省 1/N，对拍必须显式给")
    sp2.add_argument("--p-grad", type=float, default=0.1)
    sp2.add_argument("--scheme", default="UP7",
                     help="无粘格式：UP7/WENO5/WENO7/WENO7-SYMBO/NND2/"
                          "OMP6-1/2/3，或原码 in 文件的 OMP6_HR/OMP6_LD/OMP6_CD8")
    sp2.add_argument("--viscous", action="store_true",
                     help="忽略：原码 du_comput 恒含粘性项，这里也恒加")
    sp2.add_argument("--fp64", action="store_true",
                     help="忽略：对拍恒为 FP64（x64 已在入口强制开启）")
    sp2.add_argument("--ref-dir", default="dump-npz",
                     help="原版 step0 初值所在目录（dat2npz 产物）")
    sp2.add_argument("--out", default="dump-jax")
    sp3 = sub.add_parser("diff")
    sp3.add_argument("ref_dir", help="原版 npz 目录（dat2npz 产物）")
    sp3.add_argument("jax_dir", help="JAX npz 目录（jax-run 产物）")
    args = p.parse_args()
    if args.cmd == "dat2npz":
        cmd_dat2npz(args)
    elif args.cmd == "jax-run":
        cmd_jax_run(args)
    elif args.cmd == "diff":
        sys.exit(cmd_diff(args))


if __name__ == "__main__":
    main()
