"""超声速等温层流的多站位对照（Astra 第八轮口径）：剖面、Cf(x)、时间漂移分开报告。

对比对象：JAX plate-CGNS 解（Ma2、Tw=T∞、Re=1000、Pr=0.72）与
CLBL_pr072 的相似解（Re=1e6/m、板长 0.01 m、终点 Re_x=1e4）。

相似坐标 η = y·sqrt(Re_x)/x。CLBL 是零压梯度相似解，其终点剖面可按 η
用于任意站位；Cf 按当地 Re_x 从 CF_BL.dat 插值。

分开记录三个量（避免把时间发展误判为空间误差）：
  Δref   = u(t) − u_CLBL(η)          对参考的偏差
  drift  = u(t) − u(t − 10)          同期时间漂移
  Cf     的提取误差单独用"CLBL 采样到 JAX 节点 + 同一差分"分离

用法（从归档快照复现 β=4.5 的报告，在 ocfd2jax 目录下）：
    python opencfd_jax/experiment/compare_supersonic_bl_stations.py \n        --npz-dir . --mesh opencfd_jax/Grid/testcgnstest_yb45.cgns \n        --steps 20000 40000 60000 --files \n        opencfd_jax/experiment/supersonic_bl_reference_20261123/jax_yb45_step20000_t10.npz \n        opencfd_jax/experiment/supersonic_bl_reference_20261123/jax_yb45_step40000_t20.npz \n        opencfd_jax/experiment/supersonic_bl_reference_20261123/jax_yb45_step60000_t30.npz

对尚在输出目录里的运行结果，用 --npz-dir <目录> --prefix <前缀> 代替 --files。
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO.parent))
from opencfd_jax.src import mesh_cgns  # noqa: E402

REF = REPO / "experiment" / "supersonic_bl_reference_20261123"
GAMMA, AMA, RE = 1.4, 2.0, 1000.0
CV = 1.0 / (GAMMA * (GAMMA - 1.0) * AMA ** 2)
TSB = 110.4 / 288.15


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--prefix", help="运行输出目录的文件前缀（随网格名）")
    ap.add_argument("--files", nargs="+",
                    help="显式快照路径，与 --steps 一一对应；给出时不看 --prefix"
                         "（归档快照已改名，用这个复现报告）")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[20000, 40000, 60000])
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--stations", type=float, nargs="+",
                    default=[7.98, 9.09, 9.49, 9.80, 10.0])
    args = ap.parse_args()
    if args.files:
        if len(args.files) != len(args.steps):
            ap.error("--files 与 --steps 数量必须一致")
        paths = {s: Path(f) for s, f in zip(args.steps, args.files)}
    elif args.prefix:
        paths = {s: Path(args.npz_dir) / f"{args.prefix}-istep-{s:08d}.npz"
                 for s in args.steps}
    else:
        ap.error("需要 --prefix 或 --files")
    if len(args.steps) < 2:
        ap.error("--steps 至少给两个时刻：时间漂移和 Cf 趋势要相邻两时刻才能报告")

    # --mesh 先按原样解析（绝对路径或相对当前目录），不存在再拼仓库目录
    mesh_path = Path(args.mesh)
    if not mesh_path.exists() and not mesh_path.is_absolute():
        mesh_path = REPO / args.mesh
    zone = mesh_cgns.read_cgns_zone(str(mesh_path))
    y = zone.coords["Ayy"]                      # 内区 (nk,nj,ni)，与 npz 同索引
    ni = y.shape[2]
    dx = 10.0 / (ni - 1)
    kc = y.shape[0] // 2
    dy_w = float(y[kc, 1, 0] - y[kc, 0, 0])

    def load(step):
        d = np.load(paths[step])
        rho, rhou, E = d["cons_0"], d["cons_1"], d["cons_4"]
        u = rhou / rho
        T = (E / rho - 0.5 * u * u) / CV
        p = (GAMMA - 1.0) * rho * CV * T
        return rho, u, T, p

    cbl = np.loadtxt(REF / "clbl_Re1e6_uprofile_final.dat", skiprows=2)
    eta_c, u_c = cbl[:, 0] * 1e-3 * 1e4, cbl[:, 2]
    cf_tab = np.loadtxt(REF / "CF_BL.dat", skiprows=2)
    x_mm, cf_c = cf_tab[:, 0], cf_tab[:, 1]
    i_of = [min(int(round(x / dx)), ni - 1) for x in args.stations]

    print(f"=== 健康（全场）===  dy_w={dy_w:.5f}")
    for s in args.steps:
        rho, u, T, p = load(s)
        c = np.sqrt(np.maximum(GAMMA * p / rho, 0.0))
        nu = (1.0 + TSB) / RE * T ** 1.5 / (TSB + T) / rho
        print(f" t={s*args.dt:5.2f}: 有限={bool(np.isfinite(u).all())} "
              f"rho_min={rho.min():.4f} T=[{T.min():.4f},{T.max():.4f}] "
              f"p_min={p.min():.5f} 扩散数max={float(nu.max())*args.dt/dy_w**2:.3f} "
              f"CFL_y={float((np.abs(u)+c).max())*args.dt/dy_w:.3f}")

    last, prev = args.steps[-1], args.steps[-2]
    _, uL, TL, _ = load(last)
    _, uP, _, _ = load(prev)

    print(f"\n=== u 剖面（k 中面，t={last*args.dt:g}）: JAX | Δref | drift(相对 t={prev*args.dt:g}) ===")
    for x, i in zip(args.stations, i_of):
        eta = y[kc, :, i] * np.sqrt(RE * x) / x
        print(f" 站位 x={x:5.2f} (Re_x={RE*x:6.0f})")
        for e in (0.10, 0.30, 0.75, 1.50, 2.80, 4.50):
            uj = float(np.interp(e, eta, uL[kc, :, i]))
            up = float(np.interp(e, eta, uP[kc, :, i]))
            uc = float(np.interp(e, eta_c, u_c))
            print(f"   η={e:4.2f}: {uj:.4f} | {uj-uc:+.4f} | {uj-up:+.4f}")

    print(f"\n=== 近壁前 4 层节点（JAX 节点上的 u 与 CLBL 同 η 值）===")
    for x, i in zip(args.stations[:3], i_of[:3]):
        eta = y[kc, :, i] * np.sqrt(RE * x) / x
        row = "  ".join(f"η={eta[j]:.3f}: {uL[kc, j, i]:.4f}/{np.interp(eta[j], eta_c, u_c):.4f}"
                        for j in range(1, 5))
        print(f" x={x:5.2f}  {row}")

    print(f"\n=== Cf(x) 相对 CLBL 当地值 ===")
    print(f"  {'x':>5} {'Re_x':>6} | {'JAX t='+str(prev*args.dt):>11} {'JAX t='+str(last*args.dt):>11} | "
          f"{'采样提取':>8}（CLBL→JAX 节点同差分）")
    for x, i in zip(args.stations, i_of):
        def cf(u, T):
            mu_w = (1.0 + TSB) / RE * T[kc, 0, i] ** 1.5 / (TSB + T[kc, 0, i])
            return 2.0 * mu_w * (u[kc, 1, i] - u[kc, 0, i]) / dy_w
        ref = float(np.interp(RE * x / 1e6 * 1000.0, x_mm, cf_c))
        eta = y[kc, :, i] * np.sqrt(RE * x) / x
        nodes = np.interp(eta, eta_c, u_c)
        cf_s = 2.0 * (1.0 / RE) * (nodes[1] - nodes[0]) / dy_w
        a, b = cf(uP, TL) / ref - 1, cf(uL, TL) / ref - 1
        print(f"  {x:5.2f} {RE*x:6.0f} | {a*100:+10.1f}% {b*100:+10.1f}% | {cf_s/ref*100-100:+7.1f}%")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 2026-11-23 β=4.5 长跑结果（t=30，60000 步，dt=5e-4；快照归档于
# experiment/supersonic_bl_reference_20261123/jax_yb45_step*_t*.npz）
#
# 运行：全程稳定（有限，rho_min≈0.62，扩散数 0.027，y 向 CFL 0.14）。
#
# 加密效果（内站 x=7.98 / 9.09，满足 Astra 第八轮判据）：剖面加密改变量
# 0.09～0.13，是同期时间漂移（≈0.04）的 3 倍以上，且方向趋向参考——
# Δref 由 β=3 的 −0.13～−0.18 降到 −0.015～−0.047；Cf 由 −77% / −66%
# 变为 −56% / −47%。内站剖面偏差主要来自近壁分辨率。
#
# 未闭合一：内站壁面第一层速度只有参考的 44%～71%（沿流向增大），Cf
# 偏低约一半，对加密不敏感（外层剖面则很快收敛）。CLBL 采样到同节点、
# 用同一差分只得 −0.3%，排除提取方法与网格表示能力。来源未定——壁面
# 边界带公式、粘性项近壁口径是候选，尚未做开关实验。
#
# 未闭合二：出口效应。x≥9.8 剖面偏差反号并增大，出口列 Cf +290%
# （β=3 时 +192%，加密后变大）。与路线图 4.9.9 记录的原码口径
# （出口不赋值、满模板读零鬼层）相符，但未单独验证。
#
# 时间漂移：每 10 个时间单位收缩到约 0.35 倍，但 t=30 时仍与 β=4.5 的
# 内站偏差同量级（≈0.04），且方向为继续偏离参考。三个时刻不足以确定
# 定常极限；当前的 Δref 不是定常偏差。
# ---------------------------------------------------------------------------
