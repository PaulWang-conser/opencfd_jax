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
# 以下数字经 Astra 第九轮复算核对（压力比按实际坐标插值复核）。
#
# 运行：全程稳定（有限，rho_min≈0.62，扩散数 0.027，y 向 CFL 0.14）。
#
# 内站剖面随加密明显改善：同刻加密改变量 0.09～0.13，是同期时间漂移的
# 2.5～3.4 倍（x=9.09 的 η=2.8 处 2.46 倍，x=7.98 处 3.39 倍），方向趋向
# 参考；Δref 由 β=3 的 −0.13～−0.18 降到 −0.015～−0.047。
#
# 未闭合一：内站壁面梯度。Cf 在 β=3 为 −77% / −66%，β=4.5 为 −56% / −47%
# （x=7.98 / 9.09），对加密不敏感，且仍在缓慢加深（β=4.5 的 x=7.98 由
# t=10 的 −40% 到 t=30 的 −56%，每单位时间约 0.45 个百分点）。前三层节点
# 速度只有参考的 44%～71%，第二层回升、第三层回落。提取方法与网格表示
# 已排除（CLBL 采样到同节点同差分 −0.3%；换非均匀三点二次导数仍为
# −61% / −52%）。来源未定。
#
# 未闭合二：出口附近。x=9.50 壁压已比来流低 5%（Cf 仍偏低 29%，两种偏差
# 抵消），x=9.80 起 Cf 反号，出口列壁压 0.635 p∞、Cf +290%。出口列 Cf 由
# β=3 的 +192% 到 β=4.5 的 +290%，其中含首层间距由 0.0168 变到 0.0055
# 的取样效应（u1/Δy1 由 9.5 升到 12.8），且两次长跑的 dt 相差十倍，
# 不能写成"反射增强"。与 4.9.9 记录的原码出口口径相符，未单独验证。
#
# 未闭合三：外层条件偏离参考。CLBL 是零压梯度相似解（Ue=1、dUe/dx=0），
# 而 JAX 流场内 x=2～6 的壁压比来流高 4%～9%（y=1 处高 5%～17%），
# 这一段的 Cf 共同偏低约 60%；顶面附近 y=3.56 的压力比在 x≈5.5 达 1.17。
# 出口扰动的逆行时间量级：到 x=9.09、7.98 约 2、4 个时间单位（按第一层
# 局地声速），所以内站是否已受出口影响尚未确定。
#
# 已排除：展向周期长度（跨 k 的 Cf 偏差范围 <0.05 个百分点）、壁温设定
# （Tw=1）与 Sutherland 常数之差（μ 相对差 ≤0.003%）。
#
# 时间漂移：每 10 个时间单位收缩到约 0.35 倍，t=30 时仍与内站偏差同量级，
# 且内站 u 持续下降（穿过参考值的时刻各站不同）。定常极限未定。
#
# 下一步（Astra 第九轮建议）：先在 β=3 做出口开关实验（只改出口四层
# 守恒状态的延拓，同时刻 A/B 对比，以 CLBL 为量尺），再对内站做同状态的
# 近壁算子诊断后决定壁面侧实验；直接关闭壁面边界带会同时改变无粘重构、
# 粘性梯度与粘性通量散度，不宜作为判别实验。
# ---------------------------------------------------------------------------
