"""出口状态鬼层开关实验（E1）的同时刻对比：默认 keep（A）对零梯度延拓 copy（B）。

两条轨迹同一网格、同一初场、同一时间步长，只差出口四层守恒状态鬼层的
处理。主判据是同时刻配对差，以 CLBL 当地值作量尺（公共参考不引入时间漂移）：

    D(x, t) = 100 · (Cf_B − Cf_A) / Cf_CLBL(x)          单位：百分点

另报逐列壁压比 p_w/p∞（CLBL 的参考值是 1）、第一层速度 u1；站位剖面见
`compare_supersonic_bl_stations.py`。Astra 第九轮定的两级验收：
  1. 出口列同刻的正偏差减少至少一半，且 x=9.80、9.90 的壁压与剪切同步向参考移动；
  2. 影响范围：连续两次读数 |D| ≥ 2 个百分点的最上游连续站位（并列出 1、5 的门槛）。
若最后两次读数之间 D 的变化仍超过其幅值的 20%，说明尚未定型，需要延长观察。

A 的后期时刻可以用归档快照代替，前提是 A 在 t=10 与归档逐位相同——脚本先核对这一点。

用法（从归档复现，在 ocfd2jax 目录下；快照在 experiment/e1_outlet_ghost_20261123/{A,B,C}，
A 的 t=20、30 取 experiment/supersonic_bl_reference_20261123 的归档）：
    python opencfd_jax/experiment/compare_outlet_ghost.py --c-dir \
        opencfd_jax/experiment/e1_outlet_ghost_20261123/C \
        --c-prefix testcgnstest_ystr_nr
C 的文件前缀来自派生网格 testcgnstest_ystr_nr.cgns，所以要给 --c-prefix；匹配不到
快照时报错并列出目录里现有的文件。只给一个时刻（如 --steps 2000）也做三方比较。
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO.parent))
from opencfd_jax.src import mesh_cgns  # noqa: E402

REF = REPO / "experiment" / "supersonic_bl_reference_20261123"
GAMMA, RE = 1.4, 1000.0
P_INF = 1.0 / (GAMMA * 4.0)
TSB = 110.4 / 288.15
ARCHIVE_A = {4000: "jax_ystr_step4000_t20.npz", 6000: "jax_ystr_step6000_t30.npz"}
KC = 20                                   # 展向中面
STATIONS = (2.02, 4.04, 5.96, 7.98, 9.09, 9.49, 9.80, 9.90, 10.0)


def load(path):
    d = np.load(path)
    rho = d["cons_0"]
    u = d["cons_1"] / rho
    p = (GAMMA - 1.0) * (d["cons_4"] - 0.5 * rho * u * u
                         - 0.5 * (d["cons_2"] ** 2 + d["cons_3"] ** 2) / rho)
    # T = (E/rho - 动能)/Cv，Cv = 1/(gamma*(gamma-1)*Ma^2)；等价于 p/(rho*R)
    T = (d["cons_4"] / rho - 0.5 * u * u
         - 0.5 * (d["cons_2"] ** 2 + d["cons_3"] ** 2) / rho) / (1.0 / (GAMMA * (GAMMA - 1.0) * 4.0))
    return rho, u, T, p


def bits(a):
    """数组的位型视图；逐位比较用（== 把 +0.0 与 -0.0 视为相同）。"""
    return np.ascontiguousarray(a).view(f"u{a.dtype.itemsize}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-dir", default=str(REPO / "experiment" / "e1_outlet_ghost_20261123" / "A"))
    ap.add_argument("--b-dir", default=str(REPO / "experiment" / "e1_outlet_ghost_20261123" / "B"))
    ap.add_argument("--prefix", default="testcgnstest_ystr")
    ap.add_argument("--mesh", default="Grid/testcgnstest_ystr.cgns")
    ap.add_argument("--steps", type=int, nargs="+", default=[1000, 2000, 4000, 6000])
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--c-dir", default=None,
                    help="可选：第三条轨迹（无反射 nonreflecting）的快照目录")
    ap.add_argument("--c-prefix", default=None,
                    help="第三条轨迹的文件前缀（缺省同 --prefix；换过网格时文件名会不同）")
    args = ap.parse_args()

    mesh_path = Path(args.mesh)
    mesh_path = mesh_path if mesh_path.exists() else REPO / args.mesh
    y = mesh_cgns.read_cgns_zone(str(mesh_path)).coords["Ayy"]
    ni, dx = y.shape[2], 10.0 / (y.shape[2] - 1)
    dy_w = float(y[KC, 1, 0] - y[KC, 0, 0])
    x = np.arange(ni) * dx
    cf_tab = np.loadtxt(REF / "CF_BL.dat", skiprows=2)
    cf_ref = np.interp(RE * x / 1e6 * 1000.0, cf_tab[:, 0], cf_tab[:, 1])

    def snap(step):
        # A 的后期时刻优先用归档快照（t=10 与归档逐位相同时才允许）
        a = Path(args.a_dir) / f"{args.prefix}-istep-{step:08d}.npz"
        if not a.exists() and step in ARCHIVE_A:
            a = REF / ARCHIVE_A[step]
        return a, Path(args.b_dir) / f"{args.prefix}-istep-{step:08d}.npz"

    # A(t=10) 与归档逐位核对：通过才允许后期时刻使用归档
    a10, _ = snap(2000)
    fresh = np.load(Path(args.a_dir) / f"{args.prefix}-istep-00002000.npz")
    arch = np.load(REF / "jax_ystr_step2000.npz")
    # 逐位：先比 dtype 与形状，再比位型。
    keys = [f"cons_{i}" for i in range(5)]
    same = all(fresh[k].dtype == arch[k].dtype and fresh[k].shape == arch[k].shape
               and np.array_equal(bits(fresh[k]), bits(arch[k])) for k in keys)
    print(f"A(t=10) 与归档逐位相同: {same}")
    if not same:
        raise SystemExit("A(t=10) 与归档不一致：后期时刻的归档不可混用，中止")

    def wall(q):
        rho, u, T, p = q
        mu_w = (1.0 + TSB) / RE * T[KC, 0, :] ** 1.5 / (TSB + T[KC, 0, :])
        return 2.0 * mu_w * (u[KC, 1, :] - u[KC, 0, :]) / dy_w, p[KC, 0, :] / P_INF, u[KC, 1, :]

    table = {}
    for step in args.steps:
        fa, fb = snap(step)
        if not fb.exists():
            print(f"t={step*args.dt:g}: B 快照缺失，跳过")
            continue
        table[step] = (wall(load(fa)), wall(load(fb)))

    print("\n=== Cf 相对 CLBL（%）：A=keep  B=copy  D=B−A（百分点）===")
    print("  t    x    Cf_A    Cf_B       D   | p_w/p∞ A→B | u1 A→B")
    for step, (A, B) in table.items():
        t = step * args.dt
        eA = (A[0] / cf_ref - 1) * 100
        eB = (B[0] / cf_ref - 1) * 100
        D = eB - eA
        for xs in STATIONS:
            i = int(round(xs / dx))
            print(f" {t:4.1f} {xs:5.2f} {eA[i]:+7.1f} {eB[i]:+7.1f} {D[i]:+8.1f}   |"
                  f" {A[1][i]:.3f}→{B[1][i]:.3f} | {A[2][i]:.4f}→{B[2][i]:.4f}")
        print()

    steps = sorted(table)

    def extent(D, thr):
        """从出口向上游的连续达标段：返回 (最上游达标列, 段外孤立点数)。"""
        i = ni - 1
        while i >= 0 and abs(D[i]) >= thr:
            i -= 1
        return i + 1, int(np.count_nonzero(np.abs(D[:i + 1]) >= thr))

    if len(steps) >= 2:
        Ds = {s: (table[s][1][0] - table[s][0][0]) / cf_ref * 100 for s in steps}
        print("=== 影响范围：|D| 达到门槛的、从出口向上游的连续段 ===")
        for thr in (1.0, 2.0, 5.0):
            heads, isolated = {}, {}
            for s in steps:
                heads[s], isolated[s] = extent(Ds[s], thr)
            row = []
            for s in steps:
                h = heads[s]
                row.append(f"t={s*args.dt:g}: " + (f"x≥{x[h]:.2f}（{ni - h} 列）" if h < ni
                                                   else "无")
                           + (f"＋{isolated[s]} 个孤立点" if isolated[s] else ""))
            # 连续两次读数都达标的范围：取两者中较短的那段
            h2 = max(heads[prev_t] for prev_t in steps[-2:])
            both = (f"x≥{x[h2]:.2f}" if h2 < ni else "无") if all(
                heads[s] < ni for s in steps[-2:]) else "无（最近两次读数未同时达标）"
            print(f"  门槛 {thr:.0f} 个百分点：最近两次读数共同的范围 {both}")
            for r in row:
                print(f"      {r}")

        last, prev = steps[-1], steps[-2]
        change = np.abs(Ds[last] - Ds[prev])
        mag = np.abs(Ds[last])
        rel = np.full(change.shape, np.nan)
        big = mag > 1.0
        rel[big] = change[big] / mag[big]
        i10 = ni - 1
        print(f"\n=== 定型检查（t={prev*args.dt:g} → {last*args.dt:g}）===")
        print(f"  出口列: D {Ds[prev][i10]:+.1f} → {Ds[last][i10]:+.1f}，"
              f"变化占幅值 {change[i10]/mag[i10]*100:.3f}%（<20% 才算定型）")
        for xs in (7.98, 9.09, 9.49, 9.80, 9.90):
            i = int(round(xs / dx))
            print(f"  x={xs:5.2f}: D {Ds[prev][i]:+6.1f} → {Ds[last][i]:+6.1f}"
                  f"（变化 {change[i]/mag[i]*100 if mag[i] > 1 else float('nan'):6.3f}%）")
        worst = np.nanargmax(np.where(mag > 5.0, rel, np.nan)) if (mag > 5.0).any() else None
        if worst is not None:
            print(f"  |D|>5 的站位中变化比例最大：x={x[worst]:.2f}，{rel[worst]*100:.3f}%")
    # 第三条轨迹 C=nonreflecting（原码 NON_REFLETION 口径），与 A 同时刻对比。
    # 单时刻也做；显式给了 --c-dir 却一个快照都没匹配到时报错，不静默只打印 A/B。
    c_dir = Path(args.c_dir) if args.c_dir else None
    rows_c = []
    if c_dir:
        c_prefix = args.c_prefix or args.prefix
        for step in steps:
            fc = c_dir / f"{c_prefix}-istep-{step:08d}.npz"
            if fc.exists():
                rows_c.append((step, wall(load(fc))))
        if not rows_c:
            have = sorted(p.name for p in c_dir.glob("*.npz"))
            raise SystemExit(
                f"--c-dir {c_dir} 下没有前缀为 {c_prefix!r}、时刻为 {steps} 的快照；"
                f"目录里现有：{have or '无 .npz'}。换过网格时文件前缀会变，请给 --c-prefix")
    if rows_c:
        print("\n=== C=nonreflecting（原码 NON_REFLETION 口径）：Cf 相对 CLBL，DC=C−A ===")
        print("  t    x    Cf_A    Cf_C      DC   | p_w/p∞ A→C | u1 A→C")
        Dc_all = {}
        for step, C in rows_c:
            A = table[step][0]
            eA = (A[0] / cf_ref - 1) * 100
            eC = (C[0] / cf_ref - 1) * 100
            Dc = eC - eA
            Dc_all[step] = Dc
            for xs in STATIONS:
                i = int(round(xs / dx))
                print(f" {step*args.dt:4.1f} {xs:5.2f} {eA[i]:+7.1f} {eC[i]:+7.1f} {Dc[i]:+8.1f}   |"
                      f" {A[1][i]:.3f}→{C[1][i]:.3f} | {A[2][i]:.4f}→{C[2][i]:.4f}")
            print()
        print("=== DC 的影响范围（从出口向上游的连续段）===")
        for thr in (2.0, 5.0):
            parts = []
            for step in steps:
                if step not in Dc_all:
                    continue
                h, iso = extent(Dc_all[step], thr)
                parts.append(f"t={step*args.dt:g}: " + (f"x≥{x[h]:.2f}（{ni - h} 列）"
                                                       if h < ni else "无")
                             + (f"＋{iso} 个孤立点" if iso else ""))
            print(f"  门槛 {thr:.0f} 个百分点：" + "；".join(parts))
        last_c = rows_c[-1][0]
        prev_c = [st for st, _ in rows_c if st < last_c]
        if prev_c and prev_c[-1] in Dc_all:
            a_, b_ = Dc_all[prev_c[-1]], Dc_all[last_c]
            print(f"=== DC 定型检查（t={prev_c[-1]*args.dt:g} → {last_c*args.dt:g}）===")
            for xs in (7.98, 9.09, 9.49, 9.80, 9.90, 10.0):
                i = int(round(xs / dx))
                r = abs(b_[i] - a_[i]) / abs(b_[i]) * 100 if abs(b_[i]) > 1 else float("nan")
                print(f"  x={xs:5.2f}: DC {a_[i]:+7.1f} → {b_[i]:+7.1f}（变化 {r:6.3f}%）")



if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 2026-11-23 E1 结果（β=3 网格、dt=0.005；A=keep 冷启动到 t=10，与归档 t=10
# 逐位相同后 t=20、30 用归档；B=copy 冷启动到 t=30）
#
# 出口零状态鬼层是出口附近偏差的来源，这一点成立：
# - 壁压 p_w/p∞（x=9.09 / 9.49 / 9.80 / 9.90 / 10，t=30）：
#       A  0.956 / 0.895 / 0.784 / 0.700 / 0.543
#       B  1.016 / 1.019 / 1.020 / 1.020 / 1.025
# - 第一层速度 u1：A 0.0197 / 0.0295 / 0.0557 / 0.0816 / 0.1599
#                 B 0.0117 / 0.0112 / 0.0109 / 0.0104 / 0.0104
#   （与上游内站同量级，沿程连续）
# - Cf 相对 CLBL：出口列 A +191.5% → B −81.1%；B 在 x=7.98～10 都是
#   −79%～−81%，即出口的偏高消失，露出与内站相同的偏低。
# 第一级验收（Astra 第九轮原定"正偏差至少减半且壁压与剪切同步趋近"）只部分
# 满足：出口列 Cf 正向尖峰消失、9.80/9.90 壁压向 1 移动；但 9.80/9.90 的 Cf
# 绝对偏差增大（A 中 +0.5%/+48.0% → B 中 −80.4%/−81.2%）——A 里出口偏高与
# 内站偏低相互抵消，A/B 两条轨迹不足以独立辨识这两项（Astra 第十轮复算）。
#
# 影响范围（|D| ≥ 2 个百分点，连续两次读数共同达标）：t=20 与 t=30 均为
# x≥8.18（t=30 单独为 x≥7.98）；上游边缘尚未定型（x=7.98 的 D 在 t=20→30
# 变化 36%）。x ≤ 6 的 |D| 最大 1.63 个百分点（位于 x=0.30；x=2.02/4.04/5.96
# 分别为 −0.069/+0.089/−0.084）。出口列 D −271.928 → −272.629（变化 0.257%）；
# x=9.09～9.90 变化 1%～12%。
#
# 由此：出口零鬼层足以解释出口列 Cf 的正向尖峰与近出口壁压异常；内站壁面
# 梯度偏低（x=2～6 约 −75%～−92%）不因此解释。E1 检验的是普通出口
# keep/copy，无反射本身的效果见下方 C 轨迹。
#
# C=nonreflecting（原码 NON_REFLETION 口径；β=3 网格的出口边条 dom-5 改为
# BCOutflowSupersonic，跑到 t=10；快照 e1_outlet_ghost_20261123/C/）。
# 来源：网格由 Grid/override_cgns_bc.py 的缺省参数从 testcgnstest_ystr.cgns 派生
#   源文件 SHA-256       4b0841427dc7964d38f847a6314026f9c0f9b5ecdc03fc3fabc5746267104217
#   产物逻辑内容 SHA-256 db551205ec45c82850609754fe6f9e867667bdd49673dd86377e6bc2a3606b14
# （派生网格不入库。产物的文件字节哈希不稳定——HDF5 把父组修改时间写进对象头，
# 所以核对逻辑内容哈希），配置 experiment/cases/e1_outlet_nonreflecting.json；按此
# 重跑，第 1000、2000 步与归档快照逐位相同。
# 此前写的“x≤7.98 与 A 逐位相同”不成立（Astra 第十一轮更正）：该段 640000 个值里
# 606627 个位型不同。逐列取比值 max|C−A|/max|A|（五个守恒量、全部 j 和 k 上取最大），
# 该段最大为 6.13e-4（x=7.68），x=7.98 处为 4.72e-4。
# 以 CLBL 的 Cf 为量尺、2 个百分点为门槛：t=5 为 x≥9.09，t=10 为 x≥8.79
# （copy 同期 x≥8.69）。出口列 t=10：壁压比 0.564→0.955、第一层速度
# 0.165→0.022、Cf +200.4%→−59.9%；对照 copy（1.003 / 0.017 / −68.9%），
# 壁压是 copy 更接近 1，Cf 的绝对偏差是无反射更小。未测反射系数。
# ---------------------------------------------------------------------------
