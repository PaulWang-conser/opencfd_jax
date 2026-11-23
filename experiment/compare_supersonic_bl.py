"""超声速层流相似解对照：JAX (Ma2, Re_x=1e4) vs CLBL_pr072 终点剖面。

CLBL：Re=1e6/m、x=0.01 m → Re_x=1e4；y 有量纲（m），输出 mm。
JAX：Re=1000、x=10 → Re_x=1e4；y 即网格坐标（无量纲）。
相似变量统一 η = y·sqrt(Re_x)/x（两种单位各自换算后无量纲一致）。
"""
import sys

sys.path.insert(0, r"D:\Desktop\ocfd2jax")
import numpy as np

from opencfd_jax.src import config

config.enable_x64()

RE_X = 1.0e4
X_JAX, X_CBL = 10.0, 0.01          # 两边的特征长度（各自单位）
AMA, GAMMA = 2.0, 1.4
CV = 1.0 / (GAMMA * (GAMMA - 1.0) * AMA ** 2)

# ---- JAX 末态：x=10 列、k 中面（网格/末态可用环境变量切换） ----
import os

import h5py
from opencfd_jax.src import mesh_cgns

MESH = os.environ.get(
    "BL_MESH", r"D:\Desktop\ocfd2jax\opencfd_jax\Grid\testcgnstest.cgns")
NPZ = os.environ.get(
    "BL_NPZ",
    r"D:\Desktop\ocfd2jax\opencfd_jax\experiment\cases\supersonic-long-output"
    r"\testcgnstest-istep-00002000.npz")
zone = mesh_cgns.read_cgns_zone(MESH)
y_c = zone.coords["Ayy"][zone.coords["Ayy"].shape[0] // 2, :, 0]  # (nj,)
d = np.load(NPZ)
rho, rhou, rhov, E = (d[f"cons_{i}"] for i in (0, 1, 2, 4))
k_mid, i_last = rho.shape[0] // 2, rho.shape[2] - 1 - 4  # 内区末列
rho, rhou, rhov, E = (a[k_mid, :, i_last + 4] for a in (rho, rhou, rhov, E))
u_j = rhou / rho
v_j = rhov / rho
T_j = (E / rho - 0.5 * (u_j ** 2 + v_j ** 2)) / CV
eta_j = y_c * np.sqrt(RE_X) / X_JAX

# ---- CLBL 终点剖面 ----
cbl = np.loadtxt(r"D:\Desktop\clbl_run\uprofile.dat", skiprows=2)
y_cb, rho_c, u_c, v_c, T_c = cbl.T
eta_c = (y_cb * 1e-3) * np.sqrt(RE_X) / X_CBL      # mm → m

# ---- 相似坐标插值对比（eta 到 5 覆盖边界层）----
eta_grid = np.linspace(0.02, 5.0, 40)
uj_i = np.interp(eta_grid, eta_j, u_j)
uc_i = np.interp(eta_grid, eta_c, u_c)
Tj_i = np.interp(eta_grid, eta_j, T_j)
Tc_i = np.interp(eta_grid, eta_c, T_c)
du = np.abs(uj_i - uc_i)
dT = np.abs(Tj_i - Tc_i)
print(f"{'eta':>6} {'u_JAX':>8} {'u_CBL':>8} {'T_JAX':>8} {'T_CBL':>8}")
for i in range(0, len(eta_grid), 4):
    print(f"{eta_grid[i]:6.2f} {uj_i[i]:8.4f} {uc_i[i]:8.4f} "
          f"{Tj_i[i]:8.4f} {Tc_i[i]:8.4f}")
print(f"\nu 剖面：max|Δu|={du.max():.4f}（{du.max()/uc_i.max()*100:.1f}% of U∞），"
      f"RMS={np.sqrt((du**2).mean()):.4f}")
print(f"T 剖面：max|ΔT|={dT.max():.4f}，RMS={np.sqrt((dT**2).mean()):.4f}")

# ---- Cf 对比：壁面应力（JAX 一阶差分）与 CLBL 的 cf(x=终) ----
dy_w = y_c[1] - y_c[0]
mu_w = 1.0 / 1000.0                       # μ(T=1)=1/Re（Sutherland 设计值）
tau_w = mu_w * (u_j[1] - u_j[0]) / dy_w
cf_jax = 2.0 * tau_w
cf_cbl = np.loadtxt(r"D:\Desktop\clbl_run\CF_BL.dat", skiprows=2)[:, 1]
print(f"\nCf@Re_x=1e4：JAX={cf_jax:.6f}  CLBL={cf_cbl[-1]:.6f}  "
      f"相对差={(cf_jax-cf_cbl[-1])/cf_cbl[-1]*100:+.1f}%")
print(f"（参考：不可压 Blasius cf·sqrt(Re_x)=0.664；"
      f"JAX={cf_jax*np.sqrt(RE_X):.4f}  CLBL={cf_cbl[-1]*np.sqrt(RE_X):.4f}）")
# ---------------------------------------------------------------------------
# 2026-11-23 首次对照结论（Ma2 等温 Tw/T∞=1、Re_x=1e4）
#
# CLBL_pr072（jisen 提供，WSL gfortran -O2 -std=legacy 编译；输入口径：
# TINF 为 K、**TW 为无量纲 T_w/T∞**（默认 2.4765≈Ma2.87 绝热壁温可证，
# 程序提示语 "Twall(k)" 有误导）、Re 单位 /m、板长 m）。
# 终点 cf·sqrt(Re_x)=0.6548（不可压 Blasius 0.664 的 -1.4%，等温壁
# Ma2 可压修正量级合理）。
#
# JAX（dy≈0.103 均匀网格，δ99≈0.5 → 边界层仅 ~5 点）：
# u 剖面 max|Δu|=0.267（27% U∞，整体偏厚）、Cf=2.24·sqrt(Re_x)^-1
# （CLBL 的 3.4 倍）——近壁粘性底层完全无解析（第一点 η=0.02 处
# u=0.022 vs CLBL 0.0065）。诊断：**壁面分辨率不足**，非格式/边界
# 口径问题（壁面 u 严格 0、T/ρ 正、远场精确）。
#
# 定量验收的下一步：壁面加密网格（y 向 tanh 拉伸或 40→160 点，
# 保持 Laminar BL 的 y+<1 量级），CLBL 参考与对照脚本已就绪。
# ---------------------------------------------------------------------------
