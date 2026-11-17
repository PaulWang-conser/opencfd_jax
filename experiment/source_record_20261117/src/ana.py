"""诊断分析（对应 OCFD_ana.cu 的常用子集）。

- total_energy / residual：对应 ana_residual（ρE 内区平均，规约后除全局
  cell 数；主循环里 E0_d 直接指向 pf 的第 5 分量，即 rho*E）。
- check_nan：对应 ana_NAN_and_NT 的 NAN 检查部分。
"""

import numpy as np


def total_energy(cons, grid):
    """平均总能量 = mean(rho*E)（内区）。对应 ana_residual。"""
    return float(np.asarray(cons[grid.inner()][4]).mean())


def check_nan(cons):
    """全场（含鬼胞）NaN 检查。返回 True 表示出现 NaN。"""
    return bool(np.isnan(np.asarray(cons)).any())


def diagnostic_line(cons, grid, t, step, wall_time):
    """对齐原码主循环打印的一行诊断（t、步数、平均能量）。"""
    e0 = total_energy(cons, grid)
    return f"{t:.6f} of ( step {step} ), wall {wall_time:.2f}s, avg(rho*E) = {e0:.10e}, nan = {check_nan(cons)}"
