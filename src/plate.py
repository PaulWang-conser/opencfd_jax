"""平板边界层：底板无滑移等温壁 + 入口剖面 + 顶部单点远场。

执行顺序由 `boundary.BoundaryExecution.prepare` 统一编排：

  1. **鬼层**（`fill_faces`）：周期面复制对侧，INFLOW 保持，其余面（壁面、
     出口、远场）一律清零——与原码「鬼胞没人赋值就是 0」一致；
  2. **边界点**：入口由通用 INFLOW 写（边界点 + 四层鬼层 = 同一剖面），顶部
     远场由通用 FARFIELD 只写最后一层内点，出口不赋值；
  3. **特殊回调**（本模块 `apply`）：只写下壁的壁面点——ρ 由 ρT 两点外推、
     动量清零、ρE = ρ Cv Tw。回调不碰任何鬼层，也不写入口/顶部/出口。

边界带（`DIF_BOUND`）由面的 kind 派生：非周期且非 INFLOW 才开单侧带；特殊
回调继承该面的 band，自己不设。

字段口径（对齐原码）：入口剖面 `inflow` 是**守恒量** (5, nz, ny)，可以是
均匀来流常数，也可以来自 OCFD 1D 剖面文件（见 `inflow.py`）。
"""

import jax.numpy as jnp
import numpy as np

from .boundary import apply_plate_bc as apply


def far_state(rho=1.0, u=1.0, v=0.0, w=0.0, T=1.0, Cv=1.0):
    """自由来流的守恒量常数向量 (rho, rho*u, rho*v, rho*w, rho*E)。"""
    E = Cv * T + 0.5 * (u * u + v * v + w * w)
    return jnp.asarray([rho, rho * u, rho * v, rho * w, rho * E], dtype=jnp.float64)


def inflow_from_profile(rho, u, v, w, T, Cv):
    """(rho, u, v, w, T) 剖面（形状 (nz,ny)）→ 守恒量 (5, nz, ny)。"""
    rho = np.asarray(rho)
    E = Cv * np.asarray(T) + 0.5 * (np.asarray(u) ** 2 + np.asarray(v) ** 2
                                    + np.asarray(w) ** 2)
    return jnp.asarray(np.stack([rho, rho * u, rho * v, rho * w, rho * E]), dtype=jnp.float64)
