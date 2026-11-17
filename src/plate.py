"""平板边界层边界条件：入口 Dirichlet 剖面 + 底板无滑移等温壁 + 顶部自由来流。

与 IBC=200 槽道（`channel.ChannelBC`）的区别在于 **y 两侧不对称**：
  - 下侧 y=0：无滑移等温壁 → 壁面层覆写 + 镜像鬼胞（与槽道下壁同一套公式，
    见 `channel._wall_layer` / `fields.fill_wall_y` 的下壁分支）；
  - 上侧 y=H：自由来流 → Dirichlet 常数（不是镜像、也不是零梯度）。

鬼胞归属（关键）：本模块的 `apply()` 在每子步算 du 之前把 x/y 两轴的物理鬼胞
**整体写入**，因此 `fields.fill_halos_x(bc=INFLOW)` 与
`fields.fill_halos_y(bc=PLATE)` 都是**保留不重写**的 no-op；
`fill_halos_z(PERIODIC)` 照常环绕。这与 `channel.apply_wall_bc`「只管 y、
x/z 由周期填充」不同，不能混用。

字段口径（对齐原码）：入口剖面 `inflow` 是**守恒量** (5, nz, ny)；
顶部来流 `far` 是 5 分量守恒量常数向量。
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
