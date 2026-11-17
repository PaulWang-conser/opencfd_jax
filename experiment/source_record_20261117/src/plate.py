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

from .channel import primitive
from .fields import LAP, _WALL_SIGN


def far_state(rho=1.0, u=1.0, v=0.0, w=0.0, T=1.0, Cv=1.0):
    """自由来流的守恒量常数向量 (rho, rho*u, rho*v, rho*w, rho*E)。"""
    E = Cv * T + 0.5 * (u * u + v * v + w * w)
    return jnp.asarray([rho, rho * u, rho * v, rho * w, rho * E], dtype=jnp.float64)


def apply(cons, grid, phys, inflow, far, Tw=1.0):
    """写入平板的三类鬼胞/壁面层，返回新 cons。

    cons: (5, nz+8, ny+8, nx+8)；入口、壁面和顶层均包含物理点。
    inflow: (5, nz, ny) 入口守恒量剖面；far: (5,) 顶部来流常数。
    """
    nx, ny = grid.nx, grid.ny
    ny_in = LAP + ny
    nz_in = LAP + grid.nz

    # 1) x 两侧：左 = 入口 Dirichlet（含首个物理点），右 = 零梯度鬼层
    cons = cons.at[:, LAP:nz_in, LAP:ny_in, :LAP + 1].set(inflow[:, :, :, None])
    cons = cons.at[:, :, :, LAP + nx:].set(cons[:, :, :, LAP + nx - 1:LAP + nx])

    # 2) 下壁：壁面层（rho 由 ρT 两点外推、无滑移、ρE = ρ Cv Tw）
    rho, _, T = primitive(cons, phys)
    # IBC=124 的等温壁使用两点压力外推，系数与槽道边界不同。
    rho_w = (4.0 * rho[:, LAP + 1, :] * T[:, LAP + 1, :]
             - rho[:, LAP + 2, :] * T[:, LAP + 2, :]) / (3.0 * Tw)
    cons = cons.at[0, :, LAP, :].set(rho_w)
    cons = cons.at[1:4, :, LAP, :].set(0.0)
    cons = cons.at[4, :, LAP, :].set(rho_w * phys.Cv * Tw)

    # 3) 下壁镜像鬼胞：ghost[p] = f[2LAP-p] * sign（速度反号、rho/ρE 同号）
    sign = jnp.array(_WALL_SIGN).reshape(5, 1, 1, 1)
    cons = cons.at[:, :, :LAP, :].set(cons[:, :, 2 * LAP:LAP:-1, :] * sign)

    # 4) 顶部：自由来流 Dirichlet（全 x，含角鬼胞）
    cons = cons.at[:, :, ny_in - 1:, :].set(far.reshape(5, 1, 1, 1))
    return cons


def inflow_from_profile(rho, u, v, w, T, Cv):
    """(rho, u, v, w, T) 剖面（形状 (nz,ny)）→ 守恒量 (5, nz, ny)。"""
    rho = np.asarray(rho)
    E = Cv * np.asarray(T) + 0.5 * (np.asarray(u) ** 2 + np.asarray(v) ** 2
                                    + np.asarray(w) ** 2)
    return jnp.asarray(np.stack([rho, rho * u, rho * v, rho * w, rho * E]), dtype=jnp.float64)
