"""IBC=200 槽道流边界与源项（对应原码 src/channel_flow.cu）。

原码结构（OCFD_NS_Solver.cu:77-92，每个 RK 子步）：

    du_comput(KRK)  →  channel_add_body_force()  →  OCFD_time_advance(KRK)
    →  get_duvwT()（cons→pri）  →  OCFD_bc()（case 200 → bc_user_channel()＋pri→cons）

`bc_user_channel()` 作用在**原始变量** (d,u,v,w,T) 上，改的是
"壁面那一层物理点"和它的镜像鬼胞：

- 壁面位置：下壁 j=LAP（`ny_lap = ny+LAP`，物理区 j=LAP..ny+LAP-1），
  上壁 j=ny+LAP-1——即**两端壁面都是物理点，不是半格**。
- 无滑移：`u=v=w=0`。
- 壁面压力：三点单侧外推（零壁法向压力梯度）
      pw = (18·ρT[j±1] − 9·ρT[j±2] + 2·ρT[j±3]) / 11
  （外推在 ρT 上做，1/(γMa²) 是公因子，等价于外推 p）
- 等温壁（Tw>0）：T_wall = Tw、ρ_wall = pw/Tw；绝热壁（Tw≤0）：T 也三点外推。
- 镜像鬼胞：ρ、T 偶对称；u、v、w 奇对称（等价于守恒量里
  ρ/ρE 同号、ρu/ρv/ρw 反号，正是 `fields.fill_wall_y` 的 `_WALL_SIGN`）。

守恒量 ↔ 原始变量用与原码同一条链：`T=(ρE−½ρV²)/(ρ Cv)`、`p=ρT/(γMa²)`。

源项 `channel_body_force_kernel` **只加动量**：`du[1] += F·ρ`（没有 ρuF 做功项，
与 GPU 上那套 80000 步的算例逐项一致；要开做功项用 `work=True`）。
"""

import jax.numpy as jnp

from .fields import LAP, fill_wall_y


def primitive(cons, phys):
    """守恒量 → 原始变量 (rho, u(3), T)，对应原码 cons_to_pri_kernel。

    除法用 `rho>0` 保护：新 alloc 的场鬼胞是 0，直接除会出 NaN/inf。
    真实计算里调用前 x/z 鬼胞已由周期填充（见 apply_wall_bc 说明）。
    """
    rho = cons[0]
    rho_safe = jnp.where(rho > 0, rho, 1.0)
    u = cons[1:4] / rho_safe
    T = (cons[4] - 0.5 * rho_safe * jnp.sum(u * u, axis=0)) / (rho_safe * phys.Cv)
    return rho, u, T


def _wall_layer(rho, T, j, step, tw):
    """壁面那一层：返回 (rho_wall, T_wall)，均为 (nz, nx) 形状的场。

    j: 壁面所在 y 索引；step=+1 表示向 +y（下壁），-1 表示向 −y（上壁）。
    """
    j1, j2, j3 = j + step, j + 2 * step, j + 3 * step
    pt1 = rho[:, j1, :] * T[:, j1, :]
    pt2 = rho[:, j2, :] * T[:, j2, :]
    pt3 = rho[:, j3, :] * T[:, j3, :]
    pw = (18.0 * pt1 - 9.0 * pt2 + 2.0 * pt3) / 11.0
    if tw > 0.0:
        Tw = jnp.full_like(pw, tw)          # 等温（冷）壁
    else:
        Tw = (18.0 * T[:, j1, :] - 9.0 * T[:, j2, :]
              + 2.0 * T[:, j3, :]) / 11.0   # 绝热壁：T 也三点外推
    # 角鬼胞（x/z 同时是鬼胞）上 ρT=0，绝热外推会给出 Tw=0；用保护分母避免 NaN，
    # 这些位置随后会被 x/z 周期填充覆盖，不影响内区。
    return pw / jnp.where(Tw > 0, Tw, 1.0), Tw


def apply_wall_bc(cons, grid, phys, Tw_bottom=1.0, Tw_top=1.0):
    """把 IBC=200 的双壁条件写进 cons（壁面层 + 镜像鬼胞），返回新 cons。

    调用时机与原码一致：每个 RK 子步 `du_comput` **之前**（等价于原码
    `get_duvwT()`+`OCFD_bc()` 在上一子步末尾做的事）。x/z 鬼胞应已由周期
    填充（`fields.fill_halos_x/z`）；本函数只管 y 向壁面与镜像。
    """
    ny = grid.ny
    rho, _, T = primitive(cons, phys)
    Cv = phys.Cv

    for j, step, tw in ((LAP, +1, Tw_bottom), (LAP + ny - 1, -1, Tw_top)):
        rho_w, Tw = _wall_layer(rho, T, j, step, tw)
        cons = cons.at[0, :, j, :].set(rho_w)                       # rho
        cons = cons.at[1:4, :, j, :].set(0.0)                      # 无滑移
        cons = cons.at[4, :, j, :].set(rho_w * Cv * Tw)            # ρE = ρ Cv T

    # 镜像鬼胞：ρ/ρE 偶、ρu/ρv/ρw 奇（与原码 for i in 0..LAP-1 / 1..LAP 同索引）
    return fill_wall_y(cons, ny)


def cons_from_primitive(prim, grid, phys):
    """原码场文件里的 (rho, U, V, W, T) → 守恒量 cart（内区），鬼胞留 0。"""
    rho, u, v, w, T = prim
    sp = grid.inner()[1:]
    cons = grid.alloc()
    cons = cons.at[(0,) + sp].set(rho)
    cons = cons.at[(1,) + sp].set(rho * u)
    cons = cons.at[(2,) + sp].set(rho * v)
    cons = cons.at[(3,) + sp].set(rho * w)
    E = phys.Cv * T + 0.5 * (u * u + v * v + w * w)
    cons = cons.at[(4,) + sp].set(rho * E)
    return cons


def body_force(cons, grid, F, work=False):
    """恒定体积力源项：du[1] += F·ρ（原码 channel_body_force_kernel，仅动量）。

    work=True 时额外加 du[4] += F·ρu（可压 NS 的标准做功项）——原码没有这一项，
    对拍必须用 work=False。
    """
    du = jnp.zeros_like(cons)
    if F == 0.0:
        return du
    du = du.at[1].set(F * cons[0])
    if work:
        du = du.at[4].set(F * cons[1])
    return du


class ChannelBC:
    """IBC=200 的参数包（对应原码 channel_bc_parameter() 读 BC_RPARA 的三个值）。"""

    def __init__(self, Tw_bottom=1.0, Tw_top=1.0, F=0.0, work=False):
        """保存上下壁温度、流向体积力强度及是否加入体积力做功项。"""
        self.Tw_bottom = float(Tw_bottom)
        self.Tw_top = float(Tw_top)
        self.F = float(F)
        self.work = bool(work)

    def __repr__(self):
        """返回包含全部边界参数的调试字符串。"""
        return (f"ChannelBC(Tw_bottom={self.Tw_bottom}, Tw_top={self.Tw_top}, "
                f"F={self.F}, work={self.work})")

    @classmethod
    def from_infile(cls, inp, work=False):
        """从 in 文件解析出的 dict 建：需要 IBC=200 与 BC_RPARA=[Tw_bottom Tw_top F]。"""
        from . import infile
        if "IBC" not in inp or infile.as_int(inp["IBC"]) != 200:
            return None
        if "BC_RPARA" not in inp:
            raise ValueError("IBC=200 但 in 文件没有 BC_RPARA（原码会打印警告并"
                             "关掉 channel BC）")
        tp = infile.as_floats(inp["BC_RPARA"])
        if len(tp) < 3:
            raise ValueError(f"BC_RPARA 需要 3 个值 Tw_bottom/Tw_top/F，得到 {tp}")
        return cls(tp[0], tp[1], tp[2], work=work)

    def apply(self, cons, grid, phys):
        """把当前参数对应的双壁边界条件写入守恒量场。"""
        return apply_wall_bc(cons, grid, phys, self.Tw_bottom, self.Tw_top)

    def source(self, cons, grid):
        """根据当前状态计算槽道体积力产生的动量源和可选能量源。"""
        return body_force(cons, grid, self.F, self.work)
