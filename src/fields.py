"""场布局与鬼胞管理。

与 OpenCFD 对齐（head/config_parameters.h: LAP=4, NVARS=5）：
守恒变量 SoA，shape (NVARS, nz+2*LAP, ny+2*LAP, nx+2*LAP)，
变量顺序 0:rho, 1:rho*u, 2:rho*v, 3:rho*w, 4:rho*E，axis=-1 为 x 方向。

边界填充支持周期、零梯度出流、槽道壁面，以及平板入口/壁面/远场的保留式
接口；专用物理边界分别由 ``channel.py`` 和 ``plate.py`` 写入。
"""

import jax.numpy as jnp

LAP = 4
NVARS = 5


def spatial_array_axis(axis):
    """将计算方向 (i,j,k) 的 0,1,2 映射到标量数组 (k,j,i) 的轴。"""
    return 2 - axis

# 边界类型
PERIODIC = "periodic"
OUTFLOW = "outflow"
# 平板边界层的两侧非对称边界（见 plate.py）：
#   INFLOW：x 下侧 Dirichlet 剖面 + 上侧零梯度——鬼胞由 plate.apply 写入，
#           这里的填充函数只**保留**，不重写（否则入口剖面会被邻格复制冲掉）。
#   PLATE ：y 下侧无滑移等温壁（镜像）+ 上侧自由来流 Dirichlet——同上，只保留。
INFLOW = "inflow"
PLATE = "plate"


class Grid:
    """结构网格：内区 N 个 cell，全场 N+2*LAP（含鬼胞）。"""

    def __init__(self, nx, ny, nz):
        """记录三个方向的内区网格数，并推导带四层鬼胞的完整数组形状。"""
        self.nx, self.ny, self.nz = nx, ny, nz
        self.shape = (NVARS, nz + 2 * LAP, ny + 2 * LAP, nx + 2 * LAP)
        self.n_inner = (nz, ny, nx)
        # 各轴全长（含鬼胞），按轴顺序 (z, y, x) 与 shape 对齐
        self._n_full = (nz + 2 * LAP, ny + 2 * LAP, nx + 2 * LAP)

    def inner(self):
        """内区 slice（含变量轴）。"""
        return (slice(None),
                slice(LAP, LAP + self.nz),
                slice(LAP, LAP + self.ny),
                slice(LAP, LAP + self.nx))

    def alloc(self, dtype=None):
        """分配全零守恒量场，形状为 ``(5, nz+8, ny+8, nx+8)``。"""
        return jnp.zeros(self.shape, dtype=dtype)


def fill_halos_x(f, bc, nx):
    """沿 x（axis=-1）填充鬼胞。f: (..., nx+2*LAP)。"""
    if bc == PERIODIC:
        f = f.at[..., :LAP].set(f[..., nx:nx + LAP])
        f = f.at[..., LAP + nx:].set(f[..., LAP:LAP + LAP])
    elif bc == OUTFLOW:
        f = f.at[..., :LAP].set(f[..., LAP:LAP + 1])
        f = f.at[..., LAP + nx:].set(f[..., LAP + nx - 1:LAP + nx])
    elif bc == INFLOW:
        pass   # 由 plate.apply 每子步写入入口剖面与出口零梯度，此处保留原值
    else:
        raise ValueError(f"unknown bc: {bc}")
    return f


def fill_halos_y(f, bc, ny):
    """沿 y（axis=-2）填充鬼胞。"""
    if bc == PERIODIC:
        f = f.at[..., :LAP, :].set(f[..., ny:ny + LAP, :])
        f = f.at[..., LAP + ny:, :].set(f[..., LAP:LAP + LAP, :])
    elif bc == OUTFLOW:
        f = f.at[..., :LAP, :].set(f[..., LAP:LAP + 1, :])
        f = f.at[..., LAP + ny:, :].set(f[..., LAP + ny - 1:LAP + ny, :])
    elif bc == PLATE:
        pass   # 下壁镜像 + 顶部来流由 plate.apply 写入，此处保留原值
    else:
        raise ValueError(f"unknown bc: {bc}")
    return f


def fill_halos_z(f, bc, nz):
    """沿 z（axis=-3）填充鬼胞。"""
    if bc == PERIODIC:
        f = f.at[:, :LAP, :, :].set(f[:, nz:nz + LAP, :, :])
        f = f.at[:, LAP + nz:, :, :].set(f[:, LAP:LAP + LAP, :, :])
    elif bc == OUTFLOW:
        f = f.at[:, :LAP, :, :].set(f[:, LAP:LAP + 1, :, :])
        f = f.at[:, LAP + nz:, :, :].set(f[:, LAP + nz - 1:LAP + nz, :, :])
    else:
        raise ValueError(f"unknown bc: {bc}")
    return f


WALL = "wall"   # 无滑移绝热壁（速度镜像反号，rho/rhoE 镜像同号）

# var 轴符号：(rho, rho*u, rho*v, rho*w, rho*E)——速度反号、其余同号
_WALL_SIGN = (1.0, -1.0, -1.0, -1.0, 1.0)


def fill_wall_y(f, ny):
    """沿 y（axis=-2）的无滑移绝热壁镜像填充。

    壁面在 y=0（下壁，cell 0 左边界）与 y=ny*dy（上壁，cell ny-1 右边界）。
    镜像规则（对齐原码 OCFD_NS_Jacobian3d.cu:442-484 的
    boundary_symmetry_pole_vis_y_ker_m/p）：
      下壁（ker_m，y1 = 2*LAP - y）：ghost[p] = f[2*LAP - p]
      上壁（ker_p，y1 = 2*(ny+LAP-1) - y）：ghost[ny+LAP+q] = f[ny+LAP-2-q]
    速度分量（var 1,2,3）反号、rho/rhoE 同号。
    """
    sign = jnp.array(_WALL_SIGN).reshape(5, 1, 1, 1)
    mir_lo = f[:, :, 2 * LAP:LAP:-1, :]                # y 轴（-2）索引 2LAP-1..LAP（反序）
    f = f.at[:, :, :LAP, :].set(mir_lo * sign)
    # 上壁（对齐原码 ker_p 的 y1 = 2*(ny+LAP-1) - y）：ghost[ny+LAP+q] =
    # f[ny+LAP-2-q]，即索引 ny+LAP-2..ny+LAP-5（反序，避开自复制）
    mir_hi = f[:, :, ny + LAP - 2:ny + LAP - 6:-1, :]  # ny+LAP-2..ny+LAP-5（反序）
    f = f.at[:, :, ny + LAP:ny + 2 * LAP, :].set(mir_hi * sign)
    return f


def fill_halos(f, bcs, nx, ny, nz):
    """依次填充 x/y/z 鬼胞。bcs = (bc_x, bc_y, bc_z)，支持 WALL。"""
    f = fill_halos_x(f, bcs[0], nx)
    if bcs[1] == WALL:
        f = fill_wall_y(f, ny)
    else:
        f = fill_halos_y(f, bcs[1], ny)
    f = fill_halos_z(f, bcs[2], nz)
    return f


def zero_nonperiodic_ghosts(f, bcs, preserve_inflow=False):
    """清除非周期方向的鬼层，周期方向保留交换值。

    平板 IBC=124 的入口会写入低端状态鬼层；处理状态时用
    preserve_inflow=True 保留它；该函数也供粘性通量鬼层处理使用。
    f 的最后三轴为 (z, y, x)，bcs 顺序为 (x, y, z)。
    """
    for axis, bc in enumerate(bcs):
        if bc == PERIODIC:
            continue
        ax = f.ndim - 1 - axis             # 最后三轴依次为 (z, y, x)
        f = jnp.moveaxis(f, ax, -1)
        if not (preserve_inflow and bc == INFLOW):
            f = f.at[..., :LAP].set(0.0)
        f = f.at[..., -LAP:].set(0.0)
        f = jnp.moveaxis(f, -1, ax)
    return f
