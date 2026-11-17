"""场布局与鬼胞管理。

与 OpenCFD 对齐（head/config_parameters.h: LAP=4, NVARS=5）：
守恒变量 SoA，shape (NVARS, nz+2*LAP, ny+2*LAP, nx+2*LAP)，
变量顺序 0:rho, 1:rho*u, 2:rho*v, 3:rho*w, 4:rho*E；空间末轴为计算 i 方向。
计算坐标 (i,j,k) 对应 (ξ,η,ζ)，物理坐标为 (x,y,z)。现有代码中的
nx/ny/nz 和 x/y/z 轴标签依次对应计算 i/j/k，不代表曲线网格的物理方向。

边界填充的实际实现已迁到 ``boundary.py``；本模块保留 ``LAP`` / ``NVARS`` /
边界类型 / ``fill_halos_*`` 等旧导入入口，供历史测试和调用方兼容使用。
"""

import jax.numpy as jnp

from .boundary import (INFLOW, LAP, NVARS, OUTFLOW, PERIODIC, PLATE, WALL,
                       _WALL_SIGN, fill_halos, fill_halos_x, fill_halos_y,
                       fill_halos_z, fill_wall_y, zero_nonperiodic_ghosts)


def spatial_array_axis(axis):
    """将计算方向 (i,j,k) 的 0,1,2 映射到标量数组 (k,j,i) 的轴。"""
    return 2 - axis


class Grid:
    """结构网格：内区 N 个 cell，全场 N+2*LAP（含鬼胞）。"""

    def __init__(self, nx, ny, nz):
        """记录三个方向的内区网格数，并推导带四层鬼胞的完整数组形状。"""
        self.nx, self.ny, self.nz = nx, ny, nz
        self.shape = (NVARS, nz + 2 * LAP, ny + 2 * LAP, nx + 2 * LAP)
        self.n_inner = (nz, ny, nx)
        # 各轴全长（含鬼胞），按计算轴顺序 (k, j, i) 与 shape 对齐
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
