"""场布局与鬼胞管理。

与 OpenCFD 对齐（head/config_parameters.h: LAP=4, NVARS=5）：
守恒变量 SoA，shape (NVARS, nk+2*LAP, nj+2*LAP, ni+2*LAP)，
变量顺序 0:rho, 1:rho*u, 2:rho*v, 3:rho*w, 4:rho*E；空间末轴为计算 i 方向。
计算坐标 (i,j,k) 对应 (ξ,η,ζ)，物理坐标为 (x,y,z)。规范尺寸名是
ni/nj/nk；nx/ny/nz 是只读兼容别名。代码中的 x/y/z 轴标签只作为旧面名
兼容输入，依次对应计算 i/j/k，不代表曲线网格的物理方向。

边界填充的实际实现已迁到 ``boundary.py``；本模块保留 ``LAP`` / ``NVARS`` /
边界类型 / ``fill_halos_*`` 等旧导入入口，供历史测试和调用方兼容使用。
"""

import jax.numpy as jnp

from .axes import MISSING, normalize_axis, resolve_dimensions
from .boundary import (INFLOW, LAP, NVARS, OUTFLOW, PERIODIC, PLATE, WALL,
                       _WALL_SIGN, fill_halos, fill_halos_i, fill_halos_j,
                       fill_halos_k, fill_halos_x, fill_halos_y, fill_halos_z,
                       fill_wall_j, fill_wall_y, zero_nonperiodic_ghosts)

def array_axis(axis):
    """将计算方向 (i,j,k) 的 0,1,2 映射到标量数组 (k,j,i) 的轴。"""
    return 2 - normalize_axis(axis)


def spatial_array_axis(axis):
    """旧名：将计算方向映射到标量数组 (k,j,i) 的轴。"""
    return array_axis(axis)


class Grid:
    """结构网格：内区 ni/nj/nk 个 cell，全场加 2*LAP 鬼胞。

    规范字段是 ``ni/nj/nk``；``nx/ny/nz`` 是只读兼容属性。旧位置参数
    ``Grid(nx, ny, nz)`` 与新旧关键字都指向同一份尺寸，不保存两份状态。
    """

    def __init__(self, ni=MISSING, nj=MISSING, nk=MISSING, *,
                 nx=MISSING, ny=MISSING, nz=MISSING):
        values = resolve_dimensions(ni, nj, nk, nx=nx, ny=ny, nz=nz)
        self.ni, self.nj, self.nk = values
        # 各轴全长（含鬼胞），按计算轴顺序 (k, j, i) 与 shape 对齐
        self._n_full = tuple(extent + 2 * LAP for extent in values[::-1])
        self.shape = (NVARS, *self._n_full)
        self.n_inner = (self.nk, self.nj, self.ni)

    @property
    def nx(self):
        return self.ni

    @property
    def ny(self):
        return self.nj

    @property
    def nz(self):
        return self.nk

    def inner(self):
        """内区 slice（含变量轴），空间顺序仍为 (k,j,i)。"""
        return (slice(None),
                slice(LAP, LAP + self.nk),
                slice(LAP, LAP + self.nj),
                slice(LAP, LAP + self.ni))

    def alloc(self, dtype=None):
        """分配全零守恒量场，形状为 ``(5, nk+8, nj+8, ni+8)``。"""
        return jnp.zeros(self.shape, dtype=dtype)
