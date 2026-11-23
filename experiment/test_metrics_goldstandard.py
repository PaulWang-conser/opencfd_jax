"""度量-粘性算子的黄金标准回归（2026-11-23 诊断的固化）。

背景：CLBL 超声速层流对照暴露剖面偏厚；排查撤销了"Ajac 口径"与"单位"
两个误判后，用小振幅 sin(y) 解析场定位真凶——`pad_coords` 非周期轴的
**边缘复制**使鬼胞坐标恒定、近壁 ∂y/∂j=0，度量在壁面带崩坏
（handoff 第五轮"Ajac(wall) 差 2 倍"同源）。改线性外推后：
内区（距边 >4 点）du_rhou 比值精确 1.0000。

锁三件事：
  1. 内区深处：du_rhou / (-μk²A sin(ky)) 逐点 = 1（带度量，拉伸网格）；
  2. 修复前后对比的行为边界：j>=8 全部精确（边界带内 4 点走单侧公式，
     由 BC 支配，不在本测试断言范围）；
  3. pad_coords 非周期轴是线性外推（鬼胞坐标不恒定）。
"""

from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import mesh  # noqa: E402
from opencfd_jax.src import mesh_cgns  # noqa: E402
from opencfd_jax.src.viscous import PhysParams, viscous_rhs  # noqa: E402

MESH = Path(__file__).resolve().parents[1] / "Grid" / "testcgnstest.cgns"


def test_pad_coords_nonperiodic_extrapolates():
    arr = np.arange(1.0, 13.0).reshape(1, 12, 1)
    out = mesh.pad_coords(arr, periodic=(False, False, False))
    # 数组轴 1（periodic[1]）非周期：真·等差延拓 [-3,-2,-1,0 | 1..12 | 13..16]
    np.testing.assert_allclose(out[0, :8, 0], [-3, -2, -1, 0, 1, 2, 3, 4])
    np.testing.assert_allclose(out[0, -8:, 0], [9, 10, 11, 12, 13, 14, 15, 16])
    # 轴 1 周期（periodic[1]=True）：取对侧内区
    per = mesh.pad_coords(arr, periodic=(False, True, False))
    assert per[0, 0, 0] == pytest.approx(9.0)


@pytest.mark.skipif(not MESH.exists(), reason="缺少 testcgnstest.cgns")
def test_viscous_sin_goldstandard_interior_exact():
    """u=0.1·sin(y)（T=1 恒定）过粘性算子：内区精确等于 −μk²A·sin(ky)。"""
    zone = mesh_cgns.read_cgns_zone(str(MESH))
    blk = mesh_cgns.build_cgns_block(zone)
    grid, metrics, y = blk.grid, blk.metrics, zone.coords["Ayy"]
    params = PhysParams(gamma=1.4, Ama=2.0, Re=1000.0, Ref_T=288.15, Pr=0.72)
    Cv = 1.0 / (1.4 * 0.4 * 4.0)
    A, k = 0.1, 1.0
    u = A * np.sin(k * y)
    pad = lambda a: np.pad(a, 4, mode="edge")
    E = Cv + 0.5 * u ** 2
    cons = jnp.asarray(np.stack([pad(np.ones_like(y)), pad(u),
                                 pad(0 * u), pad(0 * u), pad(E)]))
    du = np.asarray(viscous_rhs(cons, grid, params,
                                ("periodic", "outflow", "periodic"),
                                axes=(1,), dx=(1.0, 1.0, 1.0), order=8,
                                metrics=metrics, dif_axis_map=(0, 1, 2)))
    exact = -(1.0 / 1000.0) * k ** 2 * u
    got = du[1][tuple(grid.inner()[1:])]
    # 内区深处（距每侧边界 >7 点，避开单侧公式带）
    sl = (slice(None), slice(8, -8), slice(8, -8))
    r = got[sl] / exact[sl]
    np.testing.assert_allclose(r, np.ones_like(r), rtol=2e-3, atol=0)
    # 修复的行为边界：j>=8 起也应精确（近壁带外第一点）
    kc, ic = y.shape[0] // 2, y.shape[2] // 2
    assert got[kc, 8, ic - 4] == pytest.approx(exact[kc, 8, ic], rel=2e-3)
