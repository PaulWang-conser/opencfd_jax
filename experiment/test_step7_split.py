"""viscous_rhs 拆成两段以后的接线（第七步方案 U1）。

锁四件事：
  1. 两段顺序组合与 viscous_rhs 逐位相同。这只查接线；与改动前是否逐位相同，由
     experiment/test_step7_baseline.py 用改动前存下的抽样回答。
  2. 第一段输出的平面数是 4·len(axes)，按 axes 的顺序排列。直角度量、线性的速度场、
     常数温度时，内区的应力平面等于解析值。
  3. 内区的 Ev 只沿三个坐标方向读守恒量：块角上的守恒量换成 NaN，内区的 Ev 逐位不变；
     面内的一层鬼胞换成 NaN，相邻内区的 Ev 出现 NaN。
  4. 内区的粘性右端项只沿三个坐标方向读 Ev：块角上的 Ev 换成 NaN，内区逐位不变；面内
     的一层鬼胞换成 NaN，相邻内区出现 NaN。

第 3、4 两条是方案 R8 在单个块上的依据。多块的接口交换只覆盖“面内、切向下标在内区
里”的鬼胞，块角从不填；这两条说明块角上是什么值都不要紧。

输入的构造复用 experiment/test_step7_baseline.py。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import viscous  # noqa: E402
from opencfd_jax.src.boundary import LAP  # noqa: E402
from opencfd_jax.src.fields import Grid  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402
import test_step7_baseline as baseline  # noqa: E402

_same = support.same_bits
pytestmark = pytest.mark.skipif(not baseline.PLATE.exists(), reason="缺少 plate 网格")

WIRING = ("small_faces_cd8", "plate_compare_cd8", "cartesian_fill_cd8",
          "cartesian_band_cd8", "small_faces_axes01_cd8", "cube_outlet_face_cd8")


def _divergence_kwargs(kwargs):
    """从 viscous_rhs 的关键字参数里取出 viscous_divergence 要的那几个，缺省的按缺省取。"""
    return {name: kwargs[name] for name in ("axes", "dx", "order")} | {
        "metrics": kwargs.get("metrics"), "dif_bound": kwargs.get("dif_bound"),
        "dif_axis_map": kwargs.get("dif_axis_map", (0, 1, 2))}


@pytest.mark.parametrize("name", WIRING)
def test_two_stages_in_sequence_match_viscous_rhs(name):
    cons, kwargs = baseline.cases()[name]
    want = np.asarray(viscous.viscous_rhs(cons, **kwargs))
    fluxes = viscous.viscous_fluxes(cons, **kwargs)
    got = np.asarray(viscous.viscous_divergence(fluxes, kwargs["grid"], **_divergence_kwargs(kwargs)))
    _same(got, want)


@pytest.mark.parametrize("name", ["small_faces_cd8", "small_faces_axes01_cd8",
                                 "small_bcs_axes0_cd8", "cartesian_fill_cd6"])
def test_flux_stage_emits_four_planes_per_direction(name):
    cons, kwargs = baseline.cases()[name]
    fluxes = np.asarray(viscous.viscous_fluxes(cons, **kwargs))
    grid = kwargs["grid"]
    assert fluxes.shape == (4 * len(kwargs["axes"]), *grid.shape[1:])
    assert fluxes.dtype == np.asarray(cons).dtype
    assert np.all(np.isfinite(fluxes))


def test_a_missing_dx_is_reported_in_each_functions_own_name():
    """不给 dx 时，viscous_rhs 的报错与拆分之前相同：报的是它自己的名字。"""
    grid = Grid(ni=13, nj=11, nk=9)
    cons, _ = baseline.cases()["cartesian_fill_cd8"]
    args = (cons, grid, viscous.PhysParams(), ("outflow",) * 3)
    with pytest.raises(ValueError, match="^viscous_rhs requires dx$"):
        viscous.viscous_rhs(*args)
    with pytest.raises(ValueError, match="^viscous_fluxes requires dx$"):
        viscous.viscous_fluxes(*args)
    with pytest.raises(ValueError, match="^viscous_divergence requires dx$"):
        viscous.viscous_divergence(np.zeros((4, *grid.shape[1:])), grid)


def test_an_iterator_of_axes_behaves_like_the_tuple():
    """拆分以前 axes 只被遍历一次，给迭代器也一样。两段各遍历一次的话，第二段拿到的
    是空的，结果悄悄少掉一些方向。"""
    cons, kwargs = baseline.cases()["cube_axes02_cd8"]
    want = np.asarray(viscous.viscous_rhs(cons, **dict(kwargs, axes=(0, 2))))
    got = np.asarray(viscous.viscous_rhs(cons, **dict(kwargs, axes=iter((0, 2)))))
    assert np.any(want != 0.0)
    _same(got, want)


def test_no_axes_gives_zeros_like_before():
    """axes 为空时拆分以前返回零：du 按守恒量的精度建，给了度量就乘一次 Ajac。"""
    for name in ("cartesian_fill_cd8", "cube_mixed_cd8"):
        cons, kwargs = baseline.cases()[name]
        want = jnp.zeros(kwargs["grid"].shape, dtype=jnp.asarray(cons).dtype)
        if kwargs.get("metrics") is not None:
            want = want * kwargs["metrics"]["Ajac"]
        got = np.asarray(viscous.viscous_rhs(cons, **dict(kwargs, axes=())))
        _same(got, np.asarray(want), name)


def test_stress_planes_equal_the_analytic_values_for_a_linear_velocity_field():
    """直角度量，速度是坐标的线性函数、温度和密度是常数。CD8 对线性函数精确，所以
    离边界足够远的内点上，应力平面就是常数的 μ 乘应变率，交叉应力为零。"""
    grid = Grid(ni=21, nj=19, nk=17)
    params = viscous.PhysParams(Re=100.0)
    dxs = (0.1, 0.2, 0.15)
    k, j, i = np.meshgrid(*(np.arange(n) for n in grid.shape[1:]), indexing="ij")
    x, y, z = ((index - LAP) * step for index, step in zip((i, j, k), dxs))
    u, v, w = 0.5 + 0.3 * x, 0.1 + 0.2 * y, 0.05 * z
    energy = params.Cv + 0.5 * (u * u + v * v + w * w)
    cons = jnp.asarray(np.stack([np.ones_like(u), u, v, w, energy]))
    fluxes = np.asarray(viscous.viscous_fluxes(
        cons, grid, params, ("outflow", "wall", "periodic"), axes=(0, 1, 2), dx=dxs,
        order=8, fill_state_halos=True))
    mu = 1.0 / 100.0                                   # T=1 时 μ(T)=1/Re
    dilatation = 0.3 + 0.2 + 0.05
    middle = (slice(LAP + 6, LAP + 11), slice(LAP + 6, LAP + 13),
              slice(LAP + 6, LAP + 15))                # 离各面至少 6 层
    shear = (2.0 / 3.0) * dilatation
    expected = {                                       # 平面号 → 解析值；能量通量随坐标变，不比
        0: (2 * 0.3 - shear) * mu, 1: 0.0, 2: 0.0,     # x 方向：s11、s12、s13
        4: 0.0, 5: (2 * 0.2 - shear) * mu, 6: 0.0,     # y 方向：s12、s22、s23
        8: 0.0, 9: 0.0, 10: (2 * 0.05 - shear) * mu,   # z 方向：s13、s23、s33
    }
    for plane, value in expected.items():
        np.testing.assert_allclose(fluxes[plane][middle], value, rtol=1e-10, atol=1e-15)


# 块角、面内鬼胞的掩码放在 step7_support 里，多块的测试（test_step7_step.py）也用


def test_corner_and_face_ghost_masks_match_their_definition():
    """掩码按定义独立摆一遍对照。只靠“内区逐位不变”测不出掩码标错：块角掩码曾经
    因为下标少了一维，标成了“全部 j 向鬼胞”，那类测试照样通过。"""
    grid = Grid(ni=8, nj=7, nk=7)
    out = [np.array([True] * LAP + [False] * n + [True] * LAP) for n in grid.n_inner]
    k, j, i = np.meshgrid(*out, indexing="ij")
    assert np.array_equal(step7.corner(grid), (k & j) | (k & i) | (j & i))
    for axis, (first, second) in enumerate(((i, j), (i, k), (j, k))):   # 数组轴 0 是 k
        for depth in (1, 3):
            low = np.zeros(grid.shape[1:], dtype=bool)
            index = [slice(None)] * 3
            index[axis] = LAP - depth
            low[tuple(index)] = True
            assert np.array_equal(step7.face_ghost(grid, axis, depth), low & ~first & ~second)


def _interior(value, grid, count):
    """含鬼胞数组（最前面一根轴是分量或平面）的内区，并核对比到的形状。

    ``grid.inner()`` 带着最前面的整段切片。只拿三个空间切片去套四维数组时，第一个切片
    会落在分量轴上，比到的只是其中一个分量、空间轴还错位；Astra 第 24 轮指出过。
    """
    picked = np.asarray(value)[grid.inner()]
    assert picked.shape == (count, *grid.n_inner), (picked.shape, count, grid.n_inner)
    return picked


@pytest.mark.parametrize("name", ["small_faces_cd8", "plate_faces_cd8"])
def test_interior_fluxes_do_not_read_corner_conservative_variables(name):
    cons, kwargs = baseline.cases()[name]
    grid = kwargs["grid"]
    inner = grid.inner()[1:]
    fluxes = viscous.viscous_fluxes(cons, **kwargs)
    corner = step7.corner(grid)
    assert corner.any() and not corner[inner].any()
    damaged = jnp.asarray(np.where(np.broadcast_to(corner, np.asarray(cons).shape), np.nan, cons))
    corrupted = np.asarray(viscous.viscous_fluxes(damaged, **kwargs))
    _same(_interior(corrupted, grid, 12), _interior(fluxes, grid, 12))
    # 对照：周期方向（k−）面内的一层鬼胞是 NaN，紧挨它的内区行上，含 k 向导数的 Ev
    # 平面出现 NaN。不选有单侧公式的物理边界：那里 4 层以内只读内区点，读不到鬼胞。
    band = step7.face_ghost(grid, 0)      # 数组轴 0 是 k；k± 是周期面，没有单侧公式
    damaged = jnp.asarray(np.where(np.broadcast_to(band, np.asarray(cons).shape), np.nan, cons))
    corrupted = np.asarray(viscous.viscous_fluxes(damaged, **kwargs))
    row = corrupted[:, LAP, LAP:LAP + grid.nj, LAP:LAP + grid.ni]
    assert np.isnan(row[[2, 6, 10, 11]]).all()
    assert not np.array_equal(_interior(corrupted, grid, 12), _interior(fluxes, grid, 12))


@pytest.mark.parametrize("name", ["small_faces_cd8", "plate_faces_cd8"])
def test_interior_viscous_rhs_does_not_read_corner_fluxes(name):
    cons, kwargs = baseline.cases()[name]
    grid = kwargs["grid"]
    fluxes = np.asarray(viscous.viscous_fluxes(cons, **kwargs))
    want = np.asarray(viscous.viscous_divergence(jnp.asarray(fluxes), grid,
                                                 **_divergence_kwargs(kwargs)))
    corner = step7.corner(grid)
    damaged = np.where(np.broadcast_to(corner, fluxes.shape), np.nan, fluxes)
    got = np.asarray(viscous.viscous_divergence(jnp.asarray(damaged), grid,
                                                **_divergence_kwargs(kwargs)))
    _same(_interior(got, grid, 5), _interior(want, grid, 5))
    # 对照：周期方向（k−）第 1 层鬼胞上、z 方向平面的 Ev 是 NaN，k=0 这一列内区的动量和
    # 能量右端项出现 NaN；离它 5 层以外的内区不受影响
    band = step7.face_ghost(grid, 0)      # 数组轴 0 是 k；k± 是周期面，没有单侧公式
    damaged = fluxes.copy()
    damaged[8:12][:, band] = np.nan
    got = np.asarray(viscous.viscous_divergence(jnp.asarray(damaged), grid,
                                                **_divergence_kwargs(kwargs)))
    rows = got[:, LAP:LAP + grid.nk, LAP:LAP + grid.nj, LAP:LAP + grid.ni]
    assert np.isnan(rows[1:5, 0]).all()
    assert not np.isnan(rows[1:5, 5:]).any()
    assert not np.array_equal(_interior(got, grid, 5), _interior(want, grid, 5))
