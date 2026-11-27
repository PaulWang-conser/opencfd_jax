"""``viscous_rhs`` 的逐位回归（第七步的护栏）。

``experiment/step7_viscous_samples.npz`` 存的是第七步改动之前，``viscous_rhs`` 在一组
固定输入上的输出抽样。第七步要把这个函数的函数体拆成“算粘性通量 Ev”和“对 Ev 求导”
两段，再让 ``viscous_rhs`` 顺序调用它们（方案 U1、KTD1）。拆完以后这些抽样必须逐位
相同。

覆盖的调用方式：

- y 向拉伸的 plate 源网格（100×40×40）和它的合成小块（15×7×7），都放上第六步的验收
  初场：带 ``faces`` 的正式路径（CD8）、``fill_state_halos`` 为真的旧接口（CD6）、只给
  逐轴 ``bcs`` 的旧路径、``src/compare.py`` 那种省略 ``dif_bound`` 和
  ``boundary_execution`` 的调用；
- 只在小块上：只有部分方向、带 ``zero_state_ghosts``、只算一个方向且 ``bcs`` 全是出口；
- 解析给出的非均匀立方体，直角度量（``metrics=None``）：CD6 与 CD8、``fill_state_halos``
  真与假、给与不给 ``dif_bound``、有不求解的方向、只有一个出口面的 ``faces``；
- ``axes`` 隔着取（``(0, 2)``）；状态是 FP32 而度量是 FP64 的立方体。后一种是 Astra 第 24
  轮指出的：拆分以前 ``du`` 按状态的精度累加，乘 ``Ajac`` 以后才升到 FP64，全是 FP64 的
  调用方式看不出这一层。

Astra 第 24 轮还指出两条不在基线里、由 ``test_step7_split.py`` 直接查的旧行为：``axes``
给迭代器时与给元组相同；``axes`` 为空时返回零。

抽样点每条轴取离两端 9 层以内的全部，加中间一个，与第六步基线的取法相同。本文件
既负责构造输入，也负责核对；存基线的一次性脚本复用这里的构造。

锁两件事：
  1. ``viscous_rhs`` 的输出在抽样点上与存下的原值逐位相同；
  2. 比较有判别力：结果副本里翻转一个位，比较必须失败。
"""

import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import init, viscous  # noqa: E402
from opencfd_jax.src.boundary import LAP  # noqa: E402
from opencfd_jax.src.fields import Grid  # noqa: E402
from opencfd_jax.src.mesh_data import FaceBoundary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
from Grid.make_cgns_fixture import build_multizone  # noqa: E402

SAMPLES = REPO / "experiment" / "step7_viscous_samples.npz"
PLATE = support.SOURCE
SMALL_MESH = support.MESH_DIR / "step7_small.cgns"
LAYERS = 9                        # 每条轴离两端这么多层以内的点全部抽样

ON_STATES = ("faces_cd8", "faces_fill_cd6", "compare_cd8", "bcs_fill_cd6")
CASE_NAMES = (
    *(f"small_{suffix}" for suffix in ON_STATES),
    "small_faces_axes01_cd8", "small_bcs_zero_ghosts_cd8", "small_bcs_axes0_cd8",
    *(f"plate_{suffix}" for suffix in ON_STATES),
    "cartesian_fill_cd8", "cartesian_fill_cd6", "cartesian_nofill_cd8",
    "cartesian_band_cd8", "cartesian_inactive_z_cd8", "cube_outlet_face_cd8",
    "cube_axes02_cd8", "cube_mixed_cd8",
)


def _picks(n: int) -> np.ndarray:
    """一条轴的抽样下标（含鬼胞数组上的位置）：两端各 ``LAYERS`` 层以内，加中间一个。"""
    near = (list(range(LAP, LAP + min(LAYERS, n)))
            + list(range(LAP + max(0, n - LAYERS), LAP + n)))
    return np.array(sorted(set(near + [LAP + n // 2])), dtype=np.int64)


def _tag(grid) -> str:
    """网格的记号，形如 ``100x40x40``。"""
    return f"{grid.ni}x{grid.nj}x{grid.nk}"


@functools.lru_cache(maxsize=None)
def _states():
    """装配好的两个单块：合成小块（15×7×7）和 plate 源网格，内区放上第六步的验收初场。

    ``start`` 是起点（鬼胞为零），``prepared`` 是走过一次物理准备的状态。plate 那块
    直接用第六步的单块参考。
    """
    build_multizone(SMALL_MESH, [support.block(
        "Zone 1", 15, 7, 7, bcs=support.outer_bcs(15, 7, 7, ("Ilo", "BCInflow"),
                                                 ("Ihi", "BCOutflow")))])
    out = {}
    for name, mesh in (("small", SMALL_MESH), ("plate", PLATE)):
        state = init.prepare(support.plate_args(mesh))
        start = jnp.asarray(support.with_field(state, support.acceptance_state))
        prepared = np.asarray(state.execution.prepare(jnp.asarray(start)))
        out[name] = SimpleNamespace(state=state, start=start, prepared=prepared)
    return out


def _cube(grid):
    """解析给出的非均匀守恒量，铺满含鬼胞的数组；每个方向都变。"""
    k, j, i = np.meshgrid(*(np.arange(n) for n in grid.shape[1:]), indexing="ij")
    x, y, z = 0.7 * i, 1.1 * j, 1.3 * k
    rho = 1.0 + 0.1 * np.sin(0.5 * x) * np.cos(0.4 * y)
    u = 0.8 + 0.2 * np.cos(0.3 * z) * np.sin(0.5 * x)
    v = 0.1 * np.sin(0.6 * y)
    w = 0.05 * np.cos(0.5 * z)
    temperature = 1.0 + 0.1 * np.sin(0.4 * (x + y)) * np.cos(0.3 * z)
    params = viscous.PhysParams()
    energy = temperature * params.Cv + 0.5 * (u * u + v * v + w * w)
    return jnp.asarray(np.stack([rho, rho * u, rho * v, rho * w, rho * energy]))


@functools.lru_cache(maxsize=None)
def cases() -> dict:
    """全部调用方式：{名字: (cons, 调用 viscous_rhs 的关键字参数)}。"""
    out = {}
    for name, item in _states().items():
        desc, execution = item.state.desc, item.state.execution
        base = dict(grid=desc.grid, params=item.state.case["params"], bcs=execution.bcs,
                    axes=desc.axes, dx=desc.dxs, metrics=desc.metrics,
                    dif_axis_map=desc.dif_axis_map)
        out[f"{name}_faces_cd8"] = (item.prepared, dict(
            base, order=8, dif_bound=execution.band, boundary_execution=execution,
            farfield=execution.farfield, faces=execution.faces, fill_state_halos=False))
        out[f"{name}_faces_fill_cd6"] = (item.start, dict(
            base, order=6, dif_bound=execution.band, boundary_execution=execution,
            farfield=execution.farfield, faces=execution.faces, fill_state_halos=True))
        out[f"{name}_compare_cd8"] = (item.prepared, dict(
            base, order=8, farfield=execution.farfield, faces=execution.faces,
            fill_state_halos=False))              # src/compare.py 的写法：不给边界带
        out[f"{name}_bcs_fill_cd6"] = (item.start, dict(base, order=6, fill_state_halos=True))

    small = _states()["small"]
    desc, execution = small.state.desc, small.state.execution
    base = dict(grid=desc.grid, params=small.state.case["params"], bcs=execution.bcs,
                axes=desc.axes, dx=desc.dxs, metrics=desc.metrics,
                dif_axis_map=desc.dif_axis_map)
    out["small_faces_axes01_cd8"] = (small.prepared, dict(
        base, order=8, axes=(0, 1), dif_bound=execution.band,
        boundary_execution=execution, farfield=execution.farfield, faces=execution.faces,
        fill_state_halos=False))
    out["small_bcs_zero_ghosts_cd8"] = (small.start, dict(
        base, order=8, fill_state_halos=True, zero_state_ghosts=True))
    out["small_bcs_axes0_cd8"] = (small.prepared, dict(
        base, order=8, axes=(0,), bcs=("outflow",) * 3, fill_state_halos=False))

    grid = Grid(ni=13, nj=11, nk=9)
    cons = _cube(grid)
    params = viscous.PhysParams()
    cartesian = dict(grid=grid, params=params, bcs=("outflow", "wall", "periodic"),
                     axes=(0, 1, 2), dx=(0.1, 0.2, 0.15))
    out["cartesian_fill_cd8"] = (cons, dict(cartesian, order=8, fill_state_halos=True))
    out["cartesian_fill_cd6"] = (cons, dict(cartesian, order=6, fill_state_halos=True))
    out["cartesian_nofill_cd8"] = (cons, dict(cartesian, order=8, fill_state_halos=False))
    out["cartesian_band_cd8"] = (cons, dict(
        cartesian, order=8, fill_state_halos=True, dif_bound=(1, 0, 0, 1, 1, 1)))
    out["cartesian_inactive_z_cd8"] = (cons, dict(
        cartesian, order=8, fill_state_halos=True, dx=(0.1, 0.2, 0.0)))   # z 向不求解
    outlet = Grid(ni=8, nj=8, nk=8)
    out["cube_outlet_face_cd8"] = (_cube(outlet), dict(
        grid=outlet, params=params, bcs=("outflow",) * 3, axes=(0,), dx=(1, 1, 1), order=8,
        faces=(FaceBoundary("i+", 0, "outflow", "halo", False),), fill_state_halos=False))
    out["cube_axes02_cd8"] = (cons, dict(cartesian, order=8, axes=(0, 2), fill_state_halos=True))
    warped = Grid(ni=13, nj=11, nk=9)                # 状态 FP32、度量 FP64
    single = jnp.asarray(np.asarray(_cube(warped), dtype=np.float32))
    out["cube_mixed_cd8"] = (single, dict(
        cartesian, grid=warped, dx=(1.0, 1.0, 1.0), order=8, metrics=_warped_metrics(warped),
        fill_state_halos=True))
    return out


def _warped_metrics(grid):
    """三个方向间距都在变的网格上的度量（FP64）。"""
    from opencfd_jax.src import mesh
    k, j, i = np.meshgrid(*(np.arange(n) for n in grid.n_inner), indexing="ij")
    raw = {"Axx": 0.10 * i + 0.002 * i * i, "Ayy": 0.20 * j + 0.0015 * j * j,
           "Azz": 0.15 * k + 0.001 * k * k}
    coords = {key: mesh.pad_coords(value, periodic=(False, False, False))
              for key, value in raw.items()}
    metrics = mesh.compute_metrics_from_coords(coords, (1.0, 1.0, 1.0), 6,
                                               periodic=(False, False, False))
    return {key: np.asarray(value) for key, value in metrics.items()}


def _sampled(name: str):
    """算一个调用方式的 ``viscous_rhs``，取抽样点上的值。

    返回 (抽样值, 网格记号, 各轴下标, 算出它的设备)。设备在取回主机以前看。
    """
    cons, kwargs = cases()[name]
    grid = kwargs["grid"]
    picks = tuple(_picks(n) for n in grid.n_inner)
    raw = viscous.viscous_rhs(cons, **kwargs)
    on = support.computed_on(raw)
    du = np.asarray(raw)
    return du[(slice(None),) + np.ix_(*picks)], _tag(grid), picks, on


pytestmark = pytest.mark.skipif(
    not SAMPLES.exists() or not PLATE.exists(), reason="缺少抽样基线文件或 plate 网格")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_viscous_rhs_matches_samples(name):
    """和 CPU 上存下的原值比：在 CPU 上算的逐位相同，在 GPU 上算的在验收的阈值内；
    状态是 FP32 的调用方式只在前一种情况下比（只检验 FP64）。"""
    got, tag, picks, on = _sampled(name)
    with np.load(SAMPLES) as data:
        want = data[f"du_{name}"]
        for axis, pick in enumerate(picks):
            np.testing.assert_array_equal(pick, data[f"picks_{('k', 'j', 'i')[axis]}_{tag}"])
    assert got.size and np.any(got != 0.0), f"{name}：抽样全是零，这个调用方式没有判别力"
    # 只检验 FP64（jisen 的决定）：状态是 FP32 的调用方式（cube_mixed_cd8）不在验收范围，
    # 在 GPU 上不和 CPU 存下的值比。在 CPU 上算的仍然逐位比。
    if np.asarray(cases()[name][0]).dtype == np.float32 and on != support.computed_on(want):
        pytest.skip("只检验 FP64：状态是 FP32 的调用方式，跨设备时不比")
    support.exact(got, want, f"{name}：viscous_rhs 与存下的原值不一致", got_on=on)


def test_comparison_notices_a_flipped_bit():
    """比较有判别力：结果副本里翻转一个位，逐位比较必须报错。"""
    got, _, _, _ = _sampled("small_faces_cd8")
    flipped = got.copy().reshape(-1)
    position = int(np.flatnonzero(flipped != 0.0)[0])
    flipped[position] = np.nextafter(flipped[position], np.inf)      # 最低位翻转
    with pytest.raises(AssertionError):
        support.same_bits(flipped.reshape(got.shape), got)
