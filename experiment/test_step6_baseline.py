"""``src/mesh.py`` 两个函数的逐位回归（第六步的护栏）。

``experiment/step6_baseline_samples.npz`` 存的是第六步改动之前的原值：

  - ``compute_metrics_from_coords``：plate 网格和 y 向拉伸网格上，三种 ``jac_bound``
    下的十个度量。每条轴取 19 个固定下标——离两端 9 层以内的全部（鬼胞、边界带、
    头一层中心差分）加中间一个——存这些下标的组合上的值；
  - ``pad_coords``：轴长 7、8、15 的立方体坐标，三种周期设置，存整场。轴长 7 走
    边缘复制，8 和 15 走奇镜像外推。

抽样下标也在文件里。第六步给这两个函数加了可选参数（逐侧的边界标志、等价
单块的轴长），不传新参数时结果必须与这里的原值逐位相同。
"""

from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import mesh, mesh_cgns  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "experiment" / "step6_baseline_samples.npz"
MESHES = {"plate": REPO / "Grid" / "testcgnstest.cgns",
          "ystr": REPO / "Grid" / "testcgnstest_ystr.cgns"}
PERIODIC = (True, False, True)          # i 周期、j 非周期、k 周期：环绕和外推两种补层都覆盖到
PERIODICS = ((True, False, True), (False, False, False), (True, True, True))

pytestmark = pytest.mark.skipif(
    not SAMPLES.exists() or not all(p.exists() for p in MESHES.values()),
    reason="缺少抽样基线文件或网格")


def _same_bits(a, b) -> bool:
    """先比类型和形状，再逐位比较。最低位翻转后数值几乎不变，所以不按数值比。"""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _cube(n):
    """沿三个方向取值各不相同的坐标立方体 (n, n, n)，轴序 (k, j, i)。"""
    k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    return 0.1 * i + 0.7 * j + 1.3 * k


def _tag(periodic) -> str:
    return "".join("1" if p else "0" for p in periodic)


@pytest.mark.parametrize("name", sorted(MESHES))
def test_metrics_match_samples(name):
    with np.load(SAMPLES) as data:
        picks = data[f"picks_{name}"]
        want = data[f"metrics_{name}"]
        jac_bounds = data["metric_jac_bound"]
        keys = [str(k) for k in data["metric_keys"]]
    zone = mesh_cgns.read_cgns_zone(MESHES[name])
    padded = {key: mesh.pad_coords(value, periodic=PERIODIC)
              for key, value in zone.coords.items()}
    blocks = np.stack([
        np.stack([np.asarray(metrics[k]) for k in keys])
        for metrics in (
            mesh.compute_metrics_from_coords(
                padded, (1.0, 1.0, 1.0), 6, jac_bound=tuple(jb), periodic=PERIODIC)
            for jb in jac_bounds)])
    got = blocks[np.ix_(np.arange(len(jac_bounds)), np.arange(len(keys)), *picks)]
    assert got.shape == want.shape
    for a, jac_bound in enumerate(jac_bounds):
        for b, key in enumerate(keys):
            assert _same_bits(got[a, b], want[a, b]), (
                f"{name} jac_bound={tuple(jac_bound)} {key}："
                "度量在抽样点上与存下的原值不逐位相同")


@pytest.mark.parametrize("n", [7, 8, 15])
@pytest.mark.parametrize("periodic", PERIODICS, ids=_tag)
def test_pad_coords_matches_samples(n, periodic):
    with np.load(SAMPLES) as data:
        want = data[f"pad_{n}_{_tag(periodic)}"]
    got = mesh.pad_coords(_cube(n), periodic=periodic)
    assert _same_bits(got, want), (
        f"n={n} periodic={periodic}：补层坐标与存下的原值不逐位相同")
