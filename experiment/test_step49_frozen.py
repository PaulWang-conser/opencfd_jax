"""单块默认口径的逐位回归（第六步的护栏）。

``experiment/step49_eps_scaled_20261122/testcgnstest-istep-00000001.npz`` 是
第六步改动之前，用 plate 4.9 的默认口径跑一步存下的五个守恒量。这里在进程内
走正式入口再跑一步，结果必须与它逐位相同。

冻结文件只回答“单块有没有被改坏”，不是数值正确性的标准（路线图第 0 步记着
jisen 的决定：不再以历史 JAX 基准作为验收条件）。

锁两件事：
  1. 正式入口跑一步，五个守恒量与冻结文件逐位相同；
  2. 比较有判别力：结果副本里翻转一个位，比较必须失败。
"""

from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import io, main, runtime_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "experiment" / "cases" / "plate_cgns_step49.json"
MESH = REPO / "Grid" / "testcgnstest.cgns"
FROZEN = (REPO / "experiment" / "step49_eps_scaled_20261122"
          / "testcgnstest-istep-00000001.npz")

pytestmark = pytest.mark.skipif(
    not FROZEN.exists() or not MESH.exists(), reason="缺少冻结文件或 plate 网格")


def _same_bits(a, b) -> bool:
    """先比类型和形状，再逐位比较。最低位翻转后数值几乎不变，所以不按数值比。"""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture(scope="module")
def one_step(tmp_path_factory):
    """走正式入口跑一步；输出放进临时目录，不写进仓库。"""
    cfg = runtime_config.load_config(str(CFG))
    args = runtime_config.resolve_args({}, cfg, main.build_parser().parse_args([]))
    args.out_dir = str(tmp_path_factory.mktemp("step49_frozen"))
    main.run(args)
    files = sorted(Path(args.out_dir).glob("*.npz"))
    assert len(files) == 1, f"跑一步应只存一个文件，得到 {[p.name for p in files]}"
    fields, _ = io.load_npz(str(files[0]))
    return fields


def test_one_step_matches_frozen_file(one_step):
    want, _ = io.load_npz(str(FROZEN))
    assert sorted(want) == [f"cons_{i}" for i in range(5)]
    assert sorted(one_step) == sorted(want)
    for key in sorted(want):
        assert _same_bits(one_step[key], want[key]), f"{key} 与冻结文件不逐位相同"


def test_comparison_detects_one_flipped_bit(one_step):
    want, _ = io.load_npz(str(FROZEN))
    assert all(_same_bits(one_step[key], want[key]) for key in want)
    damaged = {key: value.copy() for key, value in one_step.items()}
    damaged["cons_3"].view(np.uint64)[10, 10, 10] ^= np.uint64(1)
    assert damaged["cons_3"][10, 10, 10] != one_step["cons_3"][10, 10, 10]
    assert not _same_bits(damaged["cons_3"], want["cons_3"])
    assert all(_same_bits(damaged[key], want[key])
               for key in want if key != "cons_3")
