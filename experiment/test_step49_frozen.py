"""单块默认口径的回归（第六步的护栏）。

``experiment/step49_eps_scaled_20261122/testcgnstest-istep-00000001.npz`` 是
第六步改动之前，在 CPU 上用 plate 4.9 的默认口径跑一步存下的五个守恒量。这里在进程内
走正式入口再跑一步，交给存盘的内区守恒量必须与它相同：在 CPU 上算的逐位相同；在 GPU
上算的在验收的阈值内（比法见 ``step6_support.exact``）。

冻结文件只回答“单块有没有被改坏”，不是数值正确性的标准（路线图第 0 步记着
jisen 的决定：不再以历史 JAX 基准作为验收条件）。

锁两件事：
  1. 正式入口跑一步，交给存盘的内区守恒量与冻结文件相同；文件里的 ``b0`` 是它
     换算的原始变量；
  2. 比较有判别力：在 CPU 上算的，守恒量里翻转一个位，比较必须失败。不能换成原始
     变量再比：守恒量差一个最低位，换算后的五个原始量可能一个位都不变。在 GPU
     上算的，和存下来的数据本来就不逐位相同，改坏一个远大于阈值的量，比较必须失败。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import io, main, runtime_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402

CFG = REPO / "experiment" / "cases" / "plate_cgns_step49.json"
MESH = REPO / "Grid" / "testcgnstest.cgns"
FROZEN = (REPO / "experiment" / "step49_eps_scaled_20261122"
          / "testcgnstest-istep-00000001.npz")

pytestmark = pytest.mark.skipif(
    not FROZEN.exists() or not MESH.exists(), reason="缺少冻结文件或 plate 网格")


def _frozen_cons():
    """冻结文件里的五个守恒量 (5, nk, nj, ni) 和它的 _meta。"""
    want, meta = io.load_npz(str(FROZEN))
    assert sorted(want) == [f"cons_{i}" for i in range(5)]
    return np.stack([want[f"cons_{i}"] for i in range(5)]), meta


@pytest.fixture(scope="module")
def one_step(tmp_path_factory):
    """走正式入口跑一步；输出放进临时目录，不写进仓库。

    ``cons`` 是正式入口交给 ``io.primitive_field`` 的内区守恒量：把存盘的换算包一层
    记下来。``saved`` 是写出的文件里的 ``b0``。``on`` 是算出它们的设备，在取回主机以前看。
    """
    handed, where = [], []
    convert = io.primitive_field

    def recording(cons, grid, phys):
        handed.append(np.asarray(cons)[grid.inner()].copy())
        where.append(support.computed_on(cons))
        return convert(cons, grid, phys)

    cfg = runtime_config.load_config(str(CFG))
    args = runtime_config.resolve_args({}, cfg, main.build_parser().parse_args([]))
    args.out_dir = str(tmp_path_factory.mktemp("step49_frozen"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io, "primitive_field", recording)
        main.run(args)
    files = sorted(Path(args.out_dir).glob("*.npz"))
    assert len(files) == 1, f"跑一步应只存一个文件，得到 {[p.name for p in files]}"
    assert len(handed) == 1
    return SimpleNamespace(cons=handed[0], on=where[0],
                           saved=io.load_npz(str(files[0]))[0]["b0"])


def test_one_step_matches_frozen_file(one_step):
    frozen, meta = _frozen_cons()
    support.exact(one_step.cons, frozen, "交给存盘的内区守恒量", got_on=one_step.on)
    support.exact(one_step.saved, io.primitive_of(frozen, io.phys_from_meta(meta)),
                  "文件里的 b0", got_on=one_step.on)


def test_comparison_notices_damage(one_step):
    """比较有判别力。在 CPU 上算的，守恒量里翻转一个最低位，比较必须失败：换成原始变量
    以后一个位都不变，直接比才看得出来。在 GPU 上算的，和存下来的数据本来就不逐位相同，
    翻转最低位在阈值下面看不出来，所以改坏一个远大于阈值的量。"""
    frozen, meta = _frozen_cons()
    damaged = one_step.cons.copy()
    if one_step.on == support.computed_on(frozen):
        damaged[4].view(np.uint64)[0, 1, 1] ^= np.uint64(1)        # rhoE 差一个最低位
        assert damaged[4][0, 1, 1] != one_step.cons[4][0, 1, 1]
        phys = io.phys_from_meta(meta)
        support.same_bits(io.primitive_of(damaged, phys), io.primitive_of(frozen, phys))
    else:
        damaged[0, 0, 0, 0] *= 1.0 + 1e-6
    with pytest.raises(AssertionError):
        support.exact(damaged, frozen, "改坏的守恒量", got_on=one_step.on)
