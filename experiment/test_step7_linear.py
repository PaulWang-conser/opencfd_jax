"""第七步的第二层验收：五个合成小网格上的线性场（U4）。

锁五件事：
  1. 五个合成小网格放上线性场，一步的粘性右端项与各自的等价单块在阈值内一致。各块内区
     的坐标与等价单块上对应下标的坐标逐位相同（``step7_support.windows`` 核对）。
  2. 阶梯和一格宽条带两个网格上，离接口 4 层以内的点和其余的点分开比，都在阈值内。
  3. 常数来流：多块与等价单块在阈值内一致，不断言为零。物理面的状态鬼胞是清零或镜像
     的，紧挨物理面的内区点粘性右端项不为零。零值断言放在全周期的 per3 上。
  4. CD6：两块切口的小网格编译时传阶数 6，多块与等价单块在阈值内一致。线性场上 CD6 和
     CD8 的导数都精确，分不出阶数传没传对，所以这一条用验收初场；并确认 CD6 和 CD8 的
     单块参考在它上面确有可测的差别。
  5. 判别力：只把 Ev 的交换换成恒等，守恒量的交换照旧，阶梯网格上的差远超阈值。

比较项是多块第一步的粘性分项（``trace.viscous[0]``，各块内区拼回等价单块）和单块在
起点上的粘性右端项。阈值用第六步的 ``step6_support.compare``。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import multiblock  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402

DT = 1e-4
KEYS = step7.SYNTHETIC_KEYS
NAMES = step7.SYNTHETIC_NAMES
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")

_same = support.same_bits


def _run(key, field, vis_order="CD8"):
    """合成小网格上粘性打开的一步，连同它的等价单块。``vis_order`` 是粘性导数的阶数名；
    合成小网格的参考尺度只进无粘项，这里不比无粘。"""
    built = step7.synthetic(key)
    kernels = multiblock.compile_blocks(built, step7.SYNTHETIC_SCALE,
                                        vis_order=multiblock.CD_ORDER[vis_order])
    trace = multiblock.solve_step(kernels, step7.states_from_field(built, field), DT)
    return built, step7.single(key, vis_order), trace


def _back(built, note, trace):
    """多块第一步的粘性分项，各块内区拼回等价单块。"""
    return support.reassemble(note, support.interiors(built, trace.viscous[0]))


def _alone(state, field):
    """等价单块在起点上的粘性右端项（内区）。"""
    start = support.with_field(state, field)
    return step7.viscous_stage(state, state.execution.prepare(jnp.asarray(start)))[
        state.desc.grid.inner()]


# ---- 1、2. 线性场 ----

@pytest.mark.parametrize("key", KEYS, ids=NAMES)
def test_linear_field_matches_the_equivalent_single_block(key):
    built, state, trace = _run(key, step7.linear_field)
    note = step7.windows(built, state)             # 各块内区的坐标与单块逐位相同
    want, got = _alone(state, step7.linear_field), _back(built, note, trace)
    assert np.any(want[1:5] != 0.0) and np.any(got[1:5] != 0.0)
    support.compare(want, got, f"{NAMES[KEYS.index(key)]}的粘性右端项")
    if key in ("stair", "strip"):                   # 离接口 4 层以内与其余分开比
        near = step7.near_interfaces(built, note)
        assert near.any() and not near.all()
        support.compare(want, got, "接口邻域", mask=near)
        support.compare(want, got, "其余部分", mask=~near)


# ---- 3. 常数来流 ----

@pytest.mark.parametrize("key", KEYS, ids=NAMES)
def test_constant_free_stream_matches_without_being_asserted_zero(key):
    built, state, trace = _run(key, step7.free_stream)
    want = _alone(state, step7.free_stream)
    got = _back(built, step7.windows(built, state), trace)
    assert np.any(want[1:5] != 0.0)                 # 物理面旁不为零，所以不拿零当标准
    support.compare(want, got, f"{NAMES[KEYS.index(key)]}的常数来流")


def test_viscous_terms_are_zero_for_a_constant_field_on_the_fully_periodic_mesh():
    item = step7.preset("per3")
    kernels = multiblock.compile_blocks(item.built, item.scale, vis_order=8)
    trace, = multiblock.solve(
        kernels, DT, 1, initial=step7.states_from_field(item.built, step7.free_stream))
    interiors = [support.interiors(item.built, trace.viscous[stage]) for stage in range(3)]
    for stage, part in enumerate(interiors):
        for name, value in part.items():
            if stage == 0:
                assert np.all(value == 0.0), name       # 常数场本身的粘性右端项恰好是零
            else:
                # 无粘项对常数场有 1.6e-12 的舍入，之后的阶段态不再是逐位的常数
                assert np.all(np.abs(value) <= 1e-18), (name, stage)


# ---- 4. CD6 ----

def test_cd6_on_the_two_block_mesh_matches_the_equivalent_single_block():
    field = step7.acceptance_on(step7.extent(step7.synthetic("two")))
    built, state, trace = _run("two", field, vis_order="CD6")
    want = _alone(state, field)
    got = _back(built, step7.windows(built, state), trace)
    worst = support.compare(want, got, "CD6 的粘性右端项")
    # 分得出阶数：CD8 的单块参考与 CD6 的差，远大于这里用的容差
    apart = np.max(np.abs(_alone(step7.single("two"), field) - want))
    assert apart > 1000.0 * max(tolerance for _, tolerance in worst.values())


# ---- 5. 判别力 ----

def test_turning_off_only_the_ev_exchange_is_caught(monkeypatch):
    built, state, trace = _run("stair", step7.linear_field)
    note = step7.windows(built, state)
    want = _alone(state, step7.linear_field)
    right = _back(built, note, trace)
    monkeypatch.setattr(multiblock, "exchange",
                        step7.exchange_without_ev(multiblock.exchange))
    kernels = multiblock.compile_blocks(built, step7.SYNTHETIC_SCALE, vis_order=8)
    start = step7.states_from_field(built, step7.linear_field)
    broken = multiblock.solve_step(kernels, start, DT, order=list(range(len(kernels.tables))))
    got = _back(built, note, broken)
    with pytest.raises(AssertionError, match="超出容差"):
        support.compare(want, got, "不换 Ev 的阶梯网格")
    gap = np.max(np.abs(got[1:5] - want[1:5]))
    assert gap > 1.0e6 * np.max(np.abs(right[1:5] - want[1:5]))      # 远超正确的差
    for name in kernels.names:                    # 守恒量的交换没动：第一步准备且交换过的状态逐位相同
        _same(broken.prepared[0][name], trace.prepared[0][name])
