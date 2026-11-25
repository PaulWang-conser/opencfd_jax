"""第七步的第一层验收：五个验收网格打开粘性，与等价单块逐阶段一致（U6）。

两边都用验收初场，都打开粘性（CD8），各跑两个整步。单块走正式入口的装配，每个阶段的
粘性分项用 ``viscous_rhs`` 在同一阶段的准备态上另算（KTD9）；多块走求解函数，各块的内区
按旁记拼回全局再比。比较项：三个 RK 阶段的总项和粘性分项、准备之前的 q1 和 q2、整步后
的末态。阈值用第六步的；每个量分三层报告：全域、接口邻域、其余部分。

另外锁四件事：
  1. 切口共有的内区点上，各块粘性分项的两份副本一致，并与单块一致；
  2. 比较器比 ``Ev`` 时看得到第 12 个平面；拿缺省的 5 个分量去比 12 平面的数组会直接报错，
     不会悄悄只比前 5 个；
  3. 四种故意引入的错误都被发现：不换 ``Ev``、换 ``Ev`` 时取错层、切口面被套上单侧公式、
     RK 组合用了准备过的状态。每一条只动要试的那一处；
  4. 第六步的无粘验收原样通过（在 ``test_step6_acceptance.py`` 里，阈值不变）。

报告行、接口邻域和共有的顶点层用 ``test_step6_acceptance`` 里的辅助函数。
"""

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import multiblock, solver  # noqa: E402
from opencfd_jax.src.axes import side_axis  # noqa: E402
from opencfd_jax.src.boundary import LAP  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402
import test_step6_acceptance as acceptance  # noqa: E402

DT = 1e-4
STEPS = 2
PRESETS = acceptance.PRESETS
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")

_check, _neighbourhood = acceptance._check, acceptance._neighbourhood


def _kernels(preset):
    """验收网格粘性打开（CD8）的求解函数。"""
    item = step7.preset(preset)
    return multiblock.compile_blocks(item.built, item.scale, vis_order=8)


def _stages(trace, reference, inner):
    """一个整步要比的八个量：(名字, 多块的各块, 单块的内区)。"""
    pairs = [(f"du{s + 1}", trace.du[s], reference.du[s][inner]) for s in range(3)]
    pairs += [(f"粘性分项{s + 1}", trace.viscous[s], reference.viscous[s][inner])
              for s in range(3)]
    return pairs + [("q1", trace.q1, reference.inputs[1][inner]),
                    ("q2", trace.q2, reference.inputs[2][inner]),
                    ("末态", trace.final, reference.final[inner])]


@pytest.fixture(scope="module")
def chain():
    """打开粘性的单块两个整步：每步三个阶段的总项、粘性分项、RHS 的三次入参、末态。"""
    state = step7.reference()
    start = support.with_field(state, support.acceptance_state)
    staged, cons = [], start
    for _ in range(STEPS):
        trace = support.formal_trace(state, cons, DT)
        trace.viscous = tuple(
            step7.viscous_stage(state, state.execution.prepare(jnp.asarray(q)))
            for q in trace.inputs)
        staged.append(trace)
        cons = trace.final
    return SimpleNamespace(state=state, start=start, staged=staged)


@pytest.fixture(scope="module")
def runs():
    """五个验收网格各自两个整步的记录。起点是验收初场按旁记切给各块。"""
    return {preset: multiblock.solve(_kernels(preset), DT, STEPS,
                                     initial=step7.preset(preset).initial)
            for preset in PRESETS}


# ---- 五个网格逐阶段对照 ----

@pytest.mark.parametrize("preset", PRESETS)
def test_every_stage_matches_the_viscous_single_block(preset, chain, runs):
    built, note = support.blocks(preset), support.sidecar(preset)
    near = _neighbourhood(preset)
    lines = []
    inner = chain.state.desc.grid.inner()
    for n, (trace, reference) in enumerate(zip(runs[preset], chain.staged)):
        for what, pieces, alone in _stages(trace, reference, inner):
            lines.append(_check(f"{preset} 第 {n + 1} 步 {what}", alone, built, note,
                                pieces, near))
    print("\n".join(lines))
    assert all("全域" in line and "接口邻域" in line and "其余" in line for line in lines)


def test_viscous_copies_agree_at_the_shared_points(chain, runs):
    inner = chain.state.desc.grid.inner()
    for preset in PRESETS:
        for n, trace in enumerate(runs[preset]):
            for s in range(3):
                for item in acceptance._shared(preset):
                    one = np.asarray(trace.viscous[s][item.name_a])[item.a]
                    other = np.asarray(trace.viscous[s][item.name_b])[item.b]
                    where = f"{preset} 第 {n + 1} 步的粘性分项 {item.name_a} 与 {item.name_b} 的副本"
                    support.compare(one, other, where)
                    support.compare(chain.staged[n].viscous[s][inner][item.whole], one,
                                    where + "对照单块")


# ---- 比较器 ----

def test_compare_sees_the_twelfth_plane_when_asked():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(12, 6, 5, 4))
    damaged = reference.copy()
    damaged[11] *= 1.0 + 1e-9
    support.compare(reference[:5], damaged[:5])              # 只看前 5 个平面时发现不了
    support.compare(reference, reference.copy(), count=12)
    with pytest.raises(AssertionError, match=r"分量_11 .*差 [0-9.e+-]+，容差"):
        support.compare(reference, damaged, count=12)
    with pytest.raises(AssertionError, match="分量的个数"):   # 不会悄悄只比前 5 个平面
        support.compare(reference, damaged)


# ---- 四种故意引入的错误 ----

def _only_viscosity_changed(trace, kernels, correct):
    """故意引入的错误只动在粘性上：``trace`` 第一步准备且交换过的状态与正确的运行
    ``correct`` 逐位相同，它的总项减去粘性分项就是无粘右端项。"""
    for n, name in enumerate(kernels.names):
        assert np.array_equal(np.asarray(trace.prepared[0][name]),
                              np.asarray(correct.prepared[0][name]))
        inviscid = np.asarray(kernels.rhs[n](correct.prepared[0][name]))
        np.testing.assert_allclose(                 # 减法本身有舍入，只要求到舍入量级
            np.asarray(trace.du[0][name] - trace.viscous[0][name]), inviscid,
            rtol=1e-12, atol=1e-12)


def test_skipping_the_ev_exchange_is_caught(chain, runs, monkeypatch):
    built, note = support.blocks("x2"), support.sidecar("x2")
    kernels = _kernels("x2")
    monkeypatch.setattr(multiblock, "exchange",
                        step7.exchange_without_ev(multiblock.exchange))
    trace = multiblock.solve_step(kernels, step7.preset("x2").initial, DT,
                                  order=list(range(len(kernels.tables))))
    inner = chain.state.desc.grid.inner()
    with pytest.raises(AssertionError, match="粘性分项.*超出容差") as caught:
        _check("故意不换 Ev 的粘性分项", chain.staged[0].viscous[0][inner], built, note,
               trace.viscous[0], _neighbourhood("x2"))
    print(caught.value)
    _only_viscosity_changed(trace, kernels, runs["x2"][0])


def test_taking_the_wrong_donor_layer_is_caught(chain, runs, monkeypatch):
    built, note = support.blocks("x2"), support.sidecar("x2")
    kernels = _kernels("x2")
    real = multiblock.exchange

    def shifted(tables, pieces):
        """只动 Ev：供体的下标沿接口的法向挪一层，仍在内区里。"""
        first = next(iter(pieces.values()))
        if first.shape[0] != 12:
            return real(tables, pieces)
        moved = []
        for table in tables:
            it = table.interface
            sources = table.sources.copy()
            row = 2 - side_axis(it.side)                     # 数组轴
            n = built[it.donor_zone].grid.n_inner[row]
            sources[row] += -1 if sources[row].max() + 1 >= LAP + n else 1
            assert sources[row].min() >= LAP and sources[row].max() < LAP + n
            moved.append(replace(table, sources=sources))
        return real(moved, pieces)

    monkeypatch.setattr(multiblock, "exchange", shifted)
    trace = multiblock.solve_step(kernels, step7.preset("x2").initial, DT,
                                  order=list(range(len(kernels.tables))))
    inner = chain.state.desc.grid.inner()
    with pytest.raises(AssertionError, match="粘性分项.*超出容差") as caught:
        _check("故意取错供体层的粘性分项", chain.staged[0].viscous[0][inner], built, note,
               trace.viscous[0], _neighbourhood("x2"))
    print(caught.value)
    _only_viscosity_changed(trace, kernels, runs["x2"][0])


def test_a_one_sided_formula_on_the_cut_face_is_caught(chain, runs):
    """只改粘性用的切口一侧的标志：只给改了边界带的块重编粘性的两段，无粘的部分仍用
    原来那套。"""
    built, note = support.blocks("x2"), support.sidecar("x2")
    cuts = set()
    for pair in support.topology("x2").interfaces:
        if pair.kind == multiblock.CUT:
            cuts |= {(pair.zone, pair.side), (pair.donor_zone, pair.donor_side)}
    assert cuts
    changed = []
    for block in built.blocks:
        band = list(block.execution.band)
        for zone, side in cuts:
            if zone == block.name:
                band[2 * side_axis(side) + int(side.endswith("+"))] = 1
        changed.append(replace(block, execution=replace(block.execution, band=tuple(band))))
    kernels = _kernels("x2")
    hybrid = copy.copy(kernels)
    hybrid.flux = [multiblock.flux_kernel(block, 8) for block in changed]
    hybrid.divergence = [multiblock.divergence_kernel(block, 8) for block in changed]
    trace = multiblock.solve_step(hybrid, step7.preset("x2").initial, DT)
    inner = chain.state.desc.grid.inner()
    with pytest.raises(AssertionError, match="粘性分项.*超出容差") as caught:
        _check("故意在切口面套单侧公式的粘性分项", chain.staged[0].viscous[0][inner],
               built, note, trace.viscous[0], _neighbourhood("x2"))
    print(caught.value)
    _only_viscosity_changed(trace, kernels, runs["x2"][0])


def test_using_the_prepared_state_in_the_combination_is_caught(chain, runs):
    built, note = support.blocks("x2"), support.sidecar("x2")
    trace = runs["x2"][0]
    wrong = {name: solver.rk3_q1(trace.prepared[0][name], DT, trace.du[0][name])
             for name in trace.q1}
    with pytest.raises(AssertionError, match="q1"):
        _check("故意用准备过的状态组合 q1",
               chain.staged[0].inputs[1][chain.state.desc.grid.inner()],
               built, note, wrong, _neighbourhood("x2"))
