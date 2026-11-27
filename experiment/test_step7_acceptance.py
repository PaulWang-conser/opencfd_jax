"""第七步的第一层验收：五个验收网格打开粘性，与等价单块逐阶段一致（U6）。

两边都用验收初场，都打开粘性，各跑两个整步。粘性导数的阶数 CD8、CD6 各验一遍（CD6 是
补尾巴 C1 加的）。单块走正式入口的装配，每个阶段的粘性分项用 ``viscous_rhs`` 在同一阶段
的准备态上另算（KTD9）；多块走求解函数，各块的内区按旁记拼回全局再比。比较项：三个 RK
阶段的总项和粘性分项、准备之前的 q1 和 q2、整步后的末态。

阈值用第六步的；每个量分四层报告：全域、接口邻域、物理边界带、其余部分。物理边界带是
各块里紧挨着打开了单侧公式的那一面的几层（壁面、远面）。切口面不套单侧公式，所以它不在
物理边界带里，只在接口邻域里。

另外锁五件事：
  1. 切口共有的内区点上，各块粘性分项的两份副本一致，并与单块一致；
  2. 比较器比 ``Ev`` 时看得到第 12 个平面；拿缺省的 5 个分量去比 12 平面的数组会直接报错，
     不会悄悄只比前 5 个；
  3. 四种故意引入的错误都被发现：不换 ``Ev``、换 ``Ev`` 时取错层、切口面被套上单侧公式、
     RK 组合用了准备过的状态。每一条只动要试的那一处，两种阶数都验；
  4. 切口面的边界带标志都是 0；
  5. 第六步的无粘验收原样通过（在 ``test_step6_acceptance.py`` 里，阈值不变）。

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
ORDERS = ("CD8", "CD6")
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")

_check, _neighbourhood = acceptance._check, acceptance._neighbourhood
_chains, _solved = {}, {}       # 每种阶数的单块参考和多块运行，各算一次


def _kernels(preset, vis_order="CD8"):
    """验收网格粘性打开的求解函数；``vis_order`` 是粘性导数的阶数名。"""
    item = step7.preset(preset)
    return multiblock.compile_blocks(item.built, item.scale,
                                     vis_order=multiblock.CD_ORDER[vis_order])


def _stages(trace, reference, inner):
    """一个整步要比的八个量：(名字, 多块的各块, 单块的内区)。"""
    pairs = [(f"du{s + 1}", trace.du[s], reference.du[s][inner]) for s in range(3)]
    pairs += [(f"粘性分项{s + 1}", trace.viscous[s], reference.viscous[s][inner])
              for s in range(3)]
    return pairs + [("q1", trace.q1, reference.inputs[1][inner]),
                    ("q2", trace.q2, reference.inputs[2][inner]),
                    ("末态", trace.final, reference.final[inner])]


def _chain(vis_order):
    """打开粘性的单块两个整步：每步三个阶段的总项、粘性分项、RHS 的三次入参、末态。"""
    if vis_order not in _chains:
        state = step7.reference(vis_order)
        start = support.with_field(state, support.acceptance_state)
        staged, cons = [], start
        for _ in range(STEPS):
            trace = support.formal_trace(state, cons, DT)
            trace.viscous = tuple(
                step7.viscous_stage(state, state.execution.prepare(jnp.asarray(q)))
                for q in trace.inputs)
            staged.append(trace)
            cons = trace.final
        _chains[vis_order] = SimpleNamespace(state=state, start=start, staged=staged)
    return _chains[vis_order]


def _runs(vis_order):
    """五个验收网格各自两个整步的记录。起点是验收初场按旁记切给各块。"""
    if vis_order not in _solved:
        _solved[vis_order] = {
            preset: multiblock.solve(_kernels(preset, vis_order), DT, STEPS,
                                     initial=step7.preset(preset).initial)
            for preset in PRESETS}
    return _solved[vis_order]


# ---- 五个网格逐阶段对照 ----

@pytest.mark.parametrize("vis_order", ORDERS)
@pytest.mark.parametrize("preset", PRESETS)
def test_every_stage_matches_the_viscous_single_block(preset, vis_order):
    chain, runs = _chain(vis_order), _runs(vis_order)
    built, note = support.blocks(preset), support.sidecar(preset)
    near = _neighbourhood(preset)
    band = step7.physical_band(built, note)
    assert near.any() and band.any() and not (near | band).all()
    lines = []
    inner = chain.state.desc.grid.inner()
    for n, (trace, reference) in enumerate(zip(runs[preset], chain.staged)):
        for what, pieces, alone in _stages(trace, reference, inner):
            lines.append(_check(f"{preset} 第 {n + 1} 步 {what}", alone, built, note,
                                pieces, near, band))
    print("\n".join(lines))
    assert all(all(region in line for region in ("全域", "接口邻域", "物理边界带", "其余"))
               for line in lines)


def test_physical_band_follows_the_band_flags():
    """物理边界带跟着各块的边界带标志走。验收网格上只有壁面（j−）打开了单侧公式：带是紧挨
    壁面的 4 层，入口、远场、出口、周期面旁边都没有。t3 的 ``blk-rhi`` 没有壁面，它的范围里
    一个点也不在带里。"""
    built, note = support.blocks("x2"), support.sidecar("x2")
    band = step7.physical_band(built, note)
    assert band.shape == (40, 40, 100) == tuple(note["global_vertices"][::-1])
    assert band[:, :LAP, :].all()
    assert not band[:, LAP:, :].any()
    bare, note = support.blocks("t3"), support.sidecar("t3")
    without = step7.physical_band(bare, note)
    top = next(entry for entry in note["blocks"] if entry["name"] == "blk-rhi")
    window = tuple(slice(start, stop) for start, stop in zip(top["start"][::-1], top["stop"][::-1]))
    assert without.sum() == LAP * 40 * 100                 # 三块里有两块带壁面
    assert not without[window].any()


@pytest.mark.parametrize("vis_order", ORDERS)
def test_viscous_copies_agree_at_the_shared_points(vis_order):
    chain, runs = _chain(vis_order), _runs(vis_order)
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


# ---- 切口面不套单侧公式 ----

@pytest.mark.parametrize("preset", PRESETS)
def test_cut_faces_do_not_carry_the_physical_band(preset):
    """切口面用的是邻块换过来的值，不套物理边界的单侧公式：切口两侧的面上，边界带的标志
    都是 0。"""
    built = support.blocks(preset)
    checked = 0
    for pair in support.topology(preset).interfaces:
        for zone, side in ((pair.zone, pair.side), (pair.donor_zone, pair.donor_side)):
            flag = built[zone].execution.band[2 * side_axis(side) + int(side.endswith("+"))]
            assert flag == 0, (preset, zone, side)
            checked += 1
    assert checked


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


@pytest.mark.parametrize("vis_order", ORDERS)
def test_skipping_the_ev_exchange_is_caught(monkeypatch, vis_order):
    chain, runs = _chain(vis_order), _runs(vis_order)
    built, note = support.blocks("x2"), support.sidecar("x2")
    kernels = _kernels("x2", vis_order)
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


@pytest.mark.parametrize("vis_order", ORDERS)
def test_taking_the_wrong_donor_layer_is_caught(monkeypatch, vis_order):
    chain, runs = _chain(vis_order), _runs(vis_order)
    built, note = support.blocks("x2"), support.sidecar("x2")
    kernels = _kernels("x2", vis_order)
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


@pytest.mark.parametrize("vis_order", ORDERS)
def test_a_one_sided_formula_on_the_cut_face_is_caught(vis_order):
    """只改粘性用的切口一侧的标志：只给改了边界带的块重编粘性的两段，无粘的部分仍用
    原来那套。"""
    chain, runs = _chain(vis_order), _runs(vis_order)
    built, note = support.blocks("x2"), support.sidecar("x2")
    cuts = set()
    for pair in support.topology("x2").interfaces:
        if pair.kind == multiblock.CUT:
            cuts |= {(pair.zone, pair.side), (pair.donor_zone, pair.donor_side)}
    assert cuts
    order = multiblock.CD_ORDER[vis_order]
    changed = []
    for block in built.blocks:
        band = list(block.execution.band)
        for zone, side in cuts:
            if zone == block.name:
                band[2 * side_axis(side) + int(side.endswith("+"))] = 1
        changed.append(replace(block, execution=replace(block.execution, band=tuple(band))))
    kernels = _kernels("x2", vis_order)
    hybrid = copy.copy(kernels)
    hybrid.flux = [multiblock.flux_kernel(block, order) for block in changed]
    hybrid.divergence = [multiblock.divergence_kernel(block, order) for block in changed]
    trace = multiblock.solve_step(hybrid, step7.preset("x2").initial, DT)
    inner = chain.state.desc.grid.inner()
    with pytest.raises(AssertionError, match="粘性分项.*超出容差") as caught:
        _check("故意在切口面套单侧公式的粘性分项", chain.staged[0].viscous[0][inner],
               built, note, trace.viscous[0], _neighbourhood("x2"))
    print(caught.value)
    _only_viscosity_changed(trace, kernels, runs["x2"][0])


@pytest.mark.parametrize("vis_order", ORDERS)
def test_using_the_prepared_state_in_the_combination_is_caught(vis_order):
    chain, runs = _chain(vis_order), _runs(vis_order)
    built, note = support.blocks("x2"), support.sidecar("x2")
    trace = runs["x2"][0]
    wrong = {name: solver.rk3_q1(trace.prepared[0][name], DT, trace.du[0][name])
             for name in trace.q1}
    with pytest.raises(AssertionError, match="q1"):
        _check("故意用准备过的状态组合 q1",
               chain.staged[0].inputs[1][chain.state.desc.grid.inner()],
               built, note, wrong, _neighbourhood("x2"))
