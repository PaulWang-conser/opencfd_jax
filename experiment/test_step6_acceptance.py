"""第六步的验收（U6）：五个验收网格与等价单块逐阶段一致。

两边都用验收初场（R16），都关粘性项，各跑两个整步。单块走正式入口的装配，多块
走求解函数，各块的内区按旁记拼回全局再比。比较项：三个 RK 阶段的 du、准备之前
的 q1 和 q2、整步后的末态。阈值见 ``step6_support.RTOL``；每个量分三层报告：
全域、接口邻域、其余部分。

另外锁四件事：共享顶点层的各份副本在每个阶段都一致；t3 的共线处与接口段和块的
遍历顺序无关；验收有判别力（把准备过的状态用于组合、把参考尺度换成本块尺寸，
各自被抓到）；存盘文件读回并拼回后与单块末态一致。
"""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import io, main, multiblock, solver  # noqa: E402
from opencfd_jax.src.axes import side_axis  # noqa: E402
from opencfd_jax.src.boundary import LAP  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402

DT = 1e-4
STEPS = 2
PRESETS = ("x2", "y2", "z2", "xyz8", "t3")
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")


_same = support.same_bits
_interiors = support.interiors


def _window(entry):
    """旁记里一块的范围在全局内区数组 (5, Nk, Nj, Ni) 上的切片。"""
    return (slice(None),) + support.block_window(entry)


def _neighbourhood(preset, layers=LAP):
    """接口邻域：离任何一条内部切口或周期接缝不超过 ``layers`` 个顶点的全局点。"""
    note = support.sidecar(preset)
    shape = note["global_vertices"][::-1]                     # (Nk, Nj, Ni)
    start = {b["name"]: b["start"] for b in note["blocks"]}
    points = {b["name"]: [stop - begin for begin, stop in zip(b["start"], b["stop"])]
              for b in note["blocks"]}
    mask = np.zeros(shape, dtype=bool)
    for pair in support.topology(preset).interfaces:
        for zone, side in ((pair.zone, pair.side), (pair.donor_zone, pair.donor_side)):
            axis = side_axis(side)
            plane = start[zone][axis] + (0 if side.endswith("-") else points[zone][axis] - 1)
            index = [None] * 3                           # 数组轴序 (k, j, i)
            for a in range(3):
                index[2 - a] = slice(start[zone][a], start[zone][a] + points[zone][a])
            index[2 - axis] = slice(max(plane - layers + 1, 0),
                                    min(plane + layers - 1, shape[2 - axis] - 1) + 1)
            mask[tuple(index)] = True
    return mask


def _worst(worst):
    """容差用得最多的变量：名字、最大绝对差、占容差的比例。"""
    v = max(worst, key=lambda v: worst[v][0] / worst[v][1] if worst[v][1] else 0.0)
    share = worst[v][0] / worst[v][1] if worst[v][1] else 0.0
    return f"cons_{v} 差 {worst[v][0]:.2e}（容差的 {100 * share:.3g}%）"


def _check(label, reference, built, note, pieces, near, band=None):
    """先逐块比（报错时能点名块），再拼回比分层比。返回这个量的报告行。

    分层是全域、接口邻域、其余；给出 ``band``（物理边界带，第七步）时多比一层，其余也不含它。
    """
    interiors = _interiors(built, pieces)
    for entry in note["blocks"]:
        support.compare(reference[_window(entry)], interiors[entry["name"]],
                        f"{label} {entry['name']}")
    back = support.reassemble(note, interiors)
    rest = near if band is None else near | band
    layers = [("全域", None), ("接口邻域", near)]
    if band is not None:
        layers.append(("物理边界带", band))
    parts = []
    for name, mask in [*layers, ("其余", ~rest)]:
        worst = support.compare(reference, back, f"{label} {name}", mask=mask)
        parts.append(f"{name} {_worst(worst)}")
    return f"{label}：" + "；".join(parts)


def _stages(trace, reference, inner):
    """一个整步要比的六个量：(名字, 多块的各块, 单块的内区)。"""
    pairs = [(f"du{s + 1}", trace.du[s], reference.du[s][inner]) for s in range(3)]
    return pairs + [("q1", trace.q1, reference.inputs[1][inner]),
                    ("q2", trace.q2, reference.inputs[2][inner]),
                    ("末态", trace.final, reference.final[inner])]


def _grid(rng, origin=(0, 0, 0)):
    """有向范围铺成可以逐点取的下标：数组轴序 (k, j, i) 的三个广播数组。

    每条轴按书写的方向排，再加上 ``origin``（计算轴序 (i, j, k)）。
    """
    axes = [np.arange(lo, hi + 1) if hi >= lo else np.arange(lo, hi - 1, -1)
            for lo, hi in zip(rng[0], rng[1])]
    return tuple(value + shift for value, shift in
                 zip(np.meshgrid(*axes[::-1], indexing="ij"), origin[::-1]))


def _shared(preset):
    """内部切口两侧共有的顶点层，每项给出两块各自内区数组上的下标和全局下标。"""
    start = {b["name"]: b["start"] for b in support.sidecar(preset)["blocks"]}
    out = []
    for pair in support.topology(preset).interfaces:
        if pair.kind == multiblock.CUT:
            out.append(SimpleNamespace(
                name_a=pair.zone, name_b=pair.donor_zone,
                a=(slice(None),) + _grid(pair.point_range, (LAP,) * 3),
                b=(slice(None),) + _grid(pair.point_range_donor, (LAP,) * 3),
                whole=(slice(None),) + _grid(pair.point_range, start[pair.zone])))
    return out


def _assert_same_trace(one, other):
    for name in one.final:
        for stage in range(3):
            _same(other.du[stage][name], one.du[stage][name])
        _same(other.q1[name], one.q1[name])
        _same(other.q2[name], one.q2[name])
        _same(other.final[name], one.final[name])


@pytest.fixture(scope="module")
def chain():
    """单块的两个整步：逐步走的记录（三个 du、RHS 的三次入参、末态）和整步 jit 的末态。"""
    state = support.reference()
    start = support.with_field(state, support.acceptance_state)
    staged, whole, cons = [], [], start
    for _ in range(STEPS):
        trace = support.formal_trace(state, cons, DT)
        staged.append(trace)
        cons = trace.final
    cons = start
    for _ in range(STEPS):
        cons = support.formal_step(state, cons, DT)
        whole.append(cons)
    return SimpleNamespace(state=state, start=start, staged=staged, whole=whole)


@pytest.fixture(scope="module")
def runs(chain):
    """五个验收网格各自两个整步的记录。"""
    out = {}
    for preset in PRESETS:
        initial = support.split_field(
            support.blocks(preset), support.sidecar(preset),
            chain.start[chain.state.desc.grid.inner()])
        out[preset] = multiblock.solve(support.kernels(preset), DT, STEPS, initial=initial)
    return out


# ---- R16：验收初场 ----

def test_acceptance_field_violates_the_boundary_conditions(chain):
    state, start = chain.state, chain.start
    inner = state.desc.grid.inner()
    field = start[inner]
    u, v, w = (field[i] / field[0] for i in (1, 2, 3))
    temperature = (field[4] - 0.5 * field[0] * (u * u + v * v + w * w)) / (
        field[0] * state.case["params"].Cv)
    assert np.all(np.abs(field[0] - 1) < 0.10)
    assert np.all(np.abs(u - 1) < 0.05)
    assert np.all(np.abs(v) < 0.03) and np.all(np.abs(w) < 0.03)
    assert np.all(np.abs(temperature - 1) < 0.05)
    rho = field[0]                           # 三个方向都在变
    for line in (rho[0, 0, :], rho[0, :, 0], rho[:, 0, 0]):
        assert np.ptp(line) > 1e-4
    prepared = np.asarray(state.execution.prepare(jnp.asarray(start)))
    changed = (prepared != start)[inner]
    assert changed[..., 0].any()             # 入口面 i=0
    assert changed[:, 0, :, :].any()         # 壁面 j=0


# ---- 五个网格逐阶段对照 ----

def test_staged_reference_matches_the_whole_step_jit(chain):
    inner = chain.state.desc.grid.inner()
    for n, (trace, whole) in enumerate(zip(chain.staged, chain.whole)):
        support.compare(whole[inner], trace.final[inner],
                        f"单块第 {n + 1} 步：逐步走对照整步 jit")


@pytest.mark.parametrize("preset", PRESETS)
def test_every_stage_matches_the_single_block(preset, chain, runs):
    built, note = support.blocks(preset), support.sidecar(preset)
    near = _neighbourhood(preset)
    assert near.any() and not near.all()
    for block in built.blocks:               # 接口面不套边界带
        wall = any(face.kind == "wall" for face in block.faces)
        assert block.execution.band == (0, 0, int(wall), 0, 0, 0)
    lines = []
    inner = chain.state.desc.grid.inner()
    for n, (trace, ref) in enumerate(zip(runs[preset], chain.staged)):
        for what, pieces, reference in _stages(trace, ref, inner):
            lines.append(_check(f"{preset} 第 {n + 1} 步 {what}", reference,
                                built, note, pieces, near))
        for prepared in trace.prepared:      # 准备态上的接口四层是直接复制的
            for table in built.tables:
                it = table.interface
                _same(prepared[it.zone][(slice(None),) + tuple(table.slots)],
                      prepared[it.donor_zone][(slice(None),) + tuple(table.sources)])
    print("\n".join(lines))
    assert all("全域" in line and "接口邻域" in line and "其余" in line for line in lines)


@pytest.mark.parametrize("preset", PRESETS)
def test_shared_points_agree_between_copies(preset, chain, runs):
    shared = _shared(preset)
    assert shared
    inner = chain.state.desc.grid.inner()
    for n, (trace, ref) in enumerate(zip(runs[preset], chain.staged)):
        for what, pieces, reference in _stages(trace, ref, inner):
            for item in shared:
                va = np.asarray(pieces[item.name_a])[item.a]
                vb = np.asarray(pieces[item.name_b])[item.b]
                where = f"{preset} 第 {n + 1} 步 {what}：{item.name_a} 与 {item.name_b} 的副本"
                support.compare(va, vb, where)
                support.compare(reference[item.whole], va, where + "对照单块")


# ---- t3 的共线处与遍历顺序 ----

def test_t_junction_is_independent_of_traversal_order(chain):
    built = support.blocks("t3")
    initial = support.split_field(
        built, support.sidecar("t3"), chain.start[chain.state.desc.grid.inner()])
    kernels = support.kernels("t3")
    straight = multiblock.solve_step(kernels, initial, DT, audit=True)
    tables = list(range(len(kernels.tables)))
    for order in (tables[::-1], tables[2::3] + tables[0::3] + tables[1::3]):
        _assert_same_trace(straight,
                           multiblock.solve_step(kernels, initial, DT, order=order, audit=True))
    backward = multiblock.compile_blocks(                  # 块的次序倒过来另编一套
        replace(built, blocks=built.blocks[::-1], kernels={}), support.FLUX_SCALE)
    _assert_same_trace(straight, multiblock.solve_step(backward, initial, DT, audit=True))
    second = multiblock.solve(kernels, DT, 2, initial=initial, audit=True)[1]
    for audits in second.audits:
        assert len(audits) == 1             # 左块 i+ 的两段在 j=20 这条线上共线
        support.compare(np.asarray(audits[0].written), np.asarray(audits[0].candidate),
                        "t3 共线处")


# ---- 判别力 ----

def test_using_the_prepared_state_in_the_combination_is_caught(chain, runs):
    built, note = support.blocks("x2"), support.sidecar("x2")
    trace = runs["x2"][0]
    wrong = {name: solver.rk3_q1(trace.prepared[0][name], DT, trace.du[0][name])
             for name in trace.q1}
    with pytest.raises(AssertionError, match="q1"):
        _check("故意用准备过的状态组合 q1",
               chain.staged[0].inputs[1][chain.state.desc.grid.inner()],
               built, note, wrong, _neighbourhood("x2"))


def test_a_block_sized_flux_scale_is_caught(chain):
    built, note = support.blocks("x2"), support.sidecar("x2")
    initial = support.split_field(
        built, note, chain.start[chain.state.desc.grid.inner()])
    left = (1521.0, 1950.0, 1950.0)          # 左块 51×40×40 按自己的尺寸算出的尺度
    kernels = multiblock.compile_blocks(built, left)
    trace = multiblock.solve_step(kernels, initial, DT)
    with pytest.raises(AssertionError, match="du1"):
        _check("故意把参考尺度换成本块尺寸 du1",
               chain.staged[0].du[0][chain.state.desc.grid.inner()],
               built, note, trace.du[0], _neighbourhood("x2"))


# ---- 落盘 ----

def test_saved_files_reassemble_to_the_single_block_state(tmp_path):
    """正式入口的多块运行从默认初场出发，单块也从同一个初场走正式入口。"""
    state = support.reference()
    cons = state.cons0
    for _ in range(STEPS):
        cons = support.formal_step(state, cons, DT)
    out = tmp_path / "out"
    out.mkdir()
    main.run(support.run_args("x2", out_dir=str(out), end_time=STEPS * DT,
                              end_step=STEPS, kstep_save=STEPS))
    [path] = out.iterdir()
    meta, blocks, fields = io.load_checkpoint(str(path))
    assert meta["step"] == STEPS and meta["t"] == pytest.approx(STEPS * DT)
    assert meta["dt"] == pytest.approx(DT) and "zone" not in meta
    assert [entry["name"] for entry in blocks["blocks"]] == ["blk-i0j0k0", "blk-i1j0k0"]
    back = support.reassemble(blocks, {
        entry["name"]: np.asarray(io.conservative_of(field, state.case["params"]))
        for entry, field in zip(blocks["blocks"], fields)})
    support.compare(np.asarray(cons)[state.desc.grid.inner()], back, "存盘读回")
