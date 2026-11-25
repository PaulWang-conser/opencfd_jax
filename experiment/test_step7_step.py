"""第七步：多块粘性的一步，块角的鬼胞不参与，接口两侧的 Ev 一致（U3）。

锁八件事：
  1. ``_band`` 按定义对照。它选出内区里紧挨某个侧面的几层，后面的对照靠它定位。
  2. 四处注入都到了会被读的位置。守恒量和 Ev 的对照把 NaN 放在接口面内的鬼胞上，
     度量和坐标的对照放在紧挨接口的内区层上；内区紧挨接口的各层出现 NaN。
  3. 块角的鬼胞不参与内区的粘性右端项。五个合成小网格上，坐标、度量、守恒量、Ev
     四处分别把块角换成 NaN：内区三个阶段的粘性分项逐位不变；坐标那一处，内区的
     度量逐位不变并且有限。
  4. 每个阶段的粘性分项可以从准备之前的阶段态重算：准备、换守恒量、算 Ev、换 Ev、
     求导，逐位相同；q1、q2 是准备之前的值；整步后的末态只做准备和守恒量交换。
  5. 求解路径里 Ev 按交换表换过：接收槽上的值与供体内区逐位相同；周期接缝两侧取的
     是对侧内区按周期映射对应的点。
  6. 普通切口共有的内区点上，两块各自算出的 Ev 在阈值内一致；T 形共线处拥有者的 Ev
     与被让掉的候选一致。
  7. 六个面全是接口的块（per3 的中心块）跳过物理准备，Ev 的鬼胞全靠交换；跑一步，
     各阶段的粘性分项、总项和末态与等价的单块在阈值内一致。
  8. 装配时不给等价单块的全局顶点数（没有旁记的网格），短块物理面旁的坐标鬼层按自己的
     轴长选规则；一个整步的结果与给了全局顶点数时逐位相同。

度量在粘性项里是逐点用的，内区的粘性分项不读鬼胞上的度量；坐标的导数沿一条轴取，
块角（两条以上的轴同时出界）的坐标只进得到鬼胞度量。所以度量和坐标的对照不能放在
鬼胞上，放在那里不会有人读。

数组的轴序：状态是 (5, k, j, i)，Ev 是 (12, k, j, i)，平面的排法见
``viscous.viscous_fluxes``。网格、掩码和单块参考见 ``step7_support``；切口共有的顶点层
用第六步验收测试里的 ``_shared``。
"""

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import mesh, mesh_cgns, multiblock  # noqa: E402
from opencfd_jax.src.axes import side_axis  # noqa: E402
from opencfd_jax.src.boundary import LAP, PERIODIC  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402
import test_step6_acceptance as acceptance  # noqa: E402
from Grid.make_cgns_fixture import build_multizone  # noqa: E402

DT = 1e-4
CENTRAL = ((0, 0), (0, 0), (0, 0))       # 逐侧的标志全 0：边界带也走中心差分
# 含沿某条轴的导数的度量（compute_metrics_from_coords 的算式）：Ak = yj·zk − zj·yk 一类，
# 含沿 j、k 的导数；Ai 含沿 k、i 的；As 含沿 i、j 的
_DERIVED = {"i": ("Aix", "Aiy", "Aiz", "Asx", "Asy", "Asz"),
            "j": ("Akx", "Aky", "Akz", "Asx", "Asy", "Asz"),
            "k": ("Akx", "Aky", "Akz", "Aix", "Aiy", "Aiz")}
PRESETS = ("x2", "y2", "z2", "xyz8", "t3")
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")

_same = support.same_bits


def _viscous(built, scale=step7.SYNTHETIC_SCALE):
    """粘性打开（CD8）的求解函数。合成小网格的参考尺度只进无粘项，这里不比无粘。"""
    return multiblock.compile_blocks(built, scale, vis_order=8)


def _start(built):
    """合成小网格各块的起点：验收初场按各块的坐标取值。"""
    return step7.states_from_field(built, step7.acceptance_on(step7.extent(built)))


def _band(grid, side, layers):
    """内区里离某个侧面不超过 ``layers`` 层的点，形状是内区 (nk, nj, ni)。"""
    axis = side_axis(side)                          # 计算轴：0 是 i，对应数组轴 2-axis
    n = grid.n_inner[2 - axis]                      # n_inner 按数组轴 (k, j, i) 排
    along = np.zeros(n, dtype=bool)
    if side.endswith("-"):                          # 两侧各写各的：轴短时两层范围会重叠
        along[:layers] = True
    else:
        along[n - layers:] = True
    shape = [1, 1, 1]
    shape[2 - axis] = n
    return np.broadcast_to(along.reshape(shape), tuple(grid.n_inner))


def _padded(grid, inner_mask):
    """把内区形状的掩码放回含鬼胞的形状。"""
    out = np.zeros(grid.shape[1:], dtype=bool)
    out[grid.inner()[1:]] = inner_mask
    return out


def _viscous_interiors(built, trace):
    """三个阶段的粘性分项在各块内区上的值：[{块名: (5, nk, nj, ni)}]。"""
    return [support.interiors(built, trace.viscous[stage]) for stage in range(3)]


def _same_viscous(one, other, note):
    for stage in range(3):
        for name in one[stage]:
            _same(one[stage][name], other[stage][name], f"{note}：第 {stage + 1} 阶段 {name}")


def _with_metrics(kernels, metrics):
    """一份拷贝的求解函数：``metrics`` 里的块（{块名: {度量名: 数组}}）换成新的度量，
    只给这些块重编粘性的两段。无粘的那几个函数读的还是原来的度量，所以用它的几条测试
    只看粘性分项。"""
    out = copy.copy(kernels)
    changed = [replace(block, metrics=MappingProxyType(metrics[block.name]))
               if block.name in metrics else block for block in kernels.blocks]
    order = kernels.vis_order
    out.flux = [multiblock.flux_kernel(block, order) if block.name in metrics else flux
                for block, flux in zip(changed, kernels.flux)]
    out.divergence = [multiblock.divergence_kernel(block, order) if block.name in metrics else part
                      for block, part in zip(changed, kernels.divergence)]
    return out


def _poisoned(kernels, step, masks):
    """一份拷贝的求解函数。``step`` 是 "flux" 或 "divergence"：那一步的输入先按
    ``masks[块名]`` 把掩码里的点换成 NaN。``masks`` 里没有的块不动。"""
    out = copy.copy(kernels)
    if step == "flux":
        out.flux = [f if name not in masks else jax.jit(
            lambda q, f=f, w=jnp.asarray(masks[name]): f(jnp.where(w, np.nan, q)))
            for name, f in zip(kernels.names, kernels.flux)]
    else:
        out.divergence = [d if name not in masks else jax.jit(
            lambda ev, d=d, w=jnp.asarray(masks[name]): d(jnp.where(w, np.nan, ev)))
            for name, d in zip(kernels.names, kernels.divergence)]
    return out


def _ready(kernels, stage):
    """物理准备，然后同步交换守恒量；与 solve_step 里的做法相同，走急切路径。"""
    names = kernels.names
    prepared = [prepare(stage[name]) if prepare is not None else stage[name]
                for prepare, name in zip(kernels.prepare, names)]
    return multiblock.exchange(kernels.tables, dict(zip(names, prepared)))


def _viscous_of(kernels, state):
    """从准备且交换过的状态算粘性分项：算 Ev、换 Ev、求导。返回 (各块的粘性分项, 换过的 Ev)。"""
    names = kernels.names
    own = {name: flux(state[name]) for name, flux in zip(names, kernels.flux)}
    moved = multiblock.exchange(kernels.tables, own)
    return ({name: divergence(moved[name])
             for name, divergence in zip(names, kernels.divergence)}, moved)


# ---- 1. _band 按定义对照 ----

@pytest.mark.parametrize("key", step7.SYNTHETIC_KEYS, ids=step7.SYNTHETIC_NAMES)
def test_band_selects_exactly_the_layers_next_to_that_side(key):
    """``_band`` 原来先写低侧、再写高侧：轴长 7、低侧取 4 层时，第 4 层被高侧的假值覆盖，
    只选到 3 层（Astra 第 24 轮）。当时的对照恰好都取在高侧，没有暴露。"""
    for block in step7.synthetic(key).blocks:
        grid = block.grid
        for axis, letter in enumerate("ijk"):          # 计算轴，对应数组轴 2-axis
            n = grid.n_inner[2 - axis]
            for sign in "-+":
                for layers in (1, 4):
                    first, last = (0, layers) if sign == "-" else (n - layers, n)
                    want = np.zeros(tuple(grid.n_inner), dtype=bool)
                    want[tuple(slice(first, last) if b == 2 - axis else slice(None)
                               for b in range(3))] = True
                    assert np.array_equal(_band(grid, letter + sign, layers), want), (
                        block.name, letter + sign, layers)


# ---- 2. 四个阳性对照 ----

@pytest.mark.parametrize("key", step7.SYNTHETIC_KEYS, ids=step7.SYNTHETIC_NAMES)
def test_each_injection_reaches_the_viscous_terms(key):
    built = step7.synthetic(key)
    kernels = _viscous(built)
    start = _start(built)
    it = built.tables[0].interface
    name, side = it.zone, it.side
    block = built[name]
    axis = side_axis(side)
    near, layer = _band(block.grid, side, 4), _band(block.grid, side, 1)
    clean = support.interiors(built, multiblock.solve_step(kernels, start, DT).viscous[0])
    assert np.any(clean[name][1:5] != 0.0)            # 干净的运行里粘性分项不为零

    # 守恒量：这个块全部接收槽上的守恒量换成 NaN。接口旁的内区点算一阶导数要读它们
    trace = multiblock.solve_step(
        _poisoned(kernels, "flux", {name: step7.received(built, name)}), start, DT)
    part = support.interiors(built, trace.viscous[0])[name]
    assert np.isnan(part[1:5][:, near]).all()

    # Ev：同一个接口面的法向那个方向的四个平面，在接收槽上换成 NaN
    mask = np.zeros((12, *block.grid.shape[1:]), dtype=bool)
    mask[4 * axis:4 * axis + 4] = step7.received(built, name, side)
    trace = multiblock.solve_step(_poisoned(kernels, "divergence", {name: mask}), start, DT)
    part = support.interiors(built, trace.viscous[0])[name]
    assert np.isnan(part[1:5][:, near]).all()

    # 度量：紧挨这个接口的内区层上，度量换成 NaN。鬼胞上的度量没人读，所以放内区
    damaged = {metric: np.where(_padded(block.grid, layer), np.nan, np.asarray(value))
               for metric, value in block.metrics.items()}
    trace = multiblock.solve_step(_with_metrics(kernels, {name: damaged}), start, DT)
    part = support.interiors(built, trace.viscous[0])[name]
    assert np.isnan(part[1:5][:, layer]).all()

    # 坐标：同一个接口面内第一层鬼胞的坐标换成 NaN，按全中心的标志重算度量
    first = (step7.face_ghost(block.grid, 2 - axis, 1, high=side.endswith("+"))
             & step7.received(built, name, side))
    assert first.any()
    coords = {key_: np.where(first, np.nan, np.asarray(value))
              for key_, value in block.coords.items()}
    metrics = mesh.compute_metrics_from_coords(coords, (1.0, 1.0, 1.0), 6, side_flags=CENTRAL)
    for metric in _DERIVED["ijk"[axis]] + ("Ajac",):   # 含沿这个面法向那条轴的导数
        assert np.isnan(np.asarray(metrics[metric])[_padded(block.grid, layer)]).all(), metric


# ---- 3. 块角不读 ----

@pytest.mark.parametrize("key", step7.SYNTHETIC_KEYS, ids=step7.SYNTHETIC_NAMES)
def test_corner_ghosts_are_never_read_by_the_viscous_terms(key):
    built = step7.synthetic(key)
    kernels = _viscous(built)
    start = _start(built)
    clean = _viscous_interiors(built, multiblock.solve_step(kernels, start, DT))
    corners = {block.name: step7.corner(block.grid) for block in built.blocks}
    assert all(mask.any() for mask in corners.values())

    # 坐标：块角的坐标换成 NaN，按全中心的逐侧标志重算度量。内区的度量逐位不变、有限
    for block in built.blocks:
        inner = block.grid.inner()[1:]
        whole = {name: np.asarray(value) for name, value in block.coords.items()}
        damaged = {name: np.where(corners[block.name], np.nan, value)
                   for name, value in whole.items()}
        one = mesh.compute_metrics_from_coords(whole, (1.0, 1.0, 1.0), 6, side_flags=CENTRAL)
        two = mesh.compute_metrics_from_coords(damaged, (1.0, 1.0, 1.0), 6, side_flags=CENTRAL)
        for metric in one:
            assert np.all(np.isfinite(np.asarray(one[metric])[inner])), metric
            _same(np.asarray(two[metric])[inner], np.asarray(one[metric])[inner],
                  f"{block.name} 的 {metric}")

    # 度量：编核之前把块角鬼胞的度量换成 NaN
    damaged = {block.name: {metric: np.where(corners[block.name], np.nan, np.asarray(value))
                            for metric, value in block.metrics.items()}
               for block in built.blocks}
    trace = multiblock.solve_step(_with_metrics(kernels, damaged), start, DT)
    _same_viscous(clean, _viscous_interiors(built, trace), "块角的度量换成 NaN")

    # 守恒量：状态交换之后、算 Ev 之前，把块角换成 NaN
    trace = multiblock.solve_step(_poisoned(kernels, "flux", corners), start, DT)
    _same_viscous(clean, _viscous_interiors(built, trace), "块角的守恒量换成 NaN")

    # Ev：交换之后、求导之前，把块角换成 NaN
    trace = multiblock.solve_step(_poisoned(kernels, "divergence", corners), start, DT)
    _same_viscous(clean, _viscous_interiors(built, trace), "块角的 Ev 换成 NaN")


# ---- 4. 每个阶段从阶段态重算 ----

@pytest.mark.parametrize("preset", (*PRESETS, "per3"))
def test_every_stage_can_be_recomputed_from_the_stage_state(preset):
    item = step7.preset(preset)
    built, initial = item.built, item.initial
    kernels = _viscous(built, item.scale)
    trace = multiblock.solve_step(kernels, initial, DT, audit=True)
    assert len(trace.flux) == 3
    names = kernels.names
    for stage, operand in enumerate((initial, trace.q1, trace.q2)):
        state = _ready(kernels, operand)
        part, moved = _viscous_of(kernels, state)
        for n, name in enumerate(names):
            _same(trace.flux[stage][name], moved[name])
            _same(trace.viscous[stage][name], part[name])
            _same(trace.du[stage][name], kernels.rhs[n](state[name]) + part[name])   # 无粘加粘性
        if stage == 2:
            continue
        for name in names:                                          # q1、q2 是准备之前的值
            du = trace.du[stage][name]
            want = (operand[name] + DT * du if stage == 0 else
                    0.75 * initial[name] + 0.25 * (operand[name] + DT * du))
            np.testing.assert_allclose(
                trace.q1[name] if stage == 0 else trace.q2[name], want, rtol=1e-12, atol=1e-17)
    # 对照：把准备过的阶段态当成 RK 组合的操作数，q1 就变了，有物理面的块在内区也看得出来。
    # 这个错误要在 q1 上看：下一阶段还会再做一次准备，粘性分项在内区与正确的逐位相同
    # （Astra 第 24 轮）。只有块内周期面的块，准备不动内区，差别只在鬼胞。
    for name in names:
        wrong = np.asarray(kernels.q1(trace.prepared[0][name], DT, trace.du[0][name]))
        right = np.asarray(trace.q1[name])
        inside = built[name].grid.inner()
        assert not np.array_equal(wrong, right), name
        writes = any(face.kind != PERIODIC for face in built[name].faces)
        assert np.array_equal(wrong[inside], right[inside]) != writes, name
    # 整步后的末态：对第三阶段的组合结果做准备、再换守恒量，重算逐位相同
    third = {name: kernels.q3(initial[name], DT, trace.q2[name], trace.du[2][name])
             for name in names}
    final = _ready(kernels, third)
    for name in names:
        _same(trace.final[name], final[name])


# ---- 5. 求解路径里的 Ev 交换 ----

@pytest.mark.parametrize("key", ["two", "seam"], ids=["两块切口", "周期接缝"])
def test_the_solve_path_exchanges_ev_through_the_tables(key):
    built = step7.synthetic(key)
    kernels = _viscous(built)
    trace = multiblock.solve_step(kernels, _start(built), DT, audit=True)
    for stage in range(3):
        flux = {name: np.asarray(value) for name, value in trace.flux[stage].items()}
        assert all(np.ptp(value) > 0.0 for value in flux.values())    # Ev 不是常数场
        for table in built.tables:
            it = table.interface
            _same(flux[it.zone][(slice(None),) + tuple(table.slots)],
                  flux[it.donor_zone][(slice(None),) + tuple(table.sources)])
        if key == "seam":                         # 两块 8×7×7：内区 k、j 是 4..10，i 是 4..11
            one, two = flux["Zone 1"], flux["Zone 2"]
            for r in range(1, 5):
                # Zone 1 的低侧面接 Zone 2 的高侧面：第 r 层鬼胞取对端的 i = n-r
                _same(one[:, 4:11, 4:11, 4 - r], two[:, 4:11, 4:11, 12 - r])
                # Zone 2 的高侧面接 Zone 1 的低侧面：第 r 层鬼胞取对端的 i = r-1
                _same(two[:, 4:11, 4:11, 11 + r], one[:, 4:11, 4:11, 3 + r])


# ---- 6. 切口两侧的 Ev 一致 ----

@pytest.mark.parametrize("preset", PRESETS)
def test_both_sides_of_a_cut_compute_the_same_ev(preset):
    layers = acceptance._shared(preset)            # 每条普通切口一层；周期接缝没有共有的点
    assert len(layers) == sum(pair.kind == multiblock.CUT
                              for pair in support.topology(preset).interfaces)
    item = step7.preset(preset)
    kernels = _viscous(item.built, item.scale)
    trace = multiblock.solve_step(kernels, item.initial, DT, audit=True)
    for stage in range(3):
        flux = trace.flux[stage]
        for layer in layers:
            support.compare(
                flux[layer.name_a][layer.a], flux[layer.name_b][layer.b],
                f"{preset} 第 {stage + 1} 阶段 {layer.name_a} 与 {layer.name_b} 的 Ev",
                count=12)
        for candidates in multiblock.audit_collinear(kernels.tables, flux, flux):
            support.compare(
                candidates.written, candidates.candidate,
                f"{preset} 第 {stage + 1} 阶段共线处 {candidates.interface.names[0]}",
                count=12)


# ---- 7. 六个面全是接口的块 ----

def test_center_block_with_six_interfaces_matches_a_periodic_single_block():
    item = step7.preset("per3")
    built, initial = item.built, item.initial
    kernels = _viscous(built, item.scale)
    center = kernels.names.index("blk-i1j1k1")
    block = built.blocks[center]
    assert kernels.prepare[center] is None and block.faces == ()
    assert block.execution.band == (0, 0, 0, 0, 0, 0)      # 六个面都不套单侧公式
    # 中心块的每个内区点离某个面都不超过 4 层：粘性分项处处要读交换来的 Ev 鬼胞
    assert all(min(i, n - 1 - i) <= LAP for n in block.grid.n_inner for i in range(n))
    state = step7.periodic_reference()
    start = support.with_field(state, step7.acceptance_on(step7.PERIODIC_DOMAIN))
    trace, = multiblock.solve(kernels, DT, 1, initial=initial)
    formal = support.formal_trace(state, start, DT)
    note, inner = support.sidecar("per3"), state.desc.grid.inner()
    parts = [support.reassemble(note, support.interiors(built, trace.viscous[stage]))
             for stage in range(3)]
    assert np.any(parts[0][1:5] != 0.0)                     # 粘性分项在内区不为零
    for stage in range(3):
        prepared = state.execution.prepare(jnp.asarray(formal.inputs[stage]))
        support.compare(step7.viscous_stage(state, prepared)[inner], parts[stage],
                        f"per3 第 {stage + 1} 阶段的粘性分项")
        support.compare(
            formal.du[stage][inner],
            support.reassemble(note, support.interiors(built, trace.du[stage])),
            f"per3 第 {stage + 1} 阶段的总项")
    support.compare(formal.final[inner],
                    support.reassemble(note, support.interiors(built, trace.final)),
                    "per3 的末态")


# ---- 8. 不给全局顶点数的装配 ----

def test_blocks_assembled_without_the_global_sizes_give_the_same_step(tmp_path):
    """网格旁边没有旁记时，各块按自己的轴长选物理面旁坐标鬼层的规则，轴长 7 的短块走
    边缘复制；给了等价单块的全局顶点数，走奇镜像外推。两种装配的坐标鬼层不同。物理面旁
    的度量和导数用单侧公式，只读内区，所以一个整步的右端项、q1、q2、末态和粘性分项都
    逐位相同。

    合成小网格的其余测试都用在给了全局顶点数的装配上。这一条守住不给的组合（Astra 第
    25 轮建议保留）。"""
    path = tmp_path / "step7-stair-without-note.cgns"
    build_multizone(path, step7.SYNTHETIC["stair"][1]())
    plain = multiblock.build_blocks(
        multiblock.build_topology(mesh_cgns.read_cgns_zones(path)), step7.viscous_args(path))
    noted = step7.synthetic("stair")
    assert [block.name for block in plain.blocks] == [block.name for block in noted.blocks]
    for short, tall in zip(plain.blocks, noted.blocks):
        inside = short.grid.inner()[1:]
        same = [np.array_equal(np.asarray(short.coords[key]), np.asarray(tall.coords[key]))
                for key in step7.COORDS]
        assert not all(same), short.name                # 鬼胞的坐标确实不同，比较不是空转
        for key in step7.COORDS:                        # 内区的坐标相同
            assert np.array_equal(np.asarray(short.coords[key])[inside],
                                  np.asarray(tall.coords[key])[inside])
    field = step7.acceptance_on(step7.extent(noted))
    traces = [multiblock.solve_step(_viscous(built), step7.states_from_field(built, field), DT)
              for built in (plain, noted)]
    assert any(np.any(part[1:5] != 0.0)
               for part in support.interiors(plain, traces[0].viscous[0]).values())
    acceptance._assert_same_trace(*traces)
    _same_viscous(_viscous_interiors(plain, traces[0]), _viscous_interiors(noted, traces[1]),
                  "不给全局顶点数的装配")
