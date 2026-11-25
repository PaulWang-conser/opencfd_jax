"""第七步的 Ev 交换（U2）。

锁六件事：
  1. Ev 按第六步的交换表搬，12 个平面一起搬：普通切口和周期接缝的四种取法逐层核对，
     T 形的两段各取自己的供体；
  2. 表的先后不影响 Ev 的交换结果；jit 路径与急切路径逐位相同；
  3. 写进接收槽的 Ev 与供体内区对应点逐位相同，内区不动，表以外的鬼胞不动；
  4. 关闭、CD6、CD8 三种编译结果互不串用；
  5. 粘性关闭时求解函数的输出与改动 ``src/multiblock.py`` 之前存下的抽样逐位相同，
     也就是与第六步相同；
  6. 只有一块、没有接口的多块打开粘性，各阶段的粘性分项、总项、q1、q2 和末态与正式
     路径在阈值内一致。

Ev 数组的轴序是 (平面, k, j, i)。平面的排法见 ``viscous.viscous_fluxes``：i、j、k
三个计算方向各四个，依次是三个动量方向的粘性通量和能量通量。合成小网格见
``step7_support.SYNTHETIC``。

第 5 条的抽样存在 ``experiment/step7_inviscid_samples.npz``：五个验收网格从验收初场出发
跑两步，取三个阶段的右端项、q1、q2 和两步的末态；全周期的 per3 只跑一步，取第一步的
右端项、q1 和末态。各块含鬼胞的数组展平后每隔 ``STRIDE`` 个取一个。本文件既负责算这些
抽样，也负责核对。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import mesh_cgns, multiblock  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402

DT = 1e-4
SAMPLES = REPO / "experiment" / "step7_inviscid_samples.npz"
STRIDE = 379                 # 抽样间隔
PRESETS = ("x2", "y2", "z2", "xyz8", "t3")
FIELDS = ("du0", "du1", "du2", "q1", "q2", "final1", "final2")
PERIODIC_FIELDS = ("du0", "q1", "final1")
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")

_same = support.same_bits


# ---- 1 到 3. Ev 的交换 ----

def _code(index, plane, i, j, k):
    """块号、平面号、i、j、k 各占一段的编码；都是 float64 里精确的整数。"""
    return 10_000_000.0 * index + 100_000.0 * plane + 10_000.0 * i + 100.0 * j + k


def _fluxes(built):
    """各块的 Ev：内区是 _code(块号, 平面, i, j, k)，鬼胞先留成 -1。"""
    out = {}
    for index, block in enumerate(built.blocks):
        k, j, i = np.meshgrid(*(np.arange(n) for n in block.grid.n_inner),
                              indexing="ij")
        planes = np.arange(12)[:, None, None, None]
        own = np.full((12, *block.grid.shape[1:]), -1.0)
        own[(slice(None),) + block.grid.inner()[1:]] = _code(index, planes, i, j, k)
        out[block.name] = jnp.asarray(own)
    return out


def _swap(built, fluxes):
    """急切路径交换一次，转成 numpy。"""
    return {name: np.asarray(value) for name, value in
            multiblock.exchange(built.tables, fluxes).items()}


def test_twelve_planes_follow_the_four_mappings():
    built = step7.synthetic("seam")             # 两块 8×7×7：中间切口，外侧周期接缝
    after = _swap(built, _fluxes(built))
    one, two = after["Zone 1"], after["Zone 2"]    # 内区 k、j 是 4..10，i 是 4..11
    planes = np.arange(12)[:, None, None]
    k, j = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    for r in range(1, 5):
        # 普通切口：接收块的低侧面接供体的高侧面，取 n-1-r
        assert np.array_equal(two[:, 4:11, 4:11, 4 - r], _code(0, planes, 7 - r, j, k)), r
        # 普通切口：接收块的高侧面接供体的低侧面，取 r
        assert np.array_equal(one[:, 4:11, 4:11, 11 + r], _code(1, planes, r, j, k)), r
        # 周期接缝：接收块的低侧面接供体的高侧面，取 n-r
        assert np.array_equal(one[:, 4:11, 4:11, 4 - r], _code(1, planes, 8 - r, j, k)), r
        # 周期接缝：接收块的高侧面接供体的低侧面，取 r-1
        assert np.array_equal(two[:, 4:11, 4:11, 11 + r], _code(0, planes, r - 1, j, k)), r


def test_each_segment_of_a_split_face_brings_its_own_donor():
    built = step7.synthetic("tee")
    left = _swap(built, _fluxes(built))["Zone 1"]
    planes = np.arange(12)[:, None, None]
    kk = np.arange(7)[:, None]                 # 沿 k，放在第 0 轴
    for r in range(1, 5):
        # 左块（j 是 4..16）的 i+ 鬼层：j=0..6 接 Zone 2，j=7..12 接 Zone 3。
        # Zone 3 的全局 j 是 6..12，它的局部 j 比左块小 6。
        ghost = left[:, 4:11, 4:17, 11 + r]
        assert np.array_equal(ghost[:, :, :7],
                              _code(1, planes, r, np.arange(7)[None, :], kk)), r
        assert np.array_equal(ghost[:, :, 7:],
                              _code(2, planes, r, np.arange(1, 7)[None, :], kk)), r


@pytest.mark.parametrize("key", step7.SYNTHETIC_KEYS, ids=step7.SYNTHETIC_NAMES)
def test_table_order_and_the_jit_path_do_not_change_the_result(key):
    built = step7.synthetic(key)
    fluxes = _fluxes(built)
    names = [block.name for block in built.blocks]
    straight = multiblock.exchange(built.tables, fluxes)
    kernels = multiblock.compile_blocks(built, step7.SYNTHETIC_SCALE)
    compiled = dict(zip(names, kernels.swap(*(fluxes[name] for name in names))))
    tables = list(built.tables)
    for order in (tables[::-1], tables[2:] + tables[:2], tables[1::2] + tables[0::2]):
        shuffled = multiblock.exchange(order, fluxes)
        for name in names:
            _same(shuffled[name], straight[name])
            _same(compiled[name], straight[name])


@pytest.mark.parametrize("key", step7.SYNTHETIC_KEYS, ids=step7.SYNTHETIC_NAMES)
def test_written_values_equal_the_donor_interior_and_interiors_do_not_move(key):
    built = step7.synthetic(key)
    fluxes = _fluxes(built)
    after = multiblock.exchange(built.tables, fluxes)
    for table in built.tables:
        it = table.interface
        _same(np.asarray(after[it.zone])[(slice(None),) + tuple(table.slots)],
              np.asarray(fluxes[it.donor_zone])[(slice(None),) + tuple(table.sources)])
    for block in built.blocks:
        inner = block.grid.inner()
        value = np.asarray(after[block.name])
        _same(value[inner], np.asarray(fluxes[block.name])[inner])
        received = step7.received(built, block.name)
        ghost = np.ones(block.grid.shape[1:], dtype=bool)
        ghost[inner[1:]] = False
        assert np.all(value[:, received] != -1.0)                  # 接收槽都换到了
        assert np.all(value[:, ghost & ~received] == -1.0)         # 表以外的鬼胞不碰


# ---- 4. 三种编译结果 ----

def test_the_three_compile_modes_do_not_share_kernels():
    item = step7.preset("x2")
    off = multiblock.compile_blocks(item.built, item.scale, vis_order=None)
    six = multiblock.compile_blocks(item.built, item.scale, vis_order=6)
    eight = multiblock.compile_blocks(item.built, item.scale, vis_order=8)
    assert len({id(off), id(six), id(eight)}) == 3
    assert (off.vis_order, six.vis_order, eight.vis_order) == (None, 6, 8)
    assert off.flux == [] and off.divergence == []
    assert len(six.flux) == len(six.divergence) == len(item.built.blocks)
    assert len(eight.flux) == len(eight.divergence) == len(item.built.blocks)
    assert all(a is not b for a, b in zip(six.flux, eight.flux))
    assert all(a is not b for a, b in zip(six.divergence, eight.divergence))
    assert six.rhs is eight.rhs and six.swap is eight.swap     # 与粘性无关的函数共用一批
    plain, cd6, cd8 = (multiblock.solve_step(k, item.initial, DT) for k in (off, six, eight))
    assert plain.viscous == () and plain.flux == ()
    for stage in range(3):
        for name in off.names:
            assert not np.array_equal(cd6.viscous[stage][name], cd8.viscous[stage][name])


# ---- 5. 粘性关闭时与第六步逐位相同 ----

def _pick(value):
    return np.asarray(value).reshape(-1)[::STRIDE].copy()


def _samples(preset):
    """粘性关闭时求解函数的输出抽样，{字段/块名: 抽样}。per3 只取第一步的量，就只跑一步。"""
    item = step7.preset(preset)
    kernels = multiblock.compile_blocks(item.built, item.scale)
    traces = multiblock.solve(kernels, DT, 1 if preset == "per3" else 2, initial=item.initial)
    whole = {"du0": traces[0].du[0], "du1": traces[0].du[1], "du2": traces[0].du[2],
             "q1": traces[0].q1, "q2": traces[0].q2, "final1": traces[0].final}
    if len(traces) > 1:
        whole["final2"] = traces[1].final
    fields = PERIODIC_FIELDS if preset == "per3" else FIELDS
    return {f"{preset}/{field}/{name}": _pick(value)
            for field in fields for name, value in whole[field].items()}


@pytest.mark.parametrize("preset", (*PRESETS, "per3"))
def test_viscous_off_matches_the_samples(preset):
    with np.load(SAMPLES) as data:
        want = {key: data[key] for key in data.files if key.startswith(preset + "/")}
    assert want, f"抽样文件里没有 {preset} 的抽样"
    got = _samples(preset)
    assert sorted(got) == sorted(want)
    for key, value in want.items():
        assert np.any(value != 0.0), f"{key} 的抽样全是零，没有判别力"
        _same(got[key], value, key)


# ---- 6. 一块的多块打开粘性，对照正式路径 ----

def test_one_block_with_viscosity_matches_the_formal_path():
    state = step7.reference()
    zone = mesh_cgns.read_cgns_zone(support.SOURCE)
    single = multiblock.build_blocks(multiblock.build_topology((zone,)), state.args)
    assert single.tables == ()
    kernels = multiblock.compile_blocks(single, support.FLUX_SCALE, vis_order=8)
    block, = single.blocks
    start = support.with_field(state, support.acceptance_state)     # 鬼胞为零
    trace = multiblock.solve_step(kernels, {block.name: jnp.asarray(start)}, DT)
    formal = support.formal_trace(state, start, DT)
    inner = block.grid.inner()
    got = {f"第 {stage + 1} 阶段的粘性分项": trace.viscous[stage][block.name]
           for stage in range(3)}
    got.update({f"第 {stage + 1} 阶段的总项": trace.du[stage][block.name] for stage in range(3)})
    got.update({"q1": trace.q1[block.name], "q2": trace.q2[block.name], "末态": trace.final[block.name]})
    want = {f"第 {stage + 1} 阶段的粘性分项": step7.viscous_stage(
        state, state.execution.prepare(jnp.asarray(formal.inputs[stage]))) for stage in range(3)}
    want.update({f"第 {stage + 1} 阶段的总项": formal.du[stage] for stage in range(3)})
    want.update({"q1": formal.inputs[1], "q2": formal.inputs[2], "末态": formal.final})
    worst = {label: support.compare(want[label][inner], got[label][inner], label)
             for label in want}
    # 实测的差在舍入量级（FP64，验收初场，CD8）：粘性分项到 8.4e-13，总项到 1.2e-10
    # （量级 1.06e4；q1 先差 2.8e-14，被右端项放大），q1、q2 到 2.8e-14，末态到 1.4e-13
    assert max(entry[0] for label in worst for entry in worst[label].values()) < 1e-9
