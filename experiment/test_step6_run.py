"""第六步的编排、运行循环与存盘（U4）。

锁六件事：
  1. RK 组合函数与 rk3_step_stages 的三条公式逐位相同；
  2. 只有一块、没有接口的多块跑一步，末态与正式路径在 R37 的容差内一致；
  3. 每个阶段的右端项都来自所有块同一阶段的准备态，q1、q2 是准备之前的值；
  4. 参考尺度的三个分量按 i、j、k 的顺序到达 inviscid_rhs；
  5. 保存时机与单块相同，每个保存步每块一个文件，文件名带块名的 slug，``_meta``
     带 zone；显示步对每块各打印一行诊断；
  6. 全周期小网格的中心块跳过物理准备；``main.run`` 按 zone 的个数选路径；注入的
     初场被使用；比较函数能报出最差的变量、位置和数值。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import init, io, main, mesh_cgns, multiblock, solver  # noqa: E402
from opencfd_jax.src.errors import SetupError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
from Grid.make_cgns_fixture import ZoneSpec, build_multizone  # noqa: E402
from Grid.split_cgns import periodic_source  # noqa: E402
from step6_support import two_blocks  # noqa: E402

DT = 1e-4
T3_BLOCKS = ["blk-left", "blk-rlo", "blk-rhi"]
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")


_same = support.same_bits


_args = support.run_args


def _run(preset, out, **changes):
    """走正式入口跑一个多块算例。输出目录平时由 ``main.main`` 建，这里直接建好。"""
    out.mkdir(parents=True)
    args = _args(preset, out_dir=str(out), **changes)
    main.run(args)
    return args


def _initial(preset):
    """各块的起点状态：验收初场在等价单块内区上的值，按旁记切给各块。"""
    state = support.reference()
    whole = support.acceptance_state(state.desc.coords, state.case["params"])
    return support.split_field(support.blocks(preset), support.sidecar(preset), whole)


_interiors = support.interiors


# ---- 1. RK 组合 ----

def test_rk_combination_matches_rk3_step_stages():
    grid = support.reference().desc.grid
    rng = np.random.default_rng(6)
    cons = rng.normal(size=grid.shape)
    dus = [rng.normal(size=grid.shape) for _ in range(3)]
    calls = []

    def recorded(q):
        calls.append(np.asarray(q))
        return jnp.asarray(dus[len(calls) - 1])

    q3, _ = solver.rk3_step_stages(jnp.asarray(cons), DT, recorded)
    one = solver.rk3_q1(jnp.asarray(cons), DT, jnp.asarray(dus[0]))
    two = solver.rk3_q2(jnp.asarray(cons), DT, one, jnp.asarray(dus[1]))
    three = solver.rk3_q3(jnp.asarray(cons), DT, two, jnp.asarray(dus[2]))
    assert len(calls) == 3
    _same(one, calls[1])
    _same(two, calls[2])
    _same(three, q3)


def test_rk_combination_uses_the_ssp_rk3_coefficients():
    """三条公式按 SSP-RK3 的系数手写一遍对照，不经过 ``rk3_q*``。"""
    rng = np.random.default_rng(7)
    cons, first, second, du1, du2, du3 = (rng.normal(size=(5, 4, 4, 4)) for _ in range(6))
    close = dict(rtol=1e-13, atol=1e-13)             # 只容得下舍入，容不下系数的错
    np.testing.assert_allclose(
        np.asarray(solver.rk3_q1(jnp.asarray(cons), DT, jnp.asarray(du1))),
        cons + DT * du1, **close)
    np.testing.assert_allclose(
        np.asarray(solver.rk3_q2(jnp.asarray(cons), DT, jnp.asarray(first),
                                 jnp.asarray(du2))),
        0.75 * cons + 0.25 * (first + DT * du2), **close)
    np.testing.assert_allclose(
        np.asarray(solver.rk3_q3(jnp.asarray(cons), DT, jnp.asarray(second),
                                 jnp.asarray(du3))),
        cons / 3.0 + (2.0 / 3.0) * (second + DT * du3), **close)


# ---- 2. 一块的多块对照正式路径 ----

def test_one_block_without_interfaces_matches_the_formal_path():
    state = support.reference()
    zone = mesh_cgns.read_cgns_zone(support.SOURCE)
    single = multiblock.build_blocks(multiblock.build_topology((zone,)), state.args)
    assert single.tables == ()
    kernels = multiblock.compile_blocks(single, support.FLUX_SCALE)
    block, = single.blocks
    start = support.with_field(state, support.acceptance_state)
    own = np.zeros(block.grid.shape)
    own[block.grid.inner()] = start[block.grid.inner()]
    trace = multiblock.solve_step(kernels, {block.name: jnp.asarray(own)}, DT)
    formal = support.formal_step(state, start, DT)
    worst = support.compare(formal[block.grid.inner()],
                            _interiors(single, trace.final)[block.name],
                            "一块的多块")
    # 实测的差在舍入量级：ρE 到 1.4e-13（量级 202），其余到 8.9e-16
    assert max(entry[0] for entry in worst.values()) < 1e-12


# ---- 3. 每个阶段读的是同一阶段的准备态 ----

def test_each_stage_reads_the_prepared_state_of_the_same_stage():
    built = support.blocks("x2")
    kernels = support.kernels("x2")
    initial = _initial("x2")
    trace = multiblock.solve_step(kernels, initial, DT)
    names = kernels.names

    def ready(stage):
        prepared = [prepare(q) if prepare is not None else q
                    for prepare, q in zip(kernels.prepare, stage)]
        return multiblock.exchange(kernels.tables, dict(zip(names, prepared)))

    for n, name in enumerate(names):
        for stage, operand, combine in (
                (0, initial, lambda q, du: q + DT * du),
                (1, trace.q1, lambda q, du: 0.75 * initial[name] + 0.25 * (trace.q1[name] + DT * du))):
            _same(trace.du[stage][name],
                  kernels.rhs[n](ready([operand[m] for m in names])[name]))
            np.testing.assert_allclose(
                trace.q1[name] if stage == 0 else trace.q2[name],
                combine(operand[name], trace.du[stage][name]), rtol=1e-12, atol=1e-17)
        # 把准备过的阶段态当成组合的操作数，第二阶段的右端项会变：分得出来
        wrong = ready([trace.prepared[0][m] for m in names])
        assert not np.array_equal(np.asarray(kernels.rhs[n](wrong[name])),
                                  np.asarray(trace.du[1][name]))


# ---- 4. 参考尺度的顺序 ----

def test_flux_scale_reaches_the_rhs_in_i_j_k_order(tmp_path, monkeypatch):
    seen = []

    def spy(cons, grid, **keywords):
        seen.append(keywords["flux_scale"])
        return jnp.zeros_like(cons)

    monkeypatch.setattr(multiblock, "inviscid_rhs", spy)
    path = tmp_path / "mesh.cgns"
    build_multizone(path, two_blocks())
    built = multiblock.build_blocks(
        multiblock.build_topology(mesh_cgns.read_cgns_zones(path)),
        support.plate_args(support.mesh_path("x2")))
    kernels = multiblock.compile_blocks(built, (2.0, 3.0, 5.0))
    for n, block in enumerate(built.blocks):
        assert np.all(np.asarray(kernels.rhs[n](jnp.zeros(block.grid.shape))) == 0.0)
    assert seen == [(2.0, 3.0, 5.0)] * 2


# ---- 5. 存盘与诊断 ----

def test_saving_follows_the_single_block_rule(tmp_path, capsys):
    short = tmp_path / "short"
    _run("t3", short, end_time=0.00045, end_step=5, kstep_save=2, kstep_show=5)

    def names(steps):
        return sorted(f"t3-{name}-istep-{step:08d}.npz"
                      for name in T3_BLOCKS for step in steps)

    assert sorted(p.name for p in short.iterdir()) == names((2, 4, 5))
    _, meta = io.load_npz(str(short / "t3-blk-rhi-istep-00000005.npz"))
    assert meta["zone"] == "blk-rhi" and meta["step"] == 5
    assert meta["dt"] == pytest.approx(5e-5) and meta["t"] == pytest.approx(0.00045)
    _, earlier = io.load_npz(str(short / "t3-blk-rhi-istep-00000002.npz"))
    assert earlier["dt"] == pytest.approx(DT) and earlier["t"] == pytest.approx(2 * DT)
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("blk-")]
    assert sorted(line.split(":")[0] for line in lines) == sorted(T3_BLOCKS)
    assert all("avg(rho*E)" in line for line in lines)

    long = tmp_path / "long"
    _run("t3", long, end_time=0.001, end_step=5, kstep_save=2, kstep_show=5)
    assert sorted(p.name for p in long.iterdir()) == names((2, 4))


def test_block_slugs_replace_whitespace_and_reject_collisions():
    slugs = multiblock.block_slugs(["Zone  1", "blk-i0j0k0", "左 块", "a-b.c_d"])
    assert slugs == {"Zone  1": "Zone_1", "blk-i0j0k0": "blk-i0j0k0",
                     "左 块": "___", "a-b.c_d": "a-b.c_d"}
    with pytest.raises(SetupError, match="slug"):
        multiblock.block_slugs(["a b", "a_b"])


def test_block_slugs_that_differ_only_in_case_are_rejected():
    """Windows 的文件名不分大小写：slug 只差大小写的两个块会写进同一个文件。"""
    with pytest.raises(SetupError, match="slug"):
        multiblock.block_slugs(["A", "a"])
    with pytest.raises(SetupError, match="slug"):
        multiblock.block_slugs(["Zone 1", "ZONE 1"])


def test_sidecar_that_cannot_be_read_is_an_error(tmp_path):
    """没有旁记文件、或旁记里没记全局顶点数，才算不知道；文件在而内容坏了要报错，
    不能悄悄按各块自己的轴长算。"""
    mesh = tmp_path / "mesh.cgns"
    mesh.write_bytes(b"x")
    side = mesh.with_suffix(".json")
    assert multiblock._global_sizes(mesh) is None
    side.write_text('{"source": "somewhere"}', encoding="utf-8")
    assert multiblock._global_sizes(mesh) is None             # 旁记里没记这一项
    side.write_text('{"global_vertices": [8, 7, 7]}', encoding="utf-8")
    assert multiblock._global_sizes(mesh) == (8, 7, 7)
    side.write_text('{"global_vertices": [8.0, 7, 7]}', encoding="utf-8")
    assert multiblock._global_sizes(mesh) == (8, 7, 7)        # 整数值的小数可以
    for text in ("{ not json",                               # 语法坏了
                 '{"global_vertices": null}',                 # 有这一项但没写
                 '{"global_vertices": "877"}',                # 字符串
                 '{"global_vertices": 7}',                    # 不是三个数
                 '{"global_vertices": [8, 7]}',               # 只有两个
                 '{"global_vertices": [true, 7, 7]}',         # 布尔值
                 '{"global_vertices": ["8.0", 7, 7]}',        # 数字写成了字符串
                 '{"global_vertices": [0, 7, 7]}',            # 含零
                 '{"global_vertices": [-1, 7, 7]}',           # 含负数
                 '{"global_vertices": [8.5, 7, 7]}',         # 不是整数
                 '{"global_vertices": [100000000000000000000, 7, 7]}',   # 超出 32 位下标范围
                 '{"global_vertices": [1e308, 7, 7]}'):     # 同样超出下标范围的小数
        side.write_text(text, encoding="utf-8")
        with pytest.raises(SetupError, match="旁记"):
            multiblock._global_sizes(mesh)
    side.write_text("[" * 3000 + "]" * 3000, encoding="utf-8")   # 嵌套太深，解码器自己出错
    with pytest.raises(SetupError, match="旁记"):
        multiblock._global_sizes(mesh)


# ---- 6. 跳过物理准备、入口、注入初场、比较 ----

def test_center_block_skips_prepare_and_matches_a_periodic_single_block(tmp_path):
    built = support.blocks("per3")
    kernels = multiblock.compile_blocks(built, (576.0, 576.0, 576.0))   # 25×25×25 的等价单块
    center = kernels.names.index("blk-i1j1k1")
    assert kernels.prepare[center] is None and built.blocks[center].faces == ()
    # 等价单块走正式入口：把 25×25×25 的立方体写成六个面全是 BCSymmetryPolar 的单 zone
    source = periodic_source()
    n, = set(source.vertex_size)
    bcs = []
    for axis, letter in enumerate("IJK"):
        for sign in ("lo", "hi"):
            lo, hi = [1, 1, 1], [n, n, n]
            lo[axis] = hi[axis] = 1 if sign == "lo" else n
            bcs.append((f"{letter}{sign}", "BCSymmetryPolar", (tuple(lo), tuple(hi))))
    path = tmp_path / "periodic.cgns"
    build_multizone(path, [ZoneSpec(name="Zone 1", coords=source.coords, bcs=bcs)])
    state = init.prepare(support.plate_args(path))
    domain = tuple(float(np.max(value)) for value in state.desc.coords.values())
    start = support.with_field(
        state, lambda coords, params: support.acceptance_state(coords, params, domain))
    formal = support.formal_step(state, start, DT)
    note = support.sidecar("per3")
    initial = support.split_field(built, note, start[state.desc.grid.inner()])
    trace, = multiblock.solve(kernels, DT, 1, initial=initial)
    center_name = kernels.names[center]
    inner = built.blocks[center].grid.inner()
    _same(np.asarray(trace.prepared[0][center_name])[inner],
          np.asarray(initial[center_name])[inner])     # 没有物理准备，内区原样
    back = support.reassemble(note, _interiors(built, trace.final))
    support.compare(formal[state.desc.grid.inner()], back, "per3")


def test_main_run_chooses_the_path_by_zone_count(tmp_path):
    _run("x2", tmp_path / "many", end_time=DT, end_step=1)
    assert sorted(p.name for p in (tmp_path / "many").iterdir()) == [
        "x2-blk-i0j0k0-istep-00000001.npz", "x2-blk-i1j0k0-istep-00000001.npz"]
    # 显式选了 zone 就走原路径；这一块带着跨块的连接，原路径照旧拒绝它
    args = _args("x2", out_dir=str(tmp_path / "one"), end_time=DT, end_step=1,
                 cgns_zone="blk-i0j0k0", flux_scale=None)
    with pytest.raises(SetupError, match="只支持同块自连接"):
        main.run(args)
    assert not (tmp_path / "one").exists() or not list((tmp_path / "one").iterdir())


def test_injected_initial_state_is_used():
    kernels = support.kernels("x2")
    block = kernels.blocks[0]
    inside = (slice(None), slice(8, 40), slice(8, 40), slice(8, 50))   # 离各面都远的内点
    own = np.array(block.cons0)
    own[inside] = 7.0
    cons = {item.name: item.cons0 for item in kernels.blocks}
    cons[block.name] = jnp.asarray(own)
    trace = multiblock.solve_step(kernels, cons, DT)
    assert np.all(np.asarray(trace.prepared[0][block.name])[inside] == 7.0)
    np.testing.assert_allclose(
        np.asarray(trace.q1[block.name])[inside],
        7.0 + DT * np.asarray(trace.du[0][block.name])[inside], rtol=0, atol=1e-15)


def test_compare_passes_equal_fields_and_reports_the_worst_place():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(5, 6, 7, 8))
    worst = support.compare(reference, reference.copy(), "相同")
    assert [entry[0] for entry in worst.values()] == [0.0] * 5
    damaged = reference.copy()
    damaged[3, 2, 5, 4] += 1.0
    with pytest.raises(AssertionError,
                       match=r"相差已知量 超出容差：cons_3 在 \(k, j, i\) = \(2, 5, 4\) 差 1\.000e\+00"):
        support.compare(reference, damaged, "相差已知量")
    elsewhere = np.ones((6, 7, 8), dtype=bool)
    elsewhere[2, 5, 4] = False
    support.compare(reference, damaged, "只比别处", mask=elsewhere)
    damaged[4, 0, 0, 0] = np.nan
    with pytest.raises(AssertionError, match="非有限"):
        support.compare(reference, damaged)


def test_compare_gives_each_conserved_variable_its_own_tolerance():
    """每个守恒量各有相对容差和绝对容差：近零的由绝对容差控制，不跟全场最大值放大。"""
    reference = np.zeros((5, 4, 4, 4))
    reference[4] = 1.0e4                      # ρE 的量级是 1e4
    noisy = reference.copy()
    noisy[3] = 1.0e-12                        # ρw 原为零，舍入差由大分量的运算带出来
    support.compare(reference, noisy, "近零的舍入差")
    wrong = reference.copy()
    wrong[3] = 5.0e-9
    with pytest.raises(AssertionError, match="cons_3"):
        support.compare(reference, wrong, "近零的分量被改")
    # 自身不小的守恒量由相对容差控制：各分量同量级时，相对自身差 1e-9 就要报
    scaled = np.zeros_like(reference)
    scaled[0] = 1.0
    wrong = scaled.copy()
    wrong[0] = 1.0 + 1.0e-9
    with pytest.raises(AssertionError, match="cons_0"):
        support.compare(scaled, wrong, "相对误差 1e-9")
    support.compare(scaled, scaled * (1.0 + 1e-11), "相对误差 1e-11 放行")


def test_compare_reports_the_place_when_the_tolerance_is_zero():
    """参考场全零时容差是零：超差要报出位置，不能在做除法时崩掉。"""
    reference = np.zeros((5, 4, 4, 4))
    got = reference.copy()
    got[3, 1, 2, 3] = 1.0e-3
    with pytest.raises(AssertionError,
                       match=r"全零参考场 超出容差：cons_3 在 \(k, j, i\) = \(1, 2, 3\) 差 1\.000e-03"):
        support.compare(reference, got, "全零参考场")


def test_compare_reports_a_flat_index_for_point_sets():
    """共线审计拿来比较的是 (5, 点数) 的数组：超差时按它实际的下标报。"""
    reference = np.zeros((5, 28))
    reference[4] = 1.0
    got = reference.copy()
    got[3, 5] = 1.0
    with pytest.raises(AssertionError, match=r"cons_3 在下标 \(5,\) 差 1\.000e\+00"):
        support.compare(reference, got, "共线审计")


def test_compare_reports_the_largest_error_when_every_tolerance_is_zero():
    """参考场全零、几个变量同时超差：超差倍数都是无穷大，按绝对差挑最大的报。"""
    reference = np.zeros((5, 4, 4, 4))
    got = reference.copy()
    got[0, 1, 1, 1] = 0.01
    got[4, 2, 2, 2] = 7.0
    with pytest.raises(AssertionError,
                       match=r"cons_4 在 \(k, j, i\) = \(2, 2, 2\) 差 7\.000e\+00"):
        support.compare(reference, got, "全零")


def test_compare_follows_the_mask():
    """``mask`` 只比其中的点：掩码外的非有限值不管，掩码内的要管；掩码全为假直接报错。"""
    reference = np.zeros((5, 4, 4, 4))
    got = reference.copy()
    got[3, 0, 0, 0] = np.nan                       # 在掩码外
    inside = np.zeros((4, 4, 4), dtype=bool)
    inside[1:, 1:, 1:] = True
    support.compare(reference, got, "掩码外", mask=inside)
    got[3, 2, 2, 2] = np.nan                       # 在掩码内
    with pytest.raises(AssertionError, match="非有限"):
        support.compare(reference, got, "掩码内", mask=inside)
    with pytest.raises(AssertionError, match="mask"):
        support.compare(reference, reference, "空掩码", mask=np.zeros((4, 4, 4), dtype=bool))


def test_compare_requires_a_mask_shaped_like_the_field():
    """掩码比场短时会悄悄少比一些点，所以形状不对直接报错。"""
    reference = np.zeros((5, 4, 4, 4))
    with pytest.raises(AssertionError, match="mask"):
        support.compare(reference, reference, "短掩码", mask=np.ones((3, 4, 4), dtype=bool))


def test_compare_locates_the_worst_point_only_inside_the_mask():
    """定位最差的点只在掩码内做：掩码外的 inf 减 inf 不该盖住掩码内该报的错。"""
    reference = np.zeros((5, 4, 4, 4))
    got = reference.copy()
    reference[3, 0, 0, 0] = np.inf                   # 掩码外，两边都是 inf
    got[3, 0, 0, 0] = np.inf
    got[3, 2, 2, 2] = 1.0                            # 掩码内的真错
    inside = np.zeros((4, 4, 4), dtype=bool)
    inside[1:, 1:, 1:] = True
    with np.errstate(invalid="raise"), pytest.raises(
            AssertionError, match=r"cons_3 在 \(k, j, i\) = \(2, 2, 2\) 差 1\.000e\+00"):
        support.compare(reference, got, "掩码内有错", mask=inside)
