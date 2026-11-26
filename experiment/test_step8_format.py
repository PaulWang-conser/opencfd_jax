"""第八步 U1：检查点的新格式、块清单、读回、``_meta`` 和读检查点的脚本。

锁五件事：
  1. 存盘：单块和多块都存原始变量 ρ、u、v、w、T，每块一个数组 ``b{块号}``，形状
     (5, nk, nj, ni)，只有内区，各块自己的形状。多块每个保存步一个文件；``_meta`` 多出
     粘性开关、阶数、精度、网格四项，原有的项不变。
  2. 块清单：照旁记的结构写，另加每块的尺寸；按定义直接对照，不跑求解器。
  3. 读回：新格式和这一步之前的单块检查点读出来是同一个样子；形状不对、不是检查点、
     缺 ``_meta`` 都报错。
  4. 拼回：``assemble`` 按定义对照，三维的坐标数组也拼得回；块清单没有位置时报错。
  5. 脚本：``outputest/snapshot_to_plt.py`` 认新旧两种格式。
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import cases, io, main, multiblock  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402

DT = 1e-4
OLD_META = {"t": DT, "step": 1, "dt": DT, "case": "made-up", "N": 8, "scheme": "UP7",
            "gamma": 1.4, "Re": 1000.0, "Ama": 0.1, "Ref_T": 288.15, "Pr": 0.7}
OLD_KEYS = [f"cons_{i}" for i in range(5)]
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")


def _inner(grid):
    """这一块内区的形状 (nk, nj, ni)。"""
    return tuple(s.stop - s.start for s in grid.inner()[1:])


def _one_step(base, out):
    """按 base 的配置走正式入口跑一步，输出到 out（pytest 已经建好）。返回 args。"""
    args = argparse.Namespace(**{**vars(base), "out_dir": str(out),
                                 "end_step": 1, "kstep_save": 1, "kstep_show": 1})
    main.run(args)
    return args


def _meta_of(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(z["_meta"].tobytes().decode("utf-8"))


def _meta_bytes(meta):
    return np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)


def _params():
    return support.reference().case["params"]


# ---- 1. 存盘：单块 ----

def test_single_block_checkpoint_holds_primitive_variables(tmp_path):
    """文件里恰好 b0 和 _meta；b0 是这一步末态内区的原始变量。"""
    args = _one_step(step7.viscous_args(step7.SOURCE), tmp_path)
    [path] = tmp_path.iterdir()
    assert path.name == io.checkpoint_name(args.out_prefix, 1)
    state = step7.reference()
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ["_meta", "b0"]
        support.same_bits(z["b0"], io.primitive_field(
            support.formal_step(state, state.cons0, args.dt),
            state.desc.grid, state.case["params"]))
        assert z["b0"].shape == (5, 40, 40, 100)


def test_meta_lists_the_new_settings_and_keeps_the_old_ones(tmp_path):
    on, plain = tmp_path / "on", tmp_path / "plain"
    on.mkdir()
    plain.mkdir()
    args = _one_step(step7.viscous_args(step7.SOURCE), on)
    _one_step(support.plate_args(step7.SOURCE), plain)
    on_meta, off_meta = (_meta_of(part / io.checkpoint_name(args.out_prefix, 1))
                         for part in (on, plain))
    # Covers AE3：粘性打开和关闭，_meta 只差粘性开关一项
    assert on_meta["viscous"] is True and off_meta["viscous"] is False
    assert {key: value for key, value in on_meta.items() if key != "viscous"} == {
        key: value for key, value in off_meta.items() if key != "viscous"}
    assert on_meta["vis_order"] == args.vis_order and on_meta["precision"] == config.precision_name()
    assert on_meta["mesh"] == {"format": "cgns", "file": str(step7.SOURCE), "zone": "blk-1",
                               "size": [100, 40, 40]}
    # 原有的项不变
    params = step7.reference().case["params"]
    assert set(on_meta) == {"t", "step", "dt", "case", "N", "scheme", "gamma", "Re", "Ama",
                            "Ref_T", "Pr", "viscous", "vis_order", "precision", "mesh"}
    assert on_meta["step"] == 1 and on_meta["t"] == pytest.approx(DT)
    assert on_meta["dt"] == pytest.approx(DT) and on_meta["case"] == "plate"
    assert on_meta["N"] == 100 and on_meta["scheme"] == "WENO7"
    assert on_meta["gamma"] == params.gamma and on_meta["Re"] == params.Re
    assert on_meta["Ama"] == params.Ama and on_meta["Ref_T"] == params.Ref_T
    assert on_meta["Pr"] == params.Pr


def test_meta_of_builtin_and_forced_viscous_cases(tmp_path):
    """内置网格：zone 是空、尺寸取网格；强制粘性的算例 viscous 是 true。"""
    def run(case, extra):
        out = tmp_path / case
        out.mkdir()
        args = main.build_parser().parse_args(
            ["--case", case, "--dt", "1e-4", "--end-step", "1", "--kstep-save", "1",
             "--kstep-show", "1", "--out-dir", str(out), "--ni", "32",
             "--nj", "16", "--nk", "4", *extra])
        main.run(args)
        return _meta_of(out / io.checkpoint_name(case, 1)), args

    sod, args = run("sod", [])
    assert sod["viscous"] is False and sod["N"] == 32
    assert sod["mesh"] == {"format": "builtin", "file": "", "zone": "", "size": [32, 16, 4]}
    assert sod["vis_order"] == args.vis_order == "CD6"
    forced, _ = run("poiseuille", [])
    assert forced["viscous"] is True          # 算例声明强制粘性


# ---- 1. 存盘：多块 ----

def test_multiblock_checkpoint_has_one_array_per_block(tmp_path):
    """Covers AE1。"""
    out = tmp_path / "x2"
    out.mkdir()
    args = support.run_args("x2", out_dir=str(out), kstep_save=1)
    main.run(args)
    assert sorted(p.name for p in out.iterdir()) == [
        io.checkpoint_name("x2", step) for step in (1, 2)]
    built = support.blocks("x2")
    inner = [_inner(block.grid) for block in built.blocks]
    meta, blocks, fields = io.load_checkpoint(
        str(out / io.checkpoint_name("x2", 2)), None, inner)
    note = support.sidecar("x2")
    assert "N" not in meta and "zone" not in meta and meta["viscous"] is False
    assert meta["mesh"] == {"format": "cgns", "file": args.mesh}
    assert blocks["global_vertices"] == note["global_vertices"]
    assert blocks["blocks"] == [dict(entry, size=list(size)[::-1])
                                for entry, size in zip(note["blocks"], inner)]
    assert [field.shape for field in fields] == [(5, 40, 40, 51), (5, 40, 40, 50)]
    # 换成守恒量、拼回的末态与等价单块在阈值内一致
    state = support.reference()
    cons = state.cons0
    for _ in range(2):
        cons = support.formal_step(state, cons, DT)
    whole = io.assemble(blocks, {entry["name"]: field for entry, field in zip(blocks["blocks"], fields)})
    back = np.asarray(cases.cons_from_state(
        {"rho": whole[0], "u": whole[1], "v": whole[2], "w": whole[3], "T": whole[4]},
        state.case["params"]))
    support.compare(np.asarray(cons)[state.desc.grid.inner()], back, "存盘读回")


SYNTHETIC = {"stair": (144.0, 144.0, 144.0),      # 13×13×13 的等价单块，各轴 (N-1)²
             "strip": (108.0, 84.0, 252.0)}       # 15×19×7：各轴是另外两轴 (N-1) 的乘积


@pytest.mark.parametrize("name", ["x2", "t3", "stair", "strip"])
def test_each_block_array_is_that_block_s_interior(tmp_path, name):
    """块号是 zone 在网格文件里的序号；各块自己的形状，不补齐。"""
    out = tmp_path / name
    out.mkdir()
    if name in SYNTHETIC:                   # 没有旁记的合成小网格，参考尺度按等价单块给
        step7.synthetic(name)
        args = argparse.Namespace(**{
            **vars(support.plate_args(support.MESH_DIR / f"step7-{name}.cgns")),
            "flux_scale": SYNTHETIC[name], "out_dir": str(out),
            "end_step": 1, "kstep_save": 1, "kstep_show": 1})
        built, scale = step7.synthetic(name), SYNTHETIC[name]
    else:
        args = support.run_args(name, out_dir=str(out), end_step=1, kstep_save=1)
        built, scale = support.blocks(name), support.FLUX_SCALE
    main.run(args)
    kernels = multiblock.compile_blocks(built, scale)
    end = multiblock.solve_step(
        kernels, {block.name: block.cons0 for block in built.blocks}, args.dt).final
    [path] = out.iterdir()
    _, _, fields = io.load_checkpoint(str(path))
    assert len(fields) == len(built.blocks)
    params = support.reference().case["params"]
    for index, block in enumerate(built.blocks):
        support.same_bits(fields[index], io.primitive_field(end[block.name], block.grid, params))
        assert fields[index].shape == (5,) + _inner(block.grid)


def test_block_list_has_no_positions_without_a_sidecar(tmp_path):
    """没有旁记的合成网格：块清单里没有全局顶点数和起止下标，读回照常。"""
    out = tmp_path / "out"
    out.mkdir()
    step7.synthetic("stair")
    args = argparse.Namespace(**{
        **vars(support.plate_args(support.MESH_DIR / "step7-stair.cgns")),
        "flux_scale": (144.0, 144.0, 144.0),      # 13×13×13 的等价单块，各轴 (N-1)²
        "out_dir": str(out), "end_step": 1, "kstep_save": 1, "kstep_show": 1})
    main.run(args)
    built = step7.synthetic("stair").blocks
    inner = [_inner(block.grid) for block in built]
    meta, blocks, fields = io.load_checkpoint(
        str(out / io.checkpoint_name(args.out_prefix, 1)))
    assert blocks == {"blocks": [{"name": block.name, "size": list(size)[::-1]}
                                 for block, size in zip(built, inner)]}
    assert [field.shape[1:] for field in fields] == inner


def test_block_list_leaves_positions_out_when_the_sidecar_disagrees(tmp_path, capsys):
    """旁记的块名与网格对不上：块清单不记位置，打印一句提示，其余照常。"""
    mesh = tmp_path / "moved.cgns"
    shutil.copyfile(support.mesh_path("x2"), mesh)
    note = support.sidecar("x2")
    note["blocks"][0]["name"] = "not-a-zone"
    mesh.with_suffix(".json").write_text(json.dumps(note), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    args = support.run_args("x2", out_dir=str(out))
    args.mesh = str(mesh)
    main.run(args)
    assert "旁记" in capsys.readouterr().out
    _, blocks, fields = io.load_checkpoint(str(out / io.checkpoint_name("x2", 2)))
    assert blocks == {"blocks": [{"name": "blk-i0j0k0", "size": [51, 40, 40]},
                                 {"name": "blk-i1j0k0", "size": [50, 40, 40]}]}
    assert len(fields) == 2


# ---- 2. 块清单：按定义直接对照，不跑求解器 ----

CUT = {"global_vertices": [10, 4, 4], "blocks": [
    {"name": "a", "start": [0, 0, 0], "stop": [6, 4, 4]},
    {"name": "b", "start": [5, 0, 0], "stop": [10, 4, 4]}]}


def _blocks(*sizes):
    """名义上的块：只有名字和尺寸 (ni, nj, nk)。"""
    return [SimpleNamespace(name=name, grid=SimpleNamespace(ni=ni, nj=nj, nk=nk))
            for name, (ni, nj, nk) in zip(("a", "b"), sizes)]


def test_block_list_takes_positions_from_a_matching_sidecar(capsys):
    """旁记与网格对得上：块清单照旁记的结构写，另加每块的尺寸 (ni, nj, nk)。"""
    listing = multiblock._block_list(_blocks((6, 4, 4), (5, 4, 4)), CUT)
    assert listing == {"global_vertices": [10, 4, 4], "blocks": [
        {"name": "a", "start": [0, 0, 0], "stop": [6, 4, 4], "size": [6, 4, 4]},
        {"name": "b", "start": [5, 0, 0], "stop": [10, 4, 4], "size": [5, 4, 4]}]}
    assert "提示" not in capsys.readouterr().out


def test_block_list_drops_positions_that_do_not_match(capsys):
    """尺寸对不上、缺全局顶点数、没有旁记：只记名字和尺寸，不提示。"""
    plain = {"blocks": [{"name": "a", "size": [6, 4, 4]}, {"name": "b", "size": [5, 4, 4]}]}
    shifted = dict(CUT, blocks=[dict(CUT["blocks"][0], stop=[5, 4, 4]), CUT["blocks"][1]])
    for note in (shifted, {"blocks": CUT["blocks"]}, None, {}):
        assert multiblock._block_list(_blocks((6, 4, 4), (5, 4, 4)), note) == plain
        assert "提示" not in capsys.readouterr().out


def test_block_list_warns_when_the_sidecar_names_other_blocks(capsys):
    """旁记在、块名却对不上：不记位置，另外打印一句提示。"""
    plain = {"blocks": [{"name": "a", "size": [6, 4, 4]}, {"name": "b", "size": [5, 4, 4]}]}
    renamed = dict(CUT, blocks=[dict(CUT["blocks"][0], name="zone-a"), CUT["blocks"][1]])
    assert multiblock._block_list(_blocks((6, 4, 4), (5, 4, 4)), renamed) == plain
    assert "旁记" in capsys.readouterr().out


# ---- 3. 读回 ----

def test_load_checkpoint_reads_the_previous_layout(tmp_path):
    """Covers AE4。在测试里按这一步之前的格式写一份单块检查点。"""
    cons = np.random.default_rng(3).uniform(0.5, 1.5, (5, 4, 6, 8))
    old = tmp_path / "old.npz"
    io.save_npz(str(old), dict(zip(OLD_KEYS, cons)), OLD_META)
    assert sorted(io.load_npz(str(old))[0]) == OLD_KEYS
    _, blocks, fields = io.load_checkpoint(str(old))
    assert blocks == {} and len(fields) == 1
    support.same_bits(fields[0], io.primitive_of(cons, io.phys_from_meta(OLD_META)))
    support.same_bits(fields[0], io.load_checkpoint(str(old), support.reference().case["params"],
                                                    [(4, 6, 8)])[2][0])


@pytest.mark.parametrize("damage", ["shape", "foreign", "no-meta", "mixed"])
def test_load_checkpoint_rejects_damaged_files(tmp_path, damage):
    """读回时逐项核对形状，不做广播；不是检查点、缺 _meta、新旧键混在一起都报错。"""
    cons = np.random.default_rng(5).uniform(0.5, 1.5, (5, 4, 6, 8))
    path = tmp_path / "c.npz"
    meta_bytes = _meta_bytes(OLD_META)
    if damage == "foreign":
        np.savez(path, x=np.zeros(3), _meta=meta_bytes)
    elif damage == "no-meta":
        np.savez(path, b0=io.primitive_of(cons, io.phys_from_meta(OLD_META)))
    elif damage == "mixed":
        np.savez(path, b0=io.primitive_of(cons, io.phys_from_meta(OLD_META)), cons_0=cons[0], _meta=meta_bytes)
    else:
        io.save_checkpoint(str(path), {"b0": io.primitive_of(cons, io.phys_from_meta(OLD_META))}, OLD_META)
    with pytest.raises(io.CheckpointError):
        if damage == "shape":
            io.load_checkpoint(str(path), None, [(4, 6, 7)])
        else:
            io.load_checkpoint(str(path))


def test_load_checkpoint_rejects_extra_blocks_without_a_block_list(tmp_path):
    """没有块清单却不止一个数组：报错，不让调用方去解包。"""
    prim = np.broadcast_to(np.array([1.0, 1.0, 0.0, 0.0, 1.0])[:, None, None, None],
                           (5, 4, 6, 8))
    path = tmp_path / "c.npz"
    io.save_checkpoint(str(path), {"b0": prim, "b1": prim}, OLD_META)
    with pytest.raises(io.CheckpointError, match="b1"):
        io.load_checkpoint(str(path))


# ---- 4. 拼回 ----

def test_assemble_matches_its_definition():
    """后写的盖住先写的。"""
    note = support.sidecar("t3")
    pieces = {block.name: np.random.default_rng(index).standard_normal((5,) + _inner(block.grid))
              for index, block in enumerate(support.blocks("t3").blocks)}
    whole = io.assemble(note, pieces)
    want = np.full((5, 40, 40, 100), np.nan)
    for entry in note["blocks"]:
        want[(slice(None),) + io.block_window(entry)] = pieces[entry["name"]]
    support.same_bits(whole, want)


def test_assemble_takes_arrays_without_a_leading_axis():
    """坐标那样的三维数组也拼得回：最后三根轴是 (k, j, i)。形状对不上时报错，不做广播。"""
    note = support.sidecar("x2")
    left, right = support.blocks("x2").blocks
    coords = {block.name: np.full(_inner(block.grid), index, dtype=float)
              for index, block in enumerate((left, right))}
    want = np.empty((40, 40, 100))
    want[:, :, :51] = 0.0
    want[:, :, 50:] = 1.0                        # 切口那一层：后写的盖住先写的
    support.same_bits(io.assemble(note, coords), want)
    for broken in ({left.name: np.zeros((5, 5, 5)), right.name: coords[right.name]},
                   {left.name: np.zeros((5, 5, 5, 5)), right.name: coords[right.name]}):
        with pytest.raises(io.CheckpointError, match="却是"):
            io.assemble(note, broken)
    support.same_bits(
        io.assemble(note, {left.name: np.zeros((5,) + _inner(left.grid)),
                           right.name: np.zeros((5,) + _inner(right.grid))}),
        np.zeros((5, 40, 40, 100)))


def test_assemble_needs_positions():
    with pytest.raises(ValueError, match="位置"):
        io.assemble({"blocks": [{"name": "a"}]}, {"a": np.zeros((5, 2, 2, 2))})


@pytest.mark.parametrize("where, out_dir, prefix, step, wanted", [
    ("out", "out", "sod", 1, 2),             # 改过名：文件名是第 2 步，记的步号是 1
    ("out", "out", "sod", 2, None),          # 文件名与记的步号一致，以后的文件步号更大
    ("out", "out", "sod", 3, None),          # 记的步号已经不小于文件名里的
    ("elsewhere", "out", "sod", 1, None),    # 不在输出目录里
    ("out", "out", "tg", 1, None),           # 前缀不同
], ids=["改过名", "步号一致", "记的步号更大", "不在输出目录", "前缀不同"])
def test_clobber_step_matches_its_definition(tmp_path, where, out_dir, prefix, step, wanted):
    """这次的运行会不会把续算的检查点覆盖掉：只有它在输出目录里、文件名是这次运行的前缀
    加一个比它记的步号大的步号时才会。"""
    (tmp_path / "out").mkdir()
    (tmp_path / "elsewhere").mkdir()
    restart = tmp_path / where / io.checkpoint_name("sod", 2)
    restart.touch()
    assert io.clobber_step(restart, tmp_path / out_dir, prefix, step) == wanted


# ---- 5. 脚本 ----

def _snapshot_module():
    spec = importlib.util.spec_from_file_location(
        "snapshot_to_plt", REPO / "outputest" / "snapshot_to_plt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_to_plt_reads_both_layouts(tmp_path, capsys, monkeypatch):
    """转出的文件里是原始量；新旧两种格式都认。"""
    module = _snapshot_module()
    rng = np.random.default_rng(11)
    prim = np.stack([rng.uniform(0.5, 1.5, (3, 4, 5)),
                     *rng.uniform(-0.2, 0.2, (3, 3, 4, 5)),
                     rng.uniform(0.8, 1.2, (3, 4, 5))])
    cons = np.asarray(cases.cons_from_state(
        {"rho": prim[0], "u": prim[1], "v": prim[2], "w": prim[3], "T": prim[4]},
        support.reference().case["params"]))
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    io.save_npz(str(old), dict(zip(OLD_KEYS, cons)), OLD_META)
    io.save_checkpoint(str(new), {"b0": prim}, OLD_META)
    for path in (old, new):
        got, meta = module.load_snapshot(str(path))
        assert got.shape == (5, 3, 4, 5) and meta == OLD_META
    support.same_bits(module.load_snapshot(str(new))[0], prim)
    np.testing.assert_allclose(module.load_snapshot(str(old))[0], prim, rtol=0, atol=1e-12)
    monkeypatch.setattr(sys, "argv", ["snapshot_to_plt.py", str(new), "--report-only"])
    module.main()
    assert "rho" in capsys.readouterr().out


def test_snapshot_to_plt_refuses_a_multiblock_checkpoint(tmp_path):
    """多块的检查点只导出第 0 块会悄悄丢掉其余的：明确报错。"""
    module = _snapshot_module()
    prim = np.random.default_rng(13).uniform(0.5, 1.5, (5, 3, 4, 5))
    multi = tmp_path / "multi.npz"
    io.save_checkpoint(str(multi), {"b0": prim, "b1": prim}, OLD_META, blocks={
        "blocks": [{"name": "a", "size": [5, 4, 3]}, {"name": "b", "size": [5, 4, 3]}]})
    with pytest.raises(ValueError, match="多块"):
        module.load_snapshot(str(multi))
