"""第八步 U3：单块续算。

锁六件事：
  1. 从正式入口“跑一步再续一步”，与一口气跑两步相比，第二张检查点在阈值内一致，
     步号和时间接续，保存和显示按累计的步号触发；
  2. 续算那一步的三个阶段右端项、q1、q2、末态，与不中断的第二步在阈值内一致；四个
     内置算例各一条；
  3. 带特殊边界脚本：不依赖历史的脚本，续算在阈值内一致；把历史留在鬼胞里的脚本能续完，
     结果超出阈值，启动信息里有一句提示；
  4. 启动时只核对网格：算例名、zone 的名字、顶点数对不上，各报一行配置错误；这一步之前
     写的检查点只核对了算例名和 N，其余提示没核对；
  5. 多块的检查点给单块运行、检查点坏了、场放不进这一次的内区，都报一行初始化错误；
  6. JSON 的 ``time.restart`` 相对路径按 JSON 所在目录解析；文件不存在在配置阶段报错；
  7. 多块续算（U4）：粘性 CD8 的 t3、per3 和关闭粘性的 x2，续算那一步的各阶段量与不中断的
     第二步在阈值内一致；正式入口“跑一步再续一步”；拦住块数、名字、尺寸的改变；单块的
     检查点给多块运行、坏文件，都报初始化错误。

验收只做 float64。比法见方案 KTD10：阈值用 ``step6_support.compare`` 的两个数。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import init, io, main, multiblock, runtime_config  # noqa: E402
from opencfd_jax.src.errors import SetupError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402

PROJECT = REPO.parent
DT = 1e-4
needs_source = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")
STATELESS = '''
from types import SimpleNamespace


def build_special_boundary(case):
    """固定赋值：回调不读上一步留下的任何东西。"""
    def bc_apply(q):
        return q.at[0, 4, 4, 4].set(2.0)
    return SimpleNamespace(name="fixed", bc_apply=bc_apply)
'''
HISTORY = '''
from types import SimpleNamespace


def build_special_boundary(case):
    """把历史留在出口的状态鬼层里，再用它改最外面一层内区的密度。

    sod 的出口保持鬼层原值。k、j 取内区的第一层：放在周期面的鬼层里会被周期拷贝冲掉。
    """
    def bc_apply(q):
        history = q[0, 4, 4, -1]
        q = q.at[0, 4, 4, -1].set(history + 1.0)
        return q.at[0, 4, 4, -5].set(history + 2.0)
    return SimpleNamespace(name="history", bc_apply=bc_apply)
'''


def _builtin(case, out, *extra):
    """内置网格的小算例（16×8×4）的参数；终止步数、输出间隔由 extra 给。"""
    return main.build_parser().parse_args(
        ["--case", case, "--dt", "1e-4", "--ni", "16", "--nj", "8", "--nk", "4",
         "--kstep-show", "1", "--kstep-save", "1", "--out-dir", str(out), *extra])


def _restarting(args, **changes):
    return argparse.Namespace(**{**vars(args), **changes})


def _save(path, cons, args, state, step, t):
    """按新格式写一张单块检查点：``cons`` 是含鬼胞的整场。"""
    grid, params = state.desc.grid, state.case["params"]
    io.save_checkpoint(
        str(path), {"b0": io.primitive_field(cons, grid, params)},
        io.checkpoint_meta(args, params, t, step, DT,
                           viscous=bool(args.viscous or state.case.get("needs_viscous", False)),
                           mesh=main._mesh_meta(args, grid), ni=grid.ni))


def _old_save(path, cons, args, state, step, t, shape=None, **meta):
    """按这一步之前的格式写一张单块检查点：cons_0..4，``_meta`` 只有原来的项。

    ``shape`` 给出时写这个形状的数组（用来造放不进内区的场）。
    """
    grid, params = state.desc.grid, state.case["params"]
    whole = np.asarray(cons)
    interior = whole[grid.inner()]
    if shape is not None:
        interior = np.random.default_rng(0).uniform(0.5, 1.5, (5,) + shape)
    meta = {"t": t, "step": step, "dt": DT, "case": args.case, "N": grid.ni,
            "scheme": args.scheme, "gamma": params.gamma, "Re": params.Re,
            "Ama": params.Ama, "Ref_T": params.Ref_T, "Pr": params.Pr, **meta}
    io.save_npz(str(path), {f"cons_{i}": interior[i] for i in range(5)}, meta)


def _fields_of(path):
    with np.load(path, allow_pickle=False) as z:
        fields = {name: z[name] for name in z.files if name != "_meta"}
        meta = (json.loads(z["_meta"].tobytes().decode("utf-8"))
                if "_meta" in z.files else None)
    return fields, meta


def _rewritten(path, **changes):
    """改检查点的 ``_meta`` 以后另存，返回新文件。"""
    fields, meta = _fields_of(path)
    out = path.with_name("rewritten.npz")
    io.save_checkpoint(str(out), fields, dict(meta, **changes))
    return out


def _second_step(state, args, checkpoint):
    """不中断的第二步，与从 ``checkpoint`` 续出来的第二步。续算的起点走正式的恢复函数
    ``main._restart``：读盘、核对网格、以 cons0 为底换内区。返回 (不中断, 续算)。"""
    one = support.formal_step(state, state.cons0, args.dt)
    _save(checkpoint, one, args, state, 1, args.dt)
    start, t, step, messages = main._restart(_restarting(args, restart=str(checkpoint)), state)
    assert (t, step, messages) == (args.dt, 1, [])
    return (support.formal_step(state, one, args.dt),
            support.formal_step(state, start, args.dt))


def _compare_interior(plain, again, grid, label):
    inner = grid.inner()
    support.compare(np.asarray(plain)[inner], np.asarray(again)[inner], label)


# ---- 1. 正式入口：跑一步再续一步 ----

@needs_source
def test_restart_through_the_entry_matches_an_uninterrupted_run(tmp_path, capsys):
    """plate 的 CGNS 源网格，粘性 CD8。第二张检查点在阈值内一致，步号和时间接续。"""
    whole, part = tmp_path / "whole", tmp_path / "part"
    whole.mkdir()
    part.mkdir()
    base = step7.viscous_args(step7.SOURCE)
    main.run(argparse.Namespace(**{**vars(base), "out_dir": str(whole), "kstep_save": 1}))
    first, second = (io.checkpoint_name(base.out_prefix, step) for step in (1, 2))
    assert sorted(p.name for p in whole.iterdir()) == [first, second]
    capsys.readouterr()
    main.run(argparse.Namespace(**{
        **vars(base), "out_dir": str(part), "restart": str(whole / first),
        "kstep_save": 2, "kstep_show": 2}))
    assert [p.name for p in part.iterdir()] == [second]          # 保存按累计的步号触发
    meta, _, [got] = io.load_checkpoint(str(part / second))
    want_meta, _, [want] = io.load_checkpoint(str(whole / second))
    support.compare(want, got, "续算的第二步")
    assert meta["step"] == 2 and meta["t"] == want_meta["t"]
    out = capsys.readouterr().out
    started = [line for line in out.splitlines() if line.startswith("restart")]
    assert len(started) == 1 and "step=1" in started[0] and first in started[0]
    assert "( step 2 )" in out                                  # 显示也按累计的步号触发


@needs_source
def test_restart_step_matches_the_uninterrupted_stages(tmp_path):
    """续算那一步的三个阶段右端项、q1、q2、末态，与不中断的第二步在阈值内一致。

    不中断的一边第二步接在第一步含鬼胞的末态后面；续算的一边走写盘，再用正式的恢复
    函数 ``main._restart`` 读盘、核对、回填。
    """
    state = step7.reference()
    args = step7.viscous_args(step7.SOURCE)
    grid = state.desc.grid
    one = support.formal_step(state, state.cons0, DT)
    plain = support.formal_trace(state, one, DT)
    _save(tmp_path / "c.npz", one, args, state, 1, DT)
    start, t, step, messages = main._restart(
        _restarting(args, restart=str(tmp_path / "c.npz")), state)
    assert (t, step, messages) == (DT, 1, [])
    again = support.formal_trace(state, start, DT)
    inner = grid.inner()
    for stage in range(3):
        support.compare(plain.du[stage][inner], again.du[stage][inner],
                        f"第 {stage} 个阶段的右端项")
    for stage in (1, 2):                               # q1、q2 是准备之前的阶段态
        support.compare(plain.inputs[stage][inner], again.inputs[stage][inner], f"q{stage}")
    support.compare(plain.final[inner], again.final[inner], "末态")


@pytest.mark.parametrize("case", ["sod", "tg", "poiseuille", "plate"])
def test_restart_step_matches_the_uninterrupted_run_on_builtin_cases(tmp_path, case):
    """四个内置算例：续一步，与不中断的第二步在内区、在阈值内一致。"""
    args = _builtin(case, tmp_path)
    state = init.prepare(args)
    plain, again = _second_step(state, args, tmp_path / "c.npz")
    _compare_interior(plain, again, state.desc.grid, case)


# ---- 2. 特殊边界脚本 ----

def _mount(monkeypatch, tmp_path, text, name):
    folder = tmp_path / "special"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.py").write_text(text, encoding="utf-8")
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", folder)
    return f"{name}.py"


@pytest.mark.parametrize("name, text, within", [("fixed", STATELESS, True),
                                                ("history", HISTORY, False)],
                         ids=["不依赖历史", "把历史留在鬼胞里"])
def test_runs_with_a_special_boundary_script_can_restart(tmp_path, monkeypatch, capsys,
                                                          name, text, within):
    """两种脚本都能从检查点续完；不依赖历史的在阈值内一致，把历史留在鬼胞里的超出阈值。
    启动信息里有一句不保证阈值内一致的提示。"""
    case = "sod"
    args = _builtin(case, tmp_path, "--end-step", "2")
    args.specificboundary = _mount(monkeypatch, tmp_path, text, name)
    state = init.prepare(args)
    plain, again = _second_step(state, args, tmp_path / "c.npz")
    if within:
        _compare_interior(plain, again, state.desc.grid, "续算的第二步")
    else:
        with pytest.raises(AssertionError, match="超出容差"):
            _compare_interior(plain, again, state.desc.grid, "续算的第二步")
    main.run(_restarting(args, restart=str(tmp_path / "c.npz")))
    assert "不保证" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [
        "c.npz", io.checkpoint_name(case, 2)]                   # 续完了一步并存盘


# ---- 3. 累计的终止条件和保存时机 ----

def test_restart_counts_steps_and_saves_by_the_accumulated_step(tmp_path, capsys):
    """从第 2 步的检查点续、终止步数 3：只跑一步。单独到达步数上限不存盘；到达终止
    时间的那一步存盘。显示和保存都按累计的步号触发。"""
    args = _builtin("sod", tmp_path)
    state = init.prepare(args)
    _save(tmp_path / "c2.npz", state.cons0, args, state, 2, 2 * args.dt)
    out = tmp_path / "out"
    out.mkdir()
    capped = _restarting(args, out_dir=str(out), restart=str(tmp_path / "c2.npz"),
                         end_step=3, end_time=1.0, kstep_save=5, kstep_show=3)
    main.run(capped)
    assert not list(out.iterdir())                       # 单独到达步数上限不存
    first = capsys.readouterr().out
    assert "( step 3 )" in first
    main.run(_restarting(capped, end_time=3 * args.dt))  # 这一步到达终止时间
    assert [p.name for p in out.iterdir()] == [io.checkpoint_name("sod", 3)]
    meta = io.load_checkpoint(str(out / io.checkpoint_name("sod", 3)))[0]
    assert meta["step"] == 3 and meta["dt"] == pytest.approx(args.dt)
    for text in (first, capsys.readouterr().out):
        started = [line for line in text.splitlines() if line.startswith("restart")]
        assert len(started) == 1 and "step=2" in started[0] and "t=" in started[0]


# ---- 4. 只核对网格 ----

@pytest.mark.parametrize("how, words", [("case", "算例"), ("size", "尺寸"), ("zone", "zone")],
                         ids=["算例名不同", "顶点数不同", "zone 名不同"])
def test_restart_rejects_a_checkpoint_of_another_mesh(tmp_path, how, words):
    """网格对不上报配置错误，消息里点出对不上的项，没有输出文件。"""
    if how == "zone":                                   # zone 名只有 CGNS 网格才有
        if not support.SOURCE.exists():
            pytest.skip("缺少源网格")
        args = step7.viscous_args(step7.SOURCE)
    else:
        args = _builtin("sod", tmp_path)
    state = init.prepare(args)
    _save(tmp_path / "c.npz", state.cons0, args, state, 1, args.dt)
    mesh = main._mesh_meta(args, state.desc.grid)
    changes = {"case": dict(case="tg"),
               "size": dict(mesh=dict(mesh, size=[16, 8, 5])),
               "zone": dict(mesh=dict(mesh, zone="blk-other"))}[how]
    other = _rewritten(tmp_path / "c.npz", **changes)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(runtime_config.ConfigError, match=words):
        main.run(_restarting(args, out_dir=str(out), restart=str(other),
                             end_step=2, kstep_save=1, kstep_show=1))
    assert not list(out.iterdir())


def test_mesh_mismatch_is_one_line_from_the_command_line(tmp_path):
    args = _builtin("sod", tmp_path)
    state = init.prepare(args)
    _save(tmp_path / "c.npz", state.cons0, args, state, 1, args.dt)
    other = _rewritten(tmp_path / "c.npz", case="tg")
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--case", "sod", "--dt", "1e-4",
         "--ni", "16", "--nj", "8", "--nk", "4", "--restart", str(other),
         "--out-dir", str(out)], cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    reported = [line for line in done.stderr.splitlines() if "配置错误" in line]
    assert len(reported) == 1 and "算例" in reported[0], done.stderr
    assert not out.exists() or not list(out.iterdir())


def test_restart_from_a_checkpoint_written_before_this_step(tmp_path, capsys):
    """这一步之前写的检查点（cons_0..4，_meta 只记了算例名和 N）能续，在阈值内一致；
    启动信息里提示 zone 的名字和另外两维的尺寸没有核对。"""
    args = _builtin("sod", tmp_path, "--end-step", "2", "--kstep-save", "5")
    state = init.prepare(args)
    one = support.formal_step(state, state.cons0, args.dt)
    old = tmp_path / "old.npz"
    _old_save(old, one, args, state, 1, args.dt)
    plain = support.formal_step(state, one, args.dt)
    start, t, step, messages = main._restart(_restarting(args, restart=str(old)), state)
    assert (t, step) == (args.dt, 1)
    assert len(messages) == 1 and "没有核对" in messages[0]
    again = support.formal_step(state, start, args.dt)
    _compare_interior(plain, again, state.desc.grid, "旧检查点")
    main.run(_restarting(args, restart=str(old)))
    assert "没有核对" in capsys.readouterr().out


def test_default_is_not_to_restart():
    assert main.build_parser().parse_args([]).restart is None


# ---- 5. 给反了、坏了、放不进去 ----

def test_restart_rejects_a_multiblock_checkpoint(tmp_path):
    args = _builtin("sod", tmp_path, "--end-step", "2")
    state = init.prepare(args)
    grid, params = state.desc.grid, state.case["params"]
    field = io.primitive_field(state.cons0, grid, params)
    meta = io.checkpoint_meta(args, params, args.dt, 1, args.dt, viscous=False,
                              mesh={"format": "builtin", "file": ""})
    multi = tmp_path / "multi.npz"
    io.save_checkpoint(str(multi), {"b0": field, "b1": field}, meta,
                       blocks={"blocks": [{"name": "a", "size": [16, 8, 4]},
                                          {"name": "b", "size": [16, 8, 4]}]})
    with pytest.raises(SetupError, match="多块"):
        main.run(_restarting(args, restart=str(multi)))


@pytest.mark.parametrize("damage", ["not-npz", "truncated", "no-meta", "bad-meta",
                                   "no-step", "no-time"])
def test_restart_rejects_damaged_checkpoints(tmp_path, damage):
    """读不了、写到一半中断、不是检查点、缺 _meta、_meta 不是 JSON、缺步号或时间：
    初始化错误。"""
    args = _builtin("sod", tmp_path, "--end-step", "2")
    state = init.prepare(args)
    path = tmp_path / "c.npz"
    _save(path, state.cons0, args, state, 1, args.dt)
    fields, meta = _fields_of(path)
    if damage == "not-npz":
        path.write_text("这不是 npz", encoding="utf-8")
    elif damage == "truncated":
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size // 2)
    elif damage == "no-meta":
        np.savez(path, **fields)
    elif damage == "bad-meta":
        np.savez(path, **fields, _meta=np.frombuffer(b"{ not json", dtype=np.uint8))
    elif damage == "no-step":
        io.save_checkpoint(str(path), fields, {k: v for k, v in meta.items() if k != "step"})
    elif damage == "no-time":
        io.save_checkpoint(str(path), fields, {k: v for k, v in meta.items() if k != "t"})
    with pytest.raises(SetupError, match="检查点"):
        main.run(_restarting(args, restart=str(path)))


def test_damaged_checkpoint_is_one_line_from_the_command_line(tmp_path):
    out = tmp_path / "out"
    broken = tmp_path / "broken.npz"
    broken.write_text("这不是 npz", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--case", "sod", "--dt", "1e-4",
         "--ni", "16", "--nj", "8", "--nk", "4", "--restart", str(broken),
         "--out-dir", str(out)], cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    reported = [line for line in done.stderr.splitlines() if "初始化错误" in line]
    assert len(reported) == 1 and "检查点" in reported[0], done.stderr


def test_restart_rejects_a_field_that_does_not_fit_this_interior(tmp_path):
    """旧检查点的场放不进这一次的内区（ni 相同、nj 不同）：初始化错误，不做广播。"""
    args = _builtin("sod", tmp_path, "--end-step", "2")
    state = init.prepare(args)
    old = tmp_path / "old.npz"
    _old_save(old, state.cons0, args, state, 1, args.dt, shape=(4, 6, 16))
    with pytest.raises(SetupError, match="形状"):
        main.run(_restarting(args, restart=str(old)))


# ---- 6. 路径 ----

def test_json_restart_path_is_resolved_against_the_json_directory(tmp_path):
    args = _builtin("sod", tmp_path, "--end-step", "2")
    state = init.prepare(args)
    _save(tmp_path / "c.npz", state.cons0, args, state, 1, args.dt)
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"case": {"name": "sod"},
                                "time": {"dt": 1e-4, "restart": "c.npz"}}), encoding="utf-8")
    args = runtime_config.resolve_args({}, runtime_config.load_config(str(path)),
                                       main.build_parser().parse_args([]))
    assert Path(args.restart) == tmp_path / "c.npz"


def test_missing_restart_file_is_a_config_error(tmp_path, capsys):
    for how in ("json", "cli"):
        missing = tmp_path / f"missing-{how}.npz"
        with pytest.raises(SystemExit) as info:
            if how == "json":
                path = tmp_path / "case.json"
                path.write_text(json.dumps(
                    {"case": {"name": "sod"},
                     "time": {"dt": 1e-4, "restart": missing.name}}), encoding="utf-8")
                main.main(["--config", str(path), "--out-dir", str(tmp_path / "out")])
            else:
                main.main(["--case", "sod", "--dt", "1e-4", "--restart", str(missing),
                           "--out-dir", str(tmp_path / "out")])
        assert info.value.code == 2
        reported = [line for line in capsys.readouterr().err.splitlines() if "配置错误" in line]
        assert len(reported) == 1 and missing.name in reported[0]


# ---- 7. 多块续算（U4）----

FLOW = np.array([1.0, 1.0, 0.0, 0.0, 1.0])       # 均匀来流的原始变量：ρ、u、v、w、T
PLATE = argparse.Namespace(case="plate", scheme="WENO7", vis_order="CD8")


def _multi_checkpoint(path, built, final, params, step, t, viscous=True):
    """按新格式写一张多块检查点：``final`` 是 {块名: 含鬼胞的整场}；块清单只记名字和尺寸。"""
    io.save_checkpoint(
        str(path),
        {f"b{index}": io.primitive_field(final[block.name], block.grid, params)
         for index, block in enumerate(built.blocks)},
        io.checkpoint_meta(PLATE, params, t, step, DT, viscous=viscous,
                           mesh={"format": "cgns", "file": ""}),
        blocks={"blocks": [{"name": block.name,
                            "size": [block.grid.ni, block.grid.nj, block.grid.nk]}
                           for block in built.blocks]})


def _crafted(path, listing, params):
    """手搭一张多块检查点：块清单按 ``listing`` 写，数组是均匀来流。"""
    fields = {f"b{index}": np.broadcast_to(
        FLOW[:, None, None, None], (5, size[2], size[1], size[0]))
        for index, entry in enumerate(listing["blocks"]) for size in [entry["size"]]}
    meta = io.checkpoint_meta(PLATE, params, DT, 1, DT, viscous=False,
                              mesh={"format": "cgns", "file": ""})
    io.save_checkpoint(str(path), fields, meta, blocks=listing)


def _compare_stages(built, plain, again, label, viscous):
    """逐块比内区：三个阶段的右端项、准备态、粘性分项，q1、q2，末态。"""
    for name in plain.final:
        inner = built[name].grid.inner()
        for stage in range(3):
            for what in ("du", "prepared", "viscous"):
                if what == "viscous" and not viscous:
                    continue
                support.compare(
                    np.asarray(getattr(plain, what)[stage][name])[inner],
                    np.asarray(getattr(again, what)[stage][name])[inner],
                    f"{label} {name} 第 {stage} 个阶段的 {what}")
        for what in ("q1", "q2", "final"):
            support.compare(np.asarray(getattr(plain, what)[name])[inner],
                            np.asarray(getattr(again, what)[name])[inner],
                            f"{label} {name} {what}")


def _multiblock_second_step(name, viscous, tmp_path):
    """不中断的第二步，与从检查点续出来的第二步。续算的起点走正式的恢复函数
    ``multiblock._restart``：读盘、核对网格、各块以自己的 cons0 为底换内区。"""
    preset = step7.preset(name)
    built, params = preset.built, step7.reference().case["params"]
    kernels = multiblock.compile_blocks(
        built, preset.scale, multiblock.CD_ORDER["CD8"] if viscous else None)
    one = multiblock.solve_step(kernels, preset.initial, DT)
    plain = multiblock.solve_step(kernels, one.final, DT)
    _multi_checkpoint(tmp_path / "c.npz", built, one.final, params, 1, DT, viscous)
    args = support.run_args(name, restart=str(tmp_path / "c.npz"))
    start, t, step, messages = multiblock._restart(args, built, params)
    assert (t, step, messages) == (DT, 1, [])
    return built, plain, multiblock.solve_step(kernels, start, DT)


@pytest.mark.parametrize("name", ["t3", "per3"])
def test_multiblock_restart_step_matches_the_uninterrupted_stages(tmp_path, name):
    """粘性 CD8 的 t3、per3：续算那一步各块内区的三个阶段右端项、粘性分项、准备态、q1、
    q2、末态，与不中断的第二步在阈值内一致。只比内区。"""
    built, plain, again = _multiblock_second_step(name, True, tmp_path)
    _compare_stages(built, plain, again, name, viscous=True)


def test_multiblock_restart_step_matches_with_viscosity_off(tmp_path):
    built, plain, again = _multiblock_second_step("x2", False, tmp_path)
    _compare_stages(built, plain, again, "x2", viscous=False)


def test_multiblock_restart_through_the_entry_matches_an_uninterrupted_run(tmp_path, capsys):
    """x2，粘性 CD8：跑一步再续一步，第二张检查点的各块数组与一口气跑两步在阈值内一致，
    步号和时间接续。显式给出 --restart 的运行通过放行校验。"""
    whole, part = tmp_path / "whole", tmp_path / "part"
    whole.mkdir()
    part.mkdir()
    args = support.run_args("x2", out_dir=str(whole), viscous=True,
                            kstep_save=1, kstep_show=1)
    main.run(args)
    first, second = io.checkpoint_name("x2", 1), io.checkpoint_name("x2", 2)
    assert sorted(p.name for p in whole.iterdir()) == [first, second]
    capsys.readouterr()
    main.run(_restarting(args, out_dir=str(part), restart=str(whole / first),
                         kstep_save=2, kstep_show=2))
    assert [p.name for p in part.iterdir()] == [second]
    meta, _, got = io.load_checkpoint(str(part / second))
    want_meta, _, want = io.load_checkpoint(str(whole / second))
    for index in range(2):
        support.compare(want[index], got[index], f"第 {index} 块续算的第二步")
    assert meta["step"] == 2 and meta["t"] == want_meta["t"]
    out = capsys.readouterr().out
    started = [line for line in out.splitlines() if line.startswith("restart")]
    assert len(started) == 1 and "step=1" in started[0]
    assert out.count("( step 2 )") == 2                # 每块一行诊断，按累计的步号触发


@pytest.mark.parametrize("how, words", [("count", "名字"), ("name", "名字"), ("size", "尺寸")],
                         ids=["块数不同", "某块的 zone 名不同", "某块的形状不同"])
def test_multiblock_restart_rejects_a_checkpoint_of_another_mesh(tmp_path, how, words):
    """拦住的是块数、名字、尺寸的改变；坐标不同而结构相同的网格拦不住（方案 Assumptions
    第 2 条）。报配置错误，消息里点出对不上的项。"""
    args = support.run_args("x2", out_dir=str(tmp_path), end_step=1, kstep_save=1, kstep_show=1)
    built = support.blocks("x2")
    listing = {"blocks": [{"name": block.name,
                           "size": [block.grid.ni, block.grid.nj, block.grid.nk]}
                          for block in built.blocks]}
    if how == "count":
        listing["blocks"].append({"name": "blk-extra", "size": [40, 40, 40]})
    elif how == "name":
        listing["blocks"][1]["name"] = "blk-other"
    else:
        listing["blocks"][0]["size"] = [50, 40, 40]
    other = tmp_path / "other.npz"
    _crafted(other, listing, step7.reference().case["params"])
    with pytest.raises(runtime_config.ConfigError, match=words):
        main.run(_restarting(args, restart=str(other)))
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == ["other.npz"]   # 没有输出文件


def test_multiblock_restart_works_without_positions_in_the_block_list(tmp_path):
    """没有旁记的合成网格：块清单里没有位置，照常能续。"""
    step7.synthetic("stair")
    out, again = tmp_path / "out", tmp_path / "again"
    out.mkdir()
    again.mkdir()
    args = argparse.Namespace(**{
        **vars(support.plate_args(support.MESH_DIR / "step7-stair.cgns")),
        "flux_scale": (144.0, 144.0, 144.0), "out_dir": str(out),
        "end_step": 1, "kstep_save": 1, "kstep_show": 1})
    main.run(args)
    [first] = out.iterdir()
    assert "start" not in io.load_checkpoint(str(first))[1]["blocks"][0]
    main.run(_restarting(args, out_dir=str(again), restart=str(first),
                         end_step=2, kstep_save=2, kstep_show=2))
    assert [p.name for p in again.iterdir()] == [io.checkpoint_name(args.out_prefix, 2)]


def test_multiblock_restart_rejects_a_single_block_checkpoint(tmp_path):
    args = support.run_args("x2", out_dir=str(tmp_path), end_step=1, kstep_save=1, kstep_show=1)
    built = support.blocks("x2")
    single = tmp_path / "single.npz"
    field = io.primitive_field(built.blocks[0].cons0, built.blocks[0].grid,
                               step7.reference().case["params"])
    io.save_checkpoint(str(single), {"b0": field},
                       io.checkpoint_meta(PLATE, step7.reference().case["params"], DT, 1, DT,
                                          viscous=False, mesh={"format": "cgns", "file": ""}))
    with pytest.raises(SetupError, match="单块"):
        main.run(_restarting(args, restart=str(single)))


@pytest.mark.parametrize("damage", ["not-npz", "no-step"])
def test_multiblock_restart_rejects_damaged_checkpoints(tmp_path, damage):
    """坏文件、缺步号：与单块的报法相同。"""
    args = support.run_args("x2", out_dir=str(tmp_path), end_step=1, kstep_save=1, kstep_show=1)
    built = support.blocks("x2")
    params = step7.reference().case["params"]
    path = tmp_path / "c.npz"
    _multi_checkpoint(path, built, {block.name: block.cons0 for block in built.blocks},
                      params, 1, DT)
    if damage == "not-npz":
        path.write_text("这不是 npz", encoding="utf-8")
    else:
        with np.load(path, allow_pickle=False) as z:
            fields = {name: z[name] for name in z.files
                      if name not in ("_meta", "_blocks")}
            meta = json.loads(z["_meta"].tobytes().decode("utf-8"))
            blocks = json.loads(z["_blocks"].tobytes().decode("utf-8"))
        io.save_checkpoint(str(path), fields, {k: v for k, v in meta.items() if k != "step"},
                           blocks=blocks)
    with pytest.raises(SetupError, match="检查点"):
        main.run(_restarting(args, restart=str(path)))


# ---- 8. 不覆盖输入、空路径、真的换了网格、zone 的原名 ----

def test_restart_refuses_to_overwrite_its_own_checkpoint(tmp_path):
    """检查点的文件名里步号比它记的步号大（比如改过名），又在输出目录里：续算写到那一步
    会把输入覆盖。启动时就报配置错误，输入文件原样保留。"""
    args = _builtin("sod", tmp_path, "--end-step", "3", "--kstep-save", "1")
    state = init.prepare(args)
    renamed = tmp_path / io.checkpoint_name("sod", 2)
    _save(renamed, state.cons0, args, state, 1, args.dt)     # 文件名是第 2 步，记的步号是 1
    before = renamed.read_bytes()
    with pytest.raises(runtime_config.ConfigError, match="覆盖"):
        main.run(_restarting(args, restart=str(renamed)))
    assert renamed.read_bytes() == before
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [renamed.name]


def test_multiblock_restart_refuses_to_overwrite_its_own_checkpoint(tmp_path):
    args = support.run_args("x2", out_dir=str(tmp_path), end_step=3, kstep_save=1, kstep_show=1)
    built = support.blocks("x2")
    renamed = tmp_path / io.checkpoint_name("x2", 2)
    _multi_checkpoint(renamed, built, {block.name: block.cons0 for block in built.blocks},
                      step7.reference().case["params"], 1, DT)
    before = renamed.read_bytes()
    with pytest.raises(runtime_config.ConfigError, match="覆盖"):
        main.run(_restarting(args, restart=str(renamed)))
    assert renamed.read_bytes() == before


def test_an_empty_restart_path_means_no_restart(tmp_path, capsys):
    """`--restart ""` 等于没给，从头算（jisen 的决定）；单块、多块都是。"""
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--case", "sod", "--dt", "1e-4",
         "--ni", "16", "--nj", "8", "--nk", "4", "--restart", "", "--end-step", "1",
         "--kstep-save", "1", "--out-dir", str(out)], cwd=PROJECT, capture_output=True,
        text=True)
    assert done.returncode == 0, done.stderr
    assert [p.name for p in out.iterdir()] == [io.checkpoint_name("sod", 1)]
    assert not [line for line in done.stdout.splitlines() if line.startswith("restart")]
    multi = tmp_path / "multi"
    multi.mkdir()
    main.run(support.run_args("x2", out_dir=str(multi), restart="",
                              end_step=1, kstep_save=1, kstep_show=1))
    assert [p.name for p in multi.iterdir()] == [io.checkpoint_name("x2", 1)]
    assert not [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("restart")]


def test_a_checkpoint_of_another_grid_size_is_a_config_error(tmp_path):
    """正式入口写出的 16×8×4 的检查点，拿到 20×8×4 上续算：先核对网格，报配置错误。"""
    main.run(_builtin("sod", tmp_path, "--end-step", "1"))
    [checkpoint] = tmp_path.glob("*.npz")
    out = tmp_path / "out"
    out.mkdir()
    wider = _builtin("sod", out, "--ni", "20", "--end-step", "2")
    with pytest.raises(runtime_config.ConfigError, match="尺寸"):
        main.run(_restarting(wider, restart=str(checkpoint)))
    assert not list(out.iterdir())


@needs_source
def test_meta_records_the_actual_zone_name(tmp_path):
    """--cgns-zone 前后带空白时选中的还是同一个 zone，_meta 里记它的原名；自动选区的运行
    能从它续算。"""
    args = _restarting(step7.viscous_args(step7.SOURCE), cgns_zone="  blk-1  ",
                       out_dir=str(tmp_path), end_step=1, kstep_save=1, kstep_show=1)
    main.run(args)
    [checkpoint] = tmp_path.glob("*.npz")
    meta, _, _ = io.load_checkpoint(str(checkpoint))
    assert meta["mesh"]["zone"] == "blk-1"
    main.run(_restarting(args, cgns_zone=None, restart=str(checkpoint), end_step=2,
                         kstep_save=2, kstep_show=2))
