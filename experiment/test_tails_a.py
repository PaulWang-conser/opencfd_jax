"""补尾巴 A 组：报错写清出处、存盘的错误分层（RA2 到 RA4、RA9）。

对照方案里的异常出口矩阵（KTD10），每个出口一条测试。

启动时的错误只往 stderr 写一行、退出码 2，不再带命令行的用法说明：

- ``配置错误：<出处>：<原因>`` 或 ``初始化错误：<出处>：<原因>``；
- 出处是参数时写成 ``<参数名>（命令行 <选项>，JSON <键>）``，原因的后面注明这个值是
  哪一边给的；出处是阶段时沿用消息里原有的开头（初始化失败、多块模式不放行、
  多块网格没有通过校验）；
- 命令行本身解析不了时照旧由 argparse 报，带用法说明。

运行期错误分四个阶段：装配 RHS、RHS 预热、时间推进、存盘。存盘是新加的；多块原来只有
时间推进一个名目，进循环以前先用起点状态空跑一步，算 RHS 预热。

走正式入口的测试在进程内调 ``main.main(argv)``：启动错误写一行 stderr 以后 ``sys.exit(2)``，
不必另起进程。另起进程的只留一条，核对真实的退出码和 traceback 的最后一行。
读续算的检查点那几行由 ``test_step8_restart.py`` 守着，这里不重复。
"""

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import init, io, main, multiblock, runtime_config, solver  # noqa: E402
from opencfd_jax.src.errors import RuntimeExecutionError, SetupError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import test_step6_config as step6  # noqa: E402

PROJECT = REPO.parent
DT = 1e-4
needs_mesh = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")
RELEASE_NOTE = "显式给出了这个参数；多块模式只放行它的缺省值，取值与缺省相同时也算给出"


def startup(capsys, *argv):
    """走正式入口启动一次应当在启动时失败的运行，返回 stderr 的各行。"""
    with pytest.raises(SystemExit) as excinfo:
        main.main(list(argv))
    assert excinfo.value.code == 2
    return capsys.readouterr().err.splitlines()


def one_line(lines):
    """启动错误只有一行，开头不是用法说明。"""
    assert len(lines) == 1, lines
    assert not lines[0].startswith("usage:")
    return lines[0]


def _config_of(tmp_path, mesh, overrides=None):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(step6._config(mesh, overrides)), encoding="utf-8")
    return path


# ---- 1. 命令行解析不了 ----

def test_argparse_still_reports_its_own_errors_with_usage(capsys):
    lines = startup(capsys, "--no-such-option")
    assert lines[0].startswith("usage:")
    assert "no-such-option" in lines[-1]


# ---- 2. 两个互相冲突的开关 ----

@pytest.mark.parametrize("name, options, key", [
    ("viscous", "--viscous 和 --no-viscous", "numerics.viscous"),
    ("weno_tv", "--weno-tv 和 --no-weno-tv", "numerics.weno_tv"),
])
def test_conflicting_flags_report_one_line(capsys, name, options, key):
    """Covers AE9。"""
    flag = "--" + name.replace("_", "-")
    line = one_line(startup(capsys, flag, "--no-" + flag[2:], "--dt", "1e-4"))
    assert line == f"配置错误：{name}（命令行 {options}，JSON {key}）：不能同时给出"


# ---- 3. 读 JSON 配置 ----

def test_missing_config_file(tmp_path, capsys):
    path = tmp_path / "nope.json"
    line = one_line(startup(capsys, "--config", str(path)))
    assert line == f"配置错误：读 JSON 配置：{path}：找不到配置文件"


def test_config_file_that_cannot_be_read(tmp_path, capsys, monkeypatch):
    """Covers AE8。没有读的权限用猴子补丁造：Windows 上改权限位挡不住文件的属主。"""
    path = tmp_path / "case.json"
    path.write_text("{}", encoding="utf-8")
    read_text = Path.read_text

    def refuse(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)
    line = one_line(startup(capsys, "--config", str(path)))
    assert line == f"配置错误：读 JSON 配置：{path}：读不了：[Errno 13] Permission denied"


def test_config_file_that_is_not_json(tmp_path, capsys):
    path = tmp_path / "case.json"
    path.write_text("{", encoding="utf-8")
    line = one_line(startup(capsys, "--config", str(path)))
    assert line.startswith(f"配置错误：读 JSON 配置：{path}：不是合法 UTF-8 JSON：")


def test_config_file_whose_top_level_is_not_an_object(tmp_path, capsys):
    path = tmp_path / "case.json"
    path.write_text("[1]", encoding="utf-8")
    line = one_line(startup(capsys, "--config", str(path)))
    assert line == f"配置错误：读 JSON 配置：{path}：顶层必须是对象"


# ---- 4. JSON 的字段：类型不对、超出范围 ----

@pytest.mark.parametrize("raw, expected", [
    ({"output": {"kstep_save": 0}},
     "kstep_save（命令行 --kstep-save，JSON output.kstep_save）："
     "output.kstep_save 不能小于 1，得到 0（JSON 给的）"),
    ({"output": {"kstep_save": "1"}},
     "kstep_save（命令行 --kstep-save，JSON output.kstep_save）："
     "output.kstep_save 需要整数，得到 str（JSON 给的）"),
    ({"case": {"name": "plate"}, "boundary": {"wall_temperature": "1"}},
     "boundary_wall_temperature（JSON boundary.wall_temperature）："
     "boundary.wall_temperature 需要数值，得到 str（JSON 给的）"),
    ({"case": {"name": "plate"},
      "boundary": {"inflow": {"rho": "1", "u": 1, "v": 0, "w": 0, "T": 1}}},
     "boundary_inflow（JSON boundary.inflow）："
     "boundary.inflow.rho 需要数值，得到 str（JSON 给的）"),
], ids=["超出范围", "类型不对", "只有 JSON 能给的字段", "状态里的一个分量"])
def test_json_field_errors_carry_the_origin(tmp_path, raw, expected):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(runtime_config.ConfigError) as info:
        runtime_config.load_config(str(path))
    assert str(info.value) == expected


@pytest.mark.parametrize("raw, expected", [
    ({"foo": 1}, "未知顶层字段：['foo']"),
    ({"output": {"foo": 1}}, "未知字段：['output.foo']"),
], ids=["顶层", "节内"])
def test_unknown_fields_keep_their_message(tmp_path, raw, expected):
    """不认识的字段没有参数名可写，消息照旧。"""
    path = tmp_path / "case.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(runtime_config.ConfigError) as info:
        runtime_config.load_config(str(path))
    assert str(info.value) == expected


# ---- 5. 合并后的校验：注明是哪一边给的 ----

def test_merged_check_for_a_command_line_value(capsys):
    line = one_line(startup(capsys, "--ni", "0", "--dt", "1e-4"))
    assert line == ("配置错误：ni（命令行 --ni）：网格尺寸 ni 必须为正整数，"
                    "得到 0（命令行给的）")


def test_merged_check_for_a_json_value(tmp_path, capsys):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"boundary": {"outlet_ghost": "copy"}}), encoding="utf-8")
    line = one_line(startup(capsys, "--config", str(path), "--dt", "1e-4"))
    assert line == ("配置错误：boundary_outlet_ghost（命令行 --outlet-ghost，"
                    "JSON boundary.outlet_ghost）：boundary.outlet_ghost 只适用于算例 "
                    "['plate']，当前为 'sod'（JSON 给的）")


def test_merged_check_for_the_same_value_from_the_command_line(capsys):
    line = one_line(startup(capsys, "--outlet-ghost", "copy", "--dt", "1e-4"))
    assert line == ("配置错误：boundary_outlet_ghost（命令行 --outlet-ghost，"
                    "JSON boundary.outlet_ghost）：boundary.outlet_ghost 只适用于算例 "
                    "['plate']，当前为 'sod'（命令行给的）")


def test_plate_parameters_in_the_wrong_order(tmp_path, capsys):
    out = tmp_path / "out"
    line = one_line(startup(capsys, "--case", "plate", "--plate-x0", "0.5", "--dt", "1e-4",
                            "--out-dir", str(out)))
    assert line == ("配置错误：plate_x0（命令行 --plate-x0，JSON case.plate_x0）："
                    "plate_x0 必须小于 plate_x1，得到 0.5 和 0.25（命令行给的）")


# ---- 6. 输入文件不存在、dt 不合格 ----

def test_missing_restart_file(tmp_path, capsys):
    """Covers AE1。"""
    missing = tmp_path / "nope.npz"
    line = one_line(startup(capsys, "--restart", str(missing), "--dt", "1e-4"))
    assert line == ("配置错误：restart（命令行 --restart，JSON time.restart）："
                    f"命令行给的文件不存在或不可读：{missing}")


def test_restart_path_that_is_a_directory(tmp_path, capsys):
    line = one_line(startup(capsys, "--restart", str(tmp_path), "--dt", "1e-4"))
    assert line == ("配置错误：restart（命令行 --restart，JSON time.restart）："
                    f"命令行给的不是普通文件：{tmp_path}")


@pytest.mark.parametrize("value, note", [
    (None, ""), ("nan", "（命令行给的）"), ("0", "（命令行给的）"), ("-1e-4", "（命令行给的）"),
], ids=["没给", "nan", "0", "负数"])
def test_invalid_dt(capsys, value, note):
    # 负数要用“--dt=-1e-4”的写法：argparse 会把跟在后面的 -1e-4 当成选项
    argv = [] if value is None else ["--dt=" + value]
    line = one_line(startup(capsys, *argv))
    assert line == ("配置错误：dt（命令行 --dt，JSON time.dt）："
                    "固定 dt 必须显式给出，且为正的有限数" + note)


def test_missing_mesh_file(tmp_path, capsys):
    """sod 不支持 CGNS 网格，所以要带上算例 plate，才轮得到核对网格文件。"""
    missing = tmp_path / "nope.cgns"
    line = one_line(startup(capsys, "--case", "plate", "--mesh-format", "cgns",
                            "--mesh", str(missing), "--dt", "1e-4"))
    assert line == ("配置错误：mesh（命令行 --mesh，JSON mesh.file）："
                    f"命令行给的文件不存在或不可读：{missing}")


def test_cgns_format_without_a_mesh_file(capsys):
    line = one_line(startup(capsys, "--case", "plate", "--mesh-format", "cgns",
                            "--dt", "1e-4"))
    assert line == ('配置错误：mesh（命令行 --mesh，JSON mesh.file）：'
                    'mesh.format="cgns" 需要 mesh.file')


# ---- 7. 输出目录 ----

def test_output_directory_cannot_be_created(tmp_path, capsys):
    """Covers AE10。输出目录的上一级是一个文件，建不出来。"""
    (tmp_path / "file").write_text("", encoding="utf-8")
    out = tmp_path / "file" / "out"
    line = one_line(startup(capsys, "--dt", "1e-4", "--out-dir", str(out)))
    head = f"配置错误：建输出目录：{out}（命令行 --out-dir，JSON output.dir）："
    assert line.startswith(head) and len(line) > len(head)


def test_non_positive_output_interval_from_the_command_line(tmp_path, capsys):
    line = one_line(startup(capsys, "--kstep-save", "0", "--dt", "1e-4",
                            "--out-dir", str(tmp_path / "out")))
    assert line == ("配置错误：kstep_save（命令行 --kstep-save，JSON output.kstep_save）："
                    "输出间隔 kstep_save 必须是正数，得到 0（命令行给的）")


# ---- 8. 多块：放行校验和配置 ----

MULTI_RELEASE = [
    # (JSON 的改动, 命令行, 出处, 原因, 原因后面注明的来源)
    ({"numerics.scheme": "UP7"}, (), "scheme（命令行 --scheme，JSON numerics.scheme）",
     "格式是 'UP7'，只放行 WENO7", "JSON 给的"),
    ({}, ("--scheme", "UP7"), "scheme（命令行 --scheme，JSON numerics.scheme）",
     "格式是 'UP7'，只放行 WENO7", "命令行给的"),
    ({"case.name": "poiseuille", "boundary.inflow": None, "boundary.farfield": None,
      "boundary.wall_temperature": None, "numerics.flux_scale": None}, (),
     "case（命令行 --case，JSON case.name）", "算例是 'poiseuille'，只放行 plate", "JSON 给的"),
    ({"specificboundary": "trip.py"}, (),
     "specificboundary（命令行 --specificboundary，JSON specificboundary）",
     "带了特殊边界脚本 'trip.py'（specificboundary），多块暂不支持", "JSON 给的"),
    ({"numerics.weno_tv": True}, (),
     "weno_tv（命令行 --weno-tv 和 --no-weno-tv，JSON numerics.weno_tv）",
     RELEASE_NOTE, "JSON 给的"),
    ({}, ("--ni", "100"), "ni（命令行 --ni）", RELEASE_NOTE, "命令行给的"),
    ({}, ("--tw", "1.0"), "tw（命令行 --tw，JSON physics.tw）", RELEASE_NOTE, "命令行给的"),
]


@needs_mesh
@pytest.mark.parametrize("overrides, argv, origin, reason, given", MULTI_RELEASE,
                         ids=["JSON 的格式", "命令行的格式", "算例", "特殊边界脚本",
                              "JSON 的 weno_tv", "命令行的 ni", "命令行的 tw"])
def test_multiblock_release_errors_carry_the_origin(
        tmp_path, capsys, overrides, argv, origin, reason, given):
    path = _config_of(tmp_path, support.mesh_path("x2"), overrides)
    line = one_line(startup(capsys, "--config", str(path),
                            "--out-dir", str(tmp_path / "out"), *argv))
    assert line == f"初始化错误：多块模式不放行：{origin}：{reason}（{given}）"


@needs_mesh
def test_multiblock_flux_scale_missing(tmp_path, capsys):
    path = _config_of(tmp_path, support.mesh_path("x2"), {"numerics.flux_scale": None})
    line = one_line(startup(capsys, "--config", str(path), "--out-dir", str(tmp_path / "out")))
    assert line == ("配置错误：flux_scale（命令行 --flux-scale，JSON numerics.flux_scale）："
                    "多块运行必须给出三轴参考尺度：取等价单块的全局顶点数里其余两轴 (N-1) "
                    "的乘积，100×40×40 对应 1521 3861 3861")


@needs_mesh
def test_multiblock_flux_scale_filled_per_block(tmp_path, capsys):
    """按 50×20×20 的那一块填的参考尺度，i 向的分量太小。"""
    path = _config_of(tmp_path, support.mesh_path("xyz8"), {"numerics.flux_scale": None})
    line = one_line(startup(capsys, "--config", str(path), "--out-dir", str(tmp_path / "out"),
                            "--flux-scale", "361", "931", "931"))
    head = ("配置错误：flux_scale（命令行 --flux-scale，JSON numerics.flux_scale）："
            "在 i 向的分量 361 小于按块 ")
    assert line.startswith(head) and line.endswith("不能按某一块的尺寸填（命令行给的）")


# ---- 9. 多块：网格校验；初始化 ----

@needs_mesh
def test_multiblock_mesh_check_reports_one_line(tmp_path, capsys):
    path = _config_of(tmp_path, step6._broken_mesh(tmp_path))
    line = one_line(startup(capsys, "--config", str(path), "--out-dir", str(tmp_path / "out")))
    assert line.startswith("初始化错误：多块网格没有通过校验：")


def test_initialisation_failure_reports_one_line(tmp_path, capsys):
    """建算例失败（这里给一个读不出来的网格文件）：一行，原因里带着原始异常。"""
    bad = tmp_path / "bad.cgns"
    bad.write_text("这不是一个 HDF5 文件", encoding="utf-8")
    line = one_line(startup(capsys, "--case", "poiseuille", "--mesh", str(bad), "--dt", "1e-4",
                            "--out-dir", str(tmp_path / "out")))
    assert line.startswith("初始化错误：初始化失败：case=poiseuille ")
    assert "原始异常=" in line and str(bad) in line


@pytest.mark.parametrize("name, text, reason", [
    ("a/b.py", None, "纯文件名"),
    ("nope.py", None, "不存在"),
    ("noentry.py", "def other():\n    pass\n", "build_special_boundary"),
], ids=["名字带路径", "脚本不存在", "脚本缺入口"])
def test_a_misconfigured_special_boundary_script_names_the_parameter(
        tmp_path, monkeypatch, capsys, name, text, reason):
    """脚本自己的配置错也写成标准格式：初始化错误：specificboundary（命令行 --specificboundary，
    JSON specificboundary）：原因（jisen 2026-11-27 定的）。"""
    scripts = tmp_path / "special"
    scripts.mkdir()
    if text is not None:
        (scripts / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", scripts)
    line = one_line(startup(capsys, "--specificboundary", name, "--dt", "1e-4",
                            "--out-dir", str(tmp_path / "out")))
    assert line.startswith("初始化错误：specificboundary（命令行 --specificboundary，JSON specificboundary）：")
    assert reason in line and line.endswith("（命令行给的）")


# ---- 10. 运行期错误的四个阶段 ----

def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _single_args(out):
    return main.build_parser().parse_args(
        ["--ni", "16", "--nj", "8", "--nk", "4", "--dt", "1e-4", "--end-step", "1",
         "--kstep-show", "1", "--kstep-save", "1", "--out-dir", str(out)])


def test_single_block_rhs_assembly_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "build_rhs", _boom)
    with pytest.raises(RuntimeExecutionError) as info:
        main.run(_single_args(tmp_path / "out"))
    assert str(info.value) == ("运行期错误（装配 RHS）：case=sod step=0 t=0 dt=0.0001 "
                               "grid=16x8x4 原始异常=RuntimeError: boom")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_single_block_rhs_warmup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "build_rhs", lambda *args, **kwargs: _boom)
    with pytest.raises(RuntimeExecutionError) as info:
        main.run(_single_args(tmp_path / "out"))
    assert str(info.value) == ("运行期错误（RHS 预热）：case=sod step=0 t=0 dt=0.0001 "
                               "grid=16x8x4 原始异常=RuntimeError: boom")


def test_single_block_time_stepping_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "build_rhs", lambda *args, **kwargs: (lambda q: q))
    monkeypatch.setattr(main, "rk3_step", _boom)
    with pytest.raises(RuntimeExecutionError) as info:
        main.run(_single_args(tmp_path / "out"))
    assert str(info.value) == ("运行期错误（时间推进）：case=sod step=1 t=0.0 dt=0.0001 "
                               "grid=16x8x4 原始异常=RuntimeError: boom")


def test_single_block_save_failure(tmp_path, monkeypatch):
    """Covers AE2。"""
    out = tmp_path / "out"
    monkeypatch.setattr(main, "build_rhs", lambda *args, **kwargs: (lambda q: q))
    monkeypatch.setattr(io, "save_checkpoint", _boom)
    with pytest.raises(RuntimeExecutionError) as info:
        main.run(_single_args(out))
    target = f"{out}/{io.checkpoint_name('sod', 1)}"
    assert str(info.value) == ("运行期错误（存盘）：case=sod step=1 t=0.0001 dt=0.0001 "
                               f"grid=16x8x4 输出路径={target} 原始异常=RuntimeError: boom")
    assert isinstance(info.value.__cause__, RuntimeError)


def _multi_args(tmp_path):
    """x2 跑一步的参数；输出目录平时由 main.main 建，这里直接建好。"""
    out = tmp_path / "out"
    out.mkdir()
    return support.run_args("x2", out_dir=str(out), end_time=DT, end_step=1), out


def _step_that_changes_nothing(kernels, cons, dt, **kwargs):
    return SimpleNamespace(final=cons)


@needs_mesh
def test_multiblock_assembly_failure(tmp_path, monkeypatch):
    args, _ = _multi_args(tmp_path)
    setup = multiblock.enter(args)
    monkeypatch.setattr(multiblock, "build_blocks", _boom)
    with pytest.raises(SetupError, match="^多块网格装配失败：boom") as info:
        multiblock.run(args, setup)
    assert isinstance(info.value.__cause__, RuntimeError)


@needs_mesh
def test_multiblock_warmup_failure(tmp_path, monkeypatch):
    args, _ = _multi_args(tmp_path)
    setup = multiblock.enter(args)
    monkeypatch.setattr(multiblock, "solve_step", _boom)
    with pytest.raises(RuntimeExecutionError) as info:
        multiblock.run(args, setup)
    assert str(info.value) == ("运行期错误（RHS 预热）：case=plate step=0 t=0.0 dt=0.0001 "
                               "blocks=2 原始异常=RuntimeError: boom")


@needs_mesh
def test_multiblock_time_stepping_failure(tmp_path, monkeypatch):
    args, _ = _multi_args(tmp_path)
    setup = multiblock.enter(args)
    calls = []

    def second_call_fails(kernels, cons, dt, **kwargs):
        calls.append(dt)
        if len(calls) > 1:
            raise RuntimeError("boom")
        return SimpleNamespace(final=None)       # 空跑的结果是要丢掉的

    monkeypatch.setattr(multiblock, "solve_step", second_call_fails)
    with pytest.raises(RuntimeExecutionError) as info:
        multiblock.run(args, setup)
    assert str(info.value) == ("运行期错误（时间推进）：case=plate step=1 t=0.0 dt=0.0001 "
                               "blocks=2 原始异常=RuntimeError: boom")
    assert calls == [DT, DT]                     # 空跑的一步在前，用的是起点状态


@needs_mesh
def test_multiblock_save_failure(tmp_path, monkeypatch):
    args, out = _multi_args(tmp_path)
    setup = multiblock.enter(args)
    monkeypatch.setattr(multiblock, "solve_step", _step_that_changes_nothing)
    monkeypatch.setattr(io, "save_checkpoint", _boom)
    with pytest.raises(RuntimeExecutionError) as info:
        multiblock.run(args, setup)
    target = f"{out}/{io.checkpoint_name('x2', 1)}"
    assert str(info.value) == ("运行期错误（存盘）：case=plate step=1 t=0.0001 dt=0.0001 "
                               f"blocks=2 输出路径={target} 原始异常=RuntimeError: boom")


def test_save_failure_from_the_command_line_keeps_the_traceback(tmp_path):
    """Covers AE2。检查点该写的位置被一个同名的目录占着，写不出去。"""
    out = tmp_path / "out"
    out.mkdir()
    name = io.checkpoint_name("sod", 1)
    (out / name).mkdir()
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--ni", "16", "--nj", "8", "--nk", "4",
         "--dt", "1e-4", "--end-step", "1", "--kstep-save", "1", "--out-dir", str(out)],
        cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 1, done.stderr
    # 原始异常的类型随系统变（Windows 是 PermissionError，Linux 是 IsADirectoryError），
    # 所以只核对到“原始异常=”为止
    assert done.stderr.strip().splitlines()[-1].startswith(
        "opencfd_jax.src.errors.RuntimeExecutionError: "
        f"运行期错误（存盘）：case=sod step=1 t=0.0001 dt=0.0001 grid=16x8x4 "
        f"输出路径={out}/{name} 原始异常=")


# ---- 11. RK 接口不收回调 ----

def test_rk3_functions_take_no_boundary_callback():
    """边界只在 prepare 里走一遍，RK 的两个函数不收 ``bc_apply``（第 4.n 步的 R7）。
    第七步和第六步的基线逐位不变，由它们自己的验收测试守着。"""
    for function in (solver.rk3_step_stages, solver.rk3_step):
        assert "bc_apply" not in inspect.signature(function).parameters


# ---- 12. 标准的 src/input.json ----

STANDARD = REPO / "src" / "input.json"
STANDARD_CONTENT = {
    "case": {"name": "plate"},
    "mesh": {"format": "cgns", "file": "../Grid/testcgnstest.cgns"},
    "boundary": {
        "inflow": {"rho": 1, "u": 1, "v": 0, "w": 0, "T": 1},
        "farfield": {"rho": 1, "u": 1, "v": 0, "w": 0, "T": 1},
        "wall_temperature": 1.0,
        "outlet_ghost": "keep",
    },
    "physics": {"Re": 1000.0, "Ama": 0.1, "ref_t": 288.15, "pr": 0.7},
    "numerics": {"scheme": "WENO7", "viscous": True, "vis_order": "CD8"},
    "time": {"dt": 0.0001, "end_time": 0.0001, "end_step": 1},
    "output": {"dir": "../../output", "kstep_show": 1, "kstep_save": 1},
}


def _resolve_config(path, *argv):
    """按 main.main 的顺序解析：默认值、显式命令行、JSON。"""
    parser = main.build_parser(explicit_only=True)
    _, explicit = main.resolve_cli(vars(parser.parse_args(argv)))
    return runtime_config.resolve_args(explicit, runtime_config.load_config(str(path)),
                                       main.build_parser().parse_args([]))


def test_standard_input_json_has_the_agreed_content():
    """单块 plate 的全部键，一个不多一个不少；相对路径相对 JSON 所在的目录。"""
    assert json.loads(STANDARD.read_text(encoding="utf-8")) == STANDARD_CONTENT


def test_standard_input_json_writes_where_the_command_line_default_does():
    """正式入口从工作区根目录运行，命令行缺省写进工作区根目录的 ``output/``；
    标准配置的 ``output.dir`` 落在同一个地方。"""
    assert Path(_resolve_config(STANDARD).out_dir) == REPO.parent / "output"


@needs_mesh
def test_standard_input_json_runs(tmp_path):
    """Covers AE5。从工作区根目录用 ``--config opencfd_jax/src/input.json`` 能跑完。
    测试里把输出目录改到临时目录，免得往工作区里写文件。"""
    out = tmp_path / "out"
    main.main(["--config", str(STANDARD), "--out-dir", str(out)])
    assert [p.name for p in out.iterdir()] == [io.checkpoint_name("testcgnstest", 1)]


# ---- 13. 建算例阶段的错误也不漏（sol 第 1 轮指出的）----

def test_an_exception_inside_a_special_boundary_script_is_an_initialisation_error(
        tmp_path, monkeypatch, capsys):
    """脚本是用户自己的代码，抛什么异常都有可能。这里故意让它抛 RuntimeError：也要归初始化
    错误，只写一行，原因里点名脚本和原始的异常。"""
    scripts = tmp_path / "special"
    scripts.mkdir()
    (scripts / "boom.py").write_text(
        "def build_special_boundary(case):\n    raise RuntimeError('脚本里的错')\n",
        encoding="utf-8")
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", scripts)
    line = one_line(startup(capsys, "--specificboundary", "boom.py", "--dt", "1e-4",
                            "--out-dir", str(tmp_path / "out")))
    assert line.startswith("初始化错误：")
    assert "boom.py" in line and "RuntimeError: 脚本里的错" in line


@pytest.mark.parametrize("value, reason", [
    ("bad", "要写成六个 0 或 1"),
    ("0 0 0 0 1", "要写成六个 0 或 1"),
    ("0 0 0 0 2 0", "只能是 0 或 1"),
], ids=["不是数", "只有五个", "有 2"])
def test_a_malformed_plate_dif_bound_names_the_parameter(tmp_path, capsys, value, reason):
    """在配置阶段就校验，报配置错误并带出处；不再等到建算例时才报一个没有出处的错。"""
    line = one_line(startup(capsys, "--case", "plate", "--plate-dif-bound", value,
                            "--dt", "1e-4", "--out-dir", str(tmp_path / "out")))
    assert line == ("配置错误：plate_dif_bound（命令行 --plate-dif-bound）："
                    f"{reason}，得到 {value!r}（命令行给的）")


def test_a_well_formed_plate_dif_bound_is_passed_through(tmp_path, capsys):
    """合规的写法照旧能跑：六个 0 或 1，空格或逗号分开。"""
    for value in ("0 0 0 0 1 0", "0,0,0,0,1,0"):
        main.main(["--case", "plate", "--plate-dif-bound", value, "--dt", "1e-4",
                   "--ni", "20", "--nj", "32", "--nk", "6", "--end-step", "1",
                   "--kstep-save", "1", "--out-dir", str(tmp_path / "out")])
        capsys.readouterr()


def test_a_missing_inflow_profile_names_the_parameter(tmp_path, capsys):
    missing = tmp_path / "nope.txt"
    line = one_line(startup(capsys, "--case", "plate", "--inflow-profile", str(missing),
                            "--dt", "1e-4", "--out-dir", str(tmp_path / "out")))
    assert line == ("配置错误：inflow_profile（命令行 --inflow-profile，JSON case.inflow_profile）："
                    f"命令行给的文件不存在或不可读：{missing}")
