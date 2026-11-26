"""第七步的放行与启动信息（U5）。

锁五件事：
  1. 粘性打开的多块 plate 配置能走通入口。从 ``main.run`` 跑两步，保存时机、文件名和
     ``_meta`` 与无粘时相同；读回的末态拼回以后，与打开粘性的单块参考在阈值内一致。
     文件名和 ``_meta`` 相同证明不了粘性生效，所以要有后半句。
  2. 粘性打开而粘性导数不是 CD8：JSON 给的、命令行给的各拦一次，都在读坐标之前，报错
     里点出 ``vis_order``。粘性关闭而导数是 CD6 时仍然放行。
  3. 粘性打开时，别的不放行的配置照旧拦住：FP32、UP7、特殊边界脚本、参考尺度按某一块填。
  4. 粘性打开时启动信息带 Re、Ama、参考温度、Pr 和导数的阶数；参考温度取缺省值时的提示
     与单块的逐字相同，粘性关闭时多块不打印这一行、两边都不提示。
  5. 命令行入口：粘性导数是 CD6 的运行只输出一行“初始化错误”，退出码 2，没有 traceback。

JSON 配置的构造、坏网格和入口解析用 ``test_step6_config`` 里的辅助函数。
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import io, main, multiblock, runtime_config  # noqa: E402
from opencfd_jax.src.errors import SetupError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
import step7_support as step7  # noqa: E402
import test_step6_config as step6  # noqa: E402

DT = 1e-4
STEPS = 2
NAMES = ["blk-i0j0k0", "blk-i1j0k0"]
ConfigError = runtime_config.ConfigError
pytestmark = pytest.mark.skipif(not support.SOURCE.exists(), reason="缺少源网格")


def _run(out, **changes):
    """走正式入口跑 x2，返回 args。输出目录平时由 main.main 建，这里直接建好。"""
    out.mkdir(parents=True)
    args = support.run_args("x2", out_dir=str(out), **changes)
    main.run(args)
    return args


def _read_back(out, step=STEPS):
    """把 x2 各块第 ``step`` 步存下的文件读回来，拼回等价单块。返回 (末态, 各块的 _meta)。"""
    built = support.blocks("x2")
    slugs = multiblock.block_slugs(NAMES)
    pieces, meta = {}, {}
    for block in built.blocks:
        fields, meta[block.name] = io.load_npz(
            str(out / io.checkpoint_name(f"x2-{slugs[block.name]}", step)))
        pieces[block.name] = np.stack([fields[f"cons_{i}"] for i in range(5)])
    return support.reassemble(support.sidecar("x2"), pieces), meta


# ---- 1. 粘性打开的运行走通入口 ----

def test_viscous_run_through_the_entry_matches_the_viscous_single_block(tmp_path):
    on, plain = tmp_path / "on", tmp_path / "plain"
    _run(on, viscous=True)
    _run(plain)
    files = [f"x2-{name}-istep-00000002.npz" for name in NAMES]
    assert sorted(p.name for p in on.iterdir()) == files            # 两个整步，最后一步存盘
    assert sorted(p.name for p in plain.iterdir()) == files         # 与无粘时相同
    saved, meta = _read_back(on)
    _, other = _read_back(plain)
    assert meta == other and [meta[name]["zone"] for name in NAMES] == NAMES
    # 单块参考：从缺省的初场走两步。plate 的初场内区是入口状态、鬼胞为零，各块的 cons0
    # 是同一个写法
    state = step7.reference()
    inner = state.desc.grid.inner()
    start = state.cons0
    inflow = np.asarray(start)[inner][:, :1, :1, :1]
    assert np.all(np.asarray(start)[inner] == inflow)                  # 初场是常数
    for block in support.blocks("x2").blocks:
        own = np.asarray(block.cons0)
        assert np.all(own[block.grid.inner()] == own[block.grid.inner()][:, :1, :1, :1])
        np.testing.assert_array_equal(own[block.grid.inner()][:, 0, 0, 0], inflow[:, 0, 0, 0])
    final = support.formal_step(state, support.formal_step(state, start, DT), DT)
    support.compare(np.asarray(final)[inner], saved, "读回的末态")


# ---- 2. 粘性导数只放行 CD8 ----

@pytest.mark.parametrize("overrides, argv", [
    ({"numerics.viscous": True, "numerics.vis_order": "CD6"}, ()),
    ({"numerics.viscous": True}, ("--vis-order", "CD6")),
    ({"numerics.vis_order": "CD6"}, ("--viscous",)),
    ({}, ("--viscous", "--vis-order", "CD6")),
], ids=["JSON", "命令行", "JSON 的阶数加命令行的开关", "开关和阶数都在命令行"])
def test_cd6_with_viscosity_is_rejected_before_coordinates_are_read(
        tmp_path, overrides, argv):
    broken = step6._broken_mesh(tmp_path)          # 读坐标会报错；要在那之前拦住
    with pytest.raises(SetupError, match="vis_order") as info:
        multiblock.enter(step6._resolve(tmp_path, overrides, argv, mesh=broken))
    assert "'CD6'" in str(info.value) and "CD8" in str(info.value)


def test_cd6_without_viscosity_is_still_released(tmp_path):
    setup = multiblock.enter(step6._resolve(tmp_path, {"numerics.vis_order": "CD6"}))
    assert len(setup.topology.zones) == 2


# ---- 3. 粘性打开时别的不放行配置照旧拦住 ----

@pytest.mark.parametrize("overrides, argv, text, kind", [
    ({"precision": "fp32"}, (), "FP64", SetupError),
    ({}, ("--fp32",), "FP64", SetupError),
    ({"numerics.scheme": "UP7"}, (), "WENO7", SetupError),
    ({"specificboundary": "trip.py"}, (), "specificboundary", SetupError),
    ({"numerics.flux_scale": [361, 931, 931]}, (), "blk-i0j0k0", ConfigError),
], ids=["JSON 的 FP32", "命令行的 FP32", "UP7", "特殊边界脚本", "参考尺度按某一块填"])
def test_other_unreleased_settings_are_still_rejected_with_viscosity_on(
        tmp_path, overrides, argv, text, kind):
    mesh = (step6._broken_mesh(tmp_path) if kind is SetupError
            else support.mesh_path("xyz8"))        # 按 50×20×20 的那一块填的
    with pytest.raises(kind) as info:
        multiblock.enter(step6._resolve(
            tmp_path, {"numerics.viscous": True, **overrides}, argv, mesh=mesh))
    assert text in str(info.value)


# ---- 4. 启动信息 ----

def test_startup_line_lists_the_physics_only_when_viscosity_is_on(tmp_path, capsys):
    for viscous_on in (True, False):
        out = tmp_path / f"on-{viscous_on}"
        out.mkdir()
        main.run(support.run_args("x2", out_dir=str(out), viscous=viscous_on,
                                  end_time=DT, end_step=1))
        lines = capsys.readouterr().out.splitlines()
        listed = [line for line in lines if line.startswith("vis-order=")]
        assert listed == (["vis-order=CD8 Re=1000.0 Ama=0.1 Ref_T=288.15 Pr=0.7"]
                          if viscous_on else [])
        assert not any(line.startswith("WARNING: Ref_T") for line in lines)   # 给了参考温度


def test_default_reference_temperature_warns_like_the_single_block(tmp_path, capsys):
    def warning(mesh, viscous_on):
        """跑一步，取出关于 Ref_T 的提示；没有就返回 None。"""
        out = tmp_path / f"out-{mesh.stem}-{viscous_on}"
        out.mkdir()
        args = step6._resolve(tmp_path, {
            "physics.ref_t": None, "numerics.viscous": viscous_on,
            "numerics.flux_scale": None if mesh.stem.startswith("step7-") else [1521, 3861, 3861],
            "time.end_time": DT, "time.end_step": 1, "output.dir": str(out)}, mesh=mesh)
        capsys.readouterr()
        main.run(args)
        notes = [line for line in capsys.readouterr().out.splitlines()
                 if line.startswith("WARNING: Ref_T")]
        assert len(notes) <= 1
        return notes[0] if notes else None

    many, one = support.mesh_path("x2"), step7.single_path("two")
    note, alone = warning(many, True), warning(one, True)
    assert note is not None and note == alone             # 与单块的逐字相同
    assert warning(many, False) is None and warning(one, False) is None


# ---- 5. 命令行入口 ----

def test_cd6_given_on_the_command_line_reports_one_line(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(step6._config(support.mesh_path("x2"),
                                             {"numerics.viscous": True})), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--config", str(path),
         "--vis-order", "CD6"],
        cwd=step6.PROJECT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    reported = [line for line in done.stderr.splitlines() if "初始化错误" in line]
    assert len(reported) == 1 and "vis_order" in reported[0]
