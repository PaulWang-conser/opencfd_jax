"""Step 3 统一运行配置（`input.json`）的验收测试。

覆盖：默认值/JSON/显式 CLI 的合并优先级、字段与算例适用范围、槽道 `.in`
优先级、输出前缀、负向开关、自适应 dt，以及七条 Step 0 用例的 JSON 与旧
CLI 参数等价。
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
for path in (PROJECT, EXPERIMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from opencfd_jax.src import runtime_config, run     # noqa: E402
from step0_cases import CASES, config_path        # noqa: E402


def parse(argv):
    """按 run.main 的顺序解析：默认值、显式 CLI、可选 JSON。"""
    defaults = run.build_parser().parse_args([])
    parser = run.build_parser(explicit_only=True)
    config_path, explicit = run.resolve_cli(
        vars(parser.parse_args(argv)), parser)
    cfg = runtime_config.load_config(
        config_path, case=explicit.get("case")) if config_path else None
    return runtime_config.resolve_args(explicit, cfg, defaults)


def write_json(tmp_path, payload, name="input.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_defaults_unchanged():
    args = parse([])
    assert args.case == "sod"
    assert args.dt is None
    assert args.Re == 1000.0
    assert args.viscous is False
    assert args.out_prefix == "sod"
    assert args.config_path is None


def test_json_then_explicit_cli(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "physics": {"Re": 123.0},
        "time": {"dt": 1e-3},
    })
    args = parse(["--config", cfg])
    assert args.Re == 123.0
    assert args.dt == 1e-3
    assert args.config_sources["Re"] == "physics.Re"

    args = parse(["--config", cfg, "--Re", "42", "--dt", "2e-4"])
    assert args.Re == 42.0
    assert args.dt == 2e-4
    assert args.config_sources["Re"] == "cli"


def test_explicit_detection_does_not_compare_values(tmp_path):
    """JSON 值等于程序默认值，仍应记为 JSON 来源而不是 default。"""
    cfg = write_json(tmp_path, {"case": {"name": "sod"},
                                "physics": {"Re": 1000.0}})
    args = parse(["--config", cfg])
    assert args.Re == 1000.0
    assert args.config_sources["Re"] == "physics.Re"


def test_partial_json_uses_cli_case(tmp_path):
    """JSON 省略 case.name 时，按显式 --case 判断字段适用范围。"""
    cfg = write_json(tmp_path, {
        "mesh": {"infile": str(EXPERIMENT / "cases" / "channel_small.in")},
    })
    args = parse(["--config", cfg, "--case", "channel"])
    assert args.case == "channel"
    assert args.infile.endswith("channel_small.in")


def test_output_dir_relative_to_json(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "output": {"dir": "runs"},
    })
    args = parse(["--config", cfg])
    assert Path(args.out_dir) == tmp_path / "runs"


def test_plate_geometry_validated_after_merge(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "plate", "plate_x0": 1.1},
    })
    with pytest.raises(runtime_config.ConfigError, match="plate_x0"):
        parse(["--config", cfg])
    with pytest.raises(runtime_config.ConfigError, match="plate_x0"):
        parse(["--config", cfg, "--plate-x1", "0.5"])


def test_cli_case_override_rechecks_json_scope(tmp_path):
    """CLI --case 改变最终算例后，JSON 内原算例专属字段必须报错。"""
    cfg = write_json(tmp_path, {
        "case": {"name": "plate", "plate_x0": 0.1},
    })
    with pytest.raises(runtime_config.ConfigError, match="plate_x0"):
        parse(["--config", cfg, "--case", "channel"])


def test_invalid_case_name_rejected(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "not-a-case"}})
    with pytest.raises(runtime_config.ConfigError, match="case.name"):
        runtime_config.load_config(cfg)


def test_unknown_field_rejected(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "sod"}, "mesh": {"dims": [1]}})
    with pytest.raises(runtime_config.ConfigError, match="mesh.dims"):
        runtime_config.load_config(cfg)


def test_case_scope_rejected(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "sod", "p_grad": 0.1}})
    with pytest.raises(runtime_config.ConfigError, match="p_grad"):
        runtime_config.load_config(cfg)


def test_channel_requires_infile(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "channel"}})
    with pytest.raises(runtime_config.ConfigError, match="mesh.infile"):
        runtime_config.load_config(cfg)


def test_channel_infile_wins_over_config():
    """槽道 `.in` 仍然是有效物性的来源，JSON 的 physics 只做提示。"""
    args = parse(["--config", str(config_path(CASES[6]))])
    from opencfd_jax.src.cases import build_case
    case = build_case(args)
    assert args.Re == 1000.0          # 元数据仍记录命令行/默认值
    assert case["params"].Re == pytest.approx(190.7, rel=1e-3)


def test_negative_switch_overrides_json(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "numerics": {"viscous": True, "filter": True},
    })
    args = parse(["--config", cfg, "--no-viscous", "--no-filter"])
    assert args.viscous is False
    assert args.filter is False


def test_positive_and_negative_switch_conflict(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "sod"}})
    with pytest.raises(SystemExit):
        parse(["--config", cfg, "--viscous", "--no-viscous"])


def test_adaptive_dt_overrides_json(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "time": {"dt": 1e-3, "cfl": 0.4},
    })
    args = parse(["--config", cfg, "--adaptive-dt"])
    assert args.dt is None
    assert args.cfl == 0.4
    with pytest.raises(SystemExit):
        parse(["--config", cfg, "--dt", "1e-3", "--adaptive-dt"])


def test_precision_merge(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "sod"}, "precision": "fp32"})
    args = parse(["--config", cfg])
    assert args.fp32 is True and args.fp64 is False
    args = parse(["--config", cfg, "--fp64"])
    assert args.fp32 is False and args.fp64 is True


def test_output_prefix_from_mesh_and_case():
    assert runtime_config.derive_prefix(r"a\b\OCFD3d-Mesh.dat", "channel") \
        == "OCFD3d-Mesh"
    assert runtime_config.derive_prefix("mesh.v2.dat", "channel") == "mesh.v2"
    assert runtime_config.derive_prefix(None, "plate") == "plate"
    with pytest.raises(runtime_config.ConfigError):
        runtime_config.derive_prefix("dir/.dat", "channel")


def test_channel_json_paths_are_relative_to_config():
    cfg = runtime_config.load_config(config_path(CASES[5]))
    expected = PROJECT / "channel_case_iiA" / "quick" / "OCFD3d-Mesh.dat"
    assert Path(cfg.values["mesh"]) == expected
    assert cfg.values["mesh_format"] == "ocfd"
    assert cfg.values["infile"].endswith("opencfd-scu.in")


COMPARE_ATTRS = (
    "case", "N", "ny", "nz", "scheme", "vis_order", "viscous", "Re", "Ama",
    "ref_t", "pr", "forcing_work", "p_grad", "dt", "cfl", "end_time",
    "end_step", "kstep_show", "kstep_save", "plate_x0", "plate_x1", "plate_h",
    "mesh", "field", "infile", "out_prefix",
)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_json_matches_legacy_cli_arguments(case):
    legacy = parse(case["argv"])
    as_json = parse(["--config", str(config_path(case)), *case["json_argv"]])
    for attr in COMPARE_ATTRS:
        assert getattr(as_json, attr) == getattr(legacy, attr), attr


def test_channel_without_infile_cli_fails():
    proc = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.run", "--case", "channel",
         "--end-step", "0"],
        cwd=str(PROJECT), capture_output=True, text=True)
    assert proc.returncode != 0
    assert ".in" in (proc.stderr + proc.stdout)


def test_missing_cli_input_file_fails():
    proc = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.run", "--case", "sod",
         "--mesh", "Z:/definitely-missing.dat", "--end-step", "0"],
        cwd=str(PROJECT), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "不存在" in (proc.stderr + proc.stdout)


def test_cli_and_json_bitwise(tmp_path):
    """同一条 Sod 用例：旧 CLI 与 input.json 的 npz 必须逐位一致。"""
    cli_out, json_out = tmp_path / "cli", tmp_path / "json"
    base = ["--case", "sod", "--N", "16", "--ny", "4", "--nz", "4",
            "--scheme", "UP7", "--dt", "5e-4", "--end-step", "3",
            "--end-time", "0.01", "--kstep-show", "1", "--kstep-save", "1"]
    subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.run", *base,
         "--out-dir", str(cli_out)],
        cwd=str(PROJECT), check=True, capture_output=True, text=True)
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "numerics": {"scheme": "UP7"},
        "time": {"dt": 5e-4, "end_step": 3, "end_time": 0.01},
        "output": {"dir": str(json_out), "kstep_show": 1, "kstep_save": 1},
    })
    subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.run", "--config", cfg,
         "--N", "16", "--ny", "4", "--nz", "4"],
        cwd=str(PROJECT), check=True, capture_output=True, text=True)

    cli_files = sorted(cli_out.glob("*.npz"))
    json_files = sorted(json_out.glob("*.npz"))
    assert [p.name for p in cli_files] == [p.name for p in json_files]
    assert cli_files
    for left, right in zip(cli_files, json_files):
        with np.load(left, allow_pickle=False) as A, \
                np.load(right, allow_pickle=False) as B:
            assert set(A.files) == set(B.files)
            for key in A.files:
                assert A[key].tobytes() == B[key].tobytes(), f"{left.name}:{key}"
