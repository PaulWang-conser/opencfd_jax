"""Step 3 统一运行配置（`input.json`）的验收测试。

覆盖：默认值/JSON/显式 CLI 的合并优先级、字段与算例适用范围、CGNS
网格路径、输出前缀、负向开关，以及 Step 0 用例的 JSON 与 CLI 参数等价。
槽道 `.in`、CFL 自适应和旧 OCFD 网格入口已在第 4 步删除。
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

from opencfd_jax.src import runtime_config, main     # noqa: E402
from step0_cases import CASES, config_path        # noqa: E402


def parse(argv):
    """按 main.main 的顺序解析：默认值、显式 CLI、可选 JSON。"""
    defaults = main.build_parser().parse_args([])
    parser = main.build_parser(explicit_only=True)
    config_path, explicit = main.resolve_cli(
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
    assert args.mesh_format == "builtin"
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
        "mesh": {"file": str(PROJECT / "opencfd_jax/Grid/fixtures/poiseuille_box.cgns")},
    })
    args = parse(["--config", cfg, "--case", "poiseuille"])
    assert args.case == "poiseuille"
    assert args.mesh_format == "cgns"
    assert args.mesh.endswith("poiseuille_box.cgns")


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
        parse(["--config", cfg, "--case", "sod"])


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


def test_cgns_mesh_requires_file(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "poiseuille"},
                                "mesh": {"format": "cgns"}})
    with pytest.raises(runtime_config.ConfigError, match="mesh.file"):
        runtime_config.load_config(cfg)


def test_cgns_mesh_rejects_non_poiseuille(tmp_path):
    (tmp_path / "mesh.cgns").write_bytes(b"")
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "mesh": {"format": "cgns", "file": "mesh.cgns"},
    })
    with pytest.raises(runtime_config.ConfigError, match="CGNS"):
        runtime_config.load_config(cfg)


def test_negative_switch_overrides_json(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "numerics": {"viscous": True, "weno_tv": True},
    })
    args = parse(["--config", cfg, "--no-viscous", "--no-weno-tv"])
    assert args.viscous is False
    assert args.weno_tv is False


def test_positive_and_negative_switch_conflict(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "sod"}})
    with pytest.raises(SystemExit):
        parse(["--config", cfg, "--viscous", "--no-viscous"])


def test_cfl_config_field_rejected(tmp_path):
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "time": {"dt": 1e-3, "cfl": 0.4},
    })
    with pytest.raises(runtime_config.ConfigError, match="time.cfl"):
        runtime_config.load_config(cfg)


def test_adaptive_dt_flag_is_gone():
    with pytest.raises(SystemExit):
        parse(["--case", "sod", "--adaptive-dt"])


def test_precision_merge(tmp_path):
    cfg = write_json(tmp_path, {"case": {"name": "sod"}, "precision": "fp32"})
    args = parse(["--config", cfg])
    assert args.fp32 is True and args.fp64 is False
    args = parse(["--config", cfg, "--fp64"])
    assert args.fp32 is False and args.fp64 is True


def test_output_prefix_from_mesh_and_case():
    assert runtime_config.derive_prefix(r"a\b\OCFD3d-Mesh.dat", "poiseuille") \
        == "OCFD3d-Mesh"
    assert runtime_config.derive_prefix("mesh.v2.dat", "poiseuille") == "mesh.v2"
    assert runtime_config.derive_prefix(None, "plate") == "plate"
    with pytest.raises(runtime_config.ConfigError):
        runtime_config.derive_prefix("dir/.dat", "poiseuille")


def test_output_filename_uses_eight_digit_istep():
    from opencfd_jax.src import io
    assert io.checkpoint_name("sod", 1) == "sod-istep-00000001.npz"
    assert io.checkpoint_name("OCFD3d-Mesh", 1000) == "OCFD3d-Mesh-istep-00001000.npz"
    # 超过八位自然扩展，不截断
    assert io.checkpoint_name("sod", 123456789) == "sod-istep-123456789.npz"
    with pytest.raises(ValueError, match="负数"):
        io.checkpoint_name("sod", -1)


def test_cgns_json_paths_are_relative_to_config(tmp_path):
    (tmp_path / "mesh.cgns").write_bytes(b"")
    cfg = write_json(tmp_path, {
        "case": {"name": "poiseuille"},
        "mesh": {"file": "mesh.cgns"},
    })
    parsed = runtime_config.load_config(cfg)
    assert Path(parsed.values["mesh"]) == tmp_path / "mesh.cgns"
    assert parsed.values["mesh_format"] == "cgns"


COMPARE_ATTRS = (
    "case", "ni", "nj", "nk", "scheme", "vis_order", "viscous", "Re", "Ama",
    "ref_t", "pr", "forcing_work", "p_grad", "dt", "end_time",
    "end_step", "kstep_show", "kstep_save", "plate_x0", "plate_x1", "plate_h",
    "mesh", "mesh_format", "cgns_zone", "out_prefix",
)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_json_matches_cli_arguments(case):
    cli = parse(case["argv"])
    as_json = parse(["--config", str(config_path(case)), *case["json_argv"]])
    for attr in COMPARE_ATTRS:
        assert getattr(as_json, attr) == getattr(cli, attr), attr


def test_channel_case_is_gone():
    proc = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--case", "channel",
         "--end-step", "0", "--dt", "1e-3"],
        cwd=str(PROJECT), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "invalid choice" in (proc.stderr + proc.stdout)


def test_missing_cgns_file_fails():
    proc = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--case", "poiseuille",
         "--mesh", "Z:/definitely-missing.cgns", "--dt", "1e-3",
         "--end-step", "0"],
        cwd=str(PROJECT), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "不存在" in (proc.stderr + proc.stdout)


def test_cli_and_json_bitwise(tmp_path):
    """同一条 Sod 用例：新 CLI 与 input.json 的 npz 必须逐位一致。"""
    cli_out, json_out = tmp_path / "cli", tmp_path / "json"
    base = ["--case", "sod", "--ni", "16", "--nj", "4", "--nk", "4",
            "--scheme", "UP7", "--dt", "5e-4", "--end-step", "3",
            "--end-time", "0.01", "--kstep-show", "1", "--kstep-save", "1"]
    subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", *base,
         "--out-dir", str(cli_out)],
        cwd=str(PROJECT), check=True, capture_output=True, text=True)
    cfg = write_json(tmp_path, {
        "case": {"name": "sod"},
        "numerics": {"scheme": "UP7"},
        "time": {"dt": 5e-4, "end_step": 3, "end_time": 0.01},
        "output": {"dir": str(json_out), "kstep_show": 1, "kstep_save": 1},
    })
    subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--config", cfg,
         "--ni", "16", "--nj", "4", "--nk", "4"],
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
