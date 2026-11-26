"""第八步 U2：输出间隔的校验。

锁两件事：
  1. ``--kstep-show``、``--kstep-save`` 不是正整数时，单块运行在启动时报一行“配置错误”，
     退出码 2，没有 traceback，没有输出文件；
  2. 直接调用 ``main.run`` 拿到的是 ``ConfigError``。

正常值照常运行、存盘的文件名不变，由 ``test_step49_frozen.py`` 等走正式入口的测试守着。
"""

import subprocess
import sys
from pathlib import Path

import pytest

from opencfd_jax.src import config

config.enable_x64()

from opencfd_jax.src import main, runtime_config  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
BAD = [("--kstep-save", "kstep_save", "0"), ("--kstep-show", "kstep_show", "0"),
       ("--kstep-save", "kstep_save", "-1"), ("--kstep-show", "kstep_show", "-1")]
IDS = ["kstep-save=0", "kstep-show=0", "kstep-save=-1", "kstep-show=-1"]


@pytest.mark.parametrize("option, name, value", BAD, ids=IDS)
def test_non_positive_output_interval_is_rejected_at_startup(tmp_path, option, name, value):
    """Covers AE5。"""
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--dt", "1e-4", "--end-step", "1",
         option, value, "--out-dir", str(out)],
        cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    reported = [line for line in done.stderr.splitlines() if "配置错误" in line]
    assert len(reported) == 1 and name in reported[0], done.stderr
    assert not out.exists() or not list(out.iterdir())


@pytest.mark.parametrize("option, name, value", BAD, ids=IDS)
def test_main_run_raises_config_error(option, name, value):
    args = main.build_parser().parse_args(
        ["--dt", "1e-4", "--end-step", "1", option, value])
    with pytest.raises(runtime_config.ConfigError, match=name):
        main.run(args)
