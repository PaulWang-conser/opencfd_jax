"""Step 4 运行入口的 CGNS / 内置网格结果对齐测试。

两边用同一尺寸、同一物性、同一显式 dt 各跑一步，比较 NPZ 的五个守恒分量
和 `_meta`。只验证 `poiseuille` 的最小可执行路径。
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CGNS = PROJECT / "opencfd_jax" / "Grid" / "fixtures" / "poiseuille_box.cgns"


def _run(args, out_dir):
    cmd = [sys.executable, "-m", "opencfd_jax.src.main", *args,
           "--out-dir", str(out_dir), "--kstep-show", "1", "--kstep-save", "1"]
    proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc


def _only_npz(directory):
    files = sorted(Path(directory).glob("*-istep-*.npz"))
    assert len(files) == 1, files
    return files[0]


def test_cgns_and_builtin_poiseuille_one_step_match(tmp_path):
    common = [
        "--case", "poiseuille", "--ni", "16", "--nj", "16", "--nk", "8",
        "--dt", "1e-3", "--end-step", "1", "--end-time", "0.5",
        "--viscous", "--Re", "100", "--Ama", "0.05", "--p-grad", "0.1",
    ]
    _run([*common, "--mesh", str(CGNS)], tmp_path / "cgns")
    _run(common, tmp_path / "builtin")

    cgns = _only_npz(tmp_path / "cgns")
    builtin = _only_npz(tmp_path / "builtin")
    with np.load(cgns, allow_pickle=False) as A, \
            np.load(builtin, allow_pickle=False) as B:
        assert set(A.files) == set(B.files)
        for key in A.files:
            if key == "_meta":
                assert json.loads(A[key].tobytes())["N"] == 16
                assert json.loads(B[key].tobytes())["N"] == 16
                continue
            assert A[key].shape == B[key].shape
            # 两条路径是同一离散的两种写法：builtin 用均匀步长 dxs，CGNS 用
            # 坐标派生的度量（网格是均匀盒子，理论上等价）。能量方程是大量项
            # 相减，抵消放大到 ~4e-9 相对量级（实测 cons_4 max 3.0e-6，场值 7e2；
            # 把 CGNS 的 metrics 置 None 走步长路径后两者逐位相同），故容差取 1e-8。
            assert np.allclose(A[key], B[key], rtol=1e-8, atol=1e-6), key
