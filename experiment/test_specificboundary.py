"""specificboundary 配置入口的行为测试（Step 4.n ⑥，2026-11-23）。

覆盖：合法脚本挂载并走 custom 执行链、非法文件名、缺脚本、缺入口、
与 plate 互斥、load_config 顶层键解析。脚本目录用 monkeypatch 指向
临时目录，不污染 src/special_boundaries/。
这些配置错一律报 SetupError，消息带出处（补尾巴 A 组，jisen 2026-11-27 定的）。
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import boundary, init, runtime_config  # noqa: E402
from opencfd_jax.src.cases import build_case  # noqa: E402
from opencfd_jax.src.errors import SetupError  # noqa: E402
from opencfd_jax.src.mesh_data import from_case  # noqa: E402

VALID_SCRIPT = '''
from types import SimpleNamespace


def build_special_boundary(case):
    """最小合法脚本：faces 缺省继承，回调在内区一角注入固定密度。"""
    def bc_apply(q):
        return q.at[0, 4, 4, 4].set(2.0)
    return SimpleNamespace(name="demo", bc_apply=bc_apply)
'''


def _base_args(**kw):
    args = SimpleNamespace(case="sod", ni=16, nj=4, nk=4, Re=1000.0, Ama=1.0,
                           p_grad=0.1, scheme="UP7", viscous=True,
                           plate_dif_bound=None, specificboundary=None)
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _make_script(tmp_path, text=VALID_SCRIPT, name="demo.py"):
    d = tmp_path / "special"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")
    return d


def test_valid_script_mounts_custom_execution(tmp_path, monkeypatch):
    d = _make_script(tmp_path)
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", d)
    args = _base_args(specificboundary="demo.py")
    case = build_case(args)
    init.attach_specificboundary(args, case)
    assert case["special_boundary"].name == "demo"
    execution = boundary.build_execution(case, from_case(case))
    assert execution.mode == "custom"
    q = case["cons0"]
    out = execution.prepare(q)
    assert float(out[0, 4, 4, 4]) == pytest.approx(2.0)


def test_rejects_pathlike_name(tmp_path, monkeypatch):
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    args = _base_args(specificboundary="../escape.py")
    with pytest.raises(SetupError, match="纯文件名"):
        init.attach_specificboundary(args, build_case(args))


def test_rejects_missing_script(tmp_path, monkeypatch):
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    args = _base_args(specificboundary="absent.py")
    with pytest.raises(SetupError, match="不存在"):
        init.attach_specificboundary(args, build_case(args))


def test_rejects_missing_entry(tmp_path, monkeypatch):
    d = _make_script(tmp_path, text="X = 1\n", name="noentry.py")
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", d)
    args = _base_args(specificboundary="noentry.py")
    with pytest.raises(SetupError, match="build_special_boundary"):
        init.attach_specificboundary(args, build_case(args))


def test_rejects_conflict_with_plate(tmp_path, monkeypatch):
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    args = _base_args(case="plate", specificboundary="demo.py")
    with pytest.raises(SetupError, match="互斥"):
        init.attach_specificboundary(args, build_case(args))


def test_load_config_parses_top_level_string(tmp_path):
    cfg_file = tmp_path / "run.json"
    cfg_file.write_text(
        '{"case": {"name": "sod"}, "specificboundary": "demo.py",'
        ' "time": {"dt": 1e-4, "end_step": 1}}', encoding="utf-8")
    cfg = runtime_config.load_config(str(cfg_file))
    assert cfg.values["specificboundary"] == "demo.py"
    bad = tmp_path / "bad.json"
    bad.write_text('{"specificboundary": 42}', encoding="utf-8")
    with pytest.raises(Exception, match="脚本名字符串"):
        runtime_config.load_config(str(bad))


def test_prepare_end_to_end_with_specificboundary(tmp_path, monkeypatch):
    """init.prepare 全链：JIT 前挂载、精度先于数组、回调可达。"""
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    args = _base_args(specificboundary="demo.py")
    state = init.prepare(args)
    assert state.execution.mode == "custom"
    out = state.execution.prepare(state.cons0)
    assert float(out[0, 4, 4, 4]) == pytest.approx(2.0)
    assert jnp.asarray(state.cons0).dtype == jnp.float64


def test_valid_poisuille_j_wall_mounts(tmp_path, monkeypatch):
    """内置 Poiseuille（j 向 WALL）+ 只写 bc_apply 的脚本必须能装上
    （通用 WALL 无回调面是合法组合——曾 validate 被误拒，Astra 复审）。"""
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    args = _base_args(case="poiseuille", specificboundary="demo.py")
    case = build_case(args)
    init.attach_specificboundary(args, case)
    execution = boundary.build_execution(case, from_case(case))
    assert execution.mode == "custom"


def test_rejects_non_py_and_directory_escape(tmp_path, monkeypatch):
    """README.md 之类的非 .py 纯文件名必须拒绝。"""
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    for bad in ("README.md", "demo.PY ", "sub/dir.py"):
        args = _base_args(specificboundary=bad.strip() if bad != "sub/dir.py" else bad)
        with pytest.raises((SetupError,)):
            init.attach_specificboundary(args, build_case(args))


def test_rejects_cgns_plate_by_case_name(tmp_path, monkeypatch):
    """CGNS plate 无 plate marker——互斥须按 case 名兜住。"""
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", _make_script(tmp_path))
    args = _base_args(case="plate", specificboundary="demo.py",
                      mesh_format="cgns", mesh="x.cgns")
    with pytest.raises(SetupError, match="互斥"):
        init.attach_specificboundary(args, {"plate": None})  # CGNS case 无 marker


def test_cgns_rejects_hand_written_dif_bound():
    """CGNS 边条驱动（4.9.13 口径）：--plate-dif-bound 手填六元组必须显式拒绝。"""
    import pytest as _pytest
    from opencfd_jax.src.cases import build_case
    from pathlib import Path as _P
    FIXTURE = _P(__file__).resolve().parents[1] / "Grid" / "fixtures" / "poiseuille_box.cgns"
    for case in ("poiseuille", "plate"):
        kw = dict(case=case, ni=16, nj=16, nk=8, Re=1000.0, Ama=0.1,
                  p_grad=0.0, scheme="UP7", viscous=True, vis_order="CD8",
                  ref_t=288.15, pr=0.7, plate_dif_bound="0 0 0 0 1 0",
                  mesh_format="cgns", mesh=str(FIXTURE),
                  boundary_inflow=None, boundary_farfield=None,
                  boundary_wall_temperature=None)
        with _pytest.raises(ValueError, match="不支持手填"):
            build_case(_base_args(**kw))


def test_cgns_mixed_sides_inherited_and_executable(tmp_path, monkeypatch):
    """终审 P2：i- 出流 / i+ 远场的异类面挂只写回调的脚本可装配可执行。"""
    from opencfd_jax.src.fields import OUTFLOW, FARFIELD
    from opencfd_jax.src.mesh_data import FaceBoundary, from_case
    d = _make_script(tmp_path)
    monkeypatch.setattr(init, "SPECIAL_BOUNDARY_DIR", d)
    args = _base_args(specificboundary="demo.py", case="poiseuille")
    faces = (
        FaceBoundary("i-", 0, "outflow", "halo", False),
        FaceBoundary("i+", 0, "farfield", "halo", False),
        FaceBoundary("j-", 1, "outflow", "halo", False),
        FaceBoundary("j+", 1, "outflow", "halo", False),
        FaceBoundary("k-", 2, "outflow", "halo", False),
        FaceBoundary("k+", 2, "outflow", "halo", False),
    )
    case = build_case(args)
    case["faces"] = faces
    case["farfield"] = [1.0, 1.0, 0.0, 0.0, 2.5]
    init.attach_specificboundary(args, case)
    execution = boundary.build_execution(case, from_case(case))
    assert execution.mode == "custom"
    q = case["cons0"]
    out = execution.prepare(q)
    assert float(out[0, 4, 4, 4]) == pytest.approx(2.0)  # 回调生效
