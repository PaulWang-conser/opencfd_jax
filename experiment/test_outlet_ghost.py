"""出口状态鬼层开关（boundary.outlet_ghost: keep | copy）的行为测试。

背景：plate-CGNS 的出口状态鬼层缺省"保持原值"（原码口径，初始为零），
超声速层流对照中出口附近偏差很大。为判别它是不是出口零鬼层造成的，
加了可切换的零梯度延拓（copy）。开关只允许改出口的**状态**鬼层：
黏性通量 Ev 的出口鬼层、出口边界带、出口点的演化方式都必须不变；
缺省行为必须逐位不变（它是 Step 4.9 与原码逐位对齐的基础）。

锁四件事：
  1. 缺省 keep：出口鬼层原值保留；
  2. copy：每次 prepare 之后，出口四层鬼层等于边界点那一层，且幂等；
  3. copy 下 Ev 的出口鬼层仍清零；除出口鬼层外，状态与缺省逐位相同；
  4. 非法取值、非 plate 算例给出该字段时在运行前报错；CLI 可覆盖 JSON。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import boundary, init, main, runtime_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "experiment" / "cases" / "plate_cgns_step49.json"
MESH = REPO / "Grid" / "testcgnstest.cgns"
pytestmark = pytest.mark.skipif(not MESH.exists(), reason="缺少 testcgnstest.cgns")


def _resolve(cfg, extra_cli=()):
    return runtime_config.resolve_args(
        dict(extra_cli), cfg, main.build_parser().parse_args([]))


def _load_text(text):
    """把配置文本放回原配置目录解析，保证网格的相对路径不变。"""
    tmp = CFG.with_name("_tmp_outlet_ghost_test.json")
    try:
        tmp.write_text(text, encoding="utf-8")
        return runtime_config.load_config(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _state(outlet_ghost=None, extra_cli=()):
    cfg = runtime_config.load_config(str(CFG))
    if outlet_ghost is not None:
        cfg.values["boundary_outlet_ghost"] = outlet_ghost
        cfg.sources["boundary_outlet_ghost"] = "test"
    return init.prepare(_resolve(cfg, extra_cli))


def _col(q):
    """取出口附近一条流向列（k、j 都在内区中部），便于逐层比较。"""
    return q[:, 10, 10, :]


def _ghost(q, grid, layer, value):
    """写出口侧第 layer 层鬼层（1..4）；最后一个内点的下标是 4+ni-1。"""
    return q.at[:, 10, 10, 4 + grid.ni - 1 + layer].set(value)


def _layer(col, grid, layer):
    """读出口侧第 layer 层（0 为最后一个内点，1..4 为鬼层）。"""
    return col[:, 4 + grid.ni - 1 + layer]


def test_default_keeps_outlet_ghosts():
    state = _state()
    grid = state.case["grid"]
    outlet = state.execution.faces[1]
    assert outlet.kind == "outflow" and "halo" not in outlet.params
    q = jnp.asarray(state.case["cons0"])
    for layer in range(1, 5):
        q = _ghost(q, grid, layer, 7.0 + layer)
    out = _col(np.asarray(state.execution.prepare(q)))
    for layer in range(1, 5):
        np.testing.assert_array_equal(
            _layer(out, grid, layer), np.full(5, 7.0 + layer),
            err_msg="缺省应保留出口鬼层原值")


def test_copy_mirrors_boundary_point_each_prepare():
    state = _state("copy")
    grid = state.case["grid"]
    assert state.execution.faces[1].params["halo"] == "copy"
    q = jnp.asarray(state.case["cons0"])
    for layer in range(1, 5):
        q = _ghost(q, grid, layer, 7.0 + layer)       # 故意放错值
    out = _col(np.asarray(state.execution.prepare(q)))
    last = _layer(out, grid, 0)
    assert not np.any(last == 7.0 + 1), "边界点本身不应被鬼层的错值污染"
    for layer in range(1, 5):
        np.testing.assert_array_equal(_layer(out, grid, layer), last)
    full = np.asarray(state.execution.prepare(jnp.asarray(state.case["cons0"])))
    again = np.asarray(state.execution.prepare(jnp.asarray(full)))
    np.testing.assert_array_equal(again, full)       # 连续两次 prepare 幂等


def test_copy_changes_only_outlet_state_ghosts():
    keep, copy = _state(), _state("copy")
    grid = keep.case["grid"]
    q = jnp.asarray(keep.case["cons0"]).at[:, 10, 10, 4 + grid.ni + 1].set(7.0)
    a = np.asarray(keep.execution.prepare(q))
    b = np.asarray(copy.execution.prepare(q))
    changed = (a != b).any(axis=0)              # (nk, nj, ni) 含鬼胞
    assert changed[:, :, 4 + grid.ni:].any(), "copy 应改变出口鬼层"
    assert not changed[:, :, :4 + grid.ni].any(), "内区与其它面的鬼层不得改变"
    # Ev 鬼层：两种设置下出口都清零（与原版参考口径一致）
    for st in (keep, copy):
        ev = np.asarray(boundary.fill_ev_faces(
            jnp.full(grid.shape, 7.0), st.execution.faces, grid))
        assert np.all(ev[..., 4 + grid.ni:] == 0.0)
        assert np.all(ev[:, 10, 10, 4 + grid.ni - 1] == 7.0)   # 末个内点不被触碰
    assert keep.execution.band == copy.execution.band   # 出口边界带不变


def test_invalid_value_and_scope_are_rejected():
    original = CFG.read_text(encoding="utf-8")
    marker = '"wall_temperature": 1.0'
    assert original.count(marker) == 1
    bad = original.replace(marker, marker + ', "outlet_ghost": "zero"')
    with pytest.raises(runtime_config.ConfigError, match="boundary.outlet_ghost"):
        _load_text(bad)
    other = json.dumps({"case": {"name": "tg"},
                        "boundary": {"outlet_ghost": "copy"}})
    with pytest.raises(runtime_config.ConfigError, match="boundary.outlet_ghost"):
        _resolve(_load_text(other))


def test_cli_overrides_json():
    state = _state("keep", extra_cli={"boundary_outlet_ghost": "copy"})
    assert state.execution.faces[1].params["halo"] == "copy"
    state = _state("copy", extra_cli={"boundary_outlet_ghost": "keep"})
    assert "halo" not in state.execution.faces[1].params


def test_pure_cli_scope_is_rejected():
    """纯 CLI 给 sod 算例 --outlet-ghost 必须报错（此前只校验 JSON 路径）。"""
    defaults = main.build_parser().parse_args([])
    explicit_parser = main.build_parser(explicit_only=True)
    _, explicit = main.resolve_cli(
        vars(explicit_parser.parse_args(["--case", "sod", "--outlet-ghost", "copy"])))
    with pytest.raises(runtime_config.ConfigError, match="outlet_ghost"):
        runtime_config.resolve_args(explicit, None, defaults)


def test_builtin_plate_with_outlet_ghost_is_rejected():
    """该开关只有 CGNS 装配分支消费：内置网格的 plate 显式给出必须报错。

    纯 CLI、JSON、JSON+CLI 合并三个入口都要拦住（Astra 第十一轮：此前
    `--case plate --outlet-ghost copy` 解析成 plate/builtin/copy 并跑完一步）。
    """
    defaults = main.build_parser().parse_args([])
    explicit_parser = main.build_parser(explicit_only=True)
    with pytest.raises(runtime_config.ConfigError, match="outlet_ghost"):
        _, explicit = main.resolve_cli(
            vars(explicit_parser.parse_args(["--case", "plate", "--outlet-ghost", "copy"])))
        runtime_config.resolve_args(explicit, None, defaults)
    builtin = {"case": {"name": "plate"}}
    for cfg_json in (dict(builtin, boundary={"outlet_ghost": "copy"}),
                     dict(builtin, mesh={"format": "builtin"},
                          boundary={"outlet_ghost": "keep"})):
        with pytest.raises(runtime_config.ConfigError, match="outlet_ghost"):
            _resolve(_load_text(json.dumps(cfg_json)))
    with pytest.raises(runtime_config.ConfigError, match="outlet_ghost"):
        _resolve(_load_text(json.dumps(builtin)), {"boundary_outlet_ghost": "copy"})
    # 对照：不显式给出时内置 plate 照常解析
    args = _resolve(_load_text(json.dumps(builtin)))
    assert args.mesh_format == "builtin" and args.case == "plate"
