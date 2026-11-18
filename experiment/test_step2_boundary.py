"""第 2 步「现有边界迁移与扩展入口」的验收测试。

U1 部分在迁移前先冻结旧路径：周期/出流四层鬼胞与角点、槽道不同壁温及
单侧绝热、平板非对称边界、Ev 镜像、滤波不支持项和 RK 回调顺序。
U2/U3/U4 部分验证描述与执行配置一致、通用实现只有一份、运行链使用统一
配置；这些会在迁移前先红，再按计划实现。

测试放在 ``experiment/`` 下，理由和运行方式见 ``experiment/README.md``。
"""

import argparse

import jax
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

# experiment/test_step2_boundary.py -> parents[2] = D:\Desktop\ocfd2jax
_REPO_TAX = Path(__file__).resolve().parents[2]
if str(_REPO_TAX) not in sys.path:
    sys.path.insert(0, str(_REPO_TAX))

from opencfd_jax.src import boundary, cases, channel, fields, filtering
from opencfd_jax.src import mesh_data, plate, main, solver, viscous
from opencfd_jax.src.fields import (INFLOW, LAP, OUTFLOW, PERIODIC,
                                    PLATE, WALL, Grid)


def _ns(**kw):
    """构造 build_case 所需的最小 CLI 命名空间。"""
    defaults = dict(case="sod", ni=16, nj=8, nk=4, Re=10.0, Ama=0.1,
                    ref_t=None, pr=0.7, forcing_work="off")
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _plate_case():
    """较小且固定的平板算例，供边界/回调测试复用。"""
    return cases.build_case(_ns(case="plate", ni=16, nj=8, nk=4,
                                plate_x0=0.1, plate_x1=1.1, plate_h=0.12,
                                plate_dif_bound="0 0 0 0 1 0"))


def _channel_case():
    """小网格槽道算例，避免依赖大网格文件。"""
    return cases.build_case(_ns(
        case="channel",
        infile=str(_REPO_TAX / "opencfd_jax/experiment/cases/channel_small.in"),
        field=None, mesh=None))


def _field(grid):
    """非均匀、全正、无对称性的测试场；角点也能暴露索引错误。"""
    shape = grid.shape
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return jnp.asarray(values / float(values.size) + 0.25)


def _periodic_ref(f, axis, n):
    """按现有索引口径独立写的周期填充参考。"""
    out = np.array(f, copy=True)
    if axis == 0:
        out[..., :LAP] = f[..., n:n + LAP]
        out[..., LAP + n:] = f[..., LAP:LAP + LAP]
    elif axis == 1:
        out[..., :LAP, :] = f[..., n:n + LAP, :]
        out[..., LAP + n:, :] = f[..., LAP:LAP + LAP, :]
    else:
        out[:, :LAP] = f[:, n:n + LAP]
        out[:, LAP + n:] = f[:, LAP:LAP + LAP]
    return out


def _outflow_ref(f, axis, n):
    """按现有索引口径独立写的零梯度出流参考。"""
    out = np.array(f, copy=True)
    if axis == 0:
        out[..., :LAP] = f[..., LAP:LAP + 1]
        out[..., LAP + n:] = f[..., LAP + n - 1:LAP + n]
    elif axis == 1:
        out[..., :LAP, :] = f[..., LAP:LAP + 1, :]
        out[..., LAP + n:, :] = f[..., LAP + n - 1:LAP + n, :]
    else:
        out[:, :LAP] = f[:, LAP:LAP + 1]
        out[:, LAP + n:] = f[:, LAP + n - 1:LAP + n]
    return out


def _fill_axis(f, axis, bc, n):
    """调用现有单轴填充入口。"""
    if axis == 0:
        return fields.fill_halos_i(f, bc, n)
    if axis == 1:
        return fields.fill_halos_j(f, bc, n)
    return fields.fill_halos_k(f, bc, n)


def _cons_nonuniform(grid, phys):
    """构造适合壁面测试的非均匀守恒量场。"""
    nk, nj, ni = grid.nk, grid.nj, grid.ni
    z, y, x = np.indices((nk, nj, ni))
    rho = 1.0 + 0.03 * np.sin(0.3 * x + 0.2 * y + 0.1 * z)
    u = 0.02 * (1.0 + 0.15 * y)
    v = -0.01 * (1.0 + 0.1 * x)
    w = 0.005 * (1.0 + 0.1 * z)
    T = 1.0 + 0.04 * np.cos(0.2 * y) + 0.01 * x / max(ni, 1)
    return channel.cons_from_primitive(
        (rho, u, v, w, T), grid, phys)


# ---------------------------------------------------------------- U1 鬼胞/角点

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_periodic_all_axes_four_layers_and_corners(axis):
    """U1：i/j/k 周期各四层鬼胞，含跨轴角点，逐位等于独立参考。"""
    grid = Grid(9, 8, 7)
    n = (grid.ni, grid.nj, grid.nk)[axis]
    f = _field(grid)
    got = np.asarray(_fill_axis(f, axis, PERIODIC, n))
    np.testing.assert_array_equal(got, _periodic_ref(np.asarray(f), axis, n))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_outflow_all_axes_four_layers_and_corners(axis):
    """U1：i/j/k 出流四层鬼胞复制最近内点，含角点。"""
    grid = Grid(9, 8, 7)
    n = (grid.ni, grid.nj, grid.nk)[axis]
    f = _field(grid)
    got = np.asarray(_fill_axis(f, axis, OUTFLOW, n))
    np.testing.assert_array_equal(got, _outflow_ref(np.asarray(f), axis, n))


# ---------------------------------------------------------------- U1 槽道壁面

def test_channel_two_wall_temperatures_and_one_adiabatic_side():
    """U1：下壁等温 0.9、上壁绝热；壁点/鬼胞与独立三点外推一致。"""
    case = _channel_case()
    grid, phys = case["grid"], case["params"]
    nj = grid.nj
    q = _cons_nonuniform(grid, phys)
    q = fields.fill_halos_i(q, PERIODIC, grid.ni)
    q = fields.fill_halos_k(q, PERIODIC, grid.nk)
    out = channel.apply_wall_bc(q, grid, phys, Tw_bottom=0.9, Tw_top=0.0)
    out_np = np.asarray(out)

    rho, _, T = channel.primitive(q, phys)
    rho_np, T_np = np.asarray(rho), np.asarray(T)
    zi, xi = slice(LAP, LAP + grid.nk), slice(LAP, LAP + grid.ni)

    # 下壁：Tw=0.9 等温，压力三点外推，rho_w=pw/Tw
    jw = LAP
    pw_lo = (18.0 * rho_np[zi, LAP + 1, xi] * T_np[zi, LAP + 1, xi]
             - 9.0 * rho_np[zi, LAP + 2, xi] * T_np[zi, LAP + 2, xi]
             + 2.0 * rho_np[zi, LAP + 3, xi] * T_np[zi, LAP + 3, xi]) / 11.0
    np.testing.assert_allclose(out_np[0, zi, jw, xi], pw_lo / 0.9, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(out_np[1:4, zi, jw, xi], 0.0)
    np.testing.assert_allclose(out_np[4, zi, jw, xi], (pw_lo / 0.9) * phys.Cv * 0.9,
                               rtol=1e-14, atol=1e-13)

    # 上壁：Tw<=0 绝热；T 三点外推，rho_w=pw/Tw
    jw2 = LAP + nj - 1
    pw_hi = (18.0 * rho_np[zi, LAP + nj - 2, xi] * T_np[zi, LAP + nj - 2, xi]
             - 9.0 * rho_np[zi, LAP + nj - 3, xi] * T_np[zi, LAP + nj - 3, xi]
             + 2.0 * rho_np[zi, LAP + nj - 4, xi] * T_np[zi, LAP + nj - 4, xi]) / 11.0
    T_hi = (18.0 * T_np[zi, LAP + nj - 2, xi] - 9.0 * T_np[zi, LAP + nj - 3, xi]
            + 2.0 * T_np[zi, LAP + nj - 4, xi]) / 11.0
    np.testing.assert_allclose(out_np[0, zi, jw2, xi], pw_hi / T_hi, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(out_np[1:4, zi, jw2, xi], 0.0)
    np.testing.assert_allclose(out_np[4, zi, jw2, xi],
                               (pw_hi / T_hi) * phys.Cv * T_hi,
                               rtol=1e-14, atol=1e-13)

    # 镜像：rho/rhoE 偶、三个动量奇；上下壁索引与原码循环一致。
    for i in range(LAP):
        src = 2 * LAP - i
        np.testing.assert_array_equal(out_np[0, zi, i, xi], out_np[0, zi, src, xi])
        np.testing.assert_array_equal(out_np[4, zi, i, xi], out_np[4, zi, src, xi])
        np.testing.assert_array_equal(out_np[1:4, zi, i, xi],
                                      -out_np[1:4, zi, src, xi])
    for i in range(1, LAP + 1):
        src = jw2 - i
        np.testing.assert_array_equal(out_np[0, zi, jw2 + i, xi],
                                      out_np[0, zi, src, xi])
        np.testing.assert_array_equal(out_np[4, zi, jw2 + i, xi],
                                      out_np[4, zi, src, xi])
        np.testing.assert_array_equal(out_np[1:4, zi, jw2 + i, xi],
                                      -out_np[1:4, zi, src, xi])


# ---------------------------------------------------------------- U1 平板边界

def test_plate_inlet_outlet_wall_far_and_corners():
    """U1：入口写剖面+鬼层、出口鬼层清零、下壁两点外推、顶部远场单点。"""
    case = _plate_case()
    grid, phys = case["grid"], case["params"]
    ni, nj = grid.ni, grid.nj
    execution = boundary.build_execution(case)
    q = _cons_nonuniform(grid, phys)
    out = np.asarray(execution.prepare(q))
    out_q = np.asarray(q)
    nj_in = LAP + nj
    kz, ix_in = slice(LAP, LAP + grid.nk), slice(LAP, LAP + ni)

    # 入口：x- 四层鬼层 + 第一个内区物理点 = 入口剖面；y 两端随后分别被
    # 下壁无滑移/顶部远场覆盖，故比较内部物理行。
    np.testing.assert_array_equal(
        out[:, LAP:LAP + grid.nk, LAP + 1:nj_in - 1, :LAP + 1],
        np.broadcast_to(np.asarray(case["plate"]["inflow"])[:, :, 1:-1, None],
                        (5, grid.nk, nj - 2, LAP + 1)))
    # 出口：方式 A，鬼层 = 出口边界点那一列（零梯度复制），不是清零。
    ghost = np.asarray(out[:, :, :, LAP + ni:])
    np.testing.assert_array_equal(
        ghost,
        np.broadcast_to(np.asarray(out[:, :, :, LAP + ni - 1:LAP + ni]),
                        ghost.shape))

    # 下壁：两点压力外推 + 无滑移 + rhoE=rho Cv Tw（只在内区窗口内比较）
    # 壁面回调在入口写完之后执行，故外推基线要用“已写入口”的状态。
    q_in = boundary.write_faces(boundary.fill_faces(q, execution.faces, grid),
                                execution.faces, grid,
                                inflow=execution.inflow, farfield=execution.farfield)
    rho, _, T = channel.primitive(q_in, phys)
    rho_np, T_np = np.asarray(rho), np.asarray(T)
    rho_w = (4.0 * rho_np[:, LAP + 1, :] * T_np[:, LAP + 1, :]
             - rho_np[:, LAP + 2, :] * T_np[:, LAP + 2, :]) / 3.0
    np.testing.assert_allclose(out[0, kz, LAP, ix_in], rho_w[kz, ix_in],
                               rtol=0, atol=1e-13)
    np.testing.assert_array_equal(out[1:4, kz, LAP, ix_in], 0.0)
    np.testing.assert_allclose(out[4, kz, LAP, ix_in],
                               rho_w[kz, ix_in] * phys.Cv, rtol=0, atol=1e-13)

    # 顶部：远场常数写内区最后一层内点，鬼层取该边界点延拓（不是清零）。
    np.testing.assert_array_equal(
        out[:, kz, nj_in - 1, ix_in],
        np.broadcast_to(np.asarray(execution.farfield).reshape(5, 1, 1),
                        (5, grid.nk, ni)))
    np.testing.assert_array_equal(
        out[:, kz, nj_in:, ix_in],
        np.broadcast_to(np.asarray(out[:, kz, nj_in - 1:nj_in, ix_in]),
                        (5, grid.nk, LAP, ni)))

    # 壁面鬼层不再镜像，整体为 0。
    np.testing.assert_array_equal(out[:, :, :LAP, :], 0.0)


# ---------------------------------------------------------------- U1 Ev / 滤波

def test_ev_j_mirror_and_plate_top_copy():
    """U1：Ev 的 j 壁镜像符号；平板顶部 Ev 是零梯度复制，不是状态镜像。"""
    grid = Grid(8, 8, 6)
    nj = grid.nj
    signs = viscous._EV_WALL_SIGNS[1]
    f = np.zeros((4, grid.shape[1], grid.shape[2], grid.shape[3]))
    for i, s in enumerate(signs):
        line = np.arange(grid.shape[2], dtype=float) * (i + 1)
        f[i] = s * line[None, :, None] + 0.1 * i
    stack = jnp.asarray(f)

    got = np.asarray(viscous.fill_ev_wall_j(stack, nj, signs))
    # 下壁 ghost[p] = f[2LAP-p] * sign；上壁 ghost[nj+LAP+q] = f[nj+LAP-2-q] * sign。
    for i in range(4):
        for p in range(LAP):
            np.testing.assert_array_equal(got[i, :, p, :],
                                          f[i, :, 2 * LAP - p, :] * signs[i])
        for q in range(LAP):
            np.testing.assert_array_equal(
                got[i, :, nj + LAP + q, :],
                f[i, :, nj + LAP - 2 - q, :] * signs[i])

    got_plate = np.asarray(viscous.fill_ev_plate_j(stack, nj, signs))
    np.testing.assert_array_equal(got_plate[:, :, nj + LAP:, :],
                                  np.broadcast_to(
                                      got_plate[:, :, nj + LAP - 1:nj + LAP, :],
                                      got_plate[:, :, nj + LAP:, :].shape))


def test_ev_direction_tables_and_axis_dispatch():
    """U1/U3：x/y 的 Ev 符号不同；单轴派发按方向选择正确的规则。"""
    assert tuple(boundary.EV_WALL_SIGNS[0]) == (+1.0, -1.0, +1.0, -1.0)
    assert tuple(boundary.EV_WALL_SIGNS[1]) == (-1.0, +1.0, -1.0, -1.0)

    grid = Grid(8, 8, 6)
    f = jnp.zeros((4, grid.shape[1], grid.shape[2], grid.shape[3]))
    f = f.at[0].set(1.0)
    got_wall = boundary.fill_ev_axis(f, 1, WALL, grid.nj, is_ev=True)
    np.testing.assert_array_equal(np.asarray(got_wall),
                                  np.asarray(boundary.fill_ev_wall_j(
                                      f, grid.nj, boundary.EV_WALL_SIGNS[1])))
    got_plate = boundary.fill_ev_axis(f, 1, PLATE, grid.nj, is_ev=True)
    np.testing.assert_array_equal(np.asarray(got_plate),
                                  np.asarray(boundary.fill_ev_plate_j(
                                      f, grid.nj, boundary.EV_WALL_SIGNS[1])))


def test_viscous_channel_uniform_state_is_fixed_point():
    """U1：槽道均匀状态（壁温=1）过完整粘性 RHS 应为零。"""
    case = _channel_case()
    grid, phys = case["grid"], case["params"]
    q = grid.alloc().at[0].set(1.0).at[4].set(phys.Cv)
    du = np.asarray(viscous.viscous_rhs(q, grid, phys, case["bcs"],
                                        axes=(0, 1, 2), dx=case["dxs"],
                                        order=8))
    np.testing.assert_allclose(du[grid.inner()], 0.0, atol=1e-13)


def test_filter_supports_periodic_outflow_and_rejects_wall():
    """U1：滤波的周期/出流路径可用；槽道 y 壁仍按现状报 unknown bc。"""
    grid = Grid(16, 4, 4)
    q = _field(grid)
    p = q[0]
    for bcs, axes in (((OUTFLOW, PERIODIC, PERIODIC), (0,)),
                      ((PERIODIC,) * 3, (0, 1))):
        out = filtering.shock_filter(q, p, grid, s0=0.3, rth=1e-5,
                                     bcs=bcs, axes=axes)
        assert np.all(np.isfinite(np.asarray(out)))
    with pytest.raises(ValueError, match="unknown bc"):
        filtering.shock_filter(q, p, grid, bcs=(PERIODIC, WALL, PERIODIC),
                               axes=(1,))


def test_callback_order_through_build_rhs_and_rk3():
    """U1：正式路径每子步先常规边界、再特殊覆盖、然后 RHS。"""
    case = _plate_case()
    calls = []
    original = case["bc_apply"]

    def wrapper(q):
        calls.append("bc")
        return original(q)

    case["bc_apply"] = wrapper
    args = argparse.Namespace(case="plate", scheme="UP7", vis_order="CD6",
                              viscous=False, weno_tv=False)
    rhs = main.build_rhs(case, args)
    rhs(case["cons0"])
    assert calls == ["bc"]

    calls.clear()
    with jax.disable_jit():
        q, _ = solver.rk3_step_stages(case["cons0"], 1e-3, rhs)
    # 关闭 JIT 后每个子步都会真实执行一次 prepare 里的特殊回调。
    assert calls == ["bc"] * 3
    assert np.all(np.isfinite(np.asarray(q)))


# ---------------------------------------------------------- U2 描述/配置一致

def test_channel_thermal_description_tracks_each_side():
    """U2：槽道上下壁温不同、单侧绝热时描述必须反映真实热条件。"""
    case = _channel_case()
    case["channel_bc"].Tw_bottom = 0.8
    case["channel_bc"].Tw_top = -1.0
    desc = mesh_data.from_case(case)
    assert desc.face("j-").thermal == "isothermal"
    assert desc.face("j+").thermal == "adiabatic"
    assert desc.face("j-").params["Tw"] == 0.8
    assert desc.face("j+").params["Tw"] == -1.0


def test_channel_execution_rejects_stale_wall_temperature():
    case = _channel_case()
    desc = mesh_data.from_case(case)
    case["channel_bc"].Tw_bottom = -1.0
    with pytest.raises(ValueError, match="Tw|壁温"):
        boundary.build_execution(case, desc)


def test_channel_axis_bcs_conflict_with_fixed_faces_is_rejected():
    """U2：把槽道 x 轴改成出流，不能仍报告/执行周期。"""
    from opencfd_jax.src import mesh_data
    case = dict(_channel_case())
    case["bcs"] = (OUTFLOW, WALL, PERIODIC)
    with pytest.raises(ValueError, match="bcs|固定|矛盾"):
        mesh_data.from_case(case)


def test_channel_missing_callback_is_rejected():
    """U2：IBC=200 描述存在但回调缺失时，必须在 JIT 前失败。"""
    from opencfd_jax.src import mesh_data
    case = dict(_channel_case())
    case["bc_apply"] = None
    with pytest.raises(ValueError, match="callback|回调|bc_apply"):
        mesh_data.from_case(case)


def test_plate_bcs_conflict_and_missing_callback_are_rejected():
    """U2：平板固定六侧与 bcs/回调矛盾时明确失败。"""
    from opencfd_jax.src import mesh_data
    case = dict(_plate_case())
    case["bcs"] = (PERIODIC, PERIODIC, PERIODIC)
    with pytest.raises(ValueError, match="bcs|固定|矛盾"):
        mesh_data.from_case(case)

    case = dict(_plate_case())
    case["bc_apply"] = None
    with pytest.raises(ValueError, match="callback|回调|bc_apply"):
        mesh_data.from_case(case)


def test_generic_wall_is_j_only_and_unknown_callback_rejected():
    """U2/R7：通用 WALL 只在 j 向支持；无标记的未知回调用途必须报错。"""
    from opencfd_jax.src import mesh_data, boundary
    base = dict(grid=Grid(4, 4, 4), dxs=(0.1, 0.1, 0.1), axes=(0,),
                bcs=(WALL, PERIODIC, PERIODIC), dif_bound=(0,) * 6,
                dif_axis_map=(0, 1, 2))
    with pytest.raises(ValueError, match="WALL|j 方向|壁面"):
        mesh_data.from_case(base)

    case = dict(grid=Grid(4, 4, 4), dxs=(0.1, 0.1, 0.1), axes=(0,),
                bcs=(OUTFLOW, PERIODIC, PERIODIC), dif_bound=(0,) * 6,
                dif_axis_map=(0, 1, 2), bc_apply=lambda q: q)
    with pytest.raises(ValueError, match="callback|回调|特殊"):
        boundary.build_execution(case)


# ---------------------------------------------------------- U3 单一实现来源

def test_fields_reexports_boundary_implementations():
    """U3：公共填充只有 boundary.py 一份实现，fields 只导出 canonical 名。"""
    assert fields.fill_halos_i is boundary.fill_halos_i
    assert fields.fill_halos_j is boundary.fill_halos_j
    assert fields.fill_halos_k is boundary.fill_halos_k
    assert fields.fill_wall_j is boundary.fill_wall_j
    assert fields.zero_nonperiodic_ghosts is boundary.zero_nonperiodic_ghosts
    assert not hasattr(fields, "fill_halos_x")
    assert not hasattr(fields, "fill_halos_y")
    assert not hasattr(fields, "fill_halos_z")


def test_boundary_exposes_ev_rules_separately_from_state_rules():
    """U3/R4：Ev 与守恒量状态规则分开，不做同一套镜像。"""
    assert boundary.fill_ev_wall_j is not boundary.fill_wall_j
    assert boundary.EV_WALL_SIGNS != fields._WALL_SIGN
    assert boundary.fill_ev_plate_j.__module__ == boundary.__name__


# ---------------------------------------------------------- U4 运行链接入

def test_build_rhs_reads_unified_boundary_execution(monkeypatch):
    """U4：main.build_rhs 在 JIT 前构造并消费统一边界执行配置。"""
    from opencfd_jax.src import boundary as bnd
    plate_case = _plate_case()
    seen = []
    real = bnd.build_execution

    def record(case, desc=None):
        result = real(case, desc)
        seen.append(result)
        return result

    monkeypatch.setattr(bnd, "build_execution", record)
    args = argparse.Namespace(case="plate", scheme="UP7", vis_order="CD6",
                              viscous=False, weno_tv=False)
    main.build_rhs(plate_case, args)
    assert len(seen) == 1
    assert seen[0].mode == "plate"
    assert seen[0].callback is plate_case["bc_apply"]


def _special_case():
    """构造仅用于测试的 case 特殊边界：i 两侧写固定鬼层，j/k 仍走通用周期。"""
    case = dict(cases.build_case(_ns(case="sod")))
    case["bcs"] = (INFLOW, PERIODIC, PERIODIC)
    faces = (
        mesh_data.FaceBoundary("i-", 0, INFLOW, "callback", False,
                               params={"test": "fixed-left"}),
        mesh_data.FaceBoundary("i+", 0, OUTFLOW, "callback", False,
                               params={"test": "fixed-right"}),
        mesh_data.FaceBoundary("j-", 1, PERIODIC, "halo", False),
        mesh_data.FaceBoundary("j+", 1, PERIODIC, "halo", False),
        mesh_data.FaceBoundary("k-", 2, PERIODIC, "halo", False),
        mesh_data.FaceBoundary("k+", 2, PERIODIC, "halo", False),
    )

    class TestBoundary:
        name = "test-fixed-ghost"

        def __init__(self):
            self.faces = faces
            self.calls = 0
            self.grid = case["grid"]

        def apply(self, q):
            self.calls += 1
            q = q.at[..., :LAP].set(0.25)
            return q.at[..., LAP + self.grid.ni:].set(0.5)

    special = TestBoundary()
    case["special_boundary"] = special
    case["bc_apply"] = special.apply
    return case, special


def test_case_side_special_boundary_is_explicit_and_stage_connected():
    """U4/A5：case 显式提供特殊边界；六侧描述、回调与通用填充按原阶段衔接。"""
    case, special = _special_case()
    desc = mesh_data.from_case(case)
    execution = boundary.build_execution(case, desc)
    assert execution.mode == "custom"
    assert execution.callback is case["bc_apply"]
    assert execution.bcs == (INFLOW, PERIODIC, PERIODIC)

    q = special.apply(case["cons0"])
    np.testing.assert_array_equal(np.asarray(q[..., :LAP]), 0.25)
    np.testing.assert_array_equal(
        np.asarray(q[..., LAP + case["grid"].ni:]), 0.5)
    # i 的 INFLOW 是保留型 no-op；j/k 通用周期填充只能在回调之后执行。
    np.testing.assert_array_equal(
        np.asarray(fields.fill_halos_i(q, INFLOW, case["grid"].ni)), np.asarray(q))
    filled = fields.fill_halos_j(q, PERIODIC, case["grid"].nj)
    np.testing.assert_array_equal(
        np.asarray(filled[..., :LAP, :]),
        np.asarray(filled[..., case["grid"].nj:case["grid"].nj + LAP, :]))

    args = argparse.Namespace(case="sod", scheme="UP7", vis_order="CD6",
                              viscous=False, weno_tv=False)
    rhs = main.build_rhs(case, args)
    rhs(case["cons0"])
    assert special.calls >= 1


def test_invalid_special_boundary_is_rejected_before_jit():
    """U4/R7：特殊边界的六侧顺序、回调轴级 bcs 矛盾必须在 JIT 前报错。"""
    case, special = _special_case()
    bad_faces = (special.faces[0], special.faces[1], special.faces[3],
                 special.faces[2], special.faces[4], special.faces[5])
    special.faces = bad_faces
    with pytest.raises(ValueError, match="六侧|顺序"):
        mesh_data.from_case(case)

    case, special = _special_case()
    case["bcs"] = (OUTFLOW, PERIODIC, PERIODIC)
    with pytest.raises(ValueError, match="INFLOW/PLATE"):
        mesh_data.from_case(case)


def test_special_inlet_and_outflow_share_axis():
    """特殊入口与普通出流组合时，右侧鬼层由内区末点复制。"""
    case, special = _special_case()
    faces = list(special.faces)
    faces[1] = mesh_data.FaceBoundary("i+", 0, OUTFLOW, "halo", False)
    special.faces = tuple(faces)
    inlet = jnp.asarray([0.25, 0.0, 0.0, 0.0, 1.0]).reshape(5, 1, 1, 1)
    special.apply = lambda q: q.at[..., :LAP].set(inlet)
    case["bc_apply"] = special.apply
    execution = boundary.build_execution(case)
    q = case["cons0"].at[..., LAP + case["grid"].ni - 1].set(0.75)
    actual = execution.apply(q)
    np.testing.assert_array_equal(np.asarray(actual[0, ..., :LAP]), 0.25)
    # OUTFLOW 走方式 A（零梯度复制）：鬼层 = 出口边界点那一层。
    np.testing.assert_array_equal(
        np.asarray(actual[..., LAP + case["grid"].ni:]),
        np.broadcast_to(np.asarray(actual[..., LAP + case["grid"].ni - 1:
                                        LAP + case["grid"].ni]),
                        np.asarray(actual[..., LAP + case["grid"].ni:]).shape))
    args = argparse.Namespace(case="sod", scheme="UP7", vis_order="CD6",
                              viscous=False, weno_tv=False)
    got = main.build_rhs(case, args)(case["cons0"])
    assert np.all(np.isfinite(np.asarray(got)))
    ev = jnp.arange(4 * np.prod(case["grid"].shape[1:]), dtype=jnp.float64).reshape(
        (4,) + case["grid"].shape[1:])
    filled_ev = execution.fill_mixed_outflow(ev, axis=0)
    expected_ev = np.repeat(np.asarray(ev[..., LAP + case["grid"].ni - 1:LAP + case["grid"].ni]), LAP, axis=-1)
    np.testing.assert_array_equal(np.asarray(filled_ev[..., LAP + case["grid"].ni:]), expected_ev)
    args.viscous = True
    got_viscous = main.build_rhs(case, args)(case["cons0"])
    assert np.all(np.isfinite(np.asarray(got_viscous)))


def test_physical_wall_implementations_belong_to_boundary():
    assert channel.apply_wall_bc is boundary.apply_channel_wall_bc
    assert plate.apply is boundary.apply_plate_bc
