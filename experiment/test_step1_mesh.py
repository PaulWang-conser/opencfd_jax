"""第 1 步「单块网格与边界描述」的验收测试。

覆盖计划中的 U1（现状冻结）、U2（六侧语义与非法输入）、U3（消费者
入口的时序）三组场景。测试放在 ``experiment/`` 下是因为仓库根 ``tests/``
被 .gitignore 排除；运行方式见 ``experiment/README.md``。
"""

import argparse
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

# experiment/test_step1_mesh.py -> parents[2] = D:\Desktop\ocfd2jax
_REPO_TAX = Path(__file__).resolve().parents[2]
if str(_REPO_TAX) not in sys.path:
    sys.path.insert(0, str(_REPO_TAX))

from opencfd_jax.src import boundary, cases, mesh_data, main
from opencfd_jax.src.fields import (INFLOW, OUTFLOW, PERIODIC, PLATE,
                                    WALL, Grid)


def _ns(**kw):
    """构造 build_case 所需的最小 CLI 命名空间，缺省值保持仓库原状。"""
    defaults = dict(case="sod", ni=16, nj=8, nk=4, Re=10.0, Ama=0.1,
                    ref_t=None, pr=0.7, forcing_work="off")
    defaults.update(kw)
    return argparse.Namespace(**defaults)


@pytest.fixture(scope="module")
def built():
    """五个内置算例各建一次，供形状/步长/边界的现状冻结断言复用。"""
    return {
        "sod": cases.build_case(_ns(case="sod", ni=16, nj=8, nk=4)),
        "tg": cases.build_case(_ns(case="tg", ni=16, nj=16, nk=4)),
        "poiseuille": cases.build_case(
            _ns(case="poiseuille", ni=16, nj=8, nk=4, p_grad=1.0)),
        "plate": cases.build_case(
            _ns(case="plate", ni=16, nj=8, nk=4, plate_x0=0.1,
                plate_x1=1.1, plate_h=0.12,
                plate_dif_bound="0 0 0 0 1 0")),
        "channel": cases.build_case(
            _ns(case="channel",
                infile=str(_REPO_TAX / "opencfd_jax/experiment/cases/channel_small.in"),
                field=None, mesh=None)),
    }


# ---------------------------------------------------------------- U1 现状冻结

def test_existing_cases_keep_shape_steps_axes(built):
    """U1：描述接入不得改动五个算例的网格、步长与求解方向。"""
    expected = {
        "sod": ((16, 8, 4), (1.0 / 16,) * 3, (0,)),
        "tg": ((16, 16, 4), (2 * np.pi / 16, 2 * np.pi / 16, 0.0), (0, 1)),
        "poiseuille": ((16, 8, 4), (2 * np.pi / 16, 1.0 / 8, 2 * np.pi / 16), (0, 1)),
        "plate": ((16, 8, 4), (0.0625, 0.12 / 7, 0.0625), (0, 1)),
        "channel": ((24, 25, 16), (1 / 23, 1 / 24, 1 / 15), (0, 1, 2)),
    }
    assert set(built) == set(expected)
    for name, case in built.items():
        grid = case["grid"]
        assert (grid.ni, grid.nj, grid.nk) == expected[name][0], name
        assert case["dxs"] == pytest.approx(expected[name][1]), name
        assert tuple(case["axes"]) == expected[name][2], name
        assert case["grid"].alloc().shape == (5, grid.nk + 8, grid.nj + 8,
                                              grid.ni + 8), name


def test_description_reuses_case_objects(built):
    """U1：描述不复制网格，也不把直角假设伪装成真度量。"""
    for name, case in built.items():
        desc = mesh_data.from_case(case)
        assert desc.grid is case["grid"], name
        assert desc.dxs == tuple(case["dxs"]), name
        assert desc.axes == tuple(case["axes"]), name
        assert desc.bcs == tuple(case["bcs"]), name
    assert built["sod"].get("metrics") is None
    assert built["tg"].get("metrics") is None
    assert built["poiseuille"].get("metrics") is None
    assert built["plate"].get("metrics") is None
    assert built["channel"].get("metrics") is None
    assert mesh_data.from_case(built["sod"]).metrics is None
    assert mesh_data.from_case(built["sod"]).coords is None


def test_plate_axis_mapping_and_dif_bound_derived_from_faces(built):
    """U1：平板保留 dif_axis_map=(0,2,1)；边界带改由边界 kind 派生。"""
    desc = mesh_data.from_case(built["plate"])
    assert desc.dif_axis_map == (0, 2, 1)
    # 算例侧旧字段仍留档，但执行配置里的带由 faces 派生。
    assert desc.dif_bound == (0, 0, 0, 0, 1, 0)
    # 只有“WALL + 回调”开带：入口扩容、出流/远场零梯度延拓、周期复制都不开。
    assert desc.face("i-").dif_band is False
    assert desc.face("i+").dif_band is False
    assert desc.face("j-").dif_band is True
    assert desc.face("j+").dif_band is False
    assert desc.face("k-").dif_band is False
    assert desc.face("k+").dif_band is False
    execution = boundary.build_execution(built["plate"], desc)
    # 原码次序 (x-,x+,y-,y+,z-,z+)，map=(0,2,1)：i→x，j→z，k→y。
    assert execution.band == (0, 0, 0, 0, 1, 0)


def test_compare_style_step_override_is_visible(built):
    """U1/U3：对拍入口覆盖 case["dxs"] 后，下一次构造必须看到新步长。"""
    case = dict(built["plate"])
    override = (0.05, 0.02, 0.0)
    case["dxs"] = override
    desc = mesh_data.from_case(case)
    assert desc.dxs == override
    assert desc.face("j-").dif_band is True


def test_snapshot_does_not_change_after_case_mutation(built):
    """U3：描述是构造时刻的快照；闭包建立后 case 改动不回溯已建描述。"""
    case = dict(built["sod"])
    desc = mesh_data.from_case(case)
    case["dxs"] = (0.01, 0.01, 0.01)
    assert desc.dxs == (1.0 / 16,) * 3
    assert mesh_data.from_case(case).dxs == (0.01, 0.01, 0.01)


def test_build_rhs_reads_description_after_case_update(built, monkeypatch):
    """U3：build_rhs 在入口才取描述，配置更新不会被旧快照遮蔽。"""
    case = dict(built["sod"])
    case["dxs"] = (0.04, 0.04, 0.04)
    seen = []
    real = main.from_case

    def record(c):
        desc = real(c)
        seen.append(desc)
        return desc

    monkeypatch.setattr(main, "from_case", record)
    args = argparse.Namespace(case="sod", scheme="WENO7", vis_order="CD8",
                              viscous=False, weno_tv=False)
    main.build_rhs(case, args)
    assert seen and seen[-1].dxs == (0.04, 0.04, 0.04)


# ---------------------------------------------------------- U2 六侧与执行来源

def test_channel_six_faces_and_executor_source(built):
    """U2：槽道双壁、周期 x/z，壁面由回调 + 鬼胞共同执行。"""
    desc = mesh_data.from_case(built["channel"])
    kinds = {f.side: f.kind for f in desc.faces}
    assert kinds == {"i-": PERIODIC, "i+": PERIODIC, "j-": WALL, "j+": WALL,
                     "k-": PERIODIC, "k+": PERIODIC}
    for side in ("j-", "j+"):
        face = desc.face(side)
        assert face.executor == "callback+halo"
        assert face.dif_band is True  # 双壁都靠回调写点，原码 DIF_BOUND 也开带
    for side in ("i-", "i+", "k-", "k+"):
        assert desc.face(side).executor == "halo"


def test_plate_faces_are_asymmetric(built):
    """U2：平板入口/出口、下壁/上来流四侧语义不同，且都由平板回调执行。"""
    desc = mesh_data.from_case(built["plate"])
    assert desc.face("i-").kind == INFLOW
    assert desc.face("i+").kind == OUTFLOW
    assert desc.face("j-").kind == WALL
    assert desc.face("j+").kind == mesh_data.FARFIELD
    for side in ("i-", "i+", "j-", "j+"):
        assert desc.face(side).executor == "callback"
    for side in ("k-", "k+"):
        assert desc.face(side).kind == PERIODIC
        assert desc.face(side).executor == "halo"


def test_generic_sod_faces(built):
    """U2：普通算例按轴级 bc 展开；sod 的 x 出流、y/z 周期。"""
    desc = mesh_data.from_case(built["sod"])
    kinds = {f.side: f.kind for f in desc.faces}
    assert kinds == {"i-": OUTFLOW, "i+": OUTFLOW,
                     "j-": PERIODIC, "j+": PERIODIC,
                     "k-": PERIODIC, "k+": PERIODIC}
    assert all(f.executor == "halo" for f in desc.faces)


@pytest.mark.skipif(
    not (_REPO_TAX / "channel_case_iiA/quick/OCFD3d-Mesh.dat").exists(),
    reason="缺少 channel_case_iiA/quick/OCFD3d-Mesh.dat")
def test_channel_real_mesh_keeps_inner_coords_and_metrics():
    """U2：真网格时描述引用内区原始坐标，并带真实度量。"""
    quick = _REPO_TAX / "channel_case_iiA/quick"
    mesh_path = quick / "OCFD3d-Mesh.dat"
    case = cases.build_case(
        _ns(case="channel", infile=str(quick / "opencfd-scu.in"),
            field=None, mesh=str(mesh_path)))
    desc = mesh_data.from_case(case)
    grid = case["grid"]
    assert desc.has_metrics and desc.has_coords
    assert set(desc.coords) == {"Axx", "Ayy", "Azz"}
    for key in ("Axx", "Ayy", "Azz"):
        assert desc.coords[key].shape == (grid.nk, grid.nj, grid.ni)
    # case 保存的是内区原始坐标（含鬼胞的副本没有留在 case 里）。
    assert desc.coords["Axx"] is case["coords"]["Axx"]
    assert tuple(desc.metrics["Ajac"].shape) == (
        grid.nk + 8, grid.nj + 8, grid.ni + 8)


def test_channel_without_mesh_keeps_metrics_and_coords_none(built):
    """U2：不给网格时不伪造坐标或度量，保持 Ajac=1 的直角路径。"""
    desc = mesh_data.from_case(built["channel"])
    assert desc.metrics is None and desc.coords is None


def test_description_is_read_only(built):
    """U2：描述本身与包装后的映射都不可原地修改。"""
    desc = mesh_data.from_case(built["plate"])
    with pytest.raises(FrozenInstanceError):
        desc.dxs = (0.0, 0.0, 0.0)
    case = dict(built["sod"])
    shape = (built["sod"]["grid"].nk + 8, built["sod"]["grid"].nj + 8,
             built["sod"]["grid"].ni + 8)
    case["metrics"] = {k: np.zeros(shape) for k in mesh_data.METRIC_KEYS}
    wrapped = mesh_data.from_case(case).metrics
    with pytest.raises(TypeError):
        wrapped["Ajac"] = np.ones(shape)


# ---------------------------------------------------------------- U2 非法输入

def _base_case(**over):
    """最小合法算例骨架，供非法输入用例逐项覆盖。"""
    base = dict(grid=Grid(4, 4, 4), dxs=(0.1, 0.1, 0.1), axes=(0,),
                bcs=(OUTFLOW, PERIODIC, PERIODIC), dif_bound=(0,) * 6,
                dif_axis_map=(0, 1, 2))
    base.update(over)
    return base


@pytest.mark.parametrize("bad", ["bogus", None])
def test_unknown_boundary_type_is_rejected(bad):
    """U2：未知/缺失边界类型明确报错，不静默当周期处理。"""
    with pytest.raises(ValueError, match="边界类型|bcs"):
        mesh_data.from_case(_base_case(bcs=(bad, PERIODIC, PERIODIC)))


def test_inflow_without_adapter_is_rejected():
    """U2：INFLOW/PLATE 只有平板/槽道适配器能准确表达。"""
    with pytest.raises(ValueError, match="INFLOW/PLATE"):
        mesh_data.from_case(_base_case(bcs=(INFLOW, PERIODIC, PERIODIC)))


def test_duplicate_axes_are_rejected():
    """U2：求解方向重复会让步长/导数含义不清，必须拒绝。"""
    with pytest.raises(ValueError, match="axes"):
        mesh_data.from_case(_base_case(axes=(0, 0)))


def test_nonpositive_step_on_solved_axis_is_rejected():
    """U2：求解方向步长为 0/负值必须拒绝；未求解方向的 0 仍允许。"""
    with pytest.raises(ValueError, match="步长必须为正"):
        mesh_data.from_case(_base_case(dxs=(0.0, 0.1, 0.1)))
    tg_like = _base_case(axes=(0, 1), dxs=(0.1, 0.1, 0.0))
    assert mesh_data.from_case(tg_like).dxs == (0.1, 0.1, 0.0)


def test_metric_shape_mismatch_is_rejected():
    """U2：度量形状与网格不一致时报错，避免静默广播。"""
    shape = (12, 12, 12)
    metrics = {k: np.zeros(shape) for k in mesh_data.METRIC_KEYS}
    metrics["Ajac"] = np.zeros((5, 5, 5))
    with pytest.raises(ValueError, match="Ajac"):
        mesh_data.from_case(_base_case(metrics=metrics))


def test_bad_dif_bound_and_axis_map_are_rejected():
    """U2：DIF_BOUND 非 0/1 或方向映射不是排列时拒绝。"""
    with pytest.raises(ValueError, match="dif_bound"):
        mesh_data.from_case(_base_case(dif_bound=(2, 0, 0, 0, 0, 0)))
    with pytest.raises(ValueError, match="dif_axis_map"):
        mesh_data.from_case(_base_case(dif_axis_map=(0, 0, 1)))
