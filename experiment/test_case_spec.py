"""CaseSpec 算例协议的行为测试（Step 5 交付 1）。"""

from types import SimpleNamespace

import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src.cases import CaseSpec, build_case  # noqa: E402
from opencfd_jax.src.fields import Grid  # noqa: E402
from opencfd_jax.src.viscous import PhysParams  # noqa: E402


def _spec(**kw):
    grid = Grid(8, 8, 4)
    params = PhysParams(gamma=1.5, Ama=0.2, Re=100.0)
    defaults = dict(grid=grid, cons0=jnp.ones(grid.shape), bcs=("outflow",) * 3,
                    dxs=(0.1, 0.1, 0.1), axes=(0,), params=params)
    defaults.update(kw)
    return CaseSpec(**defaults)


def test_gamma_defaults_to_params():
    assert _spec().gamma == 1.5


def test_required_fields_and_shape_validated():
    with pytest.raises(ValueError, match="必选字段"):
        _spec(grid=None)
    with pytest.raises(ValueError, match="bcs"):
        _spec(bcs=("outflow", "outflow"))
    with pytest.raises(ValueError, match="axes"):
        _spec(axes=(0, 0))
    with pytest.raises(ValueError, match="cons0 形状"):
        _spec(cons0=jnp.ones((5, 4, 4, 4)))


def test_mapping_protocol_read_write_and_guards():
    spec = _spec()
    assert spec["gamma"] == 1.5 and spec.get("forcing") is None
    spec["dif_bound"] = (0, 0, 0, 0, 1, 0)      # 下标写（compare/init 兼容）
    assert spec.get("dif_bound") == (0, 0, 0, 0, 1, 0)
    assert "farfield" in spec and spec.keys()
    with pytest.raises(KeyError):
        spec["nonexistent"]
    with pytest.raises(KeyError):
        spec["nonexistent"] = 1
    with pytest.raises(NotImplementedError):
        del spec["gamma"]


def test_build_case_returns_spec_for_all_public_cases():
    for name in ("sod", "tg", "poiseuille", "plate"):
        args = SimpleNamespace(case=name, ni=16, nj=16, nk=4, Re=1000.0, Ama=0.1,
                               p_grad=0.1, scheme="UP7", viscous=True,
                               vis_order="CD8", ref_t=288.15, pr=0.7,
                               plate_dif_bound=None)
        assert isinstance(build_case(args), CaseSpec), name
