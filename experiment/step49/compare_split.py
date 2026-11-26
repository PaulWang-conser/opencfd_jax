"""比较三个 RK 阶段各方向 Steger–Warming 分裂通量。"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from opencfd_jax.src import init, main, runtime_config, schemes  # noqa: E402
from opencfd_jax.src.fields import array_axis  # noqa: E402
from compare_rhs import compare_regions  # noqa: E402


def main_compare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "opencfd_jax/experiment/cases/plate_cgns_step49.json"))
    parser.add_argument("--reference", default=str(ROOT / "opencfd_jax/experiment/step49_clean_split_20261119"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = runtime_config.load_config(args.config)
    run = runtime_config.resolve_args({}, cfg, main.build_parser().parse_args([]))
    state = init.prepare(run)
    grid, desc, execution = state.case["grid"], state.desc, state.execution
    rhs = main.build_rhs(state.case, run, execution)
    q0 = state.cons0
    d1 = rhs(q0)
    q1 = q0 + run.dt*d1
    d2 = rhs(q1)
    q2 = .75*q0 + .25*(q1 + run.dt*d2)
    nk, nj, ni = grid.nk, grid.nj, grid.ni
    kr, jr, ir = slice(4, 4+nk), slice(4, 4+nj), slice(4, 4+ni)
    regions = {"physical": (kr, jr, ir), "inlet": (kr, jr, slice(4, 8)),
               "outlet": (kr, jr, slice(ni, ni+4)), "wall": (kr, slice(4, 8), ir),
               "farfield": (kr, slice(nj, nj+4), ir),
               "inlet_ghost": (kr, jr, slice(0, 4)),
               "outlet_ghost": (kr, jr, slice(4+ni, None)),
               "wall_ghost": (kr, slice(0, 4), ir),
               "farfield_ghost": (kr, slice(4+nj, None), ir),
               "periodic_minus_ghost": (slice(0, 4), jr, ir),
               "periodic_plus_ghost": (slice(4+nk, None), jr, ir)}
    result = {"index_order": "含鬼层的(k,j,i)",
              "normalization": "原版分裂通量除以另外两方向 (N-1) 的乘积", "stages": {}}
    keys = (("Akx", "Aky", "Akz"), ("Aix", "Aiy", "Aiz"), ("Asx", "Asy", "Asz"))
    for stage, q in enumerate((q0, q1, q2), 1):
        prepared = execution.prepare(q)
        result["stages"][str(stage)] = {}
        for axis, ocfd_axis in enumerate(("x", "z", "y")):
            moved = jnp.moveaxis(prepared, 1 + array_axis(axis), -1)
            metric = tuple(jnp.moveaxis(desc.metrics[k], array_axis(axis), -1) for k in keys[axis])
            split = schemes.steger_warming(moved, state.case["gamma"], 0., axis, metric=metric)
            scale = np.prod([(ni, nj, nk)[i]-1 for i in range(3) if i != axis])
            result["stages"][str(stage)]["ijk"[axis]] = {}
            for sign, got in zip(("p", "m"), split):
                path = Path(args.reference) / f"diagnostic_split_{stage}_{ocfd_axis}{sign}.bin"
                shape = (5, nj+8, nk+8, ni+8)
                raw = np.fromfile(path, dtype="<f8")
                if raw.size != int(np.prod(shape)):
                    raise ValueError(f"{path} 数据长度不匹配：{raw.size}")
                ref = raw.reshape(shape)[[0, 1, 3, 2, 4]].transpose(0, 2, 1, 3) / scale
                actual = np.asarray(jnp.moveaxis(got, -1, 1 + array_axis(axis)))
                result["stages"][str(stage)]["ijk"[axis]][sign] = compare_regions(
                    ref, actual, ("rho", "rho_u", "rho_v", "rho_w", "rhoE"), regions)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        fd, name = tempfile.mkstemp(prefix="split_comparison_", suffix=".json", dir=args.reference)
        os.close(fd)
        output = Path(name)
        output.write_text(text, encoding="utf8")
    else:
        output = Path(args.output)
        with output.open("x", encoding="utf8") as f:
            f.write(text)
    print(f"分裂通量逐区域报告：{output}")


if __name__ == "__main__":
    main_compare()
