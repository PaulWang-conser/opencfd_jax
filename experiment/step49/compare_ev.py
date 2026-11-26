"""比较三阶段三个方向的 OCFD 与 JAX 粘性 Ev。"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from opencfd_jax.src import init, main, runtime_config, viscous  # noqa: E402
from compare_rhs import compare_regions  # noqa: E402


def main_compare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "opencfd_jax/experiment/cases/plate_cgns_step49.json"))
    parser.add_argument("--reference", default=str(ROOT / "opencfd_jax/experiment/step49_clean_ev_20261119"))
    parser.add_argument("--output", default=None, help="输出文件；不指定时创建唯一文件")
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
    regions = {"physical": (kr, jr, ir), "outlet": (kr, jr, slice(ni, ni+4)),
               "wall": (kr, slice(4, 8), ir), "farfield": (kr, slice(nj, nj+4), ir),
               "inlet_ghost": (kr, jr, slice(0, 4)),
               "outlet_ghost": (kr, jr, slice(ni+4, None)),
               "wall_ghost": (kr, slice(0, 4), ir),
               "farfield_ghost": (kr, slice(nj+4, None), ir),
               "periodic_minus_ghost": (slice(0, 4), jr, ir),
               "periodic_plus_ghost": (slice(4+nk, None), jr, ir)}
    result = {"index_order": "含鬼层的(k,j,i)",
              "normalization": "原版 Ev 除以另外两方向 (N-1) 的乘积",
              "note": "原版复用 Ev 工作数组；横向鬼层可能保留此前方向的值。", "stages": {}}
    for stage, q in enumerate((q0, q1, q2), 1):
        captured = []
        original = viscous.fill_ev_faces

        def capture(values, faces, target_grid):
            out = original(values, faces, target_grid)
            captured.append(np.asarray(out))
            return out

        viscous.fill_ev_faces = capture
        try:
            viscous.viscous_rhs(execution.prepare(q), grid, state.case["params"],
                execution.bcs, axes=desc.axes, dx=desc.dxs,
                order={"CD6": 6, "CD8": 8}[run.vis_order], metrics=desc.metrics,
                dif_bound=execution.band, dif_axis_map=desc.dif_axis_map,
                boundary_execution=execution, faces=execution.faces,
                fill_state_halos=False)
        finally:
            viscous.fill_ev_faces = original
        result["stages"][str(stage)] = {}
        for axis, ocfd_axis in enumerate(("x", "z", "y")):
            shape = (nj+8, nk+8, ni+8)
            fields = []
            for comp in (0, 2, 1, 3):
                path = Path(args.reference) / f"diagnostic_ev_{stage}_{ocfd_axis}_{comp}.bin"
                raw = np.fromfile(path, dtype="<f8")
                if raw.size != int(np.prod(shape)):
                    raise ValueError(f"{path} 数据长度不匹配：{raw.size}")
                fields.append(raw.reshape(shape).transpose(1, 0, 2))
            scale = np.prod([(ni, nj, nk)[i]-1 for i in range(3) if i != axis])
            ref = np.stack(fields) / scale
            result["stages"][str(stage)]["ijk"[axis]] = compare_regions(
                ref, captured[axis], ("momentum_x", "momentum_y", "momentum_z", "energy"), regions)
    if args.output is None:
        fd, name = tempfile.mkstemp(prefix="ev_comparison_", suffix=".json", dir=args.reference)
        os.close(fd)
        output = Path(name)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    else:
        output = Path(args.output)
        with output.open("x", encoding="utf8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Ev 逐区域报告：{output}")


if __name__ == "__main__":
    main_compare()
