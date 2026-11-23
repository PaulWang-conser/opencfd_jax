"""对比干净 OCFD 诊断与 JAX 三个 RK 右端项，输出各变量、区域的误差。"""
import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from opencfd_jax.src import boundary, init, main, runtime_config, schemes, solver  # noqa: E402
from opencfd_jax.src.solver import inviscid_rhs  # noqa: E402


def main_compare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "opencfd_jax/experiment/cases/plate_cgns_step49.json"))
    parser.add_argument("--reference", default=str(ROOT / "opencfd_jax/experiment/step49_clean_diag_20261119"))
    parser.add_argument("--output", default=None, help="输出文件；不指定时创建唯一文件")
    parser.add_argument("--ocfd-farfield-band", action="store_true", default=False,
                        help="仅诊断：远场关闭边界带，其他配置保持不变")
    parser.add_argument("--interior-face-writes", action="store_true", default=False,
                        help="仅诊断：边界点的横向写入限制到内区")
    parser.add_argument("--rescale-weno-epsilon", action="store_true", default=False,
                        help="仅诊断：按两程序参考空间步长差调整 WENO7 epsilon"
                             "（2026-11-22 起正式路径已按 flux_scale 换算且显式参数"
                             "优先，本开关在 CGNS 算例上不再改变结果，仅留作历史对照）")
    args = parser.parse_args()
    cfg = runtime_config.load_config(args.config)
    runtime = runtime_config.resolve_args({}, cfg, main.build_parser().parse_args([]))
    state = init.prepare(runtime)
    if args.interior_face_writes:
        boundary._face_lateral = lambda face, grid: boundary._interior_window(grid)
    grid = state.case["grid"]
    if args.rescale_weno_epsilon:
        original_rhs = solver._rhs_along_axis
        epsilon = schemes._WENO7_EPS
        extents = (grid.ni, grid.nj, grid.nk)

        def scaled_rhs(cons, axis, *rest, **kwargs):
            scale = np.prod([extents[i] - 1 for i in range(3) if i != axis])
            schemes._WENO7_EPS = epsilon / float(scale * scale)
            return original_rhs(cons, axis, *rest, **kwargs)

        solver._rhs_along_axis = scaled_rhs
    execution = state.execution
    if args.ocfd_farfield_band:
        band = list(execution.band)
        for face in execution.faces:
            if face.kind == "farfield":
                band[2 * state.desc.dif_axis_map[face.axis] + int(face.side.endswith("+"))] = 0
        execution = replace(execution, band=tuple(band))
    rhs = main.build_rhs(state.case, runtime, execution)
    q0 = state.cons0
    du1 = rhs(q0)
    q1 = q0 + runtime.dt * du1
    du2 = rhs(q1)
    q2 = .75 * q0 + .25 * (q1 + runtime.dt * du2)
    du3 = rhs(q2)
    stages = ((q0, du1), (q1, du2), (q2, du3))
    shape = (5, grid.nj, grid.nk, grid.ni)
    names = ("rho", "rho_u", "rho_v", "rho_w", "rhoE")
    regions = {"all": (slice(None), slice(None), slice(None)),
               "inlet": (slice(None), slice(None), slice(0, 4)),
               "outlet": (slice(None), slice(None), slice(-4, None)),
               "wall": (slice(None), slice(0, 4), slice(None)),
               "farfield": (slice(None), slice(-4, None), slice(None)),
               "interior": (slice(4, -4), slice(4, -4), slice(4, -4))}
    nk, nj, ni = grid.nk, grid.nj, grid.ni
    kr, jr, ir = slice(4, 4+nk), slice(4, 4+nj), slice(4, 4+ni)
    ghost_regions = {"physical": (kr, jr, ir),
                     "inlet_ghost": (kr, jr, slice(0, 4)),
                     "outlet_ghost": (kr, jr, slice(4+ni, None)),
                     "wall_ghost": (kr, slice(0, 4), ir),
                     "farfield_ghost": (kr, slice(4+nj, None), ir),
                     "periodic_minus": (slice(0, 4), jr, ir),
                     "periodic_plus": (slice(4+nk, None), jr, ir),
                     "outlet_wall_corner": (kr, slice(0, 5), slice(3+ni, None)),
                     "outlet_farfield_corner": (kr, slice(3+nj, None), slice(3+ni, None))}
    result = {"axis_order": "k,j,i", "relative_denominator": "区域内参考值最大绝对值；为零时为 null",
              "diagnostic_options": {"ocfd_farfield_band": args.ocfd_farfield_band,
                                     "interior_face_writes": args.interior_face_writes,
                                     "rescale_weno_epsilon": args.rescale_weno_epsilon},
              "stages": {}}
    def read_reference(kind, step):
        raw = np.fromfile(Path(args.reference) / f"diagnostic_{kind}_{step}.bin", dtype="<f8")
        if raw.size != int(np.prod(shape)):
            raise ValueError(f"第 {step} 阶段 {kind} 数据长度不匹配: {raw.size}, 期望 {np.prod(shape)}")
        return raw.reshape(shape)[[0, 1, 3, 2, 4]].transpose(0, 2, 1, 3)

    for step, (q, value) in enumerate(stages, 1):
        desc = state.desc
        prepared = execution.prepare(q)
        inv = inviscid_rhs(prepared, grid, gamma=state.case["gamma"],
                           bcs=execution.bcs, axes=desc.axes, dx=desc.dxs,
                           scheme=runtime.scheme, metrics=desc.metrics,
                           dif_bound=execution.band, dif_axis_map=desc.dif_axis_map,
                           farfield=execution.farfield, faces=execution.faces,
                           fill_state_halos=False,
                           # 与 build_rhs 同一尺度口径（2026-11-22 复审：此前分项漏传，
                           # inviscid/viscous 报告停在旧 epsilon，质量粘性项假报 0.0418）
                           flux_scale=solver.flux_scales_for(grid, desc.dxs, desc.metrics))
        got_total, got_invis = np.asarray(value[grid.inner()]), np.asarray(inv[grid.inner()])
        ref_total, ref_invis = read_reference("du", step), read_reference("invis", step)
        terms = {"total": (ref_total, got_total), "inviscid": (ref_invis, got_invis),
                 "viscous": (ref_total-ref_invis, got_total-got_invis)}
        stage = {}
        for term, (ref, got) in terms.items():
            stage[term] = compare_regions(ref, got, names, regions)
        rho, velocity, temperature = boundary.primitive(prepared, state.case["params"])
        got_primitive = np.concatenate((np.asarray(rho)[None], np.asarray(velocity),
                                        np.asarray(temperature)[None]), axis=0)
        padded = (grid.nj+8, grid.nk+8, grid.ni+8)
        fields = []
        for field in (0, 1, 3, 2, 4):
            raw = np.fromfile(Path(args.reference) / f"diagnostic_primitive_{step}_{field}.bin", dtype="<f8")
            if raw.size != int(np.prod(padded)):
                raise ValueError(f"第 {step} 阶段原始量 {field} 数据长度不匹配: {raw.size}")
            fields.append(raw.reshape(padded).transpose(1, 0, 2))
        stage["primitive"] = compare_regions(np.stack(fields), got_primitive,
                                               ("rho", "u", "v", "w", "T"), ghost_regions)
        result["stages"][str(step)] = stage
    if args.output is None:
        fd, filename = tempfile.mkstemp(prefix="jax_rhs_compare_", suffix=".json",
                                        dir=args.reference)
        os.close(fd)
        output = Path(filename)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    else:
        output = Path(args.output)
        with output.open("x", encoding="utf8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"逐区域 RHS 报告：{output}")


def compare_regions(ref, got, names, regions):
    stage = {}
    for idx, name in enumerate(names):
        stage[name] = {}
        for region, bounds in regions.items():
            r, g = ref[(idx,) + bounds], got[(idx,) + bounds]
            valid = np.isfinite(r) & np.isfinite(g)
            entry = {"nonfinite_reference": int((~np.isfinite(r)).sum()),
                     "nonfinite_jax": int((~np.isfinite(g)).sum())}
            if np.any(valid):
                delta = np.where(valid, np.abs(g-r), -1)
                loc = np.unravel_index(int(np.argmax(delta)), delta.shape)
                offset = tuple((s.start or 0) if (s.start or 0) >= 0 else r0 + s.start
                               for s, r0 in zip(bounds, ref.shape[1:]))
                maximum = float(delta[loc])
                scale = float(np.max(np.abs(r[valid])))
                entry.update(max_abs=maximum, max_abs_index=[int(x+y) for x,y in zip(loc,offset)],
                             max_abs_ref=scale, relative=maximum/scale if scale else None)
            stage[name][region] = entry
    return stage


if __name__ == "__main__":
    main_compare()
