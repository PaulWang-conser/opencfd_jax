"""正式运行入口（对应 OCFD_NS_Solver.cu 的 NS_solver_real 主循环）。

用法（从工作区根目录运行，当前源码包路径为 opencfd_jax.src）：
    python -m opencfd_jax.src.run --case sod --N 256 --scheme UP7 --end-time 0.2
    python -m opencfd_jax.src.run --case tg  --N 64  --scheme UP7 --end-time 0.5 --Re 50
    python -m opencfd_jax.src.run --case poiseuille --N 32 --ny 64 --p-grad 0.1 \
        --Re 100 --end-time 2.0 --viscous

主循环对齐原码：每步 dt_fn → RK3（+forcing）→ 每 Kstep_show 同步打印诊断
（ana_residual/NAN）→ 每 Kstep_save 存 npz。滤波按原码节奏每 nstep_filter
步一次（整步级，非每子步）。除打印/存盘步外不阻塞 host。
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import boundary, config, io
from .ana import check_nan, diagnostic_line
from .cases import apply_forcing, build_case
from .mesh_data import from_case
from .solver import compute_dt, inviscid_rhs, rk3_step
from .viscous import viscous_rhs


def build_rhs(case, args, execution=None):
    """根据算例和命令行参数构造单个 RK 子步的右端项函数 ``L(q)``。

    返回的 JIT 函数依次组合物理边界、无粘通量散度、可选粘性项、通用体积力
    和槽道专用源项，但不推进流场。激波滤波属于整步后处理，不在这个函数内。
    """
    desc = from_case(case)
    if execution is None:
        execution = boundary.build_execution(case, desc)
    grid = desc.grid
    gamma = case["gamma"]
    params = case["params"]
    bcs = execution.bcs
    dxs = desc.dxs
    axes = desc.axes

    weno_tv = None
    if args.scheme == "WENO7-SYMBO" and args.weno_tv:
        from .schemes import WENO7_TV_LIMITER_DEFAULT, WENO7_TV_MAX_DEFAULT
        weno_tv = (WENO7_TV_LIMITER_DEFAULT, WENO7_TV_MAX_DEFAULT)

    vis_order = {"CD6": 6, "CD8": 8}[args.vis_order]
    src = case.get("source")
    metrics = desc.metrics
    dif_bound = desc.dif_bound
    dif_axis_map = desc.dif_axis_map
    plate_ghosts = args.case == "plate"

    def rhs(q):
        """计算当前 RK 子步状态 ``q`` 的总右端项 ``du = L(q)``。"""
        q = execution.apply(q)   # 与原码 OCFD_bc 同位置：每子步算 du 前先上壁面条件
        du = inviscid_rhs(q, grid, gamma=gamma, bcs=bcs, axes=axes, dx=dxs,
                          scheme=args.scheme, weno_tv=weno_tv, metrics=metrics,
                          dif_bound=dif_bound, zero_state_ghosts=plate_ghosts,
                          dif_axis_map=dif_axis_map)
        if args.viscous or args.case in ("poiseuille", "channel"):
            # viscous 的 dxs 是导数步长，必须全部合法
            du = du + viscous_rhs(q, grid, params, bcs, axes=axes, dx=dxs,
                                  order=vis_order, metrics=metrics,
                                  dif_bound=dif_bound, zero_state_ghosts=plate_ghosts,
                                  dif_axis_map=dif_axis_map)
        du = du + apply_forcing(case, q)
        if src is not None:
            du = du + src(q)         # IBC=200 体积力（channel_body_force_kernel）
        return du

    return jax.jit(rhs)


def run(args):
    """执行一次完整计算：设置精度、装配初值与 RHS，然后推进、滤波、诊断并存盘。

    ``args`` 通常由 :func:`main` 的命令行解析器生成；计算到 ``end_time`` 或
    ``end_step`` 中任一条件满足时停止。输出只写物理内区，不包含鬼胞。
    """
    if getattr(args, "fp32", False):
        config.disable_x64()
    else:
        config.enable_x64()   # 缺省 FP64

    gamma = 1.4
    case = build_case(args)
    desc = from_case(case)
    execution = boundary.build_execution(case, desc)
    grid = desc.grid
    dxs = desc.dxs
    params = case["params"]

    rhs = build_rhs(case, args, execution=execution)
    cons = case["cons0"]
    bcs = execution.bcs
    bc_apply = execution.callback

    dt_mode = f"fixed({args.dt})" if args.dt is not None else \
        f"adaptive(cfl={args.cfl})"
    if args.dt is None:
        def dt_fn_(q):
            """按当前流场和 CFL 数计算自适应时间步长。"""
            return compute_dt(q, grid, gamma, args.cfl, dxs)

        dt_fn = jax.jit(dt_fn_)
    else:
        dt_fn = None
    rhs(cons)
    if dt_fn is not None:
        dt_fn(cons)   # 预热编译

    print(f"case={args.case} grid={grid.nx}x{grid.ny}x{grid.nz} "
          f"scheme={args.scheme} "
          f"viscous={args.viscous} vis-order={args.vis_order} "
          f"precision={config.precision_name()} "
          f"Re={params.Re} Ama={params.Ama} Ref_T={params.Ref_T} Pr={params.Pr} "
          f"metrics={'on' if desc.has_metrics else 'off(Ajac=1)'} "
          f"dt-mode={dt_mode}")
    if (args.viscous or args.case == "poiseuille") and args.ref_t is None:
        print("WARNING: Ref_T 用了缺省 288.15 K（空气）。Ref_T 是有量纲参考温度"
              "（对应原码 T_REF），只影响 μ 的温度曲线（μ(T=1)=1/Re 与它无关）；"
              "与原码对拍时必须用 --ref-t 给出 in 文件里的同值。")
    if args.case == "sod" and args.viscous:
        print("WARNING: sod 初值用 E=p/(gamma-1) 标准约定，与 viscous 的 "
              "OCFD Ama 缩放约定不匹配——粘性系数无物理意义（不 raise，"
              "保留灵活性）")

    from .filtering import shock_filter
    t0 = time.time()
    t = 0.0
    step = 0
    # 定 dt 时可以 jit 整步 + donate 输入缓冲（省一次全场分配，也把 3 次 RHS
    # dispatch 合成 1 次）。变 dt 时 dt 是参数，仍走同一编译体（不重新编译）。
    step_jit = jax.jit(lambda q, dt_: rk3_step(q, dt_, rhs, bc_apply=bc_apply),
                       donate_argnums=(0,))
    while t < args.end_time and step < args.end_step:
        if args.dt is not None:
            dt = min(args.dt, args.end_time - t)
        else:
            dt = float(dt_fn(cons))
            if not (np.isfinite(dt) and dt > 0.0):
                print(f"bad dt={dt} at step {step}, t={t}")
                break
            dt = min(dt, args.end_time - t)
        cons = step_jit(cons, dt)
        t += dt
        step += 1
        if args.filter and step % args.nstep_filter == 0:
            d_ = cons[0]
            kin = 0.5 * (cons[1] ** 2 + cons[2] ** 2 + cons[3] ** 2) / d_
            p_ = (gamma - 1.0) * (cons[4] - kin)          # 压力标量场（3 维）
            cons = shock_filter(cons, p_, grid, s0=args.filter_s0,
                                rth=args.filter_rth, bcs=bcs,
                                axes=desc.axes)
        if step % args.kstep_show == 0:
            print(diagnostic_line(cons, grid, t, step, time.time() - t0))
        if step % args.kstep_save == 0 or t >= args.end_time:
            io.save_npz(
                f"{args.out_dir}/{args.case}_step{step:06d}.npz",
                {f"cons_{i}": np.asarray(cons[grid.inner()][i]) for i in range(5)},
                meta={"t": t, "step": step, "dt": dt, "case": args.case,
                      "N": args.N, "scheme": args.scheme, "gamma": gamma,
                      "Re": args.Re, "Ama": args.Ama,
                      "Ref_T": params.Ref_T, "Pr": params.Pr})
    print(f"done: t={t:.4f}, steps={step}, wall={time.time() - t0:.2f}s")


def main():
    """解析命令行参数、创建输出目录，并把参数交给 :func:`run` 执行。"""
    p = argparse.ArgumentParser(description="opencfd_jax runner")
    p.add_argument("--case", default="sod",
                   choices=["sod", "tg", "poiseuille", "channel", "plate"])
    p.add_argument("--plate-x0", type=float, default=0.05)
    p.add_argument("--plate-x1", type=float, default=0.25)
    p.add_argument("--plate-h", type=float, default=0.12)
    p.add_argument("--plate-dif-bound", default="0 0 0 0 1 0")
    p.add_argument("--in", dest="infile", default=None,
                   help="原码 in 文件路径（case=channel 必需，维数与物性都从它读）")
    p.add_argument("--field", default=None,
                   help="原码场文件（opencfd.dat / OCFD000xx.dat），case=channel 初场")
    p.add_argument("--mesh", default=None,
                   help="原码 OCFD3d-Mesh.dat；给出则按 JAC_BOUND=1 算度量并接进差分主链，"
                        "不给等于 JAC_BOUND=0（Ajac=1）")
    p.add_argument("--N", type=int, default=256, help="x 方向网格")
    p.add_argument("--ny", type=int, default=64, help="poiseuille 的 y 网格")
    p.add_argument("--nz", type=int, default=4,
                   help="z 向网格数（原码 in 文件 GRID_3D 的第三个数）")
    p.add_argument("--scheme", default="UP7",
                   choices=["UP7", "WENO7-SYMBO", "NND2", "WENO5", "WENO7",
                            "OMP6-1", "OMP6-2", "OMP6-3",
                            "OMP6_HR", "OMP6_LD", "OMP6_CD8"])
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--dt", type=float, default=None,
                   help="固定时间步长（不给则用 CFL 自适应）")
    p.add_argument("--weno-tv", action="store_true",
                   help="WENO7-SYMBO 启用 TV 快速路径（光滑区线性权）")
    p.add_argument("--end-time", type=float, default=0.2)
    p.add_argument("--end-step", type=int, default=10 ** 9)
    p.add_argument("--Re", type=float, default=1.0e3)
    p.add_argument("--Ama", type=float, default=0.1)
    p.add_argument("--ref-t", type=float, default=None,
                   help="有量纲参考温度(K)，对应原码 in 文件的 T_REF"
                        "（缺省 288.15 K，与原码 read_parameters 的缺省一致）")
    p.add_argument("--pr", type=float, default=0.7, help="Prandtl 数（原码 PR）")
    p.add_argument("--vis-order", default="CD6", choices=["CD6", "CD8"],
                   help="粘性项导数格式（原码 SCHEME_VIS）")
    p.add_argument("--forcing-work", default="on", choices=["on", "off"],
                   help="体积力是否做功能量源 rho*u*G（原码 channel 模块没有该项，"
                        "对拍请用 off；run 的物理缺省为 on）")
    p.add_argument("--p-grad", type=float, default=0.1,
                   help="poiseuille 体积力（单位质量）")
    p.add_argument("--viscous", action="store_true")
    p.add_argument("--filter", action="store_true",
                   help="启用激波滤波（每 nstep-filter 步一次）")
    p.add_argument("--nstep-filter", type=int, default=100)
    p.add_argument("--filter-s0", type=float, default=1.0)
    p.add_argument("--filter-rth", type=float, default=1.0e-5)
    p.add_argument("--kstep-show", type=int, default=50)
    p.add_argument("--kstep-save", type=int, default=500)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--fp64", action="store_true",
                   help="FP64（现在已是缺省，保留仅为兼容旧命令）")
    p.add_argument("--fp32", action="store_true",
                   help="切到 FP32（仅性能基准用；对拍/验证不要用）")
    args = p.parse_args()
    if getattr(args, "fp32", False):
        config.disable_x64()
    else:
        config.enable_x64()   # 缺省 FP64
    os.makedirs(args.out_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
