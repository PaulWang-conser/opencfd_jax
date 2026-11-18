"""正式运行入口（对应 OCFD_NS_Solver.cu 的 NS_solver_real 主循环）。

用法（从工作区根目录运行，当前源码包路径为 opencfd_jax.src）：
    python -m opencfd_jax.src.main --case sod --ni 256 --scheme UP7 --end-time 0.2
    python -m opencfd_jax.src.main --case tg  --ni 64  --scheme UP7 --end-time 0.5 --Re 50
    python -m opencfd_jax.src.main --case poiseuille --ni 32 --nj 64 --p-grad 0.1 \
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

from . import boundary, config, init, io, runtime_config
from .ana import check_nan, diagnostic_line
from .cases import apply_forcing
from .mesh_data import from_case
from .solver import inviscid_rhs, rk3_step
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
                                  dif_axis_map=dif_axis_map, boundary_execution=execution)
        du = du + apply_forcing(case, q)
        if src is not None:
            du = du + src(q)         # IBC=200 体积力（channel_body_force_kernel）
        return du

    return jax.jit(rhs)


def build_parser(explicit_only=False):
    """构造命令行解析器。

    ``explicit_only=True`` 时所有选项的缺省值改为 ``SUPPRESS``，用于区分
    “用户显式给了这个参数”与“参数取了缺省值”，从而让 JSON 配置只在用户
    没有显式覆盖时生效。两套解析器共用同一份参数定义，避免默认值漂移。
    """
    p = argparse.ArgumentParser(description="opencfd_jax runner")

    def add(*names, **kwargs):
        if explicit_only:
            kwargs["default"] = argparse.SUPPRESS
        p.add_argument(*names, **kwargs)

    add("--config", default=None,
        help="input.json 运行配置；命令行显式参数优先于 JSON")
    add("--case", default="sod",
        choices=runtime_config.CASE_NAMES)
    add("--plate-x0", type=float, default=0.05)
    add("--plate-x1", type=float, default=0.25)
    add("--plate-h", type=float, default=0.12)
    add("--plate-dif-bound", default="0 0 0 0 1 0")
    add("--mesh-format", default=None, choices=runtime_config.MESH_FORMATS,
        help="网格格式：builtin（程序内置网格）或 cgns（单块 CGNS/HDF5）")
    add("--mesh", default=None,
        help="CGNS/HDF5 单块网格路径；给出时默认按 cgns 读取")
    add("--cgns-zone", default=None,
        help="CGNS 文件包含多个 zone 时必须显式给出 zone 名")
    add("--ni", type=int, default=256, help="计算 i 方向网格")
    add("--nj", type=int, default=64, help="计算 j 方向网格")
    add("--nk", type=int, default=4, help="计算 k 方向网格")
    add("--scheme", default="UP7",
        choices=runtime_config.SCHEMES)
    add("--dt", type=float, default=None,
        help="固定时间步长（必填；不再提供 CFL 自适应）")
    add("--weno-tv", action="store_true",
        help="WENO7-SYMBO 启用 TV 快速路径（光滑区线性权）")
    add("--no-weno-tv", action="store_true",
        help="显式关闭 WENO7-SYMBO 的 TV 快速路径（覆盖 JSON 的 true）")
    add("--end-time", type=float, default=0.2)
    add("--end-step", type=int, default=10 ** 9)
    add("--Re", type=float, default=1.0e3)
    add("--Ama", type=float, default=0.1)
    add("--ref-t", type=float, default=None,
        help="有量纲参考温度(K)，对应原码 in 文件的 T_REF"
             "（缺省 288.15 K，与原码 read_parameters 的缺省一致）")
    add("--pr", type=float, default=0.7, help="Prandtl 数（原码 PR）")
    add("--vis-order", default="CD6", choices=runtime_config.VIS_ORDERS,
        help="粘性项导数格式（原码 SCHEME_VIS）")
    add("--forcing-work", default="on", choices=["on", "off"],
        help="体积力是否做功能量源 rho*u*G（原码 channel 模块没有该项，"
             "对拍请用 off；run 的物理缺省为 on）")
    add("--p-grad", type=float, default=0.1,
        help="poiseuille 体积力（单位质量）")
    add("--viscous", action="store_true")
    add("--no-viscous", action="store_true",
        help="显式关闭粘性项（覆盖 JSON 的 true；channel/poiseuille 仍强制粘性）")
    add("--filter", action="store_true",
        help="启用激波滤波（每 nstep-filter 步一次）")
    add("--no-filter", action="store_true",
        help="显式关闭激波滤波（覆盖 JSON 的 true）")
    add("--nstep-filter", type=int, default=100)
    add("--filter-s0", type=float, default=1.0)
    add("--filter-rth", type=float, default=1.0e-5)
    add("--kstep-show", type=int, default=50)
    add("--kstep-save", type=int, default=500)
    add("--out-dir", default="output")
    add("--fp64", action="store_true",
        help="FP64（现在已是缺省，保留仅为兼容旧命令）")
    add("--fp32", action="store_true",
        help="切到 FP32（仅性能基准用；对拍/验证不要用）")
    return p


_NEGATIVE_FLAGS = (("no_viscous", "viscous"), ("no_filter", "filter"),
                   ("no_weno_tv", "weno_tv"))


def resolve_cli(parsed, parser):
    """把一次显式命令行解析结果整理成配置合并所需的字典。

    负向开关（``--no-*``）在这里折叠成对应布尔参数；互相冲突的显式开关直接报错。
    返回 ``(config_path, explicit)``。
    """
    explicit = dict(parsed)
    config_path = explicit.pop("config", None)
    for negative, positive in _NEGATIVE_FLAGS:
        if explicit.pop(negative, False):
            if explicit.get(positive):
                parser.error(f"--{positive.replace('_', '-')} 与 "
                             f"--{negative.replace('_', '-')} 不能同时给出")
            explicit[positive] = False
    return config_path, explicit


def _warn_forced_viscous(args):
    """poiseuille 在 RHS 里强制粘性；显式关闭时如实提示。"""
    if not getattr(args, "viscous", False) and args.case == "poiseuille":
        print(f"WARNING: {args.case} 强制粘性，显式 viscous=false 不会关闭粘性项")


def run(args):
    """执行一次完整计算：初始化状态、装配 RHS，然后推进、滤波、诊断并存盘。

    ``args`` 通常由 :func:`main` 的命令行解析器生成；计算到 ``end_time`` 或
    ``end_step`` 中任一条件满足时停止。输出只写物理内区，不包含鬼胞。
    """
    state = init.prepare(args)
    case = state.case
    desc = state.desc
    execution = state.execution
    cons = state.cons0
    gamma = 1.4
    grid = desc.grid
    params = case["params"]

    rhs = build_rhs(case, args, execution=execution)
    bcs = execution.bcs
    bc_apply = execution.callback

    if args.dt is None or not (np.isfinite(args.dt) and args.dt > 0.0):
        raise ValueError("固定 dt 必须显式给出，且为正的有限数")
    dt_mode = f"fixed({args.dt})"
    rhs(cons)

    prefix = getattr(args, "out_prefix", None) or args.case
    print(f"case={args.case} grid={grid.ni}x{grid.nj}x{grid.nk} "
          f"(ni/nj/nk) "
          f"scheme={args.scheme} "
          f"viscous={args.viscous} vis-order={args.vis_order} "
          f"precision={config.precision_name()} "
          f"Re={params.Re} Ama={params.Ama} Ref_T={params.Ref_T} Pr={params.Pr} "
          f"metrics={'on' if desc.has_metrics else 'off(Ajac=1)'} "
          f"dt-mode={dt_mode} out-prefix={prefix} "
          f"config={getattr(args, 'config_path', None) or 'cli'}")
    _warn_forced_viscous(args)
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
        dt = min(args.dt, args.end_time - t)
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
                f"{args.out_dir}/{io.checkpoint_name(prefix, step)}",
                {f"cons_{i}": np.asarray(cons[grid.inner()][i]) for i in range(5)},
                meta={"t": t, "step": step, "dt": dt, "case": args.case,
                      "N": grid.ni, "scheme": args.scheme,
                      "gamma": params.gamma, "Re": params.Re,
                      "Ama": params.Ama, "Ref_T": params.Ref_T,
                      "Pr": params.Pr})
    print(f"done: t={t:.4f}, steps={step}, wall={time.time() - t0:.2f}s")


def _validate_input_paths(args, parser):
    """启动前检查输入文件；避免进入 JIT 后才因缺文件报错。"""
    if args.dt is None or not (np.isfinite(args.dt) and args.dt > 0.0):
        parser.error("固定 dt 必须显式给出，且为正的有限数")
    if getattr(args, "mesh_format", None) == "cgns" \
            and not getattr(args, "mesh", None):
        parser.error("mesh_format=cgns 需要 --mesh / mesh.file")
    value = getattr(args, "mesh", None)
    if value and not os.path.isfile(value):
        parser.error(f"--mesh 指向的文件不存在或不可读：{value}")


def main(argv=None):
    """解析命令行与 JSON 配置，创建输出目录，并交给 :func:`run` 执行。"""
    defaults = build_parser().parse_args([])
    explicit_parser = build_parser(explicit_only=True)
    config_path, explicit = resolve_cli(
        vars(explicit_parser.parse_args(argv)), explicit_parser)

    try:
        cfg = runtime_config.load_config(
            config_path, case=explicit.get("case")) if config_path else None
        args = runtime_config.resolve_args(explicit, cfg, defaults)
    except runtime_config.ConfigError as exc:
        explicit_parser.error(f"配置错误：{exc}")

    _validate_input_paths(args, explicit_parser)
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        run(args)
    except ValueError as exc:
        explicit_parser.error(f"运行参数错误：{exc}")


if __name__ == "__main__":
    main()

