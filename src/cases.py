"""算例注册表与初值（从旧入口模块抽出；对应 OCFD_init.cu 的 init + 各算例初值）。

每个 case 返回一个 dict，核心键包括：grid / cons0 / bcs / dxs / axes / params /
forcing / gamma；按算例还可包含 bc_apply / source / metrics / dif_bound /
plate 等边界、源项、曲线网格或几何信息。主循环据此统一装配计算流程。
"""

import numpy as np

import jax.numpy as jnp

from .boundary import FARFIELD
from .fields import INFLOW, OUTFLOW, PERIODIC, PLATE, WALL, Grid
from .viscous import PhysParams


def _default_farfield(params):
    """CGNS 没有 ``BCDataSet`` 时使用的默认自由来流状态。

    默认来流沿物理 x 正方向：rho=1、u=1、v=w=0、T=1；总能量按
    ``Cv*T + 0.5*u^2`` 组装。它只用于 CGNS 远场面的缺失来流参数，
    不替代每面不同的 ``BCDataSet``。
    """
    rho, u, T = 1.0, 1.0, 1.0
    E = params.Cv * T + 0.5 * u * u
    return jnp.asarray([rho, rho * u, 0.0, 0.0, rho * E],
                       dtype=jnp.float64)


def _dimension(args, name, default):
    """读取一个网格尺寸；缺省时返回算例的默认值。"""
    value = getattr(args, name, None)
    return default if value is None else value


def make_initial_sod(grid, dx, x_diaphragm=0.5, gamma=1.4):
    """生成 Sod 激波管初值：在隔膜两侧分别写入密度、零速度和压力对应的总能量。

    返回包含鬼胞的守恒量数组；x 以外的分量保持零，供一维无粘验证使用。
    """
    c = np.arange(grid.ni + 8)
    x_cell = (c - 4 + 0.5) * dx
    left = x_cell < x_diaphragm
    cons = grid.alloc()
    cons = cons.at[0].set(jnp.where(left, 1.0, 0.125))
    cons = cons.at[4].set(jnp.where(left, 1.0, 0.1) / (gamma - 1.0))
    return cons


def make_initial_poiseuille(grid, params, p_grad, gamma=1.4):
    """层流泊肃叶初值 + 恒定体积力驱动（槽道层流验证）。

    可压低 Ma 近似（Ma^2 误差）：密度 1、T=1 均匀；体积力 G（x 向，
    单位质量）驱动，稳态精确解（不可压）：
        u(y) = G/(2*nu) * y*(h-y)
    粘性 nu = mu(T=1)/rho。
    """
    ni = grid.ni
    nj = grid.nj
    dy = 1.0 / nj                 # y 域 [0,1]
    a = np.arange(nj + 8)
    y_cell = (a - 4 + 0.5) * dy
    nu = params.amu_C0 * 1.0 ** 1.5 / (params.Tsb + 1.0)   # T=1, rho=1
    u_y = p_grad / (2.0 * nu) * y_cell * (1.0 - y_cell)    # 抛物线（稳态剖面）
    u = np.broadcast_to(u_y[:, None], (nj + 8, ni))          # (y, x)，x 向均匀
    cons = grid.alloc()
    cons = cons.at[0].set(1.0)
    cons = cons.at[(1,) + grid.inner()[1:]].set(
        jnp.broadcast_to(jnp.asarray(u[4:4 + nj])[None], (grid.nk, nj, ni)))
    # E = Cv*T + 0.5*rho*V^2（T=1）
    E = params.Cv * 1.0 + 0.5 * 1.0 * jnp.asarray(u) ** 2
    cons = cons.at[(4,) + grid.inner()[1:]].set(
        jnp.broadcast_to(E[4:4 + nj][None], (grid.nk, nj, ni)))
    return cons, dy


def make_initial_tg(grid, params, A=1.0):
    """生成二维 Taylor–Green 涡初值，并返回含鬼胞的守恒量数组。

    密度和温度均取 1，速度由周期正余弦剖面给出，总能量包含内能与动能。
    """
    ni, nj = grid.ni, grid.nj
    dx = 2.0 * np.pi / ni
    ix = (np.arange(ni) + 0.5) * dx
    iy = (np.arange(nj) + 0.5) * dx
    u = A * np.cos(ix)[None, :] * np.sin(iy)[:, None]
    v = -A * np.sin(ix)[None, :] * np.cos(iy)[:, None]
    cons = grid.alloc()
    cons = cons.at[0].set(1.0)
    cons = cons.at[(1,) + grid.inner()[1:]].set(
        jnp.broadcast_to(u[None], (grid.nk, nj, ni)))
    cons = cons.at[(2,) + grid.inner()[1:]].set(
        jnp.broadcast_to(v[None], (grid.nk, nj, ni)))
    w2 = jnp.asarray(u)[None] ** 2 + jnp.asarray(v)[None] ** 2
    E = params.Cv * 1.0 + 0.5 * 1.0 * w2
    cons = cons.at[(4,) + grid.inner()[1:]].set(
        jnp.broadcast_to(E, (grid.nk, nj, ni)))
    return cons


def build_case(args):
    """按 ``args.case`` 创建网格、物性、初始守恒量、边界和源项的完整算例字典。

    后续主循环只读取这个字典，不直接了解各算例的初始化细节；channel 还会
    读取原码 in/场/网格文件，plate 则附带边界层几何与来流信息。
    """
    gamma = 1.4
    # 横向/展向网格数：原码由 in 文件 GRID_3D 决定，这里不再写死 4。
    nk = int(_dimension(args, "nk", 4) or 4)
    # Ref_T 是**有量纲的参考温度（K）**，与原码 in 文件的 T_REF 同义：
    # Tsb = 110.4/Ref_T、amu_C0 = (1+110.4/Ref_T)/Re，所以 μ(T=1)=1/Re 与
    # Ref_T 无关，但 T≠1 的 Sutherland 曲线随 Ref_T 变——原码对拍必须同值。
    # 缺省 288.15 K 取的是原码 read_parameters（parameters.c:290）在
    # T_REF<=0 时用的空气参考温度；算例不同请用 --ref-t 显式覆盖。
    ref_t = getattr(args, "ref_t", None)
    if ref_t is None:
        ref_t = 288.15
    params = PhysParams(gamma=gamma, Ama=args.Ama, Re=args.Re,
                        Ref_T=ref_t, Pr=getattr(args, "pr", 0.7))

    if args.case == "sod":
        # 初值约定：E = p/(gamma-1)（标准可压 NS 约定，T 由该式隐含）。
        # 与 viscous.py 的 OCFD Ama 缩放约定（T = E_int/Cv）不匹配——
        # sod + --viscous 时 mu 由该 T 算出，无物理意义（main.py 有警告）。
        # 对拍/研究粘性请用 tg 或 poiseuille 算例。
        ni = int(_dimension(args, "ni", 256))
        nj = int(_dimension(args, "nj", 4) or 4)
        grid = Grid(ni=ni, nj=nj, nk=nk)
        dx = 1.0 / ni
        cons = make_initial_sod(grid, dx, gamma=gamma)
        return dict(grid=grid, cons0=cons, bcs=(OUTFLOW, PERIODIC, PERIODIC),
                    dxs=(dx, dx, dx), axes=(0,), params=params,
                    forcing=None, gamma=gamma)

    if args.case == "tg":
        ni = int(_dimension(args, "ni", 256))
        nj = int(_dimension(args, "nj", ni))
        grid = Grid(ni=ni, nj=nj, nk=nk)
        dx = 2.0 * np.pi / ni
        cons = make_initial_tg(grid, params)
        return dict(grid=grid, cons0=cons, bcs=(PERIODIC,) * 3,
                    dxs=(dx, dx, 0.0), axes=(0, 1), params=params,
                    forcing=None, gamma=gamma)

    if args.case == "poiseuille":
        mesh_format = getattr(args, "mesh_format", "builtin")
        if mesh_format == "cgns":
            from . import mesh_cgns
            if not getattr(args, "mesh", None):
                raise ValueError("mesh_format=cgns 需要 --mesh / mesh.file")
            zone = mesh_cgns.read_cgns_zone(
                args.mesh, zone=getattr(args, "cgns_zone", None), validate=True)
            explicit = getattr(args, "config_sources", {})
            for name, got in (("ni", zone.ni), ("nj", zone.nj), ("nk", zone.nk)):
                value = getattr(args, name, None)
                if explicit.get(name) == "cli" and int(value) != got:
                    raise ValueError(
                        f"显式 --{name}={value} 与 CGNS VertexSize 的 {got} 不一致")
            # CGNS 路径边条驱动（4.9.13 口径）：DIF_BOUND 由程序按 kind 派生，
            # 不接受手填六元组——显式拒绝而非静默忽略（2026-11-23 Astra 复审）。
            if getattr(args, "plate_dif_bound", None) is not None:
                raise ValueError(
                    "--plate-dif-bound 只用于内置 plate；CGNS 网格的边界带"
                    "由边条 kind 派生（WALL/FARFIELD 开带），不支持手填")
            block = mesh_cgns.build_cgns_block(zone, axes=(0, 1))
            grid = block.grid
            G = args.p_grad
            cons, dy = make_initial_poiseuille(grid, params, G, gamma)
            has_farfield = any(f.kind == FARFIELD for f in block.faces)
            farfield = _default_farfield(params) if has_farfield else None
            fw = getattr(args, "forcing_work", True)
            if isinstance(fw, str):
                fw = fw.strip().lower() not in ("off", "0", "false", "no")
            return dict(grid=grid, cons0=cons, bcs=block.bcs,
                        dxs=block.dxs, axes=(0, 1), params=params,
                        faces=block.faces,
                        forcing=("x", G), forcing_work=bool(fw), gamma=gamma,
                        metrics=block.metrics, coords=block.coords,
                        dif_bound=block.dif_bound,
                        dif_axis_map=block.dif_axis_map,
                        farfield=farfield)
        ni = int(_dimension(args, "ni", 256))
        nj = int(_dimension(args, "nj", 64))
        grid = Grid(ni=ni, nj=nj, nk=nk)
        dx = 2.0 * np.pi / ni          # 流向周期域 [0,2pi]
        G = args.p_grad               # 体积力（单位质量）
        cons, dy = make_initial_poiseuille(grid, params, G, gamma)
        fw = getattr(args, "forcing_work", True)
        if isinstance(fw, str):
            fw = fw.strip().lower() not in ("off", "0", "false", "no")
        return dict(grid=grid, cons0=cons, bcs=(PERIODIC, WALL, PERIODIC),
                    dxs=(dx, 1.0 / nj, dx), axes=(0, 1), params=params,
                    forcing=("x", G), forcing_work=bool(fw), gamma=gamma)

    if args.case == "channel":
        # IBC=200 槽道：维数与物性全部来自原码 in 文件，初场可读原码 opencfd.dat。
        from . import channel, infile, mesh
        if not getattr(args, "infile", None):
            raise ValueError(
                "case=channel 需要 .in 文件（--in / mesh.infile）；"
                "Step 3 的 JSON 不提供网格维数、物性、壁温与边界开关")
        inp = infile.read_ocfd_in(args.infile)
        ni, nj, nk, _ = infile.grid_dims_ijk(inp)
        s = infile.summarize(inp)
        params = PhysParams(gamma=s.get("GAMMA", 1.4), Ama=s["AMA"],
                            Re=s["RE"], Ref_T=s["T_REF"],
                            Pr=s.get("PR", 0.7))
        grid = Grid(ni=ni, nj=nj, nk=nk)
        bc = channel.ChannelBC.from_infile(
            inp, work=getattr(args, "forcing_work", "off") == "on")
        if getattr(args, "field", None):
            prim = mesh.read_ocfd_field(args.field, ni=ni, nj=nj, nk=nk)
            cons = channel.cons_from_primitive(prim, grid, params)
        else:
            ones = jnp.ones((nk, nj, ni))
            zeros = jnp.zeros((nk, nj, ni))
            cons = channel.cons_from_primitive(
                (ones, zeros, zeros, zeros, ones), grid, params)
        # 原码步长口径：hx = 1/(NX-1)（OCFD_init.cu:29-31）
        dxs = (1.0 / (ni - 1.0), 1.0 / (nj - 1.0), 1.0 / (nk - 1.0))
        # 原码的边界带开关：JAC_BOUND（度量，3 个）/ DIF_BOUND（导数+面通量，6 个）
        def _ints(key, default):
            """把 in 文件中的整数列表读出为元组，缺省时使用调用方给定值。"""
            raw = inp.get(key, default)
            return tuple(int(float(x)) for x in raw.replace(",", " ").split())
        jac_bound = _ints("JAC_BOUND", "1 1 1")[:3]
        # DIF_BOUND 缺省 = 全零，对齐原版（parameters.c 的 D0_bound 为静态
        # 存储期数组，缺键时不写入、六位保持零）。槽道壁面带需 in 文件显式
        # 给出（库内 channel in 均显式写了 "0 0 1 1 0 0" 类值）。
        dif_bound = _ints("DIF_BOUND", "0 0 0 0 0 0")[:6]
        bcs = (PERIODIC, WALL, PERIODIC)
        periodic = tuple(b == PERIODIC for b in bcs)
        # 度量：给了 --mesh 就用 OCFD3d-Mesh.dat 算（JAC_BOUND=1 的原码路径）；
        # 不给就等于 JAC_BOUND=0（Ajac=1，直角假设）。
        metrics = None
        coords = None
        if getattr(args, "mesh", None):
            coords = mesh.read_ocfd_mesh(args.mesh, ni=ni, nj=nj, nk=nk)
            padded = {k: mesh.pad_coords(v, periodic=periodic)
                      for k, v in coords.items()}
            metrics = mesh.compute_metrics_from_coords(
                padded, dxs, order=6, jac_bound=jac_bound, periodic=periodic)
        return dict(grid=grid, cons0=cons, bcs=bcs,
                    dxs=dxs, axes=(0, 1, 2), params=params, forcing=None,
                    gamma=params.gamma,
                    bc_apply=lambda q: bc.apply(q, grid, params),
                    source=lambda q: bc.source(q, grid),
                    metrics=metrics, coords=coords, dif_bound=dif_bound,
                    channel_bc=bc, scheme_invis=s.get("SCHEME_INVIS"),
                    scheme_vis=s.get("SCHEME_VIS"))

    if args.case == "plate":
        # 平板层流边界层（JAX 自身正确性验证）：均匀来流（入口=顶面=远场），
        # 底板无滑移等温壁，x 上侧零梯度，z 周期。边界层从入口（前缘）自然发展。
        # 无量纲口径：ρ∞=T∞=U∞=1 ⇒ μ(T=1)=1/Re（与 Ref_T 无关）、ν=1/Re、
        # Re_x = x·Re、M∞ = U∞/sqrt(γ R T) = Ama。验收基准是 blasius.py。
        from . import blasius, channel, inflow, plate
        if getattr(args, "mesh_format", "builtin") == "cgns":
            from dataclasses import replace
            from . import mesh_cgns
            if getattr(args, "plate_dif_bound", None) is not None:
                raise ValueError(
                    "--plate-dif-bound 只用于内置 plate；CGNS 网格的边界带"
                    "由边条 kind 派生（WALL 开带、FARFIELD 关带），不支持手填")
            keys = ("boundary_inflow", "boundary_farfield", "boundary_wall_temperature")
            missing = [key for key in keys if getattr(args, key, None) is None]
            if missing:
                raise ValueError(f"plate CGNS 缺少 boundary 参数: {missing}")
            if args.boundary_wall_temperature <= 0:
                raise ValueError("plate CGNS 等温壁的 boundary.wall_temperature 必须大于 0")
            zone = mesh_cgns.read_cgns_zone(args.mesh, zone=getattr(args, "cgns_zone", None))
            explicit = getattr(args, "config_sources", {})
            for name, got in (("ni", zone.ni), ("nj", zone.nj), ("nk", zone.nk)):
                value = getattr(args, name, None)
                if explicit.get(name) == "cli" and int(value) != got:
                    raise ValueError(
                        f"显式 --{name}={value} 与 CGNS VertexSize 的 {got} 不一致")
            block = mesh_cgns.build_cgns_block(zone, axes=(0, 1, 2))
            grid = block.grid
            def state(value):
                rho, u, v, w, T = (value[k] for k in ("rho", "u", "v", "w", "T"))
                return jnp.asarray([rho, rho*u, rho*v, rho*w,
                                    rho*(params.Cv*T + 0.5*(u*u + v*v + w*w))])
            initial = state(args.boundary_inflow)
            cons = grid.alloc().at[grid.inner()].set(initial.reshape(5, 1, 1, 1))
            inlet = initial.reshape(5, 1, 1)
            far = state(args.boundary_farfield)
            faces = tuple(replace(f, thermal="isothermal",
                                  params={"Tw": args.boundary_wall_temperature})
                          if f.kind == WALL else
                          replace(f, dif_band=False) if f.kind == FARFIELD else f
                          for f in block.faces)
            return dict(grid=grid, cons0=cons, bcs=block.bcs, faces=faces,
                        dxs=block.dxs, axes=block.axes, params=params,
                        forcing=None, gamma=params.gamma, metrics=block.metrics,
                        coords=block.coords, dif_bound=block.dif_bound,
                        dif_axis_map=block.dif_axis_map, inflow=inlet, farfield=far)
        ni = int(_dimension(args, "ni", 256))
        nj = int(_dimension(args, "nj", 64))
        x0 = float(getattr(args, "plate_x0", 0.1))
        x1 = float(getattr(args, "plate_x1", 1.1))
        H = float(getattr(args, "plate_h", 0.12))
        U, rho0, T0 = 1.0, 1.0, 1.0
        grid = Grid(ni=ni, nj=nj, nk=nk)
        # 原码壁面在**格心**：第一个内区格（j=LAP）即 y=0，其上依次 dy,2dy…
        dx, dy = (x1 - x0) / ni, H / (nj - 1)
        xc = x0 + (np.arange(ni) + 0.5) * dx
        yc = np.arange(nj) * dy
        nu = 1.0 / params.Re
        # 初值：全场均匀来流（前缘在 x=x0 入口），边界层由入口向下游发展。
        ones = np.ones((nk, nj, ni))
        zeros = np.zeros((nk, nj, ni))
        cons = channel.cons_from_primitive((rho0 * ones, U * ones, zeros, zeros,
                                           T0 * ones), grid, params)
        # 壁面温度：由 input.json 的 physics.tw 给；缺省与来流同温（绝热壁极限）。
        tw = getattr(args, "tw", None)
        tw = T0 if tw is None else float(tw)
        # 入口：给了剖面文件就按剖面（OCFD 1D 格式，与格点逐点对应、不插值），
        # 空着就是均匀来流常数；两种口径都只依赖壁法向 y。
        prof_path = getattr(args, "inflow_profile", None)
        if prof_path:
            prof = jnp.asarray(inflow.read_inflow_profile(prof_path, nj,
                                                          params.Cv))
            inflow_state = jnp.broadcast_to(prof[:, None, :], (5, nk, nj))
        else:
            inflow_state = plate.inflow_from_profile(
                rho0 * np.ones((nk, nj)), U * np.ones((nk, nj)),
                np.zeros((nk, nj)), np.zeros((nk, nj)),
                T0 * np.ones((nk, nj)), params.Cv)
        far = plate.far_state(rho=rho0, u=U, T=T0, Cv=params.Cv)
        raw = getattr(args, "plate_dif_bound", None)
        # 显式六开关逐侧生效；缺省按算例：plate 取 OCFD 口径 0 0 0 0 1 0
        # （远场不开带），其余算例 None = 按 kind 派生（2026-11-23：CLI 缺省
        # 曾全局生效，CGNS poiseuille 误吃 plate 口径）
        dif_bound = (0, 0, 0, 0, 1, 0) if args.case == "plate" else None
        if raw:
            dif_bound = tuple(int(float(t)) for t in str(raw).replace(",", " ").split())[:6]
        return dict(grid=grid, cons0=cons, bcs=(INFLOW, PLATE, PERIODIC),
                    dxs=(dx, dy, dx), axes=(0, 1), params=params,
                    forcing=None, gamma=params.gamma,
                    bc_apply=lambda q: plate.apply(q, grid, params,
                                                   far=far, Tw=tw),
                    inflow=inflow_state,
                    metrics=None, dif_bound=dif_bound, dif_axis_map=(0, 2, 1),
                    plate=dict(x0=x0, x1=x1, H=H, dx=dx, dy=dy, xc=xc, yc=yc,
                               U=U, nu=nu, inflow=inflow_state, far=far, Tw=tw),
                    farfield=far)

    raise ValueError(f"unknown case: {args.case}")


def apply_forcing(case, q):
    """体积力（单位质量 f）：动量源 rho*f，可选能量源 rho*f*u。

    forcing_work=True（缺省，可压 NS 的标准源项）：du[1] += rho*G、du[4] += rho*u*G。
    forcing_work=False：只加动量源，与当前 channel_case_iiA 的原码
    channel_body_force_kernel（channel_flow.cu:110-120 只做 du[1] += F*rho）
    逐项一致——对拍用这一档，否则 du[4] 必然差 rho*u*G。
    """
    f = case.get("forcing")
    if f is None:
        return jnp.zeros_like(q)
    direction, G = f
    assert direction == "x", "only x forcing implemented"
    du = jnp.zeros_like(q)
    du = du.at[1].set(q[0] * G)                       # rho*G
    if case.get("forcing_work", True):
        du = du.at[4].set(q[1] * G)                   # rho*u*G（做功项）
    return du
