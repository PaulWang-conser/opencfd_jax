"""第六步测试的辅助模块。

三部分：
  - 验收网格：``mesh_path`` 按预设生成多块网格（一个进程里每个预设只生成一次，
    放系统临时目录），``sidecar`` 读旁记，``topology`` 读入并做拓扑校验，
    ``blocks`` 装配各块，``kernels`` 编求解函数；``reassemble`` 按旁记把各块拼回
    全局，``split_field`` 反过来把全局的场切给各块；
  - 等价的单块：``reference`` 和 ``periodic_reference`` 是单块参考，``plate_args``
    和 ``run_args`` 给运行参数，``formal_step`` 和 ``formal_trace`` 走正式入口；
    ``acceptance_state`` 是验收初场，``compare`` 按阈值比较（阈值见 ``RTOL``）；
  - 合成小网格的构造（``block``、``two_blocks``、``seam_blocks``、``t_blocks``），以及
    几个测试共用的小函数（``same_bits``、``with_record``、``interiors``）。

合成网格的三个方向间距不同（i 向 1、j 向 2、k 向 3），每个块又是均匀的长
方体，所以取错轴、取错层、取错供体都会在读回的坐标上露出来。每条轴都留到
至少 7 个顶点，因为物理面所在的轴要放度量的单侧带。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Grid.make_cgns_fixture import ConnSpec, ZoneSpec  # noqa: E402
from Grid.split_cgns import DEFAULT_SRC as SOURCE  # noqa: E402
from Grid.split_cgns import PRESETS  # noqa: E402
from opencfd_jax.src.mesh import COORD_KEYS as COORDS  # noqa: E402

SPACING = (1.0, 2.0, 3.0)
MESH_DIR = Path(tempfile.gettempdir()) / "opencfd_step6_meshes"
_MESHES: dict = {}         # 预设 → 网格路径
_TOPOLOGIES: dict = {}     # 预设 → 校验过的拓扑
_BUILT: dict = {}          # 预设 → 装配好的块；另有单块参考和正式入口的编译结果


def mesh_path(preset: str) -> Path:
    """预设网格的路径；第一次调用时生成，之后返回同一个路径。"""
    if preset not in _MESHES:
        from Grid.split_cgns import write_preset
        _MESHES[preset] = write_preset(preset, MESH_DIR / preset)
    return _MESHES[preset]


def sidecar(preset: str) -> dict:
    """预设网格的旁记：源网格、全局顶点数和每块的全局起止下标。"""
    return json.loads(mesh_path(preset).with_suffix(".json").read_text(encoding="utf-8"))


def block_window(entry: dict, shift: int = 0) -> tuple:
    """旁记里一块的范围在全局数组上的切片，轴序 (k, j, i)。

    ``entry`` 是旁记 ``blocks`` 里的一项。``shift`` 同时加在起止上：含四层鬼胞的
    数组传 ``LAP``，全局内区的数组传 0。
    """
    k0, j0, i0 = (value + shift for value in entry["start"][::-1])
    k1, j1, i1 = (value + shift for value in entry["stop"][::-1])
    return (slice(k0, k1), slice(j0, j1), slice(i0, i1))


def reassemble(note: dict, pieces: dict) -> np.ndarray:
    """按旁记把各块顶点上的数组拼回全局。

    ``pieces`` 是 {块名: 数组}，数组的最后三个轴是 (k, j, i)，长度是各块的顶点数；
    前面可以再带别的轴，例如守恒量最前面的变量轴。切口那一层两块各有一份，后写
    的盖住先写的；要比各份副本就先比，再拼回。
    """
    nk, nj, ni = note["global_vertices"][::-1]
    first = next(iter(pieces.values()))
    lead = (slice(None),) * (first.ndim - 3)
    out = np.empty(first.shape[:-3] + (nk, nj, ni), dtype=first.dtype)
    for entry in note["blocks"]:
        out[lead + block_window(entry)] = pieces[entry["name"]]
    return out


def topology(preset: str):
    """读入预设网格并做拓扑校验，每个预设只做一次。"""
    if preset not in _TOPOLOGIES:
        from opencfd_jax.src import mesh_cgns, multiblock
        _TOPOLOGIES[preset] = multiblock.build_topology(
            mesh_cgns.read_cgns_zones(mesh_path(preset)))
    return _TOPOLOGIES[preset]


FLOW = {"rho": 1.0, "u": 1.0, "v": 0.0, "w": 0.0, "T": 1.0}


def plate_args(mesh) -> argparse.Namespace:
    """plate 的运行参数：无粘、WENO7、固定步长，网格取 ``mesh``。

    配置按网格文件名的主干存成 JSON，放在网格的临时目录里。文件已经存在、但里面
    记的网格路径不是这次的 ``mesh`` 时重写：同名而路径不同的网格（例如测试里临时
    生成的文件）不能读到别的网格留下的配置。各预设的块、单块参考和验收共用同一组
    物性和边条状态。
    """
    from opencfd_jax.src import main, runtime_config
    mesh = str(mesh)
    path = MESH_DIR / f"plate-{Path(mesh).stem}.json"
    try:
        stale = json.loads(path.read_text(encoding="utf-8"))["mesh"]["file"] != mesh
    except (OSError, ValueError, KeyError, TypeError):
        stale = True
    if stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "case": {"name": "plate"},
            "mesh": {"format": "cgns", "file": mesh},
            "boundary": {"inflow": dict(FLOW), "farfield": dict(FLOW),
                         "wall_temperature": 1.0},
            "physics": {"Re": 1000.0, "Ama": 0.1, "ref_t": 288.15, "pr": 0.7},
            "numerics": {"scheme": "WENO7", "vis_order": "CD8", "viscous": False},
            "time": {"dt": 0.0001, "end_time": 0.0002, "end_step": 2},
            "output": {"dir": str(MESH_DIR / "out"), "kstep_show": 1,
                       "kstep_save": 2},
        }, ensure_ascii=False), encoding="utf-8")
    return runtime_config.resolve_args(
        {}, runtime_config.load_config(str(path)),
        main.build_parser().parse_args([]))


def run_args(preset: str, **changes) -> argparse.Namespace:
    """预设网格的多块运行参数：在 ``plate_args`` 上给出参考尺度，再按需要改几项。"""
    base = vars(plate_args(mesh_path(preset)))
    return argparse.Namespace(**{**base, "flux_scale": FLUX_SCALE, **changes})


def reference():
    """等价的单块：验收网格的源（y 向拉伸的 plate，100×40×40），装配一次并缓存。"""
    if "single" not in _BUILT:
        from opencfd_jax.src import init
        _BUILT["single"] = init.prepare(plate_args(SOURCE))
    return _BUILT["single"]


def periodic_reference():
    """per3 的等价单块：25×25×25 的全周期立方体，直接补坐标、算度量。"""
    if "periodic" not in _BUILT:
        from types import SimpleNamespace
        from Grid.split_cgns import periodic_source
        from opencfd_jax.src import mesh
        zone = periodic_source()
        coords = {key: mesh.pad_coords(value, periodic=(True, True, True))
                  for key, value in zone.coords.items()}
        metrics = mesh.compute_metrics_from_coords(
            coords, (1.0, 1.0, 1.0), 6, jac_bound=(1, 1, 1),
            periodic=(True, True, True))
        _BUILT["periodic"] = SimpleNamespace(
            coords=coords, vertex_size=zone.vertex_size,
            metrics={key: np.asarray(value) for key, value in metrics.items()})
    return _BUILT["periodic"]


def blocks(preset: str):
    """预设网格装配好的各个块和交换表，装配一次并缓存。"""
    if preset not in _BUILT:
        from opencfd_jax.src import multiblock
        _BUILT[preset] = multiblock.build_blocks(
            topology(preset), plate_args(mesh_path(preset)),
            sidecar(preset)["global_vertices"])
    return _BUILT[preset]


FLUX_SCALE = (1521.0, 3861.0, 3861.0)      # plate 100×40×40 的全局参考尺度
DOMAIN = (10.0, 4.0, 3.0)                  # plate 的物理域：x、y、z 的长度


def same_bits(got, want, label=""):
    """逐位比较：先比类型和形状，再比每个位。

    最低位翻转后数值几乎不变，所以不能用 ``assert_array_equal``：它把 +0 和 −0、
    以及 NaN 判成相同。
    """
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape, (
        label, got.dtype, got.shape, want.dtype, want.shape)
    assert got.tobytes() == want.tobytes(), label


def with_record(zones, zone, conn, **changes):
    """把 ``zone`` 里名为 ``conn`` 的方向记录改掉，返回新的合成网格。"""
    return [replace(spec, connections=tuple(
        replace(c, **changes) if c.name == conn else c
        for c in spec.connections)) if spec.name == zone else spec
        for spec in zones]


def interiors(built, pieces: dict) -> dict:
    """各块状态的内区（不含鬼胞），转成 numpy。"""
    lookup = {block.name: block for block in built.blocks}
    return {name: np.asarray(value)[lookup[name].grid.inner()]
            for name, value in pieces.items()}


def kernels(preset: str, flux_scale=FLUX_SCALE):
    """预设网格编好的求解函数，每个预设只编一次。"""
    from opencfd_jax.src import multiblock
    return multiblock.compile_blocks(blocks(preset), flux_scale)


def acceptance_state(coords, params, domain=DOMAIN) -> np.ndarray:
    """验收初场（R16）：归一化坐标上的三角函数扰动，三个方向都变。

    密度 1±0.10，u 为 1±0.05，v 和 w 各 ±0.03，温度 1±0.05。守恒量用 plate 的同一个
    换算。它在入口面和壁面上不满足边界条件。``coords`` 是 {"Axx","Ayy","Azz"}，
    形状任意（内区或含鬼胞都行）；返回 (5, *坐标的形状)。
    """
    from opencfd_jax.src.cases import cons_from_state
    xi, eta, zeta = (np.asarray(coords[key]) / length
                     for key, length in zip(COORDS, domain))
    state = {
        "rho": 1.0 + 0.10 * np.sin(2.0 * np.pi * xi + 0.3) * np.cos(np.pi * eta)
               * np.cos(2.0 * np.pi * zeta),
        "u": 1.0 + 0.05 * np.cos(2.0 * np.pi * zeta) * np.sin(np.pi * xi)
             * np.cos(np.pi * eta),
        "v": 0.03 * np.sin(2.0 * np.pi * eta + 0.5) * np.cos(2.0 * np.pi * xi)
             * np.sin(2.0 * np.pi * zeta),
        "w": 0.03 * np.cos(2.0 * np.pi * xi + 0.7) * np.sin(2.0 * np.pi * zeta)
             * np.cos(np.pi * eta),
        "T": 1.0 + 0.05 * np.sin(np.pi * (xi + eta)) * np.cos(2.0 * np.pi * zeta)
             * np.cos(np.pi * xi),
    }
    return np.asarray(cons_from_state(state, params))


def with_field(state, field) -> np.ndarray:
    """把单块的起点状态换成 ``field`` 在内区上的值，鬼胞为零。

    ``state`` 是 :func:`reference` 的结果；``field(coords, params)`` 给五个守恒量。
    """
    cons = np.zeros(state.desc.grid.shape)
    cons[state.desc.grid.inner()] = field(
        {key: np.asarray(value) for key, value in state.desc.coords.items()},
        state.case["params"])
    return cons


def _formal(state):
    """正式入口的右端项函数和整步 jit（与 main.run 的写法相同），每个单块只编一次。"""
    import jax
    from opencfd_jax.src import main, solver
    key = ("formal", id(state))
    if key not in _BUILT:
        rhs = main.build_rhs(state.case, state.args, execution=state.execution)
        step = jax.jit(
            lambda q, dt_: state.execution.prepare(solver.rk3_step(q, dt_, rhs)))
        _BUILT[key] = rhs, step
    return _BUILT[key]


def formal_step(state, cons, dt) -> np.ndarray:
    """正式入口的一整步（整步一个 jit），``cons`` 是起点状态。"""
    import jax.numpy as jnp
    return np.asarray(_formal(state)[1](jnp.asarray(cons), dt))


def formal_trace(state, cons, dt):
    """正式入口的装配下逐步走一整步，记下三个阶段的右端项和 RHS 的三次入参。

    入参的第 2、3 次就是准备之前的 ``q1``、``q2``；第 1 次是整步起点。
    """
    from types import SimpleNamespace
    import jax.numpy as jnp
    from opencfd_jax.src import solver
    rhs = _formal(state)[0]
    seen = []

    def recording(q):
        seen.append(np.asarray(q))
        return rhs(q)

    q3, du = solver.rk3_step_stages(jnp.asarray(cons), dt, recording)
    return SimpleNamespace(du=tuple(np.asarray(value) for value in du), inputs=seen,
                           final=np.asarray(state.execution.prepare(q3)))


def block(name, ni, nj, nk, origin=(0.0, 0.0, 0.0), bcs=(), connections=()):
    """一个均匀长方体 zone；坐标是 (nk, nj, ni)。"""
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni),
                          indexing="ij")
    coords = {"Axx": origin[0] + SPACING[0] * i,
              "Ayy": origin[1] + SPACING[1] * j,
              "Azz": origin[2] + SPACING[2] * k}
    return ZoneSpec(name=name, coords=coords, bcs=tuple(bcs),
                    connections=tuple(connections))


def outer_bcs(ni, nj, nk, low=None, high=None,
              bottom="BCWall", top="BCFarfield"):
    """板状块外表面的边条：i−、i+ 两面由 ``low``、``high`` 给出（(名字, 类型)）；
    j−、j+ 缺省是壁面和远场，k± 是一对块内周期的 ``BCSymmetryPolar``。
    ``bottom``、``top`` 传 None 时那一面不写边条，留给接口。"""
    bcs = []
    if low is not None:
        bcs.append((low[0], low[1], ((1, 1, 1), (1, nj, nk))))
    if high is not None:
        bcs.append((high[0], high[1], ((ni, 1, 1), (ni, nj, nk))))
    if bottom is not None:
        bcs.append(("Jlo", bottom, ((1, 1, 1), (ni, 1, nk))))
    if top is not None:
        bcs.append(("Jhi", top, ((1, nj, 1), (ni, nj, nk))))
    bcs += [("Klo", "BCSymmetryPolar", ((1, 1, 1), (ni, nj, 1))),
            ("Khi", "BCSymmetryPolar", ((1, 1, nk), (ni, nj, nk)))]
    return tuple(bcs)


def cut_pair(a, b, recv, give, name="cut", transform=None, periodic=None):
    """``a`` 的范围 ``recv`` 接 ``b`` 的范围 ``give``，返回两条互逆的方向记录
    （给 ``a`` 的在前）。范围是 1-based 闭区间。"""
    return (ConnSpec(name, b.name, recv, give, transform, periodic),
            ConnSpec(name, a.name, give, recv, transform, periodic))


WHOLE = ((1, 1, 1), (1, 7, 7))        # 8×7×7 的块在 i−、i+ 上的整面
OUTER = ((8, 1, 1), (8, 7, 7))


def two_blocks():
    """两块 8×7×7，中间一条 i 向的普通切口。"""
    left = block("Zone 1", 8, 7, 7)
    right = block("Zone 2", 8, 7, 7, origin=(7.0, 0.0, 0.0))
    to_right, to_left = cut_pair(left, right, OUTER, WHOLE)
    return [replace(left, bcs=outer_bcs(8, 7, 7, ("Ilo", "BCInflow")),
                    connections=(to_right,)),
            replace(right, bcs=outer_bcs(8, 7, 7, high=("Ihi", "BCOutflow")),
                    connections=(to_left,))]


def seam_blocks():
    """两块 8×7×7：中间一条普通切口，外侧的两个 i 面是跨块的周期接缝，平移 14。"""
    left = block("Zone 1", 8, 7, 7)
    right = block("Zone 2", 8, 7, 7, origin=(7.0, 0.0, 0.0))
    to_right, to_left = cut_pair(left, right, OUTER, WHOLE)
    periodic = {"RotationCenter": (0.0, 0.0, 0.0),
                "RotationAngle": (0.0, 0.0, 0.0)}
    return [
        replace(left, bcs=outer_bcs(8, 7, 7), connections=(
            to_right,
            ConnSpec("seam", "Zone 2", WHOLE, OUTER,
                     periodic=dict(periodic, Translation=(14.0, 0.0, 0.0))))),
        replace(right, bcs=outer_bcs(8, 7, 7), connections=(
            to_left,
            ConnSpec("seam", "Zone 1", OUTER, WHOLE,
                     periodic=dict(periodic, Translation=(-14.0, 0.0, 0.0))))),
    ]


def t_blocks():
    """三块的 T 形：左块 8×13×7 的 i+ 面分两段，下段接 Zone 2、上段接 Zone 3，
    两段在 j=6 这条线上共线；Zone 2 的 j+ 面与 Zone 3 的 j− 面在右半段相接。"""
    left = block("Zone 1", 8, 13, 7)
    low = block("Zone 2", 8, 7, 7, origin=(7.0, 0.0, 0.0))
    high = block("Zone 3", 8, 7, 7, origin=(7.0, 12.0, 0.0))
    cut_low, from_low = cut_pair(left, low, ((8, 1, 1), (8, 7, 7)), WHOLE,
                                 name="cut lo")
    cut_high, from_high = cut_pair(left, high, ((8, 7, 1), (8, 13, 7)), WHOLE,
                                   name="cut hi")
    cut_mid, to_mid = cut_pair(low, high, ((1, 7, 1), (8, 7, 7)),
                               ((1, 1, 1), (8, 1, 7)), name="cut mid")
    return [replace(left, bcs=outer_bcs(8, 13, 7, ("Ilo", "BCInflow")),
                    connections=(cut_low, cut_high)),
            replace(low, bcs=outer_bcs(8, 7, 7, high=("Ihi", "BCOutflow"),
                                       top=None),
                    connections=(from_low, cut_mid)),
            replace(high, bcs=outer_bcs(8, 7, 7, high=("Ihi", "BCOutflow"),
                                        bottom=None),
                    connections=(from_high, to_mid))]


def _farfield_faces(ni, nj, nk, faces):
    """几个整面的 ``BCFarfield`` 边条；``faces`` 每项是 (轴, 侧)，轴取 i、j、k，侧取 -、+。
    壁面只能出现在 j 向，这两个网格只看拓扑和交换表，外表面一律写成远场。"""
    out = []
    for axis, sign in faces:
        lo, hi = [1, 1, 1], [ni, nj, nk]
        at = "ijk".index(axis)
        lo[at] = hi[at] = 1 if sign == "-" else (ni, nj, nk)[at]
        out.append((axis + sign, "BCFarfield", (tuple(lo), tuple(hi))))
    return tuple(out)


def stair_blocks():
    """四块的阶梯。Zone 1 在下，它的 k+ 面分三段：一长两短，长段接 Zone 2，它的 i=6
    这条棱对着两个短段（接 Zone 3、Zone 4）。上面三块之间还有三处接口：Zone 2 的 i+
    面分两段接 Zone 3、Zone 4，是个 T 形；Zone 3 的 j+ 面接 Zone 4 的 j− 面。"""
    under = block("Zone 1", 13, 13, 7)
    wide = block("Zone 2", 7, 13, 7, origin=(0.0, 0.0, 18.0))
    low = block("Zone 3", 7, 7, 7, origin=(6.0, 0.0, 18.0))
    high = block("Zone 4", 7, 7, 7, origin=(6.0, 12.0, 18.0))
    low_k = ((1, 1, 1), (7, 7, 1))           # 7×7 块的 k- 面
    low_i = ((1, 1, 1), (1, 7, 7))           # 7×7 块的 i- 面
    to_wide, from_wide = cut_pair(under, wide, ((1, 1, 7), (7, 13, 7)),
                                  ((1, 1, 1), (7, 13, 1)), name="stair wide")
    to_low, from_low = cut_pair(under, low, ((7, 1, 7), (13, 7, 7)), low_k,
                                name="stair low")
    to_high, from_high = cut_pair(under, high, ((7, 7, 7), (13, 13, 7)), low_k,
                                  name="stair high")
    split_low, split_back = cut_pair(wide, low, ((7, 1, 1), (7, 7, 7)), low_i,
                                     name="split lo")
    split_high, split_top = cut_pair(wide, high, ((7, 7, 1), (7, 13, 7)), low_i,
                                     name="split hi")
    side, side_back = cut_pair(low, high, ((1, 7, 1), (7, 7, 7)),
                               ((1, 1, 1), (7, 1, 7)), name="side")
    return [replace(under, bcs=_farfield_faces(
               13, 13, 7, [("i", "-"), ("i", "+"), ("j", "-"), ("j", "+"), ("k", "-")]),
               connections=(to_wide, to_low, to_high)),
            replace(wide, bcs=_farfield_faces(
               7, 13, 7, [("i", "-"), ("j", "-"), ("j", "+"), ("k", "+")]),
               connections=(from_wide, split_low, split_high)),
            replace(low, bcs=_farfield_faces(
               7, 7, 7, [("i", "+"), ("j", "-"), ("k", "+")]),
               connections=(from_low, split_back, side)),
            replace(high, bcs=_farfield_faces(
               7, 7, 7, [("i", "+"), ("j", "+"), ("k", "+")]),
               connections=(from_high, split_top, side_back))]


def strip_blocks():
    """四块，T 形处有一格宽的条带。Zone 1（8×13×7）的 i+ 面分两段：下段十一格接
    Zone 2，最上面一格接 Zone 3。Zone 3 的 i− 面同样分两段：最下面一格接 Zone 1，
    其余六格接 Zone 4；Zone 2 的 j+ 面接 Zone 3 的 j− 面，Zone 1 的 j+ 面接 Zone 4
    的 j− 面。"""
    lower = block("Zone 1", 8, 13, 7)
    middle = block("Zone 2", 8, 12, 7, origin=(7.0, 0.0, 0.0))
    upper = block("Zone 3", 8, 8, 7, origin=(7.0, 22.0, 0.0))
    left = block("Zone 4", 8, 7, 7, origin=(0.0, 24.0, 0.0))
    tall, tall_back = cut_pair(lower, middle, ((8, 1, 1), (8, 12, 7)),
                               ((1, 1, 1), (1, 12, 7)), name="tall")
    sliver, sliver_back = cut_pair(lower, upper, ((8, 12, 1), (8, 13, 7)),
                                   ((1, 1, 1), (1, 2, 7)), name="sliver")
    cap, cap_back = cut_pair(upper, left, ((1, 2, 1), (1, 8, 7)),
                             ((8, 1, 1), (8, 7, 7)), name="cap")
    shoulder, shoulder_back = cut_pair(middle, upper, ((1, 12, 1), (8, 12, 7)),
                                       ((1, 1, 1), (8, 1, 7)), name="shoulder")
    rim, rim_back = cut_pair(lower, left, ((1, 13, 1), (8, 13, 7)),
                             ((1, 1, 1), (8, 1, 7)), name="rim")
    return [replace(lower, bcs=_farfield_faces(
               8, 13, 7, [("i", "-"), ("j", "-"), ("k", "-"), ("k", "+")]),
               connections=(tall, sliver, rim)),
            replace(middle, bcs=_farfield_faces(
               8, 12, 7, [("i", "+"), ("j", "-"), ("k", "-"), ("k", "+")]),
               connections=(tall_back, shoulder)),
            replace(upper, bcs=_farfield_faces(
               8, 8, 7, [("i", "+"), ("j", "+"), ("k", "-"), ("k", "+")]),
               connections=(sliver_back, cap, shoulder_back)),
            replace(left, bcs=_farfield_faces(
               8, 7, 7, [("i", "-"), ("j", "+"), ("k", "-"), ("k", "+")]),
               connections=(cap_back, rim_back))]


# ---- 一致性阈值（R37）与比较 ----
# 实测（FP64，验收初场，五个验收网格各两个整步，拼回后对照单块，共 4130 处比较）。
# 每个守恒量的最大绝对差，和它相对自身最大值的差：
#
#   ======  ===========  ==================
#   守恒量  最大绝对差    相对自身最大值
#   ======  ===========  ==================
#   ρ       3.5e-13       8.0e-13
#   ρu      2.4e-12       6.1e-14
#   ρv      4.6e-12       1.1e-12
#   ρw      1.9e-12       6.7e-13
#   ρE      1.6e-10       9.0e-13
#   ======  ===========  ==================
#
# 近零的守恒量（有的比较里 ρv、ρw 自身的最大值只有 1e-5 量级）相对自身的差没有意义：
# 它们的舍入差由同一行算式里的大分量带出来，绝对差除以全场最大值最大是 5.1e-16。
# 所以每个守恒量各有相对容差和绝对容差：相对容差乘自身最大值，绝对容差乘全场最大
# 值，近零的量由后者控制。相对取 1e-11、绝对取 1e-13 时，实测最紧的一处用到容差的
# 9%，近零的守恒量用到 0.5%。参考尺度按某一块的尺寸误填时，du 的差在 1e-5 的相对
# 量级，远在容差之外。差别来自舍入：RK 组合在单独的 jit 里，块形状不同时向量化
# 循环的尾部也不同。
RTOL = 1e-11      # 相对容差：乘上该守恒量在参考场里的最大值
ATOL = 1e-13      # 绝对容差：乘上参考场五个分量的最大值


def compare(reference, got, label="", mask=None, count=5) -> dict:
    """按阈值比较两个 (分量, nk, nj, ni) 的场，超差或出现非有限值时报 AssertionError。

    每个分量的容差是 ``RTOL`` 乘它在参考场里的最大值，加上 ``ATOL`` 乘参考场全部分量
    的最大值；近零的分量由后一项控制。``mask`` 是 (nk, nj, ni) 的布尔数组，
    给出时只比其中的点，非有限值也只在其中查。返回 {分量号: (最大绝对差, 容差)}。
    ``count`` 是最前面一根轴上分量的个数：缺省 5 是守恒量；比粘性通量 ``Ev`` 时传 12
    （第七步）。报错时先挑超出容差倍数最大的分量，倍数相同（比如容差都是零）时挑
    绝对差最大的，再指出差得最多的下标。最后几个轴不是三维时（例如共线审计拿来
    比较的 (5, 点数) 数组），下标按它实际的维数报。
    """
    reference, got = np.asarray(reference), np.asarray(got)
    assert reference.shape == got.shape, (label, reference.shape, got.shape)
    assert reference.shape[0] == count, (label, "分量的个数", reference.shape[0], count)
    where = (np.ones(reference.shape[1:], dtype=bool) if mask is None
             else np.asarray(mask, dtype=bool))
    assert where.shape == reference.shape[1:], (label, "mask 的形状", where.shape)
    if not where.any():
        raise AssertionError(f"{label} 的 mask 里没有一个点，没有可比较的")
    inside = np.nonzero(where)                 # 掩码内各点的下标，按 C 序
    picked = reference[(slice(None),) + inside]
    found = got[(slice(None),) + inside]
    if not (np.all(np.isfinite(picked)) and np.all(np.isfinite(found))):
        raise AssertionError(f"{label} 出现非有限值")
    biggest = float(np.max(np.abs(picked)))
    worst = {v: (float(np.max(np.abs(found[v] - picked[v]))),
                 RTOL * float(np.max(np.abs(picked[v]))) + ATOL * biggest)
             for v in range(count)}
    bad = [v for v in worst if worst[v][0] > worst[v][1]]
    if bad:
        def excess(v):                         # 超出容差多少倍；容差是零时记无穷大
            ratio = worst[v][0] / worst[v][1] if worst[v][1] > 0.0 else np.inf
            return (ratio, worst[v][0])        # 倍数相同时，绝对差大的在前

        v = max(bad, key=excess)
        far = int(np.argmax(np.abs(found[v] - picked[v])))     # 只在掩码内找，掩码外的不做减法
        index = tuple(int(axis[far]) for axis in inside)
        place = f" (k, j, i) = {index}" if len(index) == 3 else f"下标 {index}"
        name = f"cons_{v}" if count == 5 else f"分量_{v}"
        raise AssertionError(
            f"{label} 超出容差：{name} 在{place} 差 "
            f"{worst[v][0]:.3e}，容差 {worst[v][1]:.3e}；参考值 {reference[(v,) + index]!r}，"
            f"实测值 {got[(v,) + index]!r}")
    return worst


def split_field(built, note: dict, whole) -> dict:
    """把全局内区上的场 (5, Nk, Nj, Ni) 按旁记切给各块，鬼胞为零。"""
    import jax.numpy as jnp
    lookup = {block.name: block for block in built.blocks}
    states = {}
    for entry in note["blocks"]:
        grid = lookup[entry["name"]].grid
        own = np.zeros(grid.shape)
        own[grid.inner()] = whole[(slice(None),) + block_window(entry)]
        states[entry["name"]] = jnp.asarray(own)
    return states
