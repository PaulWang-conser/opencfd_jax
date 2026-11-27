"""第七步测试的辅助模块：粘性专有的部分。

第六步的辅助模块 ``step6_support`` 管验收网格、验收初场、拼回和无粘的单块参考，
这里直接用它们。第七步多出来的部分放在这里：

  - 打开粘性的配置和单块参考：``viscous_args``；``reference`` 是验收网格的源，
    ``periodic_reference`` 是 per3 的等价单块；``viscous_stage`` 算单块一个阶段的粘性分项；
  - 起点状态和初场：``acceptance_on``、``acceptance_start``、``states_from_field``，
    ``preset`` 把一个验收网格（或 per3）的块、参考尺度、起点放在一起；
  - 含鬼胞数组上的掩码：``outside``、``corner``、``face_ghost``、``received``。判断块角的
    鬼胞有没有被读到、接口交换写到了哪里时用；
  - 五个合成小网格（``SYNTHETIC``、``synthetic``）和它们各自的等价单块（``single``），
    线性场和常数来流（``linear_field``、``free_stream``），把各块拼回等价单块用的旁记
    （``windows``）和接口邻域（``near_interfaces``）；
  - “故意不换 Ev”用的交换函数（``exchange_without_ev``）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import step6_support as support  # noqa: E402
from Grid.split_cgns import DEFAULT_SRC as SOURCE  # noqa: E402
from opencfd_jax.src.mesh import COORD_KEYS as COORDS  # noqa: E402

_BUILT: dict = {}            # 缓存：打开粘性的单块参考、各预设的块和起点、合成网格的文件

PERIODIC_SCALE = (576.0, 576.0, 576.0)      # per3 的等价单块是 25×25×25
PERIODIC_DOMAIN = (24.0, 48.0, 72.0)        # 它的边长：25 个顶点，间距 (1, 2, 3)


# ---- 打开粘性的单块 ----

def viscous_args(mesh, vis_order="CD8"):
    """打开粘性的 plate 配置：在第六步的 ``plate_args`` 上只改粘性开关和导数的阶数。"""
    base = vars(support.plate_args(mesh))
    return argparse.Namespace(**{**base, "viscous": True, "vis_order": vis_order})


def reference(vis_order: str = "CD8"):
    """打开粘性的单块参考：验收网格的源（y 向拉伸的 plate，100×40×40），装配一次并缓存。
    ``vis_order`` 是粘性导数的阶数名，每种阶数各建一份。

    第六步的单块参考缓存着按无粘编好的右端项，它的配置不能原地改，所以另建一份。
    """
    if ("viscous", vis_order) not in _BUILT:
        from opencfd_jax.src import init
        _BUILT[("viscous", vis_order)] = init.prepare(viscous_args(SOURCE, vis_order))
    return _BUILT[("viscous", vis_order)]


def periodic_reference(vis_order: str = "CD8"):
    """per3 的等价单块，粘性打开：25×25×25 的全周期立方体，走正式入口的装配。

    把立方体写成六个面全是 ``BCSymmetryPolar`` 的单 zone，与第六步的同名测试做法相同。
    """
    if ("periodic", vis_order) not in _BUILT:
        from Grid.split_cgns import periodic_source
        from opencfd_jax.src import init
        source = periodic_source()
        n, = set(source.vertex_size)
        bcs = []
        for axis, letter in enumerate("IJK"):
            for sign in ("lo", "hi"):
                lo, hi = [1, 1, 1], [n, n, n]
                lo[axis] = hi[axis] = 1 if sign == "lo" else n
                bcs.append((f"{letter}{sign}", "BCSymmetryPolar", (tuple(lo), tuple(hi))))
        path = _written(support.MESH_DIR / "step7-periodic.cgns", [
            _zone("Zone 1", source.coords, bcs)])
        _BUILT[("periodic", vis_order)] = init.prepare(viscous_args(path, vis_order))
    return _BUILT[("periodic", vis_order)]


def viscous_stage(state, prepared):
    """单块一个阶段的粘性分项：对准备过的状态调用 ``viscous_rhs``。

    ``prepared`` 是这一阶段交给右端项函数的状态做过一次物理准备的结果。实参与
    ``main.build_rhs`` 调用 ``viscous_rhs`` 时逐项相同，阶数取 ``state.args.vis_order``。
    返回 (5, nk+8, nj+8, ni+8) 的数组。
    """
    from opencfd_jax.src import multiblock, viscous
    desc, execution = state.desc, state.execution
    return np.asarray(viscous.viscous_rhs(
        prepared, desc.grid, state.case["params"], bcs=execution.bcs, axes=desc.axes,
        dx=desc.dxs, order=multiblock.CD_ORDER[state.args.vis_order], metrics=desc.metrics,
        dif_bound=execution.band if execution.band is not None else desc.dif_bound,
        zero_state_ghosts=False, dif_axis_map=desc.dif_axis_map,
        boundary_execution=execution, farfield=execution.farfield, faces=execution.faces,
        fill_state_halos=False))


# ---- 初场和起点状态 ----

def acceptance_on(domain):
    """验收初场，归一化用的边长取 ``domain``。返回 ``field(coords, params)``。"""
    return lambda coords, params: support.acceptance_state(coords, params, domain)


def extent(built) -> tuple:
    """各块坐标的最大值 (x, y, z)。合成小网格没有等价单块的坐标数组，拿它当初场归一化
    用的边长。"""
    return tuple(float(max(np.asarray(block.coords[key]).max() for block in built.blocks))
                 for key in COORDS)


def states_from_field(built, field) -> dict:
    """各块的起点状态：``field(coords, params)`` 在各块内区坐标上的值，鬼胞为零。

    场是坐标的函数。各块共有的顶点坐标相同，所以这与在等价单块上取值再切给各块一样。
    """
    import jax.numpy as jnp
    params = reference().case["params"]
    out = {}
    for block in built.blocks:
        inner = block.grid.inner()[1:]
        own = np.zeros(block.grid.shape)
        own[(slice(None),) + inner] = field(
            {key: np.asarray(value)[inner] for key, value in block.coords.items()}, params)
        out[block.name] = jnp.asarray(own)
    return out


def acceptance_start(preset: str) -> dict:
    """验收网格各块的起点状态：验收初场在等价单块内区上的值，按旁记切给各块。"""
    state = reference()
    whole = support.acceptance_state(state.desc.coords, state.case["params"])
    return support.split_field(support.blocks(preset), support.sidecar(preset), whole)


def preset(name: str):
    """验收网格（或 per3）的三样东西：装配好的块 ``built``、参考尺度 ``scale``、从验收
    初场出发的起点 ``initial``。每个预设只做一次。

    per3 是全周期的立方体，没有 plate 那样的源网格：参考尺度取 25×25×25 的等价单块，
    初场按各块自己的坐标取。
    """
    if ("preset", name) not in _BUILT:
        built = support.blocks(name)
        if name == "per3":
            scale, initial = PERIODIC_SCALE, states_from_field(
                built, acceptance_on(PERIODIC_DOMAIN))
        else:
            scale, initial = support.FLUX_SCALE, acceptance_start(name)
        _BUILT[("preset", name)] = SimpleNamespace(built=built, scale=scale, initial=initial)
    return _BUILT[("preset", name)]


# ---- 掩码：出界、块角、面内的一层鬼胞、接收槽 ----

def outside(grid, axis) -> np.ndarray:
    """一条数组轴（按 k、j、i 数：0 是 k，2 是 i）上出界（在鬼胞里）的掩码。"""
    from opencfd_jax.src.boundary import LAP
    n = grid.n_inner[axis]
    flags = np.ones(grid.shape[1:], dtype=bool)
    flags[(slice(None),) * axis + (slice(LAP, LAP + n),)
          + (slice(None),) * (2 - axis)] = False
    return flags


def corner(grid) -> np.ndarray:
    """块角：至少两条轴同时出界。

    接口的交换只写“面内、另外两个下标在内区里”的鬼胞，块角从不填，里面留着本地的
    无意义值。
    """
    flags = [outside(grid, axis) for axis in range(3)]
    return (flags[0] & flags[1]) | (flags[0] & flags[2]) | (flags[1] & flags[2])


def face_ghost(grid, axis, depth=1, high=False) -> np.ndarray:
    """某条数组轴低侧（``high`` 为真时是高侧）的第 ``depth`` 层鬼胞，另外两条轴取内区：
    接口交换会写到的那种点。"""
    from opencfd_jax.src.boundary import LAP
    inside = [~outside(grid, other) for other in range(3) if other != axis]
    band = np.zeros(grid.shape[1:], dtype=bool)
    index = [slice(None)] * 3
    index[axis] = grid.shape[1 + axis] - LAP + depth - 1 if high else LAP - depth
    band[tuple(index)] = True
    return band & inside[0] & inside[1]


def received(built, name: str, face: str | None = None) -> np.ndarray:
    """一个块的接收槽（接口面上四层鬼胞）的掩码；给出 ``face``（如 ``"i+"``）时只取那
    一个面。"""
    out = np.zeros(built[name].grid.shape[1:], dtype=bool)
    for table in built.tables:
        it = table.interface
        if it.zone == name and (face is None or it.side == face):
            out[tuple(table.slots)] = True
    return out


# ---- 五个合成小网格和它们的等价单块 ----

# 标识 → (显示名, 各块的构造, 等价单块的顶点数 (Ni, Nj, Nk), 外表面的 i−、i+ 两面)。
# 最后一项是 None 的两个网格六面全是远场；其余的外表面是 j− 壁面、j+ 远场、k± 块内周期。
SYNTHETIC = {
    "two": ("两块切口", support.two_blocks, (15, 7, 7),
            (("Ilo", "BCInflow"), ("Ihi", "BCOutflow"))),
    "seam": ("周期接缝", support.seam_blocks, (15, 7, 7),
             (("Ilo", "BCSymmetryPolar"), ("Ihi", "BCSymmetryPolar"))),
    "tee": ("T 形", support.t_blocks, (15, 13, 7),
            (("Ilo", "BCInflow"), ("Ihi", "BCOutflow"))),
    "stair": ("阶梯", support.stair_blocks, (13, 13, 13), None),
    "strip": ("一格宽的条带", support.strip_blocks, (15, 19, 7), None),
}
SYNTHETIC_KEYS = tuple(SYNTHETIC)
SYNTHETIC_NAMES = [item[0] for item in SYNTHETIC.values()]
SYNTHETIC_SCALE = (1.0, 1.0, 1.0)       # 合成小网格的参考尺度：只进无粘项


def _written(path, zones):
    """写出合成网格；已经写过就不再写。网格放在系统临时目录，各次测试共用。"""
    from Grid.make_cgns_fixture import build_multizone
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        build_multizone(path, zones)
    return path


def _zone(name, coords, bcs):
    from Grid.make_cgns_fixture import ZoneSpec
    return ZoneSpec(name=name, coords=coords, bcs=tuple(bcs))


def synthetic(key: str):
    """合成小网格装配成的块，装配一次并缓存。

    装配时给出等价单块的全局顶点数：物理面旁的坐标鬼层按全局轴长选规则，与单块相同。
    装配不读粘性开关和阶数，阶数只在编求解函数时给。
    """
    if ("multi", key) not in _BUILT:
        from opencfd_jax.src import mesh_cgns, multiblock
        _, make, whole, _ = SYNTHETIC[key]
        path = _written(support.MESH_DIR / f"step7-{key}.cgns", make())
        _BUILT[("multi", key)] = multiblock.build_blocks(
            multiblock.build_topology(mesh_cgns.read_cgns_zones(path)),
            viscous_args(path), whole)
    return _BUILT[("multi", key)]


def single_path(key: str):
    """合成小网格的等价单块的网格文件：一个均匀的长方体 zone。第一次调用时写出。"""
    if ("single-path", key) not in _BUILT:
        _, _, (ni, nj, nk), ends = SYNTHETIC[key]
        bcs = (support.outer_bcs(ni, nj, nk, *ends) if ends is not None else
               support._farfield_faces(ni, nj, nk, [(a, s) for a in "ijk" for s in "-+"]))
        _BUILT[("single-path", key)] = _written(
            support.MESH_DIR / f"step7-single-{key}.cgns",
            [_zone("Zone 1", support.block("Zone 1", ni, nj, nk).coords, bcs)])
    return _BUILT[("single-path", key)]


def single(key: str, vis_order: str = "CD8"):
    """合成小网格的等价单块：一个均匀的长方体 zone，走正式入口的装配，粘性打开。"""
    if ("single", key, vis_order) not in _BUILT:
        from opencfd_jax.src import init
        _BUILT[("single", key, vis_order)] = init.prepare(
            viscous_args(single_path(key), vis_order))
    return _BUILT[("single", key, vis_order)]


def linear_field(coords, params):
    """线性场（AE4）：速度和温度是坐标的线性函数，三个方向都变；密度是常数 1。"""
    from opencfd_jax.src.cases import cons_from_state
    x, y, z = (np.asarray(coords[key]) for key in COORDS)
    field = {"rho": np.ones_like(x), "u": 0.5 + 0.03 * x, "v": -0.2 + 0.05 * y,
             "w": 0.1 + 0.02 * z, "T": 1.0 + 0.01 * x - 0.02 * y + 0.03 * z}
    return np.asarray(cons_from_state(field, params))


def free_stream(coords, params):
    """常数来流：各处都是入口状态。"""
    from opencfd_jax.src.cases import cons_from_state
    cons = np.asarray(cons_from_state(support.FLOW, params))
    return np.broadcast_to(cons[:, None, None, None], (5, *np.shape(coords["Axx"])))


def windows(built, state) -> dict:
    """各块在等价单块里的位置，拼成 ``step6_support.reassemble`` 要的旁记。

    顺带核对一件事（U4）：按算出的窗口从等价单块上切出来的坐标，与各块内区的坐标逐位
    相同。只查“每个坐标值在单块的轴线上找得到”不够，要按窗口逐点比。
    """
    whole = {key: np.asarray(value) for key, value in state.desc.coords.items()}   # 内区形状
    entries = []
    for block in built.blocks:
        inside = block.grid.inner()[1:]
        own = {key: np.asarray(value)[inside] for key, value in block.coords.items()}
        start = [int(np.rint((own[key].min() - whole[key].min()) / step))
                 for key, step in zip(COORDS, support.SPACING)]             # 计算轴序 (i, j, k)
        count = block.grid.n_inner[::-1]                                      # (Ni, Nj, Nk)
        window = tuple(slice(s, s + n) for s, n in zip(start[::-1], count[::-1]))
        for key in COORDS:                                                    # 数组轴序 (k, j, i)
            assert np.array_equal(whole[key][window], own[key]), (
                f"{block.name} 的 {key} 坐标与等价单块上同一窗口的不同："
                f"起点 {start}，顶点数 {count}")
        entries.append({"name": block.name, "start": start,
                        "stop": [s + n for s, n in zip(start, count)]})
    return {"global_vertices": list(state.desc.grid.n_inner[::-1]), "blocks": entries}


def near_interfaces(built, note: dict, layers: int = 4) -> np.ndarray:
    """等价单块上离任何一条接口不超过 ``layers`` 个顶点的点，形状 (Nk, Nj, Ni)。"""
    from opencfd_jax.src.axes import side_axis
    shape = note["global_vertices"][::-1]
    mask = np.zeros(shape, dtype=bool)
    for entry in note["blocks"]:
        block = built[entry["name"]]
        sides = {table.interface.side for table in built.tables
                 if table.interface.zone == entry["name"]}
        for side in sides:
            axis = side_axis(side)                      # 计算轴
            plane = entry["start"][axis] + (
                0 if side.endswith("-") else block.grid.n_inner[2 - axis] - 1)
            index = [None] * 3                          # 数组轴序 (k, j, i)
            for a in range(3):
                index[2 - a] = slice(entry["start"][a], entry["stop"][a])
            index[2 - axis] = slice(max(plane - layers + 1, 0),
                                    min(plane + layers - 1, shape[2 - axis] - 1) + 1)
            mask[tuple(index)] = True
    return mask


def physical_band(built, note: dict, layers: int = None) -> np.ndarray:
    """等价单块上的物理边界带：各块里紧挨着打开了单侧公式的那一面的 ``layers`` 层内区点，
    形状 (Nk, Nj, Ni)。``layers`` 缺省取鬼胞的层数 LAP，与 :func:`viscous.band_one_sided`
    覆盖的层数相同。

    哪一面开了单侧公式看 ``block.execution.band``。验收网格上只有壁面是开的，切口面和
    周期面不开。
    """
    from opencfd_jax.src.boundary import LAP
    layers = LAP if layers is None else layers
    shape = note["global_vertices"][::-1]
    mask = np.zeros(shape, dtype=bool)
    for entry in note["blocks"]:
        for side, flag in enumerate(built[entry["name"]].execution.band):
            if not flag:
                continue
            axis, high = side // 2, side % 2              # 计算轴，0 是低侧、1 是高侧
            count = entry["stop"][axis] - entry["start"][axis]
            plane = entry["start"][axis] + (count - 1 if high else 0)
            index = [slice(start, stop) for start, stop in                  # 数组轴序 (k, j, i)
                     zip(entry["start"][::-1], entry["stop"][::-1])]
            index[2 - axis] = (slice(plane - layers + 1, plane + 1) if high
                               else slice(plane, plane + layers))
            mask[tuple(index)] = True
    return mask


# ---- 故意引入的错误 ----

def exchange_without_ev(real):
    """给“故意不换 Ev”的测试用：返回一个用来替换 ``multiblock.exchange`` 的函数，Ev
    （12 个平面的数组）原样返回，守恒量照旧走 ``real``。用法：
    ``monkeypatch.setattr(multiblock, "exchange", exchange_without_ev(multiblock.exchange))``。
    """
    def keep_ev(tables, pieces):
        first = next(iter(pieces.values()))
        return dict(pieces) if first.shape[0] == 12 else real(tables, pieces)
    return keep_ev
