"""把单块 plate 网格按索引面切开，写出第六步的多块 CGNS 网格和旁记。

用法（任意目录，缺省路径相对本脚本所在目录）：
    python Grid/split_cgns.py                                  # 七个预设全写
    python Grid/split_cgns.py --preset xyz8 --out-dir 某个目录

切口约定：切在下标 m 的顶点层，这一层归相邻两块共有。x 在 i=50 切，左块是
51×40×40。块与块相接的面写两条互逆的 1to1 方向记录，Transform 是 (1,2,3)。
周期方向被切开时，露出一端的面与露出另一端的块配成周期接缝，写上 Periodic_t，
平移量取供体坐标减接收坐标。外表面上的边条从源网格的边条裁剪而来。

每个网格配一份同名旁记 JSON，记下源网格、全局顶点数和每块的全局起止下标
（从 0 开始、左闭右开）。测试用它把各块拼回全局；多块运行读其中的
``global_vertices``，给物理面旁的坐标鬼层选与等价单块相同的规则（见
``src/multiblock.py`` 的 ``_global_sizes``：这一项要是三个 1 到 2³¹−1 的整数，
旁记文件在而它读不出来时报错，不悄悄按各块自己的轴长算）。

预设：x2、y2、z2、xyz8、t3 从 plate 网格切出；x2i6 是给 U3、U4 用的定向网格；
per3 是 3×3×3 切开的全周期小网格（每块 9×9×9），直接生成，不从 plate 派生。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for path in (REPO.parent, REPO):        # 工作区根（opencfd_jax 包）和仓库根（Grid 包）
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Grid.make_cgns_fixture import ConnSpec, ZoneSpec, build_multizone  # noqa: E402
from opencfd_jax.src import mesh_cgns  # noqa: E402
from opencfd_jax.src.axes import AXIS_NAMES, SIDES, side_axis  # noqa: E402
from opencfd_jax.src.boundary import PERIODIC  # noqa: E402
from opencfd_jax.src.mesh import COORD_KEYS  # noqa: E402

DEFAULT_SRC = HERE / "testcgnstest_ystr.cgns"      # y 向拉伸的 plate，100×40×40

# 各轴的切口下标（0-based 的顶点层），针对 100×40×40 的 plate 网格
_CUTS = {"x2": ([50], [], []), "y2": ([], [20], []), "z2": ([], [], [20]),
         "xyz8": ([50], [20], [20]), "x2i6": ([6], [], [])}
PER3_SIZE = (25, 25, 25)                 # 25 个顶点、24 个间隔是一个周期
PER3_CUTS = ([8, 16], [8, 16], [8, 16])
PRESETS = (*_CUTS, "t3", "per3")


def _parts(cuts, size):
    """把各轴的切口下标变成分段（左闭右开）；切口那一层归相邻两块共有。"""
    parts = []
    for axis in range(3):
        if (list(cuts[axis]) != sorted(set(cuts[axis]))
                or any(not 0 < cut < size[axis] for cut in cuts[axis])):
            raise ValueError(
                f"{AXIS_NAMES[axis]} 向的切口 {list(cuts[axis])} 不在 1 到 "
                f"{size[axis] - 1} 之间，或有重复")
        edges = [0, *cuts[axis], size[axis]]
        parts.append([(edges[n], edges[n + 1] + (1 if n < len(edges) - 2 else 0))
                      for n in range(len(edges) - 1)])
    return parts


def _cartesian(parts):
    """由各轴的分段拼出全部块，返回 (块名, (i0, i1, j0, j1, k0, k1)) 的列表。"""
    return [(f"blk-i{p}j{q}k{r}", ir + jr + kr)
            for r, kr in enumerate(parts[2])
            for q, jr in enumerate(parts[1])
            for p, ir in enumerate(parts[0])]


def preset_blocks(name, size):
    """预设的各个块，下标是全局顶点下标、左闭右开。"""
    if name == "t3":                     # 左块全高，右半在 j=20 分成上下两块
        ni, nj, nk = size
        return [("blk-left", (0, 51, 0, nj, 0, nk)),
                ("blk-rlo", (50, ni, 0, 21, 0, nk)),
                ("blk-rhi", (50, ni, 20, nj, 0, nk))]
    if name == "per3":
        return _cartesian(_parts(PER3_CUTS, PER3_SIZE))
    return _cartesian(_parts(_CUTS[name], size))


def _overlap(a, b, skip):
    """两块在 ``skip`` 以外的两条轴上的公共范围（左闭右开）。

    范围按顶点给，相邻两块在切口那一层各算一个顶点，所以至少要有 2 个顶点的
    公共范围才是一个面元的接触。只在一条棱上相接（公共的只有那一层）不算，
    这时返回 None；有一条轴上完全没有公共范围时也返回 None。
    """
    box = []
    for axis in range(3):
        if axis == skip:
            continue
        lo, hi = max(a[2 * axis], b[2 * axis]), min(a[2 * axis + 1], b[2 * axis + 1])
        if hi - lo < 2:
            return None
        box.append((lo, hi))
    return box


def _face_range(block, axis, local_index, box):
    """块 ``block`` 在 ``axis`` 侧面上、覆盖 ``box`` 的面片，1-based 闭区间。"""
    lo, hi = [1, 1, 1], [1, 1, 1]
    lo[axis] = hi[axis] = local_index + 1
    for other, (blo, bhi) in zip((a for a in range(3) if a != axis), box):
        lo[other] = blo - block[2 * other] + 1
        hi[other] = bhi - 1 - block[2 * other] + 1
    return tuple(lo), tuple(hi)


def _extent(block, axis):
    """块在 ``axis`` 向的顶点数。"""
    return block[2 * axis + 1] - block[2 * axis]


def _point(coords, index):
    """全局下标 (i, j, k) 处的坐标。"""
    return np.array([coords[key][index[2], index[1], index[0]] for key in COORD_KEYS])


def _periodic_axes(zone) -> tuple[int, ...]:
    """源网格里两侧都是 ``BCSymmetryPolar`` 的轴。"""
    polar = set()
    for bc in zone.bcs:
        if mesh_cgns.bc_kind(bc.bc_type) == PERIODIC:
            lo, hi = bc.point_range
            axis = mesh_cgns.face_axes(lo, hi)[0]
            polar.add((axis, "-" if lo[axis] == 0 else "+"))
    return tuple(axis for axis in range(3)
                 if (axis, "-") in polar and (axis, "+") in polar)


def periodic_source():
    """per3 的“源”：25×25×25 的均匀立方体，三个方向的间距不同。"""
    k, j, i = np.meshgrid(*(np.arange(n) for n in PER3_SIZE), indexing="ij")
    return SimpleNamespace(vertex_size=PER3_SIZE, bcs=(), coords={
        "Axx": 0.5 * i, "Ayy": 0.7 * j, "Azz": 1.1 * k})


def _link_cuts(blocks, records):
    """相接的块配成普通切口：a 的最后一层顶点就是 b 的第一层。"""
    for a in blocks:
        for b in blocks:
            for axis in range(3):
                if a[1][2 * axis + 1] - 1 != b[1][2 * axis]:
                    continue
                box = _overlap(a[1], b[1], axis)
                if box is None:
                    continue
                recv = _face_range(a[1], axis, _extent(a[1], axis) - 1, box)
                give = _face_range(b[1], axis, 0, box)
                name = f"cut {a[0]} {b[0]}"
                records[a[0]].append(ConnSpec(name, b[0], recv, give, transform=(1, 2, 3)))
                records[b[0]].append(ConnSpec(name, a[0], give, recv, transform=(1, 2, 3)))


def _link_seams(zone, blocks, records, periodic_axes):
    """周期方向上，露出低端的块面与露出高端的块配成周期接缝；返回这些面。"""
    size, seams = zone.vertex_size, set()
    for axis in periodic_axes:
        end = size[axis]
        for a in blocks:
            if a[1][2 * axis] != 0 or a[1][2 * axis + 1] == end:
                continue                   # a 的低侧面露在全局边界上，高侧面不在
            for b in blocks:
                if b[1][2 * axis + 1] != end or b[1][2 * axis] == 0:
                    continue               # b 的高侧面露在全局边界上，低侧面不在
                box = _overlap(a[1], b[1], axis)
                if box is None:
                    continue
                index = [0, 0, 0]
                for other, (lo, _hi) in zip((x for x in range(3) if x != axis), box):
                    index[other] = lo
                receiving = _point(zone.coords, index)
                index[axis] = end - 1
                shift = _point(zone.coords, index) - receiving       # 供体减接收
                periodic = {"RotationCenter": (0.0, 0.0, 0.0),
                            "RotationAngle": (0.0, 0.0, 0.0)}
                recv = _face_range(a[1], axis, 0, box)
                give = _face_range(b[1], axis, _extent(b[1], axis) - 1, box)
                name = f"seam {a[0]} {b[0]}"
                records[a[0]].append(ConnSpec(
                    name, b[0], recv, give, transform=(1, 2, 3),
                    periodic=dict(periodic, Translation=tuple(shift))))
                records[b[0]].append(ConnSpec(
                    name, a[0], give, recv, transform=(1, 2, 3),
                    periodic=dict(periodic, Translation=tuple(-shift))))
                seams |= {(a[0], AXIS_NAMES[axis] + "-"),
                          (b[0], AXIS_NAMES[axis] + "+")}
        for name, block in blocks:
            for exposed, sign in ((block[2 * axis] == 0 and block[2 * axis + 1] != end, "-"),
                                  (block[2 * axis + 1] == end and block[2 * axis] != 0, "+")):
                if exposed and (name, AXIS_NAMES[axis] + sign) not in seams:
                    raise ValueError(
                        f"块 {name} 的 {AXIS_NAMES[axis]}{sign} 面露在周期方向的一端，"
                        "却找不到露出另一端的块与它配成周期接缝")
    return seams


def _outer_faces(zone, blocks, seams):
    """把源网格的边条裁剪到各块的外表面上，返回 {块名: [(名字, 类型, 范围)]}。"""
    size = zone.vertex_size
    patches = []
    for bc in zone.bcs:
        lo, hi = bc.point_range
        axes = mesh_cgns.face_axes(lo, hi)
        side = None if axes is None else mesh_cgns.side_of_face(axes[0], lo, size)
        if side is None:
            raise ValueError(f"源网格的边条 {bc.name} 不是落在某个侧面上的面片")
        patches.append((side, bc.name, bc.bc_type,
                        tuple((min(lo[a], hi[a]), max(lo[a], hi[a]) + 1)
                              for a in range(3))))
    out = {name: [] for name, _ in blocks}
    for name, block in blocks:
        box = tuple(zip(block[0::2], block[1::2]))
        for side in SIDES:
            axis = side_axis(side)
            low = side.endswith("-")
            if not (block[2 * axis] == 0 if low else block[2 * axis + 1] == size[axis]):
                continue                   # 这个侧面不在全局边界上
            if (name, side) in seams:
                continue                   # 周期接缝，不要边条
            others = [a for a in range(3) if a != axis]
            area = np.prod([box[a][1] - box[a][0] for a in others])
            covered, pieces = 0, []
            for patch_side, patch_name, bc_type, patch in patches:
                if patch_side != side:
                    continue
                clip = [(max(box[a][0], patch[a][0]), min(box[a][1], patch[a][1]))
                        for a in others]
                if any(hi - lo < 1 for lo, hi in clip):
                    continue
                pieces.append((patch_name, bc_type, _face_range(
                    block, axis, 0 if low else _extent(block, axis) - 1, clip)))
                covered += np.prod([hi - lo for lo, hi in clip])
            if covered != area:
                raise ValueError(
                    f"块 {name} 的 {side} 面在全局边界上，源网格却只在那儿给了 "
                    f"{covered} 个面元的边条（共 {area} 个）")
            out[name] += pieces
    return out


def split_zone(zone, blocks, periodic_axes=()):
    """把 ``zone`` 切成若干块，返回各块的 ``ZoneSpec``。

    ``blocks`` 是 ``(块名, (i0, i1, j0, j1, k0, k1))`` 的列表，下标是全局顶点下标、
    左闭右开。块与块相接的面配成普通切口。``periodic_axes`` 里的轴按周期处理：
    只露出低端的块面与露出高端的块配成周期接缝；两端都在同一个块里的，沿用
    源网格在那里的边条。
    """
    size = tuple(zone.vertex_size)
    for name, block in blocks:
        for axis in range(3):
            if not 0 <= block[2 * axis] < block[2 * axis + 1] <= size[axis]:
                raise ValueError(
                    f"块 {name} 在 {AXIS_NAMES[axis]} 向的范围 "
                    f"{block[2 * axis]}..{block[2 * axis + 1]} 超出源网格的 "
                    f"{size[axis]} 个顶点，或为空")
    coords = {}
    for name, block in blocks:
        i0, i1, j0, j1, k0, k1 = block
        coords[name] = {key: np.ascontiguousarray(value[k0:k1, j0:j1, i0:i1])
                        for key, value in zone.coords.items()}
    records = {name: [] for name, _ in blocks}
    _link_cuts(blocks, records)
    seams = _link_seams(zone, blocks, records, periodic_axes)
    bcs = _outer_faces(zone, blocks, seams)
    return [ZoneSpec(name=name, coords=coords[name], bcs=tuple(bcs[name]),
                     connections=tuple(records[name])) for name, _ in blocks]


def _replace(out: Path, write) -> None:
    """先写同目录的临时文件，写成功后才替换 ``out``。"""
    tmp = out.with_name(out.name + ".tmp")
    try:
        write(tmp)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def write_preset(preset, out_dir, src=None) -> Path:
    """写出一个预设：``<out_dir>/<preset>.cgns`` 和旁记 ``<preset>.json``。

    返回网格的路径。输出不能放到源网格上。
    """
    out_dir = Path(out_dir)
    out = out_dir / f"{preset}.cgns"
    if preset == "per3":
        zone, source, periodic = periodic_source(), None, (0, 1, 2)
    else:
        source = Path(src or DEFAULT_SRC)
        if out.resolve() == source.resolve():
            raise SystemExit("输出路径不能等于源网格：派生网格不得覆盖源文件")
        zone = mesh_cgns.read_cgns_zone(source, validate=False)
        periodic = _periodic_axes(zone)
    blocks = preset_blocks(preset, zone.vertex_size)
    specs = split_zone(zone, blocks, periodic)
    sidecar = {"preset": preset,
               "source": None if source is None else source.name,
               "global_vertices": list(zone.vertex_size),
               "blocks": [{"name": name, "start": [b[0], b[2], b[4]],
                           "stop": [b[1], b[3], b[5]]} for name, b in blocks]}
    out_dir.mkdir(parents=True, exist_ok=True)
    _replace(out, lambda path: build_multizone(path, specs))
    _replace(out.with_suffix(".json"), lambda path: path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="从 plate 单块网格切出第六步的多块网格")
    ap.add_argument("--preset", default="all", choices=("all", *PRESETS),
                    help="预设名；all 写全部七个")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="源网格（单块 CGNS）")
    ap.add_argument("--out-dir", default=str(HERE / "step6_meshes"),
                    help="输出目录；每个预设写出 <名字>.cgns 和 <名字>.json")
    args = ap.parse_args(argv)
    for preset in (PRESETS if args.preset == "all" else (args.preset,)):
        print(f"written: {write_preset(preset, Path(args.out_dir), Path(args.src))}")


if __name__ == "__main__":
    main()
