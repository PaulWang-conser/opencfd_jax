"""IO：npz 快速读写（自用缓存/存档）+ 原版 OCFD dat reader（对拍用）。

快速格式：未压缩 npz（numpy 原生、零依赖、顺序读取快）。度量缓存走这里。
检查点每个保存步一个 NPZ：每块一个数组 ``b{块号}``，形状 (5, nk, nj, ni)，第一根轴
依次是原始变量 ρ、u、v、w、T，只有内区，各块用自己的形状；另带 ``_meta``（JSON 的
UTF-8 字节存成 uint8 数组），多块再带 ``_blocks``（块清单，编码与 ``_meta`` 相同）。

原版 dat 格式（OCFD3d-Jacobi.dat / OCFD3d-Mesh.dat，OCFD_Comput_Jacobian3d.cu:119-175
与 OCFD_IO.c 的 read_3d1）：字段序列逐个存放，每场 = 逐 z 层堆叠，
每层前置 2 个 int（MPI 并行写的层尺寸头），随后 NX_GLOBAL*NY_GLOBAL 个 REAL
（double，C 序 x 最快）。字段顺序（Jacobi.dat）：Axx, Ayy, Azz, Akx, Aky,
Akz, Aix, Aiy, Aiz, Asx, Asy, Asz, Ajac（末尾字段以基准文件实测为准）。
"""

import json
import os
import re
import zipfile

import numpy as np


class CheckpointError(ValueError):
    """检查点读不了、不是检查点、块清单或数组与它对不上。"""


_BLOCK_KEY = re.compile(r"b(\d+)\Z")


def checkpoint_name(prefix: str, step: int) -> str:
    """返回八位步号的 NPZ 文件名：``<prefix>-istep-00000001.npz``。

    步号超过八位时自然扩展，不截断；负步号或非整数直接报错。
    """
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step 必须是整数，得到 {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step 不能为负数，得到 {step}")
    return f"{prefix}-istep-{step:08d}.npz"


def time_to_save(step: int, kstep_save: int, finished: bool) -> bool:
    """保存条件：步号是保存间隔的整数倍，或者到达了终止时间。单独到达步数上限不保存。"""
    return step % kstep_save == 0 or finished


def primitive_of(cons, phys):
    """内区的守恒量 (5, nk, nj, ni) 换成原始变量 (5, nk, nj, ni)：依次是 ρ、u、v、w、T。"""
    import jax.numpy as jnp

    from .boundary import primitive
    rho, u, T = primitive(jnp.asarray(cons), phys)
    return np.stack([rho, u[0], u[1], u[2], T])


def primitive_field(cons, grid, phys):
    """存盘用的一块内区原始变量 (5, nk, nj, ni)。"""
    return primitive_of(np.asarray(cons[grid.inner()]), phys)


def conservative_of(field, phys):
    """读回用：原始变量 (5, nk, nj, ni) 换成守恒量，第一根轴依次是 ρ、ρu、ρv、ρw、ρE。"""
    from .cases import cons_from_state
    return cons_from_state(
        {"rho": field[0], "u": field[1], "v": field[2], "w": field[3], "T": field[4]}, phys)


def phys_from_meta(meta):
    """按检查点自己记的物性搭一个物性参数；换算只用到 Cv。"""
    from .viscous import PhysParams
    try:
        return PhysParams(gamma=meta["gamma"], Ama=meta["Ama"])
    except KeyError as exc:
        raise CheckpointError(
            f"检查点的 _meta 里没有 {exc.args[0]!r}，换算不了原始变量") from exc


def save_checkpoint(path, fields, meta=None, blocks=None):
    """检查点：fields 是 {b{块号}: 数组}；多块另带块清单 ``_blocks``，编码与 ``_meta`` 相同。"""
    arrays = dict(fields)
    if blocks is not None:
        arrays["_blocks"] = _json_array(blocks)
    save_npz(path, arrays, meta)


def _json_array(value):
    return np.frombuffer(json.dumps(value).encode("utf-8"), dtype=np.uint8)


def load_checkpoint(path, phys=None, inner=None):
    """读回检查点，返回 (meta, 块清单, 各块的数组)。

    各块的数组是按块号排的列表，每项是那一块的 (5, nk, nj, ni) 原始变量；单块的列表
    只有一项，块清单是空字典。这一步之前写的单块检查点（``cons_0`` 到 ``cons_4``）在这里
    换成原始变量：物性没给时，用它自己 ``_meta`` 里的 ``gamma`` 和 ``Ama`` 现搭一个。
    ``inner`` 给出时逐块核对形状，依次是各块内区的 (nk, nj, ni)；不同就报错，不做广播。
    """
    try:
        with np.load(path) as data:
            if "_meta" not in data.files:
                raise CheckpointError(f"{path} 缺少 _meta")
            meta = json.loads(data["_meta"].tobytes().decode("utf-8"))
            listed = "_blocks" in data.files
            keys = [name for name in data.files if name not in ("_meta", "_blocks")]
            if keys and all(_BLOCK_KEY.fullmatch(name) for name in keys):
                order = sorted(keys, key=lambda name: int(name[1:]))
                if [int(name[1:]) for name in order] != list(range(len(order))):
                    raise CheckpointError(f"{path} 的块号不连续：{sorted(keys)}")
                if not listed and len(order) != 1:
                    raise CheckpointError(
                        f"{path} 没有块清单，却不止一个数组：{sorted(keys)}；"
                        "单块的检查点只能有 b0")
                fields = [data[name] for name in order]
            elif sorted(keys) == [f"cons_{i}" for i in range(5)] and not listed:
                cons = np.stack([data[f"cons_{i}"] for i in range(5)])
                # 换成文件自己的类型：换算在 float32 的运行里会降精度，续算要靠它提示
                fields = [primitive_of(cons, phys if phys is not None else phys_from_meta(meta))
                          .astype(cons.dtype)]
            else:
                raise CheckpointError(f"{path} 不是检查点：数组的名字是 {sorted(keys)}")
            blocks = (json.loads(data["_blocks"].tobytes().decode("utf-8")) if listed else {})
    except CheckpointError:
        raise
    except (OSError, ValueError, KeyError, TypeError, EOFError,
            zipfile.BadZipFile) as exc:          # BadZipFile：写到一半中断的文件
        raise CheckpointError(f"{path} 读不出来：{type(exc).__name__}: {exc}") from exc
    _check_shapes(path, blocks, fields, inner)
    return meta, blocks, fields


def _check_shapes(path, blocks, fields, inner):
    """数组都是 (5, nk, nj, ni)；多块的与块清单逐块对照；``inner`` 给出时与这一次的内区对照。"""
    shapes = []
    for index, field in enumerate(fields):
        if field.ndim != 4 or field.shape[0] != 5:
            raise CheckpointError(
                f"{path} 的 b{index} 形状是 {field.shape}，应为 (5, nk, nj, ni)")
        shapes.append(tuple(int(value) for value in field.shape[1:]))
    entries = blocks.get("blocks", []) if blocks else []
    if blocks and (not isinstance(entries, list) or len(entries) != len(fields)):
        raise CheckpointError(f"{path} 的块清单里，块的个数与数组的个数对不上")
    for index, (entry, shape) in enumerate(zip(entries, shapes)):
        size = entry.get("size") if isinstance(entry, dict) else None
        if size != list(shape[::-1]):
            raise CheckpointError(
                f"{path} 的块清单里第 {index} 块记的 size={size!r}，"
                f"数组内区 (nk, nj, ni) 却是 {shape}")
    if inner is not None:
        wanted = [tuple(int(value) for value in shape) for shape in inner]
        if wanted != shapes:
            raise CheckpointError(
                f"{path} 里各块内区的形状 (nk, nj, ni) 是 {shapes}，"
                f"这一次网格的内区是 {wanted}")


def block_window(entry, shift=0):
    """块清单里一块的范围在全局数组上的切片，轴序 (k, j, i)。起止是 (i, j, k)。"""
    k0, j0, i0 = (value + shift for value in entry["start"][::-1])
    k1, j1, i1 = (value + shift for value in entry["stop"][::-1])
    return (slice(k0, k1), slice(j0, j1), slice(i0, i1))


def assemble(note, pieces):
    """按块清单把各块内区的数组拼回等价单块的内区。

    ``pieces`` 是 {块名: 数组}，数组的最后三个轴是 (k, j, i)，长度是各块的内区；
    前面可以再带别的轴。切口那一层两块各有一份，后写的盖住先写的。块清单里没有
    位置时报错：那样的块清单拼不出全局场。
    """
    if ("global_vertices" not in note
            or any("start" not in entry or "stop" not in entry for entry in note["blocks"])):
        raise CheckpointError("块清单里没有各块的位置，拼不回全局场")
    nk, nj, ni = note["global_vertices"][::-1]
    first = next(iter(pieces.values()))
    lead = (slice(None),) * (first.ndim - 3)
    out = np.empty(first.shape[:-3] + (nk, nj, ni), dtype=first.dtype)
    for entry in note["blocks"]:
        window = block_window(entry)
        piece = pieces[entry["name"]]
        if tuple(piece.shape[-3:]) != tuple(s.stop - s.start for s in window):
            raise CheckpointError(
                f"{entry['name']!r} 在块清单里的范围是 {entry['start']} 到 {entry['stop']}，"
                f"数组内区 (ni, nj, nk) 却是 {piece.shape[-3:][::-1]}")
        out[lead + window] = piece
    return out


def restart_clock(meta) -> tuple[int, float]:
    """检查点 ``_meta`` 里的步号和时间。步号不是非负整数、时间不是有限数的报 ``CheckpointError``。"""
    step, t = meta.get("step"), meta.get("t")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointError("_meta 里的 step 不是非负整数")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t):
        raise CheckpointError("_meta 里的 t 不是有限数")
    return int(step), float(t)


def clobber_step(restart, out_dir, prefix, step):
    """续算的检查点会不会被这次运行写出的文件覆盖；会就返回那一步的步号，不会返回 None。

    会覆盖的情形：检查点在输出目录里，文件名是这次运行的前缀加一个比它记的步号大的
    步号，比如被改过名。文件名与记的步号一致时不会发生：以后的文件步号都更大。
    """
    name = os.path.basename(str(restart))
    head, marker, tail = name.rpartition("-istep-")
    if not marker or head != str(prefix) or not tail.endswith(".npz"):
        return None
    digits = tail[:-4]
    if not digits.isdigit() or int(digits) <= step:
        return None
    target = os.path.join(str(out_dir), name)
    if os.path.normcase(os.path.abspath(target)) != os.path.normcase(os.path.abspath(str(restart))):
        return None
    return int(digits)


def check_mesh(meta, case, zones, blocks=None) -> tuple[list, list]:
    """核对检查点记的网格与这一次的是不是同一个：算例名，以及每个 zone 的名字和顶点数。

    ``zones`` 是按文件顺序排的 [(zone 名或 None, (ni, nj, nk))]，内置网格的 zone 名是
    None。多块的检查点另给 ``blocks``（块清单里各项，含名字和尺寸）。只核对结构，不核对
    坐标：名字和顶点数都相同的两个网格算同一个。返回 (对不上的项, 没核对到的项)：检查点
    里记了的项就比，没记的列进第二样。
    """
    problems, unchecked = [], []
    if "case" in meta:
        if meta["case"] != case:
            problems.append(f"算例名不同：检查点是 {meta['case']!r}，这一次是 {case!r}")
    else:
        unchecked.append("算例名")
    if blocks is not None:
        found_names = [entry.get("name") for entry in blocks]
        found_sizes = [entry.get("size") for entry in blocks]
        if found_names != [zone for zone, _ in zones]:
            problems.append(f"块的名字不同：检查点是 {found_names}，"
                            f"这一次是 {[zone for zone, _ in zones]}")
        if found_sizes != [list(size) for _, size in zones]:
            problems.append(f"块的尺寸不同：检查点是 {found_sizes}，"
                            f"这一次是 {[list(size) for _, size in zones]}")
        return problems, unchecked
    zone, size = zones[0]
    mesh = meta.get("mesh")
    if isinstance(mesh, dict) and isinstance(mesh.get("size"), list):
        if mesh["size"] != list(size):
            problems.append(f"网格尺寸不同：检查点是 {mesh['size']}，这一次是 {list(size)}")
        if mesh.get("format") != ("builtin" if zone is None else "cgns"):
            problems.append(f"网格格式不同：检查点是 {mesh.get('format')!r}，"
                             f"这一次是 {'builtin' if zone is None else 'cgns'}")
        name = mesh.get("zone")
        if zone is not None:
            if name:
                if name != zone:
                    problems.append(f"zone 的名字不同：检查点是 {name!r}，这一次是 {zone!r}")
            else:
                unchecked.append("zone 的名字")
    elif "N" in meta:
        if meta["N"] != size[0]:
            problems.append(f"网格尺寸不同：检查点的 N={meta['N']}，这一次的 ni={size[0]}")
        unchecked.append("zone 的名字和另外两维的尺寸" if zone is not None else "另外两维的尺寸")
    else:
        unchecked.append("网格尺寸")
    return problems, unchecked


def checkpoint_meta(args, params, t, step, dt, viscous, mesh, ni=None):
    """检查点 ``_meta`` 的内容：原有的项不变，另加粘性开关、导数的阶数、精度和网格。
    单块给 ``ni``（记作 ``N``）；多块不给。"""
    from . import config
    meta = {"t": t, "step": step, "dt": dt, "case": args.case,
            "scheme": args.scheme, "gamma": params.gamma, "Re": params.Re,
            "Ama": params.Ama, "Ref_T": params.Ref_T, "Pr": params.Pr,
            "viscous": bool(viscous), "vis_order": args.vis_order,
            "precision": config.precision_name(), "mesh": mesh}
    if ni is not None:
        meta["N"] = ni
    return meta


def save_npz(path, fields: dict, meta: dict | None = None):
    """未压缩 npz 存档。fields: {name: ndarray}; meta 任意可 json 化对象。"""
    arrays = {k: np.asarray(v) for k, v in fields.items()}
    if meta is not None:
        arrays["_meta"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)


def load_npz(path) -> tuple[dict, dict | None]:
    """读 npz，返回 (fields, meta)。_meta 若存在则解析为 dict 并从 fields 剔除。"""
    data = np.load(path)
    fields = {}
    meta = None
    for k in data.files:
        if k == "_meta":
            meta = json.loads(data[k].tobytes().decode("utf-8"))
        else:
            fields[k] = data[k]
    return fields, meta


def read_ocfd_dat(path, shape: tuple[int, int, int], n_fields: int,
                  dtype=np.float64, offset: int = 0):
    """读原版 OCFD dat 文件（MPI 并行写格式）为全场数组列表。

    布局：逐场、逐 z 层，每层 = 2 个 int 头 + nx*ny 个 dtype（x 最快）。
    shape 为全局 (nx, ny, nz)。返回 [n_fields 个 (nz, ny, nx) numpy 数组]。
    注意：2 int 头的确切含义（层尺寸/哨兵）待首个基准文件核对后补充断言；
    层头大小 8 字节由 OCFD_Comput_Jacobian3d.cu:127 的 offset 步进公式
    (2*sizeof(int) + NX*NY*sizeof(REAL))*NZ 得出。
    """
    nx, ny, nz = shape
    data_bytes = nx * ny * np.dtype(dtype).itemsize
    arrays = []
    with open(path, "rb") as fh:
        fh.seek(offset)
        for _ in range(n_fields):
            field = np.empty((nz, ny, nx), dtype=dtype)
            for k in range(nz):
                fh.seek(8, 1)                          # 跳过层头（2 int）
                layer = np.fromfile(fh, dtype=dtype, count=nx * ny)
                field[k] = layer.reshape(ny, nx)
            arrays.append(field)
    return arrays
