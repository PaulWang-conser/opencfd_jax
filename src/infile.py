"""读写原码的 in 文件（`opencfd-scu.in`），保证两侧传参同源。

原码用 `read_parameters`（src/parameters.c）逐行扫 `KEY = value`：整行注释 `#`，
值可以是整数/浮点/字符串/空格分隔的列表。这里做同样的事，解析结果给
`cases.build_channel_case` / `compare.py` 用，避免手抄参数抄错。

约定：**不做任何单位换算、不做缺省填充**——缺省值交给调用处显式决定，
因为原码的缺省（如 T_REF=288.15）必须和 JAX 侧逐字对齐。
"""

from __future__ import annotations


def read_ocfd_in(path):
    """解析 in 文件 → {KEY: 原始字符串值}。键统一大写，行内 `#` 之后忽略。"""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().upper()
            if not key:
                continue
            out[key] = val.strip()
    return out


def _words(val):
    """把空格或逗号分隔的原始参数值拆成字符串列表。"""
    return val.replace(",", " ").split()


def as_int(val):
    """读取参数值的第一个字段并转换为整数。"""
    return int(float(_words(val)[0]))


def as_float(val):
    """读取参数值的第一个字段并转换为浮点数。"""
    return float(_words(val)[0])


def as_ints(val):
    """把完整参数值转换为整数列表，兼容逗号和空格分隔。"""
    return [int(float(w)) for w in _words(val)]


def as_floats(val):
    """把完整参数值转换为浮点数列表，兼容逗号和空格分隔。"""
    return [float(w) for w in _words(val)]


def as_str(val):
    """读取参数值的第一个字段，保留为字符串。"""
    return _words(val)[0]


def grid_dims(inp):
    """GRID_3D / PARALLEL_3D → (nx, ny, nz, (npx, npy, npz))，并校验整除。"""
    dims = as_ints(inp["GRID_3D"])
    if len(dims) != 3:
        raise ValueError(f"GRID_3D 需要 3 个整数，得到 {dims}")
    npx, npy, npz = (as_ints(inp["PARALLEL_3D"])
                     if "PARALLEL_3D" in inp else [1, 1, 1])
    nx, ny, nz = dims
    if nx % npx or ny % npy or nz % npz:
        raise ValueError(
            f"GRID_3D={dims} 不能被 PARALLEL_3D={[npx, npy, npz]} 整除")
    return nx, ny, nz, (npx, npy, npz)


def summarize(inp):
    """取出对拍真正需要对齐的那些键（别的键不参与 JAX 侧计算）。"""
    keys_int = ("GRID_3D", "PARALLEL_3D")
    keys_float = ("RE", "AMA", "GAMMA", "PR", "T_REF", "EPSL_SW", "DT")
    keys_str = ("SCHEME_INVIS", "SCHEME_VIS")
    out = {}
    for k in keys_int:
        if k in inp:
            out[k] = as_ints(inp[k])
    for k in keys_float:
        if k in inp:
            out[k] = as_float(inp[k])
    for k in keys_str:
        if k in inp:
            out[k] = as_str(inp[k])
    for k in ("IPERIODIC", "JAC_BOUND", "IBC", "BC_NPARA"):
        if k in inp:
            out[k] = as_ints(inp[k])
    if "BC_RPARA" in inp:
        out["BC_RPARA"] = as_floats(inp["BC_RPARA"])
    return out
