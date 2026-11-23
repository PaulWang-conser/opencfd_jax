"""`input.json` 运行配置：读取、校验、合并与输出前缀推导。

设计口径见 `docs/plans/2026-11-17-step3-input-requirements.md`：

- JSON 只承载运行参数，不设置网格维数、边界类型或 `DIF_BOUND/JAC_BOUND` 开关；
- 优先级为 程序默认值 → JSON → 用户显式命令行；
- 网格只支持程序内置网格或 CGNS 单块文件；CGNS 的维数由文件 `VertexSize` 决定；
- JSON 内相对路径按配置文件所在目录解析，输出前缀取网格文件名主干。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMES = ("UP7", "WENO7-SYMBO", "NND2", "WENO5", "WENO7",
           "OMP6-1", "OMP6-2", "OMP6-3", "OMP6_HR", "OMP6_LD", "OMP6_CD8")
VIS_ORDERS = ("CD6", "CD8")
# 出口状态鬼层：keep=保持原值（原码口径，缺省）；copy=零梯度延拓
OUTLET_GHOSTS = ("keep", "copy")
from .cases import CGNS_CASES, PUBLIC_CASE_NAMES

CASE_NAMES = PUBLIC_CASE_NAMES
MESH_FORMATS = ("builtin", "cgns")


class ConfigError(ValueError):
    """配置字段、类型或来源冲突错误；消息自带 JSON 字段路径。"""


# section -> field -> (kind, argparse dest)
_SCHEMA = {
    "mesh": {
        "format": ("str", "mesh_format"),
        "file": ("path", "mesh"),
    },
    "physics": {
        "Re": ("number", "Re"),
        "Ama": ("number", "Ama"),
        "ref_t": ("number", "ref_t"),
        "pr": ("number", "pr"),
        "tw": ("number", "tw"),
    },
    "boundary": {
        "inflow": ("state", "boundary_inflow"),
        "farfield": ("state", "boundary_farfield"),
        "wall_temperature": ("number", "boundary_wall_temperature"),
        "outlet_ghost": ("str", "boundary_outlet_ghost"),
    },
    "numerics": {
        "scheme": ("str", "scheme"),
        "vis_order": ("str", "vis_order"),
        "viscous": ("bool", "viscous"),
        "weno_tv": ("bool", "weno_tv"),
    },
    "time": {
        "dt": ("dt", "dt"),
        "end_time": ("number", "end_time"),
        "end_step": ("int", "end_step"),
    },
    "case": {
        "name": ("str", "case"),
        "inflow_profile": ("path", "inflow_profile"),
        "p_grad": ("number", "p_grad"),
        "forcing_work": ("forcing", "forcing_work"),
        "plate_x0": ("number", "plate_x0"),
        "plate_x1": ("number", "plate_x1"),
        "plate_h": ("number", "plate_h"),
    },
    "output": {
        "dir": ("outpath", "out_dir"),
        "kstep_show": ("int", "kstep_show"),
        "kstep_save": ("int", "kstep_save"),
    },
}

# 字段只在列出的算例中生效；其余算例出现即报错，避免静默忽略。
_CASE_SCOPE = {
    "mesh.file": {"poiseuille", "plate"},
    "boundary.inflow": {"plate"},
    "boundary.farfield": {"plate"},
    "boundary.wall_temperature": {"plate"},
    "boundary.outlet_ghost": {"plate"},
    "case.p_grad": {"poiseuille"},
    "case.forcing_work": {"poiseuille"},
    "case.plate_x0": {"plate"},
    "case.plate_x1": {"plate"},
    "case.plate_h": {"plate"},
}


@dataclass
class Config:
    """一份已校验的 JSON 配置。"""

    path: Path
    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)


def _check_string(value: Any, dotted: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{dotted} 需要非空字符串，得到 {value!r}")
    return value


def _check_number(value: Any, dotted: str, *, minimum: float | None = None,
                  maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{dotted} 需要数值，得到 {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{dotted} 需要有限数值，得到 {value!r}")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{dotted} 不能小于 {minimum:g}，得到 {result:g}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{dotted} 不能大于 {maximum:g}，得到 {result:g}")
    return result


def _check_int(value: Any, dotted: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{dotted} 需要整数，得到 {type(value).__name__}")
    if value < minimum:
        raise ConfigError(f"{dotted} 不能小于 {minimum}，得到 {value}")
    return value


def _check_bool(value: Any, dotted: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{dotted} 需要布尔值 true/false，得到 {value!r}")
    return value


def _resolve_path(value: str, directory: Path, dotted: str) -> str:
    """相对路径按配置文件目录解析；绝对路径原样返回。"""
    path = Path(value)
    resolved = path if path.is_absolute() else (directory / path)
    resolved = Path(os.path.normpath(str(resolved)))
    if not resolved.exists():
        raise ConfigError(f"{dotted} 指向的文件不存在：{resolved}")
    if not resolved.is_file():
        raise ConfigError(f"{dotted} 不是普通文件：{resolved}")
    return str(resolved)


def _resolve_out_dir(value: str, directory: Path) -> str:
    """输出目录允许尚不存在，启动时再创建。"""
    path = Path(value)
    resolved = path if path.is_absolute() else (directory / path)
    return str(Path(os.path.normpath(str(resolved))))


def _validate_value(kind: str, value: Any, dotted: str, directory: Path) -> Any:
    if kind == "state":
        keys = ("rho", "u", "v", "w", "T")
        if not isinstance(value, dict) or set(value) != set(keys):
            raise ConfigError(f"{dotted} 需要 rho/u/v/w/T 五个字段")
        return {k: _check_number(value[k], f"{dotted}.{k}") for k in keys}
    if kind == "str":
        return _check_string(value, dotted)
    if kind == "path":
        return _resolve_path(_check_string(value, dotted), directory, dotted)
    if kind == "outpath":
        return _resolve_out_dir(_check_string(value, dotted), directory)
    if kind == "number":
        return _check_number(value, dotted)
    if kind == "int":
        return _check_int(value, dotted)
    if kind == "bool":
        return _check_bool(value, dotted)
    if kind == "dt":
        if value is None:
            return None
        return _check_number(value, dotted, minimum=1.0e-15)
    if kind == "forcing":
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, str) and value.lower() in ("on", "off"):
            return value.lower()
        raise ConfigError(f"{dotted} 需要 true/false 或 \"on\"/\"off\"，得到 {value!r}")
    raise AssertionError(f"未实现的字段类型 {kind}")


def _validate_applicability(values: dict[str, Any], case: str) -> None:
    """JSON 中已给出的字段是否适用于最终生效的算例。"""
    fmt = values.get("mesh_format")
    if fmt == "cgns" and case not in CGNS_CASES:
        raise ConfigError(
            f"{case} 不支持 CGNS 网格；mesh.format=\"cgns\" 只适用于 poiseuille/plate")
    for dotted, allowed in _CASE_SCOPE.items():
        section, name = dotted.split(".")
        dest = _SCHEMA[section][name][1]
        if dest in values and case not in allowed:
            raise ConfigError(
                f"{section}.{name} 只适用于算例 {sorted(allowed)}，当前为 {case!r}")


def _semantic_checks(values: dict[str, Any], sources: dict[str, str],
                     fallback_case: str = "sod") -> None:
    """字段取值域与算例适用范围校验。"""
    case = values.get("case", fallback_case)
    if case not in CASE_NAMES:
        raise ConfigError(
            f"case.name 需要 {CASE_NAMES} 之一，得到 {case!r}")
    if "scheme" in values and values["scheme"] not in SCHEMES:
        raise ConfigError(
            f"numerics.scheme 不支持 {values['scheme']!r}；可选 {SCHEMES}")
    if "boundary_outlet_ghost" in values and values["boundary_outlet_ghost"] not in OUTLET_GHOSTS:
        raise ConfigError(
            f"boundary.outlet_ghost 不支持 {values['boundary_outlet_ghost']!r}；可选 {OUTLET_GHOSTS}")
    if "vis_order" in values and values["vis_order"] not in VIS_ORDERS:
        raise ConfigError(
            f"numerics.vis_order 不支持 {values['vis_order']!r}；可选 {VIS_ORDERS}")

    fmt = values.get("mesh_format")
    if fmt is not None and fmt not in MESH_FORMATS:
        raise ConfigError(f"mesh.format 需要 {MESH_FORMATS} 之一，得到 {fmt!r}")
    if fmt is None:
        fmt = "cgns" if "mesh" in values else "builtin"
        values["mesh_format"] = fmt
        sources["mesh_format"] = "mesh.format(推断)"
    if "mesh" in values and fmt != "cgns":
        raise ConfigError("mesh.file 只在 mesh.format=\"cgns\" 时可用")
    if fmt == "cgns" and "mesh" not in values:
        raise ConfigError("mesh.format=\"cgns\" 需要 mesh.file")
    if fmt == "cgns" and case == "plate":
        missing = [k for k in ("boundary_inflow", "boundary_farfield", "boundary_wall_temperature") if k not in values]
        if missing:
            raise ConfigError(f"plate CGNS 缺少 boundary 参数: {missing}")

    _validate_applicability(values, case)

    if values.get("plate_x0") is not None and values.get("plate_x1") is not None:
        if values["plate_x0"] >= values["plate_x1"]:
            raise ConfigError(
                f"case.plate_x0 必须小于 plate_x1，得到 {values['plate_x0']:g} "
                f"和 {values['plate_x1']:g}")
    if "end_step" in values and values["end_step"] < 1:
        raise ConfigError("time.end_step 至少为 1")


def load_config(path: str | os.PathLike[str],
                case: str | None = None) -> Config:
    """读取并校验 JSON；错误信息包含具体字段路径。

    ``case`` 是显式命令行给出的算例名，用于 JSON 省略 ``case.name`` 的
    部分配置：适用范围按“最终生效的算例”判断。
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"找不到配置文件：{config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{config_path} 不是合法 UTF-8 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} 顶层必须是对象")

    directory = config_path.resolve().parent
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    allowed_top = set(_SCHEMA) | {"precision", "specificboundary"}
    unknown = sorted(set(raw) - allowed_top)
    if unknown:
        raise ConfigError(f"未知顶层字段：{unknown}")

    # specificboundary 是顶层字符串（src/special_boundaries/ 下的脚本名，
    # 4.n 规格）：不走节→字段机制，单独解析。
    if "specificboundary" in raw:
        value = raw["specificboundary"]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                "specificboundary 需要是 src/special_boundaries/ 下的脚本名字符串")
        values["specificboundary"] = value.strip()
        sources["specificboundary"] = "specificboundary"

    for section, spec in _SCHEMA.items():
        block = raw.get(section)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ConfigError(f"{section} 需要对象，得到 {type(block).__name__}")
        unknown_fields = sorted(set(block) - set(spec))
        if unknown_fields:
            raise ConfigError(f"未知字段：{[f'{section}.{u}' for u in unknown_fields]}")
        for name, value in block.items():
            kind, dest = spec[name]
            dotted = f"{section}.{name}"
            values[dest] = _validate_value(kind, value, dotted, directory)
            sources[dest] = dotted

    if "precision" in raw:
        precision = raw["precision"]
        if precision not in ("fp64", "fp32"):
            raise ConfigError(
                f"precision 需要 \"fp64\" 或 \"fp32\"，得到 {precision!r}")
        values["fp32"] = precision == "fp32"
        values["fp64"] = precision == "fp64"
        sources["fp32"] = "precision"
        sources["fp64"] = "precision"

    if "case" in values and values["case"] not in CASE_NAMES:
        raise ConfigError(
            f"case.name 必须是 {', '.join(CASE_NAMES)} 之一，"
            f"得到 {values['case']!r}")

    _semantic_checks(values, sources, case or "sod")
    return Config(path=config_path, values=values, sources=sources)


def derive_prefix(mesh_file: str | None, case_name: str) -> str:
    """输出前缀：网格文件名去最后一个扩展名，否则算例名。"""
    if mesh_file:
        name = os.path.basename(str(mesh_file))
        prefix, _ = os.path.splitext(name)
        prefix = prefix.strip()
        if not prefix or prefix in (".", "..") or prefix.startswith(".") \
                or "/" in prefix or "\\" in prefix:
            raise ConfigError(f"无法从网格文件名推导输出前缀：{mesh_file!r}")
        return prefix
    if not case_name:
        raise ConfigError("没有网格文件时必须提供算例名作为输出前缀")
    return str(case_name)


def resolve_args(explicit: Mapping[str, Any], config: Config | None,
                 defaults: argparse.Namespace) -> argparse.Namespace:
    """合并 默认值 → JSON → 显式 CLI，并记录每个参数的来源。"""
    merged = dict(vars(defaults))
    sources = {key: "default" for key in merged}
    if config is not None:
        merged.update(config.values)
        sources.update({key: config.sources[key] for key in config.values})
    merged.update(explicit)
    sources.update({key: "cli" for key in explicit})

    # precision 是互斥开关：显式 --fp32/--fp64 必须压掉另一侧。
    if explicit.get("fp32"):
        merged["fp32"], merged["fp64"] = True, False
        sources["fp32"], sources["fp64"] = "cli", "cli"
    elif explicit.get("fp64"):
        merged["fp32"], merged["fp64"] = False, True
        sources["fp32"], sources["fp64"] = "cli", "cli"

    # 适用范围按"JSON 显式值 ∪ 命令行显式值"校验，纯 CLI 也执行：
    # explicit 只含用户真给过的参数（缺省值不在内），合并缺省值会把
    # p_grad=0.1 之类的缺省误判为越界。此前只在有 JSON 时校验且只查
    # config.values——`--case sod --outlet-ghost copy` 能通过解析（Astra
    # 第十轮）。
    given = {**(config.values if config is not None else {}), **explicit}
    _validate_applicability(given, merged.get("case", "sod"))

    fmt = merged.get("mesh_format")
    if fmt is None:
        fmt = "cgns" if merged.get("mesh") else "builtin"
        merged["mesh_format"] = fmt
        sources["mesh_format"] = "mesh.format(推断)"
    if fmt not in MESH_FORMATS:
        raise ConfigError(f"mesh.format 需要 {MESH_FORMATS} 之一，得到 {fmt!r}")
    if merged.get("mesh") and fmt != "cgns":
        raise ConfigError("mesh.file 只在 mesh.format=\"cgns\" 时可用")
    if fmt == "cgns" and not merged.get("mesh"):
        raise ConfigError("mesh.format=\"cgns\" 需要 mesh.file")
    if fmt == "cgns" and merged.get("case", "sod") not in CGNS_CASES:
        raise ConfigError(
            f"{merged.get('case')} 不支持 CGNS 网格；mesh.format=\"cgns\" 只适用于 poiseuille/plate")
    # outlet_ghost 只有 CGNS 装配分支消费。内置网格的 plate 显式给出时此前
    # 静默通过，用户会以为跑了零梯度出口（Astra 第十一轮）。
    if fmt != "cgns" and "boundary_outlet_ghost" in given:
        raise ConfigError(
            "boundary.outlet_ghost 只适用于 CGNS 网格的 plate：内置网格不消费该开关")

    # 网格尺寸只保留 i/j/k 入口。JSON 不提供维数（Step 3 口径），
    # 命令行与默认值必须给出正整数。
    for canonical in ("ni", "nj", "nk"):
        if merged[canonical] < 1:
            raise ConfigError(
                f"网格尺寸 {canonical} 必须为正整数，得到 {merged[canonical]!r}")

    if merged.get("case") == "plate":
        if merged["plate_x0"] >= merged["plate_x1"]:
            raise ConfigError(
                f"plate_x0 必须小于 plate_x1，得到 {merged['plate_x0']:g} "
                f"和 {merged['plate_x1']:g}")
        if merged["plate_h"] <= 0.0:
            raise ConfigError(f"plate_h 必须为正数，得到 {merged['plate_h']:g}")

    case_name = merged.get("case", "sod")
    merged["out_prefix"] = derive_prefix(merged.get("mesh"), case_name)
    sources["out_prefix"] = "mesh.file" if merged.get("mesh") else "case.name"

    ns = argparse.Namespace(**merged)
    ns.config_path = str(config.path) if config is not None else None
    ns.config_sources = sources
    return ns
