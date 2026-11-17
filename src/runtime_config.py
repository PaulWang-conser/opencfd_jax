"""`input.json` 运行配置：读取、校验、合并与输出前缀推导。

设计口径见 `docs/plans/2026-11-17-step3-input-requirements.md`：

- JSON 只承载运行参数，不设置网格维数、边界类型或 `DIF_BOUND/JAC_BOUND` 开关；
- 优先级为 程序默认值 → JSON → 用户显式命令行；
- 槽道仍从 OCFD `.in` 取维数、物性、壁温与边界开关，JSON 不覆盖这些字段；
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
CASE_NAMES = ("sod", "tg", "poiseuille", "channel", "plate")
MESH_FORMATS = ("builtin", "ocfd")


class ConfigError(ValueError):
    """配置字段、类型或来源冲突错误；消息自带 JSON 字段路径。"""


# section -> field -> (kind, argparse dest)
_SCHEMA = {
    "mesh": {
        "format": ("str", "mesh_format"),
        "file": ("path", "mesh"),
        "field": ("path", "field"),
        "infile": ("path", "infile"),
    },
    "physics": {
        "Re": ("number", "Re"),
        "Ama": ("number", "Ama"),
        "ref_t": ("number", "ref_t"),
        "pr": ("number", "pr"),
    },
    "numerics": {
        "scheme": ("str", "scheme"),
        "vis_order": ("str", "vis_order"),
        "viscous": ("bool", "viscous"),
        "weno_tv": ("bool", "weno_tv"),
        "filter": ("bool", "filter"),
        "nstep_filter": ("int", "nstep_filter"),
        "filter_s0": ("number", "filter_s0"),
        "filter_rth": ("number", "filter_rth"),
    },
    "time": {
        "dt": ("dt", "dt"),
        "cfl": ("number", "cfl"),
        "end_time": ("number", "end_time"),
        "end_step": ("int", "end_step"),
    },
    "case": {
        "name": ("str", "case"),
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
    "mesh.file": {"channel"},
    "mesh.field": {"channel"},
    "mesh.infile": {"channel"},
    "case.p_grad": {"poiseuille"},
    "case.forcing_work": {"channel", "poiseuille"},
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
    if fmt == "ocfd" and case != "channel":
        raise ConfigError(
            f"{case} 使用程序内置网格，mesh.format 只能是 \"builtin\"，得到 {fmt!r}")
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
    if "vis_order" in values and values["vis_order"] not in VIS_ORDERS:
        raise ConfigError(
            f"numerics.vis_order 不支持 {values['vis_order']!r}；可选 {VIS_ORDERS}")

    fmt = values.get("mesh_format")
    if fmt is not None and fmt not in MESH_FORMATS:
        raise ConfigError(f"mesh.format 需要 {MESH_FORMATS} 之一，得到 {fmt!r}")
    if fmt is None:
        fmt = "ocfd" if "mesh" in values else "builtin"
        values["mesh_format"] = fmt
        sources["mesh_format"] = "mesh.format(推断)"
    if "mesh" in values and fmt != "ocfd":
        raise ConfigError("mesh.file 只在 mesh.format=\"ocfd\" 时可用")
    if fmt == "ocfd" and "mesh" not in values:
        raise ConfigError("mesh.format=\"ocfd\" 需要 mesh.file")

    if case == "channel":
        if "infile" not in values:
            raise ConfigError(
                "channel 在 Step 3 仍需要 mesh.infile（OCFD .in 提供维数、物性、"
                "壁温与边界开关）")
    _validate_applicability(values, case)

    if values.get("plate_x0") is not None and values.get("plate_x1") is not None:
        if values["plate_x0"] >= values["plate_x1"]:
            raise ConfigError(
                f"case.plate_x0 必须小于 plate_x1，得到 {values['plate_x0']:g} "
                f"和 {values['plate_x1']:g}")
    if "end_step" in values and values["end_step"] < 1:
        raise ConfigError("time.end_step 至少为 1")
    if "cfl" in values and values["cfl"] <= 0.0:
        raise ConfigError("time.cfl 必须为正数")


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

    allowed_top = set(_SCHEMA) | {"precision"}
    unknown = sorted(set(raw) - allowed_top)
    if unknown:
        raise ConfigError(f"未知顶层字段：{unknown}")

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

    if config is not None:
        _validate_applicability(config.values, merged.get("case", "sod"))

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
