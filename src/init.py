"""一次运行的初始状态组装。

入口职责边界：
- ``main.py`` 负责 CLI/JSON 解析、RHS 总装、时间推进与输出；
- ``init.py`` 只负责精度设置、算例构造、网格/边界描述和初场组装。

``prepare(args)`` 必须在任何 JAX 数组创建之前调用，以便 ``--fp32`` 生效。
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import boundary, config
from .cases import build_case
from .mesh_data import BlockMesh, from_case

# specificboundary 脚本目录（4.n 规格：值只填该目录下的脚本名）
SPECIAL_BOUNDARY_DIR = Path(__file__).resolve().parent / "special_boundaries"


@dataclass(frozen=True)
class InitState:
    """一次运行在进入时间推进前所需的只读状态。"""

    args: Any
    case: Mapping
    desc: BlockMesh
    execution: boundary.BoundaryExecution
    cons0: Any


def configure_precision(args) -> None:
    """在创建任何 JAX 数组之前设置 FP64/FP32。"""
    if getattr(args, "fp32", False):
        config.disable_x64()
    else:
        config.enable_x64()


def attach_specificboundary(args, case) -> None:
    """按 ``args.specificboundary`` 加载特殊边界脚本并挂到 case（JIT 前）。

    脚本契约（4.n）：入口 ``build_special_boundary(case)`` 返回带
    ``name``/``bc_apply`` 的对象（``faces`` 可选，缺省继承已解析六侧）。
    非法文件名、非 .py、目录外目标（symlink 逃逸）、缺脚本、缺入口、
    与 plate/channel 冲突都在此报错。

    信任边界：文件名校验不提供执行隔离——脚本由 ``exec_module`` 以求解器
    进程权限执行，**必须**由运行者信任并部署到 ``SPECIAL_BOUNDARY_DIR``；
    该目录不可由不受信任方写入。
    """
    name = getattr(args, "specificboundary", None)
    if not name:
        return
    if (name != Path(name).name or "/" in name or "\\" in name
            or name in (".", "..")):
        raise ValueError(
            f"specificboundary 只接受 {SPECIAL_BOUNDARY_DIR} 下的纯文件名，"
            f"得到 {name!r}")
    if not name.endswith(".py"):
        raise ValueError(
            f"specificboundary 只接受 .py 脚本，得到 {name!r}")
    case_name = str(getattr(args, "case", "") or "").strip().lower()
    if (case.get("plate") is not None or case.get("channel_bc") is not None
            or case_name in ("plate", "channel")):
        raise ValueError(
            "specificboundary 与 plate/channel 特殊边界互斥"
            f"（case={getattr(args, 'case', None)!r}）")
    path = SPECIAL_BOUNDARY_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"specificboundary 脚本不存在：{path}（只接受该目录下的 .py 文件名）")
    # resolve 后仍在目录内：挡 symlink 指向目录外的执行（Astra 复审）
    resolved = path.resolve()
    if resolved.parent != SPECIAL_BOUNDARY_DIR.resolve():
        raise ValueError(
            f"specificboundary 解析到目录外：{resolved}（不允许 symlink 逃逸）")
    spec = importlib.util.spec_from_file_location(
        f"opencfd_jax_special_boundaries.{path.stem}", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法为 {resolved} 构建模块加载器")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, "build_special_boundary", None)
    if not callable(entry):
        raise ValueError(f"{path} 缺少入口 build_special_boundary(case)")
    special = entry(case)
    callback = getattr(special, "bc_apply", None)
    if not callable(callback):
        raise ValueError(f"{path} 的 build_special_boundary 返回对象缺少 bc_apply")
    case["special_boundary"] = special
    case["bc_apply"] = callback


def prepare(args) -> InitState:
    """设置精度，装配算例、网格描述、边界执行配置和初场。"""
    configure_precision(args)
    case = build_case(args)
    attach_specificboundary(args, case)
    desc = from_case(case)
    execution = boundary.build_execution(case, desc)
    return InitState(args=args, case=case, desc=desc, execution=execution,
                     cons0=case["cons0"])
