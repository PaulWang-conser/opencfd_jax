"""一次运行的初始状态组装。

入口职责边界：
- ``main.py`` 负责 CLI/JSON 解析、RHS 总装、时间推进与输出；
- ``init.py`` 只负责精度设置、算例构造、网格/边界描述和初场组装。

``prepare(args)`` 必须在任何 JAX 数组创建之前调用，以便 ``--fp32`` 生效。
"""

from dataclasses import dataclass
from typing import Any, Mapping

from . import boundary, config
from .cases import build_case
from .mesh_data import BlockMesh, from_case


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


def prepare(args) -> InitState:
    """设置精度，装配算例、网格描述、边界执行配置和初场。"""
    configure_precision(args)
    case = build_case(args)
    desc = from_case(case)
    execution = boundary.build_execution(case, desc)
    return InitState(args=args, case=case, desc=desc, execution=execution,
                     cons0=case["cons0"])
