"""OpenCFD-SCU 的 JAX 求解模块：守恒量场、数值格式、空间离散与时间推进。"""

from . import config, fields, schemes, solver

__all__ = ["config", "fields", "schemes", "solver"]
