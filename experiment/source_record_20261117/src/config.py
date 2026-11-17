"""精度开关。

决策（2026-11-15 起）：默认 **FP64**。理由：这套代码的现状是"逐点对拍、判定线
1e-12"，FP32 下舍入噪声就把误差锁在 1e-6 量级，任何对拍都无意义；性能要单独
用 FP32 档位（`--fp32`）去测，不要拿它当默认。

本模块必须在入口文件第一行 import——任何数组创建之前。
"""

import jax

# 默认 FP64
jax.config.update("jax_enable_x64", True)


def enable_x64():
    """切到 FP64（缺省档位，保留此函数给显式调用）。"""
    jax.config.update("jax_enable_x64", True)


def disable_x64():
    """切到 FP32（仅性能基准用）。"""
    jax.config.update("jax_enable_x64", False)


def precision_name():
    """返回当前 JAX 浮点精度名称，供启动日志显示。"""
    return "float64" if jax.config.jax_enable_x64 else "float32"
