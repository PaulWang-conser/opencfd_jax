"""运行入口的异常分层。

``SetupError`` 只覆盖 ``init.prepare`` 的初始化失败，由 ``main.main`` 转成
启动错误；``RuntimeExecutionError`` 覆盖时间推进阶段，保留原始异常链和
运行位置上下文，由未捕获异常路径输出完整 traceback。两者都不改变底层
模块抛出的原始异常类型，只在上层边界包装。
"""


class SetupError(Exception):
    """启动前初始化失败；消息应包含 case、网格来源和原始异常类型。"""


class RuntimeExecutionError(RuntimeError):
    """时间推进失败；通过 ``raise ... from exc`` 保留原始异常链。"""
