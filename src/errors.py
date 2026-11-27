"""运行入口的异常分层。

``SetupError`` 覆盖启动时的初始化失败：``init.prepare``、多块的放行校验和网格校验、
装配各块、读续算的检查点；由 ``main.main`` 写成一行“初始化错误”，退出码 2。
``RuntimeExecutionError`` 覆盖进入计算以后的四个阶段：装配 RHS、RHS 预热、时间推进、
存盘；保留原始异常链和运行位置上下文，由未捕获异常路径输出完整 traceback。
两者都不改变底层模块抛出的原始异常类型，只在上层边界包装。
"""


class SetupError(Exception):
    """启动时初始化失败；消息应包含 case、网格来源和原始异常类型。"""


class RuntimeExecutionError(RuntimeError):
    """装配 RHS、RHS 预热、时间推进或存盘失败；通过 ``raise ... from exc`` 保留原始异常链。"""
