# special_boundaries/ —— specificboundary 脚本目录

`input.json` 顶层键 `"specificboundary"`（或 CLI `--specificboundary`）的值
只填本目录下的**纯 .py 文件名**；脚本由 `init.attach_specificboundary` 在
JIT 之前加载，与 `plate`/`channel` 特殊边界互斥。

## 信任边界（必读）

文件名校验**不提供执行隔离**：脚本经 `exec_module` 以求解器进程权限执行
（顶层代码在入口检查之前就会运行）。本目录中的脚本必须由运行者亲自审查、
部署；目录本身不可由不受信任方写入。symlink 指向目录外会被拒绝。

## 脚本契约

```python
from types import SimpleNamespace


def build_special_boundary(case):
    """返回特殊边界对象；case 是 build_case 的算例字典（只读，勿改）。"""
    def bc_apply(q):
        """边界回调：接收守恒量数组，返回同形状、同 dtype 的守恒量。

        必须可被 JAX 追踪（纯 jnp 操作，无 Python 分支依赖数值）。
        在每个 RK 子步的常规 halo 填充之后、RHS 之前调用。
        """
        return q.at[0, :, :, :4].set(1.0)      # 示例：i- 侧四层写密度 1

    return SimpleNamespace(
        name="my_jet",       # 必需：字符串
        bc_apply=bc_apply,   # 必需：上面的普通函数（不要用未绑定的实例方法）
        faces=(),            # 可选：六侧 FaceBoundary；缺省继承已解析的六侧
    )                        # （CGNS 路径保留异类正负面/壁温/逐侧带标志）；
                             # 给出时须与派生六侧逐侧一致
```

## 校验时机与顺序

- 文件名 / `.py` 后缀 / 目录归属 / 互斥 / 缺入口：`init.attach_specificboundary`
  （在 `build_case` 之后、`from_case` 之前——初场已由 `build_case` 创建，
  精度设置更早，在 `prepare` 开头）；
- `faces` 六侧一致性、WALL 方向限制：`boundary.build_execution` 的 custom
  分支与 `validate_special_sides`（同样 JIT 前）；
- 回调本身在每个 RK 子步执行（`BoundaryExecution.prepare` 的
  "写边界点 → 填鬼层 → 特殊回调"顺序，整步结束后再 `prepare` 一次）。
