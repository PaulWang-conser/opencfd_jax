---
title: Step 4.n 边界执行扩展与特殊边界顺序 - Plan
type: feat
date: 2026-11-18
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Step 4.n 边界执行扩展、特殊边界顺序与错误分层 - Plan

## 实施进度（2026-11-18）

通用 `FARFIELD`（R1–R4 的边界填充、状态校验、`Ev` 零梯度）与 CGNS `BCFarfield` 到通用远场的适配已实现并有回归测试；R5–R20 的四阶段 `prepare`、`specificboundary` 顶层入口和 `main.py` 错误分层仍未实现。本计划其余内容保持为待实施设计。

## Goal Capsule

在 Step 4 的 CGNS 单块读取之后，补上四项能力：通用边界支持 `FARFIELD`（它属于通用边界，不通过特殊边界脚本实现）；特殊边界改为在常规 halo 填充之后叠加；`input.json` 增加正式顶层入口 `specificboundary`；`main.py` 区分初始化/配置错误与运行期异常，不再把 `run()` 的任意 `ValueError` 吞成启动参数错误。

- **Authority.** 本计划以本轮已确认的四项要求为产品约束：远场进入通用边界；特殊边界在常规边界之后执行；`specificboundary` 是正式 JSON key；运行期异常必须保留原始类型、上下文和完整 traceback。Step 4 的 CGNS 读取范围不变，本计划不实现 CGNS `BCFarfield` 的 `BCDataSet` 解析。
- **Execution profile.** 只改边界执行、RHS 调用顺序、JSON 配置入口、主入口错误分层、测试与文档；不改变无粘/粘性离散公式，不改变网格来源，不引入新物理模型，不扩展到多块。
- **Stop conditions.** 若顺序调整导致 `plate`/`channel` 的内区结果无法保持、若 `specificboundary` 必须依赖固定目录外的脚本或任意文件路径才能工作、若远场状态无法在 JIT 前确定、或错误分层必须吞掉运行期 traceback 才能保持旧测试，停止并报告，不用临时兼容分支掩盖。
- **Tail ownership.** 在本机完成实现、验证和本地 Git 提交；不推送远端、不创建 PR。

---

## Product Contract

### Summary

当前通用边界只有周期、出流和计算 j 向壁面；`FARFIELD` 只是平板顶部回调的描述，不是 `GENERIC_KINDS` 成员。特殊边界的执行顺序是“case 回调先、逐轴 halo 后”，与“特殊边界在常规边界之后”相反；`rk3_step_stages` 还会在子步前后重复调用回调。`input.json` 也没有加载特殊边界的入口。`main.py` 又在 `run()` 外层捕获任意 `ValueError` 并转成 `parser.error`，把运行期的 JAX/数值错误伪装成启动参数错误，丢失 traceback。

Step 4.n 把边界执行统一成一次 `prepare`：先按六侧 `executor` 填常规 halo，再叠加特殊回调，再执行 `callback+halo` 的专属填充，最后对未被特殊面接管的常规面做一次角点清理。`FARFIELD` 作为通用边界成员，采用均匀自由来流守恒量的 Dirichlet 口径；它不是特殊边界，也不由 `specificboundary` 脚本提供。`specificboundary` 只填特殊边界脚本名，加载器按固定目录和固定入口函数装配现有的 `special_boundary` 对象。错误处理分成三层：配置解析错误、初始化错误、运行期异常；只有前两层转成启动错误，运行期异常保留原始异常链和运行位置上下文。

### Problem Frame

四个缺口互相牵连，不能只改一个分支：

- **远场缺口.** `src/boundary.py` 定义了 `FARFIELD` 常量，但 `GENERIC_KINDS` 不含它，`fill_halos_i/j/k` 也没有分支；`src/mesh_data.py` 的 `_generic_faces` 不允许它。通用远场如果只加一个“复制相邻内点”的分支，会把平板顶部的 Dirichlet 口径误当成通用远场。
- **顺序缺口.** `src/main.py::build_rhs` 先 `execution.apply(q)`，再进入 `src/solver.py::inviscid_rhs`；后者内部又调用 `fill_halos_i/j/k`。`src/solver.py::rk3_step_stages` 在 stage1 前和 q1/q2/q3 后各调一次 `bc_apply`，而 `rhs` 又调一次；平板/槽道回调每个 RK 步最多进入约 7 次。`src/viscous.py::viscous_rhs` 也自行填状态鬼层。仅把 `rhs` 里的两行对调不够，因为轴级 `bcs` 在“一侧特殊、另一侧普通出流”时会整体 no-op，普通出流侧只能靠回调后的 `fill_mixed_outflow` 补。
- **入口缺口.** `src/runtime_config.py` 的 `_SCHEMA` 顶层只有 `mesh` / `physics` / `numerics` / `time` / `case` / `output`，`allowed_top` 额外允许 `precision`；未知顶层字段直接报错。因此 `specificboundary` 目前没有正式入口，也不能靠 `case` 下的临时键绕过。
- **错误分层缺口.** `src/main.py::main` 在 `run()` 外层用 `except ValueError` 统一转成 `parser.error("运行参数错误：...")`。`init.prepare`、`build_rhs` 和真正的 JAX 时间推进都可能抛 `ValueError`；现有代码无法区分“启动前配置/初始化错误”和“时间推进中的运行错误”，后者被吞掉 traceback，定位只能靠复跑。不能简单地扩大捕获范围或打印完整 traceback 到 `stderr` 后继续，必须按阶段分层并把原始异常链保留下来。

顺序问题的根因是“轴级 `bcs` 决定整轴行为”，而执行意图已经存在于六侧 `FaceBoundary.executor`。只要把常规填充改成按面执行，才能表达同一轴一侧回调、另一侧出流的组合；也才能把 `callback+halo` 的壁面镜像和周期角点按正确顺序分开。

### Requirements

**通用远场**

- R1. `FARFIELD` 加入 `GENERIC_KINDS`，`src/mesh_data.py::_generic_faces` 能把轴级 `FARFIELD` 展开成六侧 `FaceBoundary(kind=FARFIELD, executor="halo")`。`FARFIELD` 是通用边界条件，不是特殊边界；不得要求用户通过 `specificboundary` 脚本实现它。
- R2. 通用远场的数值语义与平板顶部一致：把自由来流守恒量写到该侧的边界层内点加 `2*LAP` 鬼层；粘性 `Ev` 鬼层对该侧采用零梯度复制。不实现特征边界、不按法向马赫切换、不读 `BCDataSet`。
- R3. 远场状态由算例在 JIT 前提供为 5 分量守恒量；缺键、形状不对、含非有限值或密度非正时，在构造 `BoundaryExecution` 时直接报错。
- R4. `zero_state_ghosts=True` 与通用 `FARFIELD` 同时出现时，必须明确说明谁覆盖谁；默认保持平板历史口径（清零在后），并在文档中标注该组合会让远场鬼层归零。

**执行顺序**

- R5. `BoundaryExecution` 提供唯一的 `prepare(q)`，顺序固定为：常规 `executor="halo"` 面填充 → 特殊回调 → `executor="callback+halo"` 面填充 → 常规面角点清理。
- R6. `prepare` 的常规填充按 `FaceBoundary` 的 side/axis 执行，不再依赖轴级 `fill_halos_i/j/k` 对整轴同时生效；同一轴一侧特殊、另一侧 `OUTFLOW` 的组合必须能正确表达。
- R7. `rhs` 在每次计算 `du` 前调用一次 `prepare`；`rk3_step` / `rk3_step_stages` 不再接收或调用 `bc_apply`，所有仓库内消费者（含 `src/compare.py`）迁移到新口径。
- R8. `inviscid_rhs` 与 `viscous_rhs` 增加“状态已准备”的入口，跳过内部状态 halo 填充；`zero_state_ghosts` 仍在准备之后按现有语义执行。粘性 `Ev` 鬼层继续使用现有 `fill_ev_axis` / `fill_ev_plate_j` / `fill_ev_wall_j` 语义，只为通用 `FARFIELD` 增加零梯度分支；不在本步重写其他公式。
- R9. 状态路径不再使用 `fill_mixed_outflow(callback(q))`；`fill_mixed_outflow` 只保留给粘性 `Ev` 的混合出流侧，并在文档中说明保留原因。

**`specificboundary` 入口**

- R10. `input.json` 顶层正式 key 为 `specificboundary`，值只填一个特殊边界脚本名（例如 `"my_boundary"`，也可写 `"my_boundary.py"`），不填对象、模块路径、函数名或参数。这不改变 Step 3 的“JSON 不设置常规边界类型或 `DIF_BOUND` 开关”口径；`specificboundary` 只是特殊边界脚本的显式加载入口。
- R11. 加载器按固定目录 `src/special_boundaries/` 解析这个脚本名，拒绝绝对路径、路径分隔符、`..`；不带 `.py` 时补上 `.py`，带其他后缀则报错。脚本必须定义公开入口 `build_special_boundary(case)`，返回带 `name`、`faces`、`apply` 的对象。脚本参数由脚本自身或算例提供，JSON 不再传 `params`。
- R12. `init.prepare` 在 `build_case` 之后、`from_case` 之前加载 `specificboundary`，把返回对象挂到 `case["special_boundary"]`，把 `apply` 挂到 `case["bc_apply"]`；`from_case` 继续负责六侧与轴级 `bcs` 校验。
- R13. `specificboundary` 与 `plate` / `channel_bc` / 已有 `special_boundary` 互斥；冲突在配置解析或初始化阶段报错，不进入 JIT。本步不提供 CLI 覆盖入口。

**文档与验证**

- R14. 更新 `src/README.md` 第 7.2 节、路线图 Step 4.n 段和 Step 4 计划的 deferred 条目，使 `specificboundary`、通用 `FARFIELD`、执行顺序与错误分层在文档中一致。
- R15. 新增 Step 4.n 测试，覆盖远场填充、顺序、`specificboundary` 加载/拒绝、`plate`/`channel` 一步结果保持。
- R16. `--filter` 与含 `FARFIELD` 的 `bcs` 同时启用时，在 JIT 前报错并提示 filtering 尚未迁移；本步不重写 `shock_filter` 的逐轴填充。

**错误分层**

- R17. `main.py` 不再用 `except ValueError` 包住整个 `run()`。配置解析错误继续由 `runtime_config.ConfigError` 转成 `配置错误`；初始化错误由独立的 `SetupError` 转成 `初始化错误`；运行时 `ValueError` 不得再被转成 `parser.error`。
- R18. `run()` 的初始化段（`init.prepare`）把 `ValueError`/`OSError`（含 `CgnsError`）包装成 `SetupError`，消息至少包含 `case`、`mesh_format`、`mesh` 路径和原始异常类型；`main` 只捕获 `SetupError`，错误信息保留原始原因。
- R19. `run()` 的时间推进段（`build_rhs`、RHS 预热、RK 循环、滤波、存盘）捕获 `Exception` 后抛出 `RuntimeExecutionError`，消息至少包含 `case`、`step`、`t`、`dt`、网格尺寸和原始异常类型；必须使用 `raise ... from exc` 保留原始 traceback 与 `__cause__`，不得打印后继续或降级成参数错误。
- R20. 新增 `src/errors.py` 存放 `SetupError` 与 `RuntimeExecutionError`；`main.py` 只做异常分类，不把运行期异常转成 `SystemExit`。启动错误退出码仍由 `parser.error` 提供，运行期异常按未捕获异常退出，traceback 可见。

### Scope Boundaries

**In scope**

- `src/boundary.py` 的通用远场、按面常规填充、`prepare` 四阶段顺序、`FARFIELD` 的 Ev 零梯度。
- `src/mesh_data.py` 的 `FARFIELD` 六侧描述与 `validate_special_sides` 放宽。
- `src/main.py`、`src/solver.py`、`src/viscous.py` 的调用顺序与“状态已准备”入口。
- `src/runtime_config.py`、`src/init.py`、新增 `src/specific_boundary.py` 的 JSON 入口与加载器。
- `src/errors.py`、`src/main.py`、`src/init.py` 的错误分层；`experiment/test_step4n_boundary.py` 中的启动/运行错误测试。
- `experiment/test_step4n_boundary.py`、相关旧测试迁移、`src/README.md`、路线图、Step 4 计划 deferred。

**Deferred to follow-up work**

- CGNS `BCFarfield` 的 `BCDataSet` 读取和来流参数来源；CGNS patch 到通用远场的类型映射已实现，但不解析 `BCDataSet`，默认状态由 `poiseuille` 从初场内区两端平均得到。
- 把 `src/` 下所有底层 `ValueError` 改造成完整的自定义异常层级；本步只建立主入口的 `SetupError` / `RuntimeExecutionError` 分界。
- 运行期异常的结构化日志、JSON 错误报告和自动复现包；本步保留 traceback 并加运行位置上下文。
- 同侧多个 farfield patch、每面不同来流状态、子区远场。
- 特征边界（按法向马赫进/出口切换）、无反射远场、总压/总焓/熵进口。
- `specificboundary` 的 CLI 覆盖、脚本参数化（`params` 或脚本参数块）、路径式脚本加载、跨仓库脚本。
- 特殊边界对 `Ev` 的全新物理模型；本步只保持现有 `Ev` 分支，并为通用 `FARFIELD` 增加零梯度。
- `filtering.shock_filter` 的逐轴填充迁移；本步对 `--filter` + `FARFIELD` 直接报错，不静默运行。
- 多块、并行、完整 Pointwise BC 表。

**Outside this product's identity**

- 非结构网格、重叠网格、自动块分解。
- 把 `specificboundary` 做成动态注册的插件市场。
- 用 JSON 直接设置常规边界类型或 `DIF_BOUND` 开关。

### Acceptance Examples

- AE1. 构造 `bcs=(PERIODIC, FARFIELD, PERIODIC)` 的小网格，给出 `case["farfield"]`；`prepare` 后 `j-`/`j+` 的边界层内点与 `2*LAP` 鬼层都等于该守恒量，`i`/`k` 周期鬼层仍按周期值填充。
- AE2. 构造 i- 特殊回调、i+ 普通出流的 custom case；回调执行时 i+ 出流鬼层已经被常规填充；回调只改 i-；`prepare` 结束后 i- 保持特殊值、i+ 保持出流值、j/k 角点回到周期值。
- AE3. `plate` 与 `channel` 的一步 `cons`、三个子步 `du` 与改动前 NPZ 逐位一致；若不能逐位一致，必须把差异定位到具体鬼层/角点并报告，不能放宽断言掩盖。
- AE4. 合法 `specificboundary` 脚本名被加载并挂到 case；脚本不存在、名字含路径、带其他后缀、缺少 `build_special_boundary(case)`、返回对象缺 `apply`、与 `plate`/`channel` 冲突分别在 JIT 前报错。
- AE5. `specificboundary` 与另一轴通用 `FARFIELD` 同时存在时能通过 `from_case` 和 `prepare`，且回调次数不再是每步 7 次。
- AE6. 显式 `--ni` 与 CGNS `VertexSize` 冲突时，`main.py` 输出 `初始化错误：...` 和原始异常类型，退出码为 2，不打印误导性的 `运行参数错误`；非启动路径缺 `--mesh` 仍由 `_validate_input_paths` 报启动错误。
- AE7. 在时间推进中注入一个 `ValueError`，`main.py` 不再把它转成 `parser.error`；抛出 `RuntimeExecutionError`，stderr 同时包含运行位置（`case/step/t/dt`）、原始异常类型和完整链式 traceback（`The above exception was the direct cause...`）。

---

## Planning Contract

### Key Technical Decisions

- KTD1. **按面填充替代轴级整轴填充.** `FaceBoundary` 已经保存 `side/axis/kind/executor`，执行层应按面生成填充计划。轴级 `bcs` 继续作为兼容与校验字段保留，但不再直接决定整轴 `fill_halos_i/j/k` 的行为。这样同一轴一侧特殊、另一侧出流才能在不靠 `mixed_outflow` 的情况下表达。
- KTD2. **四阶段 `prepare`.** 顺序是常规 halo → 特殊回调 → `callback+halo` → 常规角点清理。第三阶段保留壁面镜像等“回调后仍需常规填充”的面；第四阶段只在 `executor="halo"` 的面上重跑，用于恢复被特殊回调跨轴写到的周期角点。平板/槽道的现有最终状态因此可以保持。
- KTD3. **`rhs` 是唯一状态准备点.** `build_rhs` 内部调用 `execution.prepare(q)`，然后以“状态已准备”模式调用无粘/粘性 RHS。`rk3_step` / `rk3_step_stages` 删除 `bc_apply` 参数；`main.run` 在整步 JIT 的外层再调用一次 `execution.prepare`，保证输出状态也带边界层。编译图中每步状态准备从约 7 次降到 3 次 RHS + 1 次最终准备；Python 侧具体可见次数由测试用 `jax.disable_jit()` 固定。
- KTD4. **通用远场采用均匀 Dirichlet.** 不把平板顶部的“顶部来流”语义抽象成特征边界；通用 `FARFIELD` 只做“把给定守恒量写到该侧边界层+鬼层”。这与当前平板顶部 `far_state` 的赋值口径一致，也避免在 Step 4.n 引入未验证的无反射公式。
- KTD5. **远场状态由 case 提供.** `BoundaryExecution` 从 `case["farfield"]` 读取 5 分量守恒量；本步不把远场参数塞进 `physics` JSON，也不从 CGNS `BCDataSet` 推断。CGNS 适配和 JSON 暴露留给后续计划。
- KTD6. **`specificboundary` 只放脚本名.** JSON 值是一个脚本名（可带 `.py`），目录、入口函数和对象形状由加载器约定；脚本负责构造六侧 `FaceBoundary` 与 `apply`。加载器不执行任意表达式，不做动态注册，不把 I/O 放进 JIT。参数化留给后续，不在本步给 JSON 增加第二个键。
- KTD7. **固定脚本目录.** 本步只允许 `src/special_boundaries/` 下的 `.py` 脚本，避免 `input.json` 变成任意代码执行入口。路径式脚本加载、跨仓库脚本和脚本参数化留给后续；如果用户需要仓库外脚本，必须先扩展这一层并单独评审。
- KTD8. **`fill_mixed_outflow` 只保留给 Ev.** 状态路径由按面常规填充解决；粘性 `Ev` 仍按现有混合出流侧补零梯度，并为通用 `FARFIELD` 增加零梯度分支。其他 `Ev` 公式不重写。
- KTD9. **错误按阶段分层，不按异常类型猜测.** 配置错误在 `resolve_args` 阶段已经由 `ConfigError` 标识；初始化错误在 `init.prepare` 边界包装成 `SetupError`；时间推进段包装成 `RuntimeExecutionError`。`main` 只处理前两层，运行期异常始终走 Python 未捕获异常路径。这样不依赖“哪个模块抛了哪种 `ValueError`”的脆弱分类，也不会吞掉 JAX 的真实栈。
- KTD10. **运行期上下文用异常链增强，不改写原异常.** `RuntimeExecutionError` 只携带 `case/step/t/dt/grid` 和原始异常类型；原始异常用 `raise ... from exc` 保留。禁止把 `traceback.format_exc()` 打印到 stdout 或返回码 2 的 `parser.error`，否则调用方仍无法区分启动失败与运行失败。

### Assumptions

- `case["farfield"]` 是守恒量 `(rho, rho*u, rho*v, rho*w, rho*E)`，不是原始量；这样执行层不需要再读 `Cv` 或 `gamma`。
- `specificboundary` 的值是 `src/special_boundaries/` 下的裸脚本名（可带 `.py`），不是模块路径、函数名或参数对象。
- 脚本入口固定为 `build_special_boundary(case)`，在 `build_case` 之后调用，可以使用 `case["grid"]`、`case["bcs"]`、`case["params"]` 等已装配对象，但不能要求 `BlockMesh` 已存在。
- `plate` / `channel` 的一步基线以现有 NPZ 为准；本步不接受“只比较内区、忽略边界层”的放宽。
- `RuntimeExecutionError` 只改变主入口的异常分类与上下文，不改变异常退出状态；现有脚本不应依赖“运行期 ValueError 也返回 2”的旧行为。

### Risks & Dependencies

- **角点顺序风险.** 平板和槽道回调会跨 k 写鬼层；如果第四阶段清理不覆盖这些角点，`plate`/`channel` 结果会漂移。缓解：AE3 逐位对比；U2 先写顺序测试再迁移主链。
- **`zero_state_ghosts` 交互风险.** 平板历史口径会在准备后清零非周期鬼层；通用远场如果误用同一开关，会把 Dirichlet 鬼层清掉。缓解：R4 明确组合语义，U1 加显式断言。
- **脚本加载安全风险.** 裸文件名如果拼接成路径，可能跳出固定目录。缓解：KTD7 的目录与文件名校验；错误信息只打印文件名，不执行目录扫描。
- **签名迁移风险.** 删除 `rk3_step(_stages)` 的 `bc_apply` 会影响 `experiment/compare_ocfd_jax.py` 与旧测试。缓解：U2 一次性迁移所有仓库内调用，并用 `rg` 复核无残留。
- **Ev 顺序风险.** 状态顺序改变后，`viscous_rhs` 的 Ev 填充仍按轴级 `bcs`，混合出流侧依赖 `fill_mixed_outflow`。缓解：本步不重写 Ev 公式；AE3 只要求状态与 `du` 保持，Ev 差异必须单独报告。
- **错误分层误伤风险.** `init.prepare` 的 `SetupError` 边界如果包住 `build_rhs` 或首个 JIT 预热，会把运行期错误误标成初始化错误；反过来，运行期包装如果放在 `init.prepare` 外层，会把 CGNS 读取错误降级成 traceback 而丢失启动语义。缓解：分层点固定在 `init.prepare` 返回处，`build_rhs` 与 `rhs(cons)` 预热归运行段；AE6/AE7 分别用真实启动冲突和注入运行异常验证。
- **traceback 丢失风险.** 用 `parser.error`、`logging.exception` 或只打印 `str(exc)` 都会丢掉原始栈。缓解：只允许 `raise RuntimeExecutionError(...) from exc`；测试断言 `__cause__` 是原始异常且 stderr 含链式 cause 提示。

### Sequencing

1. U1 先建立按面填充与通用远场，保持现有 `prepare` 尚未接管主链。
2. U2 再引入四阶段 `prepare`，迁移 `rhs`/RK/粘性调用，跑 `plate`/`channel` 一步基线。
3. U3 在顺序稳定后接入 `specificboundary`，避免把配置加载和数值顺序混在同一次改动里。
4. U5 在 `run`/RK 调用稳定后建立初始化与运行期错误分层，避免错误边界测试被后续签名迁移打乱。
5. U4 最后统一文档、旧测试迁移、错误测试并入和全量验证。

---

## Implementation Units

### U1. 通用 FARFIELD 与按面常规填充

**Goal.** 让 `FARFIELD` 成为可执行的通用边界，并建立按 `FaceBoundary` 执行的常规填充函数，为 U2 的顺序改造提供基础。

**Requirements.** R1, R2, R3, R4, AE1.

**Dependencies.** None.

**Files.** `src/boundary.py`; `src/mesh_data.py`; `experiment/test_step4n_boundary.py`.

**Approach.** 在 `src/boundary.py` 增加 `FARFIELD` 到 `GENERIC_KINDS`，新增 `fill_face_halo(f, face, grid, farfield=None)`：periodic 按 side 复制对侧内点，outflow 按 side 复制相邻内点，wall 只允许 j 向并复用镜像符号，farfield 把 `farfield` 写到该侧边界层内点加 `2*LAP` 鬼层。`fill_ev_axis` 增加 `FARFIELD` 分支，对该侧 Ev 鬼层做零梯度复制。`BoundaryExecution` 增加 `farfield` 字段与 `fill_generic` / `fill_post` 方法；`fill_generic` 只处理 `executor="halo"`，`fill_post` 只处理 `executor="callback+halo"`。`src/mesh_data.py::_generic_faces` 允许 `FARFIELD`；`validate_special_sides` 在无回调轴上允许 `FARFIELD`，不再只允许 `PERIODIC`/`OUTFLOW`。

**Patterns to follow.** `src/boundary.py::fill_halos_i/j/k` 的边界索引；`src/boundary.py::fill_wall_j` 的镜像符号；`src/plate.py::far_state` 的守恒量顺序；`src/mesh_data.py::_generic_faces` 的六侧展开。

**Test Scenarios.**

- `FARFIELD` 在 `GENERIC_KINDS` 中，`_generic_faces` 生成六侧 `executor="halo"`。
- `fill_face_halo` 对 j-/j+ farfield 写入 1 个边界内点 + `2*LAP` 鬼层；i/k 周期不受影响。
- `fill_ev_axis` 对 `FARFIELD` 侧做零梯度复制；`plate`/`channel`/`OUTFLOW` 的 Ev 结果不变。
- 缺少 `case["farfield"]`、形状不是 `(5,)`、含 NaN/inf、密度 <= 0 时 `build_execution` 报错。
- `zero_state_ghosts=True` + farfield 的行为与文档一致；测试显式断言鬼层被清零，防止误以为仍保留 Dirichlet。
- `--filter` 与含 `FARFIELD` 的 `bcs` 同时启用时，在 JIT 前报错并给出 filtering 未迁移的提示。

**Verification.** `python -m pytest experiment/test_step4n_boundary.py -q`。

### U2. 四阶段 `prepare` 与 RHS/RK 迁移

**Goal.** 把状态边界执行统一到 `BoundaryExecution.prepare`，实现“常规 → 特殊 → callback+halo → 角点清理”，并去掉重复回调。

**Requirements.** R5, R6, R7, R8, R9, AE2, AE3.

**Dependencies.** U1.

**Files.** `src/boundary.py`; `src/main.py`; `src/compare.py`; `src/solver.py`; `src/viscous.py`; `experiment/compare_ocfd_jax.py`; `experiment/test_step2_boundary.py`; `experiment/test_step4n_boundary.py`.

**Approach.** `BoundaryExecution.prepare(q)` 按 KTD2 四阶段执行；删除只做回调的公开 `apply`，特殊回调只作为 `prepare` 的第二阶段内部调用，避免再出现“只叠回调不填常规”的旧入口。`src/main.py::build_rhs` 改为 `q = execution.prepare(q)`，并向 `inviscid_rhs` / `viscous_rhs` 传 `fill_state_halos=False`。`src/solver.py::_rhs_along_axis` 与 `inviscid_rhs` 增加 `fill_state_halos=True` 关键字；为 False 时跳过 `fill_halos_i/j/k` 和 `fill_wall_j`，但仍执行 `zero_state_ghosts`。`src/viscous.py::viscous_rhs` 增加同样关键字，跳过状态 halo，保留派生温度与 Ev 的现有处理。`rk3_step` / `rk3_step_stages` 删除 `bc_apply` 参数；`src/main.py::run` 的 `step_jit` 改为 `execution.prepare(rk3_step(q, dt_, rhs))`；`experiment/compare_ocfd_jax.py` 在 `rk3_step_stages` 后对 `q1` 调一次 `execution.prepare`；`src/compare.py::rhs` 同样改为 `execution.prepare` 后调用 `fill_state_halos=False` 的无粘/粘性 RHS。`experiment/test_step2_boundary.py` 的回调计数断言更新为新口径。

**Patterns to follow.** `src/main.py::build_rhs` 的现有组合顺序；`src/solver.py::_rhs_along_axis` 的单轴填充位置；`src/viscous.py::viscous_rhs` 的 `mixed_axis` 分支；`src/boundary.py::validate_special_sides` 的 callback/halo 分类。

**Test Scenarios.**

- custom case：i- 回调、i+ 出流；回调执行前 i+ 鬼层已填充，回调后 i- 保持特殊值，j/k 角点回到周期值。
- `plate` / `channel` 的一步 `cons` 与三个 `du` 和改动前 NPZ 逐位一致；若不一致，测试输出差异位置并失败。
- `build_rhs` 每次调用只触发一次 `prepare`；`rk3_step_stages` 不再调用 `bc_apply`；在 `jax.disable_jit()` 下整步状态准备次数为 3 次 RHS + 1 次最终 prepare = 4 次，JIT 下不再出现旧的 RK 外层重复调用。
- `fill_state_halos=False` 时 `inviscid_rhs` 不再重填状态鬼层；`zero_state_ghosts=True` 仍清零非周期鬼层。
- `experiment/compare_ocfd_jax.py` 的 `q1` 在比较前已准备边界层，且不重复回调。
- `src/compare.py::rhs` 与 `src/main.py::build_rhs` 使用同一 `execution.prepare` 和 `fill_state_halos=False` 口径；一步结果一致。

**Verification.** `python -m pytest experiment/test_step2_boundary.py experiment/test_step4n_boundary.py -q`；再跑一份 `plate`/`channel` 一步 NPZ 对比。

### U3. `specificboundary` JSON 与脚本加载器

**Goal.** 为 `input.json` 增加正式 `specificboundary` 入口，并在 JIT 前把脚本返回的特殊边界接入现有校验链。

**Requirements.** R10, R11, R12, R13, AE4, AE5.

**Dependencies.** U1, U2.

**Files.** `src/runtime_config.py`; `src/init.py`; `src/specific_boundary.py`; `src/special_boundaries/__init__.py`; `src/special_boundaries/README.md`; `experiment/test_step4n_boundary.py`; `experiment/test_step3_config.py`.

**Approach.** 在 `src/runtime_config.py` 的 `allowed_top` 增加 `specificboundary`，按独立顶层值解析，不并入 `_SCHEMA`：值必须是非空字符串，写入 `values["specificboundary"]` 与 `sources`；不是对象，因此没有 `module` / `function` / `params` 子字段。在 `resolve_args` 合并后，如果最终 `case` 是 `plate` 或 `channel` 且存在 `specificboundary`，直接报冲突。新增 `src/specific_boundary.py::load_specific_boundary(name, case)`：校验 `name` 是裸脚本名（无目录、无 `..`，不带后缀时补 `.py`，带其他后缀报错），在固定目录 `src/special_boundaries/` 下解析并确认父目录未跳出；用 `importlib.util.spec_from_file_location` 以唯一加载名加载，取公开入口 `build_special_boundary(case)`，调用后校验返回对象的 `name` / `faces` / `apply`。`src/init.py::prepare` 在 `build_case` 后、`from_case` 前调用加载器，检查 `plate` / `channel_bc` / `special_boundary` 互斥，然后挂载 `case["special_boundary"]` 与 `case["bc_apply"]`。

**Patterns to follow.** `src/runtime_config.py::load_config` 的未知字段错误风格；`src/runtime_config.py::resolve_args` 的来源记录；`src/mesh_data.py::from_case` 对 `special_boundary` 的现有校验；`src/README.md` 第 7.2 节的 `SimpleNamespace` 示例。测试脚本可放在 `experiment/fixtures/special_boundaries/`，通过 monkeypatch `SPECIAL_BOUNDARY_DIR` 指向测试目录，避免向生产目录写测试文件。

**Test Scenarios.**

- 合法的 `specificboundary` 文件名字符串加载测试脚本，`from_case` 通过，`case["bc_apply"]` 是脚本返回对象的 `apply`。
- 文件名为空、是对象、含 `/` 或 `\`、含 `..`、带非 `.py` 后缀、文件不存在、脚本缺少 `build_special_boundary(case)`、返回对象缺 `apply` 分别报错。
- `specificboundary` 与 `plate` / `channel` 同时出现时在 JIT 前报错。
- `specificboundary` + 另一轴 `FARFIELD` 能通过 `from_case` 与 `prepare`。
- JSON 顶层出现 `boundary.specific` 或 `specific boundary` 仍按未知字段报错，确保正式 key 唯一。

**Verification.** `python -m pytest experiment/test_step3_config.py experiment/test_step4n_boundary.py -q`。

### U5. 初始化与运行期错误分层

**Goal.** 把 `main.py` 的 `except ValueError` 拆成“配置错误 / 初始化错误 / 运行期异常”三层，运行期异常保留原始类型、位置上下文和完整 traceback。

**Requirements.** R17, R18, R19, R20, AE6, AE7.

**Dependencies.** U2（`run` 的 RK 调用稳定后）。

**Files.** `src/errors.py`；`src/init.py`；`src/main.py`；`experiment/test_step4n_boundary.py`；`experiment/test_step3_config.py`。

**Approach.** 新建 `src/errors.py`，定义 `SetupError(RuntimeError)` 和 `RuntimeExecutionError(RuntimeError)`。`init.prepare` 在 `configure_precision`、`build_case`、`from_case`、`build_execution`、`cons0` 装配周围捕获 `ValueError`/`OSError`（`CgnsError` 是 `ValueError` 子类），包装成 `SetupError`，消息记录 `case`、`mesh_format`、`mesh` 和原始异常类型，并用 `raise ... from exc` 保留原因。`main.run` 把 `init.prepare` 放在运行包装层之外；`build_rhs`、`rhs(cons)` 预热、RK 循环、滤波和存盘放入 `try`，捕获 `Exception` 后抛出 `RuntimeExecutionError`，消息记录 `case`、`step`、`t`、`dt`、`grid` 和原始异常类型，仍用 `raise ... from exc`。`main.main` 只保留 `except runtime_config.ConfigError` 和 `except SetupError`；删除 `except ValueError`。`_validate_input_paths` 的 `parser.error` 保持为启动参数错误。

**Patterns to follow.** `src/runtime_config.py::ConfigError` 的子类化方式；`src/main.py::main` 现有的 `parser.error` 分层；Python 异常链 `raise ... from exc`。

**Test Scenarios.**

- 显式 `--ni` 与 CGNS `VertexSize` 冲突时，`main` 退出码为 2，stderr 含 `初始化错误`、`CgnsError`/`ValueError` 类型和冲突值，不含 `运行参数错误`，且不出现 traceback。
- `--mesh-format cgns` 缺 `--mesh` 仍由 `_validate_input_paths` 报启动错误，不进入 `init.prepare`。
- 用 monkeypatch 让 `rk3_step` 在第一步抛 `ValueError("synthetic boom")`，调用 `main.run` 得到 `RuntimeExecutionError`；`exc.__cause__` 是那个 `ValueError`，消息含 `case`、`step=1`、`t`、`dt` 和原始类型；`main.main` 不会把该异常转成 `parser.error`。
- `init.prepare` 内 `CgnsError` 被包装成 `SetupError`；运行段同期抛出的同类型 `ValueError` 不被包装成初始化错误。
- 无 `RuntimeExecutionError` 的 `ValueError` 仍按未捕获异常退出；测试不得断言退出码 2。
- 配置层 `ConfigError` 仍打印 `配置错误`，不变成 `初始化错误`。

**Verification.** `python -m pytest experiment/test_step4n_boundary.py experiment/test_step3_config.py -q`；再用真实 CGNS 尺寸冲突命令检查退出码和 stderr 分类。

### U4. 文档、旧测试迁移与整体验证

**Goal.** 把本步四项能力写进项目文档，迁移所有旧调用，完成本步验证并本地提交。

**Requirements.** R14, R15, R20, AE3, AE5, AE6.

**Dependencies.** U1, U2, U3, U5.

**Files.** `src/README.md`; `11-16 我建议先对齐的.md`; `docs/plans/2026-11-18-001-feat-cgns-single-block-read-plan.md`; `experiment/test_step2_boundary.py`; `experiment/test_step4n_boundary.py`.

**Approach.** 更新 `src/README.md` 第 7.2 节：通用边界列表加入 `FARFIELD`；说明 `prepare` 四阶段；把 `special_boundary` 的“回调后由执行配置填充”改成“常规填充后叠加特殊回调”；把 `input.json` 的 `specificboundary` 写成正式入口，并标注“只填 `src/special_boundaries/` 下的脚本名（可带 `.py`），入口函数固定为 `build_special_boundary(case)`”。同时新增“错误分类”说明：`配置错误`、`初始化错误` 走 `parser.error`（退出码 2），运行期异常抛 `RuntimeExecutionError` 并保留原始 traceback，不转成启动错误。路线图在第 4 步与第 5 步之间插入“第 4.n 步”摘要，指向本计划。Step 4 计划的 deferred 条目把 `boundary.specific` 改为 `specificboundary`，并指向本计划；`BCFarfield` 的读取层实现仍保留在 Step 4 后续，不写成已解决。迁移 `test_step2_boundary.py` 中 `execution.apply` 与 `rk3_step_stages(bc_apply=...)` 的旧断言。最后用 `rg` 复核 `bc_apply=` / `execution.apply` / `boundary.specific` 无残留（`case["bc_apply"]` 数据字段本身保留）。

**Patterns to follow.** `src/README.md` 第 7.1/7.2 节的条目结构；路线图第 4 步/第 5 步的“当前进度/验收标准”格式；`docs/plans/2026-11-18-001-...` 的 deferred 列表。

**Test Scenarios.**

- 文档中的入口、顺序、限制与代码一致；`specificboundary` 不再写成 `boundary.specific`。
- 文档明确 `FARFIELD` 是通用边界、不是特殊脚本；错误分类写明启动错误退出码 2、运行期异常保留 traceback。
- `rg` 只在 `src/cases.py`、`src/mesh_data.py`、测试数据构造和文档示例中看到 `bc_apply` 数据字段，不再看到 `rk3_step_stages(..., bc_apply=...)`。
- 全量 `experiment` 测试通过；若某个旧测试仍依赖已删除入口，按新入口迁移而不是恢复旧别名。
- 本地提交只包含本步计划、实现、测试和文档，不混入无关工作区修改。

**Verification.** `python -m pytest experiment -q`；`git diff --check`；`git status --short`。

---

## Verification Contract

| Gate | Command | Applies to | Done signal |
|---|---|---|---|
| G1 | `python -m pytest experiment/test_step4n_boundary.py -q` | U1, U2, U3, U5 | 远场、顺序、`specificboundary`、错误分层测试全绿 |
| G2 | `python -m pytest experiment/test_step2_boundary.py experiment/test_step4n_boundary.py -q` | U2 | 旧边界测试迁移后全绿，回调计数为 4 |
| G3 | `python -m pytest experiment/test_step3_config.py experiment/test_step4n_boundary.py -q` | U3 | JSON 合法/非法入口均按预期 |
| G4 | `python -m pytest experiment -q` | U4 | 全量 tracked experiment 测试通过；跳过项/已知失败单独列出 |
| G5 | `python -m pytest experiment/test_step1_mesh.py -q` | U2, U4 | 网格描述与 `build_rhs` 入口未回归 |
| G6 | `git diff --check` | U4 | 无空白错误 |
| G7 | `rg -n "bc_apply=|execution\\.apply|boundary\\.specific" src experiment docs/plans/2026-11-18-001-feat-cgns-single-block-read-plan.md` | U4 | 仅剩允许的数据字段/文档说明，无旧调用 |
| G8 | `python -m opencfd_jax.src.main --case poiseuille --mesh opencfd_jax/Grid/fixtures/poiseuille_box.cgns --ni 32 --dt 1e-3 --end-step 0 --end-time 0.5` | U5 | 退出码 2，stderr 含 `初始化错误` 与冲突值，不含 `运行参数错误`，不打印 traceback |
| G9 | `python -m pytest experiment/test_step4n_boundary.py -q` 的运行期注入用例 | U5 | `RuntimeExecutionError.__cause__` 保留原始 `ValueError`，消息含 `case/step/t/dt`，未被 `parser.error` 吞掉 |

---

## Definition of Done

- `FARFIELD` 是 `GENERIC_KINDS` 成员，`_generic_faces` 和 `validate_special_sides` 接受它；缺少/非法 `case["farfield"]` 在 JIT 前失败；粘性 `Ev` 对该侧零梯度。`FARFIELD` 不由 `specificboundary` 脚本实现。
- `BoundaryExecution.prepare` 按“常规 → 特殊 → callback+halo → 常规角点清理”执行；同一轴一侧特殊、另一侧出流的 custom case 通过。
- `build_rhs` 是唯一状态准备点；`rk3_step` / `rk3_step_stages` 不再接收 `bc_apply`；在 `jax.disable_jit()` 下整步状态准备次数为 4，JIT 下不再出现旧的 RK 外层重复调用。
- `plate` / `channel` 的一步 `cons` 与三个 `du` 与改动前 NPZ 逐位一致；若不一致，报告具体差异位置并停止，不静默放宽。
- `input.json` 的 `specificboundary` 正式可用，值只填一个脚本名；合法脚本加载，非法名字/缺失入口/冲突在 JIT 前报错。
- `main.py` 不再把运行期 `ValueError` 转成 `parser.error`；初始化错误输出 `初始化错误`，运行期错误抛 `RuntimeExecutionError` 并保留原始 traceback 与 `__cause__`，消息含 `case/step/t/dt/grid`。
- `--filter` + `FARFIELD` 的未迁移组合在 JIT 前报错，不出现运行时 `fill_halos_i` 未知边界错误。
- `src/README.md`、路线图、Step 4 deferred 三处文档一致；不再出现 `boundary.specific` 作为正式入口。
- `experiment` 全量测试通过；已知跳过/失败项在报告中如实列出。
- 改动已在本地 Git 提交；不推送、不创建 PR。

---

## Appendix

### 建议的 `specificboundary` JSON 形状

```json
{
  "specificboundary": "my_boundary"
}
```

脚本放在 `src/special_boundaries/my_boundary.py`，固定入口返回现有 `special_boundary` 契约：

```python
def build_special_boundary(case):
    return SimpleNamespace(
        name="user-special",
        faces=(FaceBoundary(...), ...),  # 必须按 i-/i+/j-/j+/k-/k+ 六侧给出
        apply=my_apply,                  # 纯 JAX 可追踪函数
    )
```

脚本参数（如来流状态、壁面温度）由脚本自身或 `case` 提供；本步不在 JSON 里再开 `params` 键。

### 本步不解决的问题

- `BCFarfield` 的 `BCDataSet` 来流参数读取。
- 每面不同远场状态。
- 特征边界/无反射远场。
- `specificboundary` 的 CLI 覆盖、脚本参数化与仓库外脚本加载。
- 特殊边界的 `Ev` 新物理模型。
- `src/` 全量自定义异常类型改造、结构化错误日志和自动复现包。

### 错误分层口径

```text
ConfigError            -> main: parser.error("配置错误：...")，退出码 2
SetupError             -> main: parser.error("初始化错误：...")，退出码 2
RuntimeExecutionError  -> 不捕获，Python 打印链式 traceback，非 2 退出
```

`SetupError` 只覆盖 `init.prepare`；`RuntimeExecutionError` 只覆盖 `build_rhs` 之后的时间推进段。两者都用 `raise ... from exc` 保留原始异常。