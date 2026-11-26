---
title: Step 3.6 运行入口与初始化整理 - Plan
type: refactor
date: 2026-11-17
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Step 3.6 运行入口与初始化整理 - Plan

## Goal Capsule

把一次运行的初始化集中到 `src/init.py`，把正式入口和主循环集中到 `src/main.py`，删除 `src/run.py`。在接入 CGNS 前完成一次职责整理，不改变网格维数来源、边界公式、初始场、时间推进、滤波节奏或 NPZ 输出内容。

- **Authority.** 本计划的 Product Contract 与已确认的 grilling 决策是产品约束；Implementation Units 只决定执行顺序与文件落点。
- **Execution profile.** 只调整职责、导入路径和入口命令；不改数值算法、配置优先级、输出步号格式或元数据字段集合。
- **Stop conditions.** 新入口无法启动、七例 NPZ 结果或 `_meta` 出现非预期差异、旧 `run.py` 仍被生产代码导入、或需要改变边界/初值/时间推进时停止并报告。
- **Tail ownership.** 在本地完成实现、验证与提交；不推送远端、不创建 PR。

## Product Contract

### Summary

`src/main.py` 成为唯一正式入口，负责 CLI/JSON 解析、RHS 总装、时间推进与输出；`src/init.py` 负责精度设置、算例/网格/边界执行配置和初场组装。`src/run.py` 删除，所有当前消费方改用 `python -m opencfd_jax.src.main` 或 `main` 模块 API。

### Problem Frame

当前 `src/run.py` 同时承担 CLI、JSON 合并、精度设置、算例构造、网格描述、边界执行配置、RHS 总装、时间循环和 NPZ 输出。初始化无法单独测试，入口与执行职责混在同一模块，文档和基准命令也把 `run.py` 当作唯一入口。Step 3.5 后旧尺寸 API 已移除，继续保留 `run.py` 会扩大旧入口的存在时间。

### Requirements

**入口与模块职责**

- R1. 正式入口位于 `src/main.py`，调用命令为 `python -m opencfd_jax.src.main`；不承诺 `python main.py` 直接执行。
- R2. `src/main.py` 负责 CLI 解析、JSON 合并、输入路径校验、输出目录创建、`build_rhs`、时间推进、滤波、诊断和 NPZ 写盘。
- R3. `src/init.py` 提供唯一的初始化入口 `prepare(args)`，接收已解析的 `args`，返回包含 `args`、`case`、`desc`、`execution`、`cons0` 的 `InitState`。
- R4. `src/run.py` 删除；生产代码、实验测试和当前文档不得再导入它或调用 `python -m opencfd_jax.src.run`。

**精度与初始化顺序**

- R5. 精度设置在 `build_case` 和任何 JAX 数组创建之前生效；默认 FP64，`--fp32` 必须在 `prepare` 内切换成功。
- R6. 初始化不得改变现有输入优先级：显式 CLI 覆盖 JSON，JSON 覆盖默认值；槽道 `.in` 的维数、物性和壁温仍按现有规则生效。

**行为保持**

- R7. 不改网格维数来源、边界公式、初始场、时间步长算法、滤波节奏、诊断文本或 NPZ 文件名规则。
- R8. 七个 Step 0 用例只通过新入口运行；比较 NPZ 的 `cons_0..cons_4` 以及 `_meta` 中现有检查器使用的字段，不要求旧入口双跑。
- R9. 保留 `main.build_parser`、`main.resolve_cli`、`main.build_rhs`、`main.run` 作为模块级可测试 API；它们不是额外的命令行入口。

### Key Decisions

- **入口唯一化。** `main.py` 是唯一命令入口，`init.py` 不提供 CLI；删除 `run.py` 而不是保留转发门面，避免两份入口长期并存。
- **初始化停在初场。** `init.prepare` 返回 `InitState` 后即结束；`build_rhs` 和时间循环归 `main.py`，保持 Step 3.6 只整理职责。
- **结果口径沿用现有检查器。** “只对比结果”解释为 NPZ 场与 `_meta`，不比较日志、`case.json` 或命令行文本；这与 `experiment/check_step0.py` 的现有判定一致。
- **历史快照不回写。** `experiment/source_record_20261117/` 和旧计划文档保留原貌；当前 README、链路入口和 experiment 测试更新。

### Scope Boundaries

本步不实现 CGNS、多块、GPU、并行通信或新的物理边界，不重构 `boundary.py`、`solver.py`、`viscous.py`、`cases.py` 的内部公式，不改变 NPZ 元数据字段集合，不新增 `python main.py` 直接执行方式。

### Acceptance Examples

- AE1. `python -m opencfd_jax.src.main --case sod --ni 8 --nj 4 --nk 4 ...` 能完成一次短运行，输出文件名仍为八位 `-istep-` 格式。
- AE2. `--fp32` 下 `init.prepare` 返回的 `cons0` 为 float32；默认模式下为 float64，且切换发生在数组创建之前。
- AE3. 七个基线用新入口运行后，五个守恒量场与 `_meta` 的现有判定字段和基线一致。
- AE4. 代码库中不再存在 `from opencfd_jax.src.run` 或 `python -m opencfd_jax.src.run` 的当前使用（历史快照除外）。

### Dependencies / Assumptions

- `config.py` 的模块导入仍默认开启 FP64；`init.prepare` 显式调用 `enable_x64/disable_x64`。
- `cases.build_case` 仍是唯一算例装配入口；`mesh_data.from_case` 仍生成 `BlockMesh` 快照。
- `boundary.build_execution` 仍是边界执行配置的唯一构造点。
- `experiment/check_step0.py` 的 `_meta` 判定口径不变。

### Outstanding Questions

- **Deferred to Planning.** `InitState` 各字段的精确类型标注、`main.run(args)` 是否长期保留、以及 `tests/` 本地迁移是否单独提交，均在执行中按现有代码结构决定。
- **Resolved.** 正式入口、删除 `run.py`、`init` 入参与边界、结果比较口径、当前文档/测试迁移范围已由 grilling 确定。

## Implementation Units

### U1. 新增初始化模块

**Goal.** 把精度设置、算例构造、网格/边界描述和初场组装集中到 `src/init.py`，并以 `InitState` 固定返回结构。

**Requirements.** R3, R5, R6, AE2.

**Dependencies.** None.

**Files.** `src/init.py`; `src/config.py` (only import/read behavior); `src/cases.py` (read-only reference); `src/mesh_data.py` (read-only reference); `src/boundary.py` (read-only reference).

**Approach.** `prepare(args)` 首先按 `args.fp32` 调用现有精度开关，然后调用 `build_case(args)`、`from_case(case)`、`boundary.build_execution(case, desc)`，返回 `InitState(args, case, desc, execution, cons0)`。`InitState` 使用冻结数据类，字段名固定。初始场直接取 `case["cons0"]`，不复制、不重算。

**Patterns to follow.** `src/mesh_data.py` 的冻结数据类；`src/cases.py::build_case` 的现有装配顺序；`src/config.py` 的精度开关。

**Test scenarios.**

- `prepare` 返回的 `args/case/desc/execution/cons0` 字段完整，`desc` 是 `BlockMesh`，`cons0` 形状与内区一致。
- 默认参数下 `config.precision_name()` 为 float64，`cons0` dtype 为 float64。
- `args.fp32=True` 时 `prepare` 在 `build_case` 前切换精度，返回的 `cons0` dtype 为 float32。
- `case=channel` 且带 `.in` 时，`prepare` 保留现有 `.in` 物性/维数结果；CLI 显式 `Re` 不静默覆盖 `.in`。

**Verification.** `init.prepare` 可单独调用，不需要进入时间循环；精度、算例和边界执行配置均可从返回值读取。

---

### U2. 新建正式入口并删除旧入口

**Goal.** 把 `run.py` 的 CLI、RHS 总装、主循环和输出逻辑迁到 `src/main.py`，删除 `src/run.py`，保证命令入口只有 `python -m opencfd_jax.src.main`。

**Requirements.** R1, R2, R4, R7, R8, R9, AE1, AE4.

**Dependencies.** U1.

**Files.** `src/main.py` (new); `src/run.py` (delete); `src/init.py`; `src/io.py` (read-only reference).

**Approach.** `main.py` 保留现有 `build_parser`、`resolve_cli`、`_warn_channel_overrides`、`_warn_forced_viscous`、`build_rhs`、`run(args)`、`_validate_input_paths`、`main(argv)` 的职责；`run(args)` 先调用 `init.prepare(args)`，再从返回的 `InitState` 取 `case/desc/execution/cons0` 构造 RHS 并推进。`main.main` 继续负责解析、校验、创建输出目录，然后调用 `run(args)`。模块末尾只保留 `if __name__ == "__main__": main()`。

**Patterns to follow.** 现有 `src/run.py` 的解析与循环顺序；`src/io.py::checkpoint_name` 的八位步号；`src/ana.py::diagnostic_line` 的输出格式。

**Test scenarios.**

- `python -m opencfd_jax.src.main --help` 正常退出并显示现有参数。
- 短 Sod 用例通过新入口运行，输出一个 `-istep-00000001.npz`，`cons_*` 形状和 `_meta` 字段与旧行为一致。
- `main.build_parser`/`main.resolve_cli` 的默认值、显式 CLI、JSON 合并和负向开关行为与迁移前一致。
- `main.build_rhs` 对 Sod/平板用例仍返回可调用 JIT 函数；`from_case` 的 monkeypatch 目标改到 `main`。
- `src/run.py` 删除后，生产代码中不存在 `opencfd_jax.src.run` 的当前导入。

**Verification.** 新入口完成一次短运行；`src/run.py` 不存在；入口命令和输出文件名不再引用旧模块。

---

### U3. 迁移 experiment 消费方与当前文档

**Goal.** 把所有当前 experiment 入口、测试和 README 的 `run.py` 引用切换到 `main.py`，保留历史快照不动。

**Requirements.** R4, R8, AE4.

**Dependencies.** U2.

**Files.** `experiment/run_step0.py`; `experiment/step0_cases.py`; `experiment/test_step3_config.py`; `experiment/test_step3_5_axes.py`; `experiment/test_step1_mesh.py`; `experiment/test_step2_boundary.py`; `experiment/compare_ocfd_jax.py`; `experiment/README.md`; `src/README.md`; `11-16 我建议先对齐的.md`.

**Approach.** 将子进程命令改为 `python -m opencfd_jax.src.main`；测试中的 `run` 导入/属性访问改为 `main`；`test_step1_mesh` 的 monkeypatch 目标改为 `main.from_case`；README 的入口说明和 Step 3.6 当前状态同步更新。`experiment/source_record_20261117/`、旧 `docs/plans/` 和历史对拍记录不改。

**Patterns to follow.** `experiment/run_step0.py` 的“真实入口子进程”原则；`experiment/test_step3_config.py` 的 parser 合并顺序。

**Test scenarios.**

- `experiment/run_step0.py` 的每条命令行都使用新入口，`--phase candidate` 能生成候选目录。
- `test_step3_config.py` 的解析测试和 CLI 子进程测试通过，JSON/CLI 优先级不变。
- `test_step3_5_axes.py` 的 parser 测试和新旧尺寸拒绝测试通过。
- `test_step1_mesh.py` 的 `from_case` 调用记录仍能观测到 `build_rhs` 的调用。
- `test_step2_boundary.py` 和 `compare_ocfd_jax.py` 使用 `main.build_rhs` 后行为不变。

**Verification.** `rg` 在非历史快照中不再发现 `src.run`/`run.py` 的当前入口引用；README 的示例命令可直接复制。

---

### U4. 本地测试迁移与七例结果对齐

**Goal.** 迁移受 `.gitignore` 排除的本地 `tests/`，并验证新入口与七个基线在结果层面一致。

**Requirements.** R7, R8, AE3.

**Dependencies.** U2, U3.

**Files.** `tests/test_plate_boundaries.py`; `experiment/test_step3_config.py`; `experiment/test_step3_5_axes.py`; `experiment/test_step1_mesh.py`; `experiment/test_step2_boundary.py`; `experiment/check_step0.py` (read-only reference).

**Approach.** 本地 `tests/test_plate_boundaries.py` 改用 `main.build_rhs`；运行 `tests/` 与 `experiment/` 的相关测试。七例对比只用新入口，比较 NPZ 场和 `_meta`；不把日志或 `case.json` 作为判定条件。该目录仍受 `.gitignore` 排除，不把测试迁移强行加入本次提交，但在交付中记录实际运行结果。

**Patterns to follow.** `experiment/check_step0.py` 的逐 NPZ、逐场和 `_meta` 判定；`experiment/README.md` 的候选/基线目录约定。

**Test scenarios.**

- `tests/test_plate_boundaries.py` 导入 `main` 后平板边界测试通过。
- `experiment/test_step1_mesh.py`、`test_step2_boundary.py`、`test_step3_config.py`、`test_step3_5_axes.py` 全部通过。
- 七个 Step 0 用例通过新入口运行，`cons_0..cons_4` 与基线零差异；`_meta` 的现有判定字段一致。
- 旧 `run.py` 不存在时，任何生产/实验测试导入都不会再成功找到它。

**Verification.** 记录测试通过数、七例结果差异和任何未纳入验证的缺口；不把“测试文件已迁移”误报为“已提交”。

## Verification Contract

| Gate | Applies to | Done signal |
|---|---|---|
| Import audit | U2, U3 | 非历史快照中没有 `from opencfd_jax.src.run`、`import ... run` 或 `python -m opencfd_jax.src.run` 的当前引用 |
| CLI smoke | U2 | `python -m opencfd_jax.src.main --help` 与短 Sod 运行正常 |
| Precision | U1 | FP64 默认、`--fp32` 切换均在数组创建前生效 |
| Config compatibility | U2, U3 | CLI/JSON/`.in` 优先级测试保持通过 |
| Seven-case result | U4 | 七个用例的 `cons_*` 与 `_meta` 判定字段和基线一致 |
| Local tests | U4 | `tests/` 与 `experiment/` 中相关测试通过；缺失项如实记录 |

## Definition of Done

- `src/main.py` 是唯一正式入口，`src/init.py` 提供 `InitState` 和 `prepare(args)`。
- `src/run.py` 已删除，当前生产代码、experiment 和 README 不再使用旧入口。
- 短 Sod 运行、配置测试、边界测试和七例结果对齐完成；未验证项在交付中明确列出。
- 代码改动和文档改动已提交到本地 Git，没有推送远端。

## Sources / Research

- `src/run.py` 当前职责与调用顺序。
- `src/config.py` 的 FP64 默认与精度开关。
- `src/cases.py::build_case` 的数组创建位置。
- `src/mesh_data.py::from_case` 与 `src/boundary.py::build_execution`。
- `experiment/run_step0.py`、`experiment/check_step0.py`、`experiment/test_step3_config.py`、`experiment/test_step3_5_axes.py`、`experiment/test_step1_mesh.py`、`experiment/test_step2_boundary.py`。
- 已确认的 grilling 边界：`src/main.py` 唯一入口、删除 `run.py`、`init.prepare(args)`、只比较 NPZ 结果、同步当前消费方与 README。
