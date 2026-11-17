# OpenCFD-JAX 计算流程说明

这套代码用 **JAX + NumPy** 求解五个守恒变量的可压缩流动方程，并提供 Sod 激波管、Taylor–Green 涡、Poiseuille 槽道流、三维槽道流和平板边界层五种算例。代码包含无粘通量、可选粘性项、物理边界、体积力、三阶 Runge–Kutta 时间推进、激波滤波、诊断和 NPZ 存盘。

阅读时只需要先记住一条主线：



```
命令行参数
   ↓
cases.py：建立网格、物性、初值、边界和源项
   ↓
run.py：组装右端项 L(Q) = 无粘项 + 粘性项 + 源项
   ↓
solver.py：CFL 定步长 + SSP-RK3 时间推进
   ↓
filtering.py / ana.py：整步后的可选滤波与诊断
   ↓
io.py：只保存物理内区的守恒量 NPZ
```



***

## 1. 先跑一个最简单的算例

依赖：Python、JAX、NumPy。

在**工作区根目录** `D:\Desktop\ocfd2jax` 下运行：



```
python -m opencfd_jax.src.run --case sod --ni 256 --scheme UP7 --end-time 0.2
```

当前目录结构中，源码包名是 `opencfd_jax.src`，所以不要写成 `python -m opencfd_jax.run`。

计算结果默认写到当前目录的 `output/`。例如：



```
output/sod-istep-00000050.npz
output/sod-istep-00000100.npz
...
```

输出文件名以 `-istep-` 加至少八位步号结尾；旧基准的 `_stepNNNNNN` 文件保持原名，由
`experiment/check_step0.py` 按步号映射比较。

程序默认使用 FP64。常用控制参数：



| 参数                        | 含义                                                      |
| ------------------------- | ------------------------------------------------------- |
| `--case`                  | 算例：`sod` / `tg` / `poiseuille` / `channel` / `plate`    |
| `--ni --nj --nk`          | 计算 i/j/k 方向内区网格数，缺省 256/64/4                          |
| `--scheme`                | 无粘重构格式，默认 `UP7`                                         |
| `--dt`                    | 固定时间步长；不提供时按 `--cfl` 自适应计算                              |
| `--cfl`                   | 自适应步长的 CFL 数，默认 0.4                                     |
| `--end-time`              | 终止物理时间                                                  |
| `--end-step`              | 最大整步数；与终止时间谁先满足就停止                                      |
| `--viscous`               | 对 Sod、Taylor–Green、平板启用粘性项；Poiseuille 和 channel 始终计算粘性项 |
| `--vis-order`             | 粘性导数格式：`CD6` 或 `CD8`                                    |
| `--Re --Ama --ref-t --pr` | Reynolds 数、参考 Mach 数、有量纲参考温度和 Prandtl 数                 |
| `--filter`                | 启用整步后的激波选择性滤波                                           |
| `--kstep-show`            | 每隔多少整步打印一次诊断                                            |
| `--kstep-save`            | 每隔多少整步保存一次；最后一个整步也会保存                                   |
| `--out-dir`               | 输出目录，默认 `output`                                        |
| `--fp32`                  | 改用 FP32，主要用于性能测试；逐点验证建议保留默认 FP64                        |

查看全部参数：



```
python -m opencfd_jax.src.run --help
```

### 1.1 用 `input.json` 统一配置（第 3 步）

`--config path/to/input.json` 可以一次给出运行参数；命令行仍是最高优先级，不写
JSON 的字段保持程序默认值：

```powershell
python -m opencfd_jax.src.run --config experiment\cases\sod_1step.json
```

JSON 只分六组，外加顶层 `precision`：

| 组 | 字段 |
|---|---|
| `mesh` | `format`（`builtin`/`ocfd`）、`file`、`field`、`infile` |
| `physics` | `Re`、`Ama`、`ref_t`、`pr` |
| `numerics` | `scheme`、`vis_order`、`viscous`、`weno_tv`、`filter`、`nstep_filter`、`filter_s0`、`filter_rth` |
| `time` | `dt`（`null` 表示自适应）、`cfl`、`end_time`、`end_step` |
| `case` | `name`、`p_grad`、`forcing_work`、`plate_x0`、`plate_x1`、`plate_h` |
| `output` | `dir`、`kstep_show`、`kstep_save` |

优先级：程序默认值 → JSON → 显式命令行。显式与否由“参数是否出现在命令行”判断，
不靠与默认值比较。需要显式关闭 JSON 里的布尔项时用 `--no-viscous`、`--no-filter`、
`--no-weno-tv`；需要用自适应步长覆盖 JSON 的固定 `dt` 时用 `--adaptive-dt`
（与 `--dt` 同时给出会报错）。

网格文件名决定输出前缀：给了 `--mesh`/`mesh.file` 就用文件名去最后一个扩展名
（`OCFD3d-Mesh.dat` → `OCFD3d-Mesh-istep-00000001.npz`），否则用算例名。JSON 内相对
路径按 JSON 所在目录解析，命令行路径按当前工作目录解析。

`channel` 的网格维数、`RE/AMA/T_REF`、可选 `GAMMA/PR`、`IBC` 与
`JAC_BOUND/DIF_BOUND` 仍以 `mesh.infile` 的 `.in` 为准；JSON/命令行给了不同的
物性只会打警告，不改变有效值。JSON 不接受网格维数、边界开关和 `gamma`，出现未知
字段或不属于当前算例的字段会在启动前报错。



***

## 2. 程序保存和推进的是什么量

### 2.1 五个守恒变量

内存和输出中的变量顺序固定为：



| 分量 | 守恒变量    | 含义      |
| -- | ------- | ------- |
| 0  | `rho`   | 密度      |
| 1  | `rho*u` | x 方向动量  |
| 2  | `rho*v` | y 方向动量  |
| 3  | `rho*w` | z 方向动量  |
| 4  | `rho*E` | 单位体积总能量 |

速度、压力、温度、声速等原始变量都是在计算过程中由这五个量临时恢复的，主时间循环始终推进守恒量。

### 2.2 数组布局和鬼胞

对应文件：`fields.py`。

完整守恒量数组形状为：



```
(5, nk + 8, nj + 8, ni + 8)
```



* 计算坐标为 `(i, j, k) = (ξ, η, ζ)`，空间数组轴顺序是 `(k, j, i)`。
  规范尺寸名是 `ni/nj/nk`；`nx/ny/nz` 是兼容只读别名，旧位置参数仍按
  计算 `i/j/k` 顺序解释。物理笛卡尔坐标 `x/y/z` 与计算坐标通过网格度量联系。
* 原码度量键 `Akx/Aky/Akz`、`Aix/Aiy/Aiz`、`Asx/Asy/Asz` 分别属于
  计算 `i/j/k` 方向；键末尾的 `x/y/z` 表示物理坐标分量，不是计算轴名。

* 每个方向有 `LAP = 4` 层鬼胞，为高阶模板和边界处理提供数据。

* `Grid.inner()` 选出不含鬼胞的物理内区。

* 诊断和无粘 / 粘性右端项可以操作完整数组，但最终存盘只保存 `(nk, nj, ni)` 内区。

`boundary.py` 负责周期环绕、零梯度出流、壁面镜像和 Ev 等通用填充；`fields.py`
只保留数组布局、`Grid` 和旧导入入口的兼容层。平板的专用入口、壁面和顶部条件仍由
`plate.py` 写入，槽道双壁由 `channel.py` 写入。



***

## 3. 完整计算流程：每一步在哪个文件

主入口是 `run.py` 的 `main()` 和 `run()`。实际执行顺序如下。



| 步骤 | 做什么             | 文件和关键函数                            | 处理方式                                            |
| -- | --------------- | ---------------------------------- | ----------------------------------------------- |
| 1  | 读取命令行并设置精度      | `run.py::main`、`config.py`         | 默认开启 JAX x64；`--fp32` 时关闭                       |
| 2  | 装配算例            | `cases.py::build_case`             | 创建 `Grid`、`PhysParams`、初始守恒量、边界类型、求解方向、步长和源项    |
| 3  | 读取外部输入（仅需要时）    | `infile.py`、`mesh.py`、`channel.py` | channel 可读取原码 in、场和网格文件；把原始变量转换为守恒量，必要时计算曲线网格度量 |
| 4  | 组装单个 RK 子步的右端项  | `run.py::build_rhs`                | 形成 JIT 函数 `rhs(q)`，组合边界、无粘、粘性和源项；滤波不放入其中        |
| 5  | 确定时间步长          | `solver.py::compute_dt`            | 固定 `dt` 或按各求解方向的 \`(                            |
| 6  | 执行一个 SSP-RK3 整步 | `solver.py::rk3_step_stages`       | 三次计算右端项，三次组合状态；每个子步都重新施加专用物理边界                  |
| 7  | 可选激波滤波          | `filtering.py::shock_filter`       | 每个 `nstep_filter` 整步执行一次，不在 RK 子步内执行            |
| 8  | 打印诊断            | `ana.py::diagnostic_line`          | 输出时间、步数、耗时、内区平均 `rho*E` 和全场 NaN 标志              |
| 9  | 保存结果            | `io.py::save_npz`                  | 将五个内区分量及时间、步数、格式和物性元信息写入 NPZ                    |
| 10 | 重复推进            | `run.py::run`                      | 回到步骤 5，直到达到终止时间或最大步数                            |

下面展开最耗计算量的第 5～7 步。



***

## 4. 一个时间整步内发生了什么

### 4.1 时间步长：固定或 CFL 自适应

文件：`solver.py`；函数：`compute_dt()`。



* 给出 `--dt`：使用该固定值，最后一步会缩短以准确到达 `end_time`。

* 不给 `--dt`：从当前流场恢复三个速度和声速，按求解方向累计最大谱半径：



```
dt = CFL / Σ_direction max[(|u_direction| + sound_speed) / h_direction]
```

`dxs` 中小于等于零的方向不参与求和，因此二维算例不会错误地加入 z 向限制。

### 4.2 SSP-RK3：三个子步推进守恒量

文件：`solver.py`；函数：`rk3_step_stages()`、`rk3_step()`。

一个整步采用三阶 SSP Runge–Kutta 组合：



```
Q*     = Qn + dt L(Qn)
Q**    = 3/4 Qn + 1/4 [Q*  + dt L(Q*)]
Q(n+1) = 1/3 Qn + 2/3 [Q** + dt L(Q**)]
```

`rk3_step_stages()` 还会返回三个 `du`，供 `compare.py` 与原版逐子步对拍；正式运行使用的 `rk3_step()` 只保留推进后的状态。

### 4.3 每个 RK 子步的右端项组装顺序

文件：`run.py`；内部函数：`rhs(q)`。

顺序固定为：



```
1. 若算例提供 bc_apply，先施加专用物理边界
2. 计算无粘右端项 inviscid_rhs
3. 若算例需要，叠加 viscous_rhs
4. 叠加 cases.apply_forcing 的通用体积力
5. 若算例提供 source，再叠加专用源项（channel 体积力）
返回 du，但不在这里更新 Q
```

边界函数也会在 RK 状态组合后再次调用，保证下一子步使用满足边界条件的状态。



***

## 5. 无粘项怎样计算

核心文件：`solver.py`、`schemes.py`、`boundary.py`。

对 `axes` 中的每个计算方向分别执行：



```
守恒量 Q
  ↓ fields.py：填该方向所需鬼胞
  ↓ schemes.py::steger_warming
正通量 f+ 和负通量 f-
  ↓ 所选格式分别做 P/M 界面重构
单元界面通量 F(i+1/2)
  ↓ solver.py：相邻界面通量作差并除以步长
该方向的无粘右端项
  ↓
累加所有求解方向；有曲线度量时乘 Ajac
```

### 5.1 通量分裂

`schemes.py::steger_warming()` 使用 Steger–Warming 分裂，把格点通量分成正、负两部分，供迎风重构使用。



* 直角网格：该方向度量退化为单位向量。

* 曲线网格：把对应余子式 `(A1, A2, A3)` 传入分裂过程，使界面通量包含面面积权重。

* 平板中全零的非周期状态鬼层有专门保护，避免派生量计算出现无意义的 0/0。

### 5.2 可选重构格式



| `--scheme`       | 处理方式                                   |
| ---------------- | -------------------------------------- |
| `UP7`            | 七阶迎风模板，分别重构正、负分裂通量                     |
| `WENO7-SYMBO`    | 带宽优化对称 WENO；`--weno-tv` 可在光滑区使用线性权快速路径 |
| `NND2`           | 三点 NND/minmod 限制重构                     |
| `WENO5`          | 五点 WENO-Z 非线性加权重构                      |
| `WENO7`          | 七点、四候选模板 WENO-Z 重构                     |
| `OMP6-1/2/3`     | OMP6 参数档，包含极值检测和限制器                    |
| `OMP6_HR/LD/CD8` | 原码 in 文件使用的 OMP6 名称别名                  |

界面通量最终按守恒形式作差：



```
du = -(F_right - F_left) / h
```

曲线网格还会按照 `mesh.py` 提供的 `Ajac` 做相应缩放。

### 5.3 物理边界附近的处理



* `boundary.py` 提供周期、出流、壁面镜像和 Ev 鬼胞；`fields.py` 只重导出这些
  兼容入口，实际实现只在 `boundary.py` 一处。

* `schemes.py::flux_band_faces()` 根据 `DIF_BOUND` 在指定边界带改用原码单侧 / 降阶面通量公式。
  各格式分别声明原码 P/M 内核的 `(ka1, kb1)` 模板范围；新增格式可使用
  `schemes.make_flux_rhs(face_P, face_M, p_template, m_template, m_boundary="m")`
  包装并注册到 `SOLVE_SCHEMES`。模板偏移相对各自的 CUDA 内核节点，M 内核
  若调用 P 边界宏则传 `m_boundary="p"`。超出当前原码边界宏定义的范围会报错。

* 边界带公式的原码核对笔记已移到 `docs/boundary_scheme_notes.md`。



***

## 6. 粘性项怎样计算

核心文件：`viscous.py`；物性参数：`PhysParams`。

计算链为：



```
守恒量
  ↓ 恢复 u/v/w、温度和 Sutherland 粘性系数 μ(T)
  ↓ CD6 或 CD8 计算速度梯度、温度梯度
  ↓ 组装粘性应力 s11...s23 和能量通量 E1/E2/E3
  ↓ 对粘性通量填鬼胞，再沿各求解方向求散度
  ↓ 正号累加到动量和能量右端项 du[1:5]
```

具体处理：



* 空间导数使用六阶或八阶中心差分，对应 `--vis-order CD6/CD8`。

* `DIF_BOUND` 指定的边界带使用 `band_one_sided()` 的单侧差分。

* 曲线网格下，先求参考方向导数，再通过度量组成物理梯度，并构造面积加权粘性通量。

* Sutherland 公式使用 `Ref_T`；`Re` 控制粘性大小，`Pr` 进入热传导能量通量。

注意：Sod 初值采用常见的 `E = p/(gamma-1)` 验证约定，与粘性模块的 OCFD `Ama` 无量纲约定不同。程序保留 `sod + --viscous` 的灵活性，但会打印警告；研究粘性时应优先使用 Taylor–Green、Poiseuille、channel 或 plate 的相应约定。



***

## 7. 边界和源项在哪里处理



| 类型        | 文件                        | 具体处理                                 |
| --------- | ------------------------- | ------------------------------------ |
| 周期边界      | `boundary.py`             | 从对侧内区环绕复制鬼胞                          |
| 零梯度出流     | `boundary.py`             | 复制最靠近边界的内点值                          |
| 槽道双壁      | `channel.py`              | 壁面无滑移；压力 / 温度按壁面类型外推；写入等温或绝热壁状态及镜像鬼胞 |
| 平板入口 / 出口 | `plate.py`                | 入口写 Dirichlet 守恒量剖面，出口鬼层零梯度          |
| 平板下壁 / 顶部 | `plate.py`                | 下壁无滑移等温并镜像，顶部写均匀自由来流                 |
| 通用体积力     | `cases.py::apply_forcing` | 动量源 `rho*G`，可选能量做功源 `rho*u*G`        |
| 槽道专用体积力   | `channel.py::body_force`  | 按 `ChannelBC.F` 生成源项；是否做功由 `work` 控制 |

与原码 channel 逐项对拍时使用 `--forcing-work off`，因为原码对应内核只添加动量源。正式 `run.py` 的默认值为 `on`。

### 7.1 单块网格与边界描述（第 1 步）

文件：`mesh_data.py`。

`run.py`（`build_rhs` / `run`）和 `compare.py`（`cmd_jax_run`）在消费入口由
`from_case(case)` 即时构造只读快照 `BlockMesh`，而不是另外保存一份配置。描述包含：

* `grid`、`dxs`、`axes`、`bcs`：直接复用算例字典里的对象/值；
* 六侧 `FaceBoundary`：每侧语义（periodic/outflow/wall/inflow/farfield）、当前
  执行来源（`halo` / `callback` / `callback+halo`）和该侧的 DIF_BOUND 开关；
* `dif_bound`、`dif_axis_map`：保留原码六标志与 JAX 轴映射（平板仍是 `(0,2,1)`）；
* `metrics`、`coords`：只有算例真的提供时才引用；没有就保持 `None`，不伪造。

描述现在由 `boundary.build_execution` 转成唯一一份 `BoundaryExecution`，`run.py`
和 `compare.py` 都用它决定轴级 `bcs` 与 case 回调；描述不写回算例字典，避免出现
两份可独立修改的状态。构造必须发生在创建 JIT 闭包之前；之后修改 `case` 不会回溯
已经建立的闭包。

### 7.2 统一边界执行与扩展入口（第 2 步）

文件：`boundary.py`、`mesh_data.py`、`run.py`。

通用边界（周期、出流、计算 j 方向壁面镜像、非周期鬼层清零）只在 `boundary.py` 实现；
`fields.py` 只重导出旧名，`channel.py` / `plate.py` 保留壁面和回调的兼容入口，
壁面及平板的已有状态赋值公式由 `boundary.py` 提供。`build_execution(case, desc=None)` 在 JIT 前完成：

* 校验 `case["bcs"]`、六侧 `FaceBoundary` 与 `bc_apply` 是否同源；
* 拒绝“有回调但无专属标记”“多个专属标记同时存在”“固定六侧与 bcs 矛盾”
  “通用 WALL 不在计算 j 方向”等组合，未知组合直接报错，不静默改成出流；
* 返回 `BoundaryExecution(mode, bcs, callback, faces, ...)`，`mode` 为
  `generic` / `channel` / `plate` / `custom`。

**新增通用边界**：在 `boundary.py` 的 `_GENERIC_AXIS_FILLERS` 注册填充函数并加入
`GENERIC_KINDS`，再让 `mesh_data._generic_faces` 能描述对应语义。通用 WALL 目前仅
支持计算 j 方向；`filtering.py` 的滤波也仍只支持周期/出流。

**新增 case 特殊边界**：case 侧显式提供一个对象，并同时把它的 `apply` 设为
`case["bc_apply"]`：

```python
from types import SimpleNamespace

special = SimpleNamespace(
    name="my-boundary",
    faces=(FaceBoundary(...), ...),  # 必须是 i-/i+/j-/j+/k-/k+ 六侧
    apply=my_apply,                  # 纯 JAX 可追踪函数
)
case["special_boundary"] = special
case["bc_apply"] = special.apply
```

约束：回调面所在轴的轴级 `bcs` 必须是 `INFLOW`/`PLATE`（no-op 占位）；同轴另一侧
可设为 `OUTFLOW` halo，回调后由执行配置填充该侧状态鬼层，并在粘性通量阶段
填充对应出流侧的 Ev 鬼层；特殊侧的 Ev 暂时保留原值，其他物理条件需另行实现和验证。没有回调的轴只能是
`PERIODIC`/`OUTFLOW`，其 halo 面的 `kind` 必须等于轴级 `bcs`；
`WALL` 仍只允许计算 j 方向。`special_boundary` 不能与 `plate`/`channel_bc` 同时出现。
当前这是**显式导入**入口，不做动态注册；回调只允许数组运算，不得把 I/O 放进 JIT。
槽道三点压力外推和平板两点外推在 `boundary.py` 中分别保留原有公式。

***

## 8. 整步后的滤波、诊断和输出

### 8.1 激波选择性滤波

文件：`filtering.py`。

只有提供 `--filter` 才启用，并且每隔 `--nstep-filter` 个**整步**执行一次：



1. 从守恒量恢复压力。

2. 用压力二阶差分构造 Ducros 式激波传感器。

3. 在传感器达到阈值的区域施加四点选择性耗散。

4. 按 `axes` 逐方向处理；周期方向环绕，其他方向采用端点复制。

5. 可用 `mask` 限制滤波区域；正式运行默认全内区。

相关强度参数为 `--filter-s0` 和 `--filter-rth`。

### 8.2 诊断

文件：`ana.py`。



* `total_energy()`：计算内区平均 `rho*E`，不是总内能或动能。

* `check_nan()`：检查含鬼胞的全场是否出现 NaN。

* `diagnostic_line()`：生成主循环打印的一行诊断。

### 8.3 NPZ 输出

文件：`io.py`、`run.py`。

每个结果文件包含：



```
cons_0 ... cons_4：五个守恒变量的物理内区，形状均为 (nk, nj, ni)
_meta            ：JSON 元信息，包括 t、step、dt、case、N（实际 i 方向网格）、scheme、gamma、Re、Ama、Ref_T、Pr；槽道的 Re/Ama 取 `.in` 解析后的实际物性
```

读取示例：



```
import numpy as np

with np.load("output/sod-istep-00000100.npz") as data:
    rho = data["cons_0"]
    rho_u = data["cons_1"]
    print(rho.shape, rho_u.shape)
```



***

## 9. 五种算例分别怎样进入主流程

所有算例都由 `cases.py::build_case()` 统一返回 `case` 字典，后续主循环不再区分初值细节。



| 算例           | 初值和输入                                 | 求解方向与边界                  | 额外处理               | 验证辅助                              |
| ------------ | ------------------------------------- | ------------------------ | ------------------ | --------------------------------- |
| `sod`        | `make_initial_sod()` 生成左右常数态激波管       | x；x 出流，y/z 周期            | 默认无粘，可选滤波          | `exact_riemann.py` 提供 Riemann 精确解 |
| `tg`         | `make_initial_tg()` 生成周期正余弦速度场        | x/y；周期边界                 | 可叠加粘性              | `tests/test_taylor_green.py`      |
| `poiseuille` | `make_initial_poiseuille()` 生成抛物线速度剖面 | x/y；x/z 周期，y 双壁          | 始终计算粘性；x 向恒体积力     | 稳态层流剖面                            |
| `channel`    | 从原码 in 读取维数和物性；可读取原码初场与网格             | i/j/k；i/k 周期，j 双壁        | 始终计算粘性；槽道源项；可选曲线度量 | `compare.py` 逐子步对拍                |
| `plate`      | 均匀来流，边界层从入口向下游发展                      | x/y；入口 / 出口、下壁 / 顶部、z 周期 | 使用平板专用边界和鬼层口径      | `blasius.py` 提供相似解和 Cf 基准         |

### 9.1 channel 的外部文件链



```
--in opencfd-scu.in
   → infile.py：解析 GRID_3D、RE、AMA、PR、T_REF、BC_RPARA、JAC_BOUND、DIF_BOUND 等

--field opencfd.dat（可选）
   → mesh.py::read_ocfd_field：读取 rho/U/V/W/T
   → channel.py::cons_from_primitive：转换为五个守恒量
   不提供 --field 时使用 rho=T=1、速度为 0 的内区初值

--mesh OCFD3d-Mesh.dat（可选）
   → mesh.py::read_ocfd_mesh：读取 X/Y/Z 坐标
   → pad_coords：按周期/非周期规则补鬼胞
   → compute_metrics_from_coords：CD6 计算坐标导数、Jacobi 和余子式
   不提供 --mesh 时 metrics=None，按 Ajac=1 的直角假设计算
```

channel 命令示例：



```
python -m opencfd_jax.src.run --case channel `
  --in path/to/opencfd-scu.in `
  --field path/to/opencfd.dat `
  --mesh path/to/OCFD3d-Mesh.dat `
  --scheme OMP6_CD8 --vis-order CD8 --forcing-work off `
  --dt 0.0001 --end-step 1
```

### 9.2 平板与 Blasius 基准

`blasius.py` 不参与主求解循环。它通过打靶、二分壁面二阶导数和定步长 RK4 求解 Blasius 方程，提供：



* 任意相似变量 `eta` 上的插值；

* 平板速度剖面 `u/v`；

* 壁面摩擦系数 `Cf = 0.664/sqrt(Re_x)`。

平板算例命令示例：



```
python -m opencfd_jax.src.run --case plate `
  --ni 200 --nj 256 --nk 6 `
  --scheme OMP6_CD8 --viscous --vis-order CD8 `
  --Re 10000 --Ama 0.0955 --ref-t 1 `
  --dt 0.000009051060300912213 --end-step 1
```



***

## 10. 文件导航：想找某一步就看哪个文件



| 文件                 | 责任                                |
| ------------------ | --------------------------------- |
| `run.py`           | 命令行、正式主循环、RHS 总装、滤波 / 诊断 / 存盘节奏   |
| `runtime_config.py`| `input.json` 读取 / 校验 / 合并 / 输出前缀（第 3 步） |
| `cases.py`         | 五种算例的网格、物性、初值、边界与通用体积力            |
| `fields.py`        | 守恒量数组布局、`Grid`、旧边界导入兼容层          |
| `axes.py`          | 计算 i/j/k 与旧 x/y/z 的唯一词表和尺寸/面标签映射      |
| `boundary.py`      | 通用鬼胞填充、壁面镜像、Ev 与统一边界执行配置       |
| `solver.py`        | 单方向 / 多方向无粘 RHS、声速、CFL 步长、SSP-RK3 |
| `schemes.py`       | Steger–Warming 分裂、各对流格式、边界面通量带    |
| `viscous.py`       | 物性、CD6/CD8、单侧导数、应力与粘性通量散度         |
| `channel.py`       | 槽道原始变量转换、双壁条件、槽道体积力               |
| `plate.py`         | 平板入口、出口、下壁和顶部自由来流条件               |
| `mesh.py`          | 原码网格 / 场读取、坐标补层、曲线网格度量与缓存         |
| `mesh_data.py`     | 单块网格与六侧边界的只读描述（第 1 步）             |
| `infile.py`        | 原码 in 文件的键值解析和网格维数校验              |
| `filtering.py`     | 压力传感器与整步后的选择性激波滤波                 |
| `ana.py`           | 平均总能量、NaN 检查和诊断行                  |
| `io.py`            | NPZ 读写和通用原码 dat 读取                |
| `compare.py`       | 原版 dump 转 NPZ、JAX 对照跑、逐场误差报告      |
| `exact_riemann.py` | Sod 的 Riemann 精确解，仅用于验证           |
| `blasius.py`       | 平板 Blasius 相似解，仅用于验证              |
| `config.py`        | JAX FP64/FP32 开关                  |
| `__init__.py`      | 包级说明和核心模块导出                       |



***

## 11. 与原版 OpenCFD 逐点对拍

文件：`compare.py`。对拍链与正式运行分开，便于定位误差出现在哪个 RK 子步、哪个守恒分量。



```
原版文本 dump
  ↓ dat2npz
原版 NPZ（含 step0 初值和各 RK 子步 du）
  ↓ jax-run 使用同一 step0、dt、网格和物性运行 JAX
JAX NPZ（逐子步 du、逐整步 cons）
  ↓ diff
逐场 max|diff|、RMS、最大误差位置和 PASS/FAIL
```

在工作区根目录执行：



```
python -m opencfd_jax.src.compare dat2npz path/to/original-dump --out dump-npz

python -m opencfd_jax.src.compare jax-run `
  --case sod --ni 128 --nj 4 --nk 4 `
  --steps 3 --dt 0.001 --scheme UP7 `
  --ref-dir dump-npz --out dump-jax

python -m opencfd_jax.src.compare diff dump-npz dump-jax
```

对拍工具的 `jax-run` 始终计算无粘和粘性项，以对应原版 `du_comput`；其 `--viscous` 参数只为兼容旧命令保留。逐点验证使用 FP64，并以最大误差小于 `1e-12` 为通过标准。



***

## 12. 最容易混淆的几个点



1. **RHS 不等于状态更新**：`run.build_rhs()` 只构造 `L(Q)`；真正更新状态的是 `solver.rk3_step*()`。

2. **滤波不在 RK 子步内**：它在整个 RK3 完成后按整步频率执行。

3. **输出不含鬼胞**：内存数组每方向多 8 层，NPZ 只有物理内区。

4. **空间数组的计算轴顺序是 `(k,j,i)`**：计算方向按 `(i,j,k)` 排列，代码通过 `spatial_array_axis()` 映射；命令行/接口的旧 `x/y/z` 标签依次对应计算 `i/j/k`。

5. **无粘和粘性是两条独立空间离散链**：前者走 “通量分裂 → 界面重构 → 通量差分”，后者走 “梯度 → 应力 / 热流通量 → 再求导”。

6. **专用边界先于每个子步的 RHS**：channel 和 plate 不只是计算结束后修边，而是在每个 RK 子步中维护边界状态。

7. **channel 的曲线度量是可选的**：提供 `--mesh` 才读取并计算；否则走直角假设。

8. **验证模块不参与正式求解**：`exact_riemann.py`、`blasius.py`、`compare.py` 都是基准或对拍工具。
