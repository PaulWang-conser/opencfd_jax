# OpenCFD-JAX 计算流程说明

这套代码用 **JAX + NumPy** 求解五个守恒变量的可压缩流动方程，提供 Sod 激波管、Taylor–Green 涡、Poiseuille 槽道流和平板边界层四个命令行算例。代码包含无粘通量、可选粘性项、物理边界、体积力、三阶 Runge–Kutta 时间推进和 NPZ 存盘（激波选择性滤波已于 2026-11-23 删除）。单块 CGNS/HDF5 结构网格在 `poiseuille` 与 `plate` 上接入；多块 CGNS 在 `plate` 上以无粘、单进程的方式接入（7.1.2）；`channel` 不再是命令行算例，只保留为内部壁面库。

阅读时只需要先记住一条主线：



```
命令行参数
   ↓
cases.py：建立网格、物性、初值、边界和源项
   ↓
main.py：组装右端项 L(Q) = 无粘项 + 粘性项 + 源项
   ↓
solver.py：显式固定 dt + SSP-RK3 时间推进
   ↓
   ↓
io.py：只保存物理内区的守恒量 NPZ
```



***

## 1. 先跑一个最简单的算例

依赖：Python、JAX、NumPy。

在**工作区根目录** `D:\Desktop\ocfd2jax` 下运行：



```
python -m opencfd_jax.src.main --case sod --ni 256 --scheme UP7 --dt 1e-3 --end-time 0.2
```

当前目录结构中，源码包名是 `opencfd_jax.src`，所以不要写成 `python -m opencfd_jax.run`。

计算结果默认写到当前目录的 `output/`。例如：



```
output/sod-istep-00000050.npz
output/sod-istep-00000100.npz
...
```

输出文件名以 `-istep-` 加至少八位步号结尾。历史 JAX 基准已从工作区删除，不再用于验收。

程序默认使用 FP64。常用控制参数：



| 参数                        | 含义                                                      |
| ------------------------- | ------------------------------------------------------- |
| `--case`                  | 算例：`sod` / `tg` / `poiseuille` / `plate`；`channel` 只保留为内部库    |
| `--ni --nj --nk`          | 计算 i/j/k 方向内区离散点数，缺省 256/64/4；CGNS 时按 `VertexSize` 校验 |
| `--mesh-format`           | `builtin`（默认）或 `cgns`；只给 `--mesh` 时自动推断为 `cgns`       |
| `--mesh`                  | CGNS/HDF5 文件路径；单 zone 的文件走单块路径，多 zone 且没给 `--cgns-zone` 时走多块路径（7.1.2） |
| `--cgns-zone`             | 只读文件里这一个 zone；不给时读全部 zone，`plate` 上不止一个就走多块路径          |
| `--flux-scale`            | 多块必填：三轴参考尺度 `SX SY SZ`，取法见 7.1.2；对应 JSON 的 `numerics.flux_scale`   |
| `--scheme`                | 无粘重构格式，默认 `UP7`                                         |
| `--dt`                    | 固定时间步长，**必填且必须为正的有限数**；CFL 自适应已删除              |
| `--end-time`              | 终止物理时间                                                  |
| `--end-step`              | 最大整步数；与终止时间谁先满足就停止                                      |
| `--viscous`               | 对 Sod、Taylor–Green、平板启用粘性项；Poiseuille 始终计算粘性项；多块 plate 也可以，导数须为 `CD8`（7.1.2） |
| `--vis-order`             | 粘性导数格式：`CD6` 或 `CD8`；多块下粘性打开时只放行 `CD8`              |
| `--Re --Ama --ref-t --pr` | Reynolds 数、参考 Mach 数、有量纲参考温度和 Prandtl 数                 |
| `--kstep-show`            | 每隔多少整步打印一次诊断                                            |
| `--kstep-save`            | 每隔多少整步保存一次；最后一个整步也会保存                                   |
| `--out-dir`               | 输出目录，默认 `output`                                        |
| `--fp32`                  | 改用 FP32，主要用于性能测试；逐点验证建议保留默认 FP64                        |

查看全部参数：



```
python -m opencfd_jax.src.main --help
```

### 1.1 用 `input.json` 统一配置（第 3 步）

`--config path/to/input.json` 可以一次给出运行参数；命令行仍是最高优先级，不写
JSON 的字段保持程序默认值：

```powershell
python -m opencfd_jax.src.main --config experiment\cases\sod_1step.json
```

JSON 只分六组，外加顶层 `precision`：

| 组 | 字段 |
|---|---|
| `mesh` | `format`（`builtin`/`cgns`）、`file` |
| `physics` | `Re`、`Ama`、`ref_t`、`pr`、`tw`（壁温，无量纲，缺省与来流同温） |
| `numerics` | `scheme`、`vis_order`、`viscous`、`weno_tv`、`flux_scale`（多块的三轴参考尺度，见 7.1.2） |
| `time` | `dt`（必填）、`end_time`、`end_step` |
| `case` | `name`、`inflow_profile`（OCFD 1D 剖面文件，空着用自由来流常数）、`p_grad`、`forcing_work`、`plate_x0`、`plate_x1`、`plate_h` |
| `output` | `dir`、`kstep_show`、`kstep_save` |

优先级：程序默认值 → JSON → 显式命令行。显式与否由“参数是否出现在命令行”判断，
不靠与默认值比较。需要显式关闭 JSON 里的布尔项时用 `--no-viscous`、
`--no-weno-tv`。`time.cfl`、`--cfl`、`--adaptive-dt` 均已删除；运行必须给出固定 `dt`。

网格文件名决定输出前缀：给了 `--mesh`/`mesh.file` 就用文件名去最后一个扩展名
（`OCFD3d-Mesh.dat` → `OCFD3d-Mesh-istep-00000001.npz`），否则用算例名。JSON 内相对
路径按 JSON 所在目录解析，命令行路径按当前工作目录解析。

`channel` 的 `.in`、场文件和旧 `OCFD3d-Mesh.dat` 入口已从 CLI/JSON 删除；`channel.py`、
`infile.py`、`mesh.read_ocfd_mesh/read_ocfd_field` 只保留为内部库和旧测试依赖，不再从正式入口进入。
CGNS 路径适用于 `poiseuille` 和 `plate`：`mesh.file` 指向 HDF5/CGNS 文件，`mesh.format="cgns"` 可由
`file` 自动推断；`VertexSize` 是离散点数，`CellSize` 只校验 `+1`，不做全局八点平均。未知字段、
不属于当前算例的字段、`CellCenter`、`Rind>0`、`PointList`、二维 zone 和单侧对称面都会在启动前
或 reader 中报错。多 zone 的文件在没有给 `--cgns-zone` 时走多块路径：只放行 `plate`，
并且要给 `numerics.flux_scale`（7.1.2）；给了 `--cgns-zone` 就只读那一块，照旧走单块。



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
  尺寸只使用 `ni/nj/nk`；旧 `nx/ny/nz` 构造/属性不再提供。物理笛卡尔坐标
  `x/y/z` 与计算坐标通过网格度量联系。
* 原码度量键 `Akx/Aky/Akz`、`Aix/Aiy/Aiz`、`Asx/Asy/Asz` 分别属于
  计算 `i/j/k` 方向；键末尾的 `x/y/z` 表示物理坐标分量，不是计算轴名。

* 每个方向有 `LAP = 4` 层鬼胞，为高阶模板和边界处理提供数据。

* `Grid.inner()` 选出不含鬼胞的物理内区。

* 诊断和无粘 / 粘性右端项可以操作完整数组，但最终存盘只保存 `(nk, nj, ni)` 内区。

`boundary.py` 负责周期环绕、出口鬼层（缺省保持原值）、壁面镜像和 Ev 等通用填充；`fields.py`
只保留数组布局、`Grid` 和 canonical 填充入口的重新导出。平板的专用入口、壁面和顶部条件仍由
`plate.py` 写入，槽道双壁由 `channel.py` 写入。



***

## 3. 完整计算流程：每一步在哪个文件

主入口是 `main.py` 的 `main()`，初始化集中在 `init.py` 的 `prepare()`，时间推进由 `main.py::run()` 负责。实际执行顺序如下。



| 步骤 | 做什么             | 文件和关键函数                            | 处理方式                                            |
| -- | --------------- | ---------------------------------- | ----------------------------------------------- |
| 1  | 读取命令行并设置精度      | `main.py::main`、`init.py::prepare`、`config.py` | 默认开启 JAX x64；`--fp32` 时关闭；精度在数组创建前设置 |
| 2  | 装配算例            | `init.py::prepare`、`cases.py::build_case` | 创建 `Grid`、`PhysParams`、初始守恒量、边界类型、求解方向、步长和源项    |
| 3  | 读取外部网格（仅 CGNS 需要时） | `mesh_cgns.py` | `poiseuille` 读 CGNS/HDF5，补鬼胞坐标并算曲线度量；旧 `.in`/场/OCFD 网格入口已删除 |
| 4  | 组装单个 RK 子步的右端项  | `main.py::build_rhs`                | 形成 JIT 函数 `rhs(q)`，组合边界、无粘、粘性和源项；滤波不放入其中        |
| 5  | 确定时间步长          | `main.py::run`                    | 只接受显式固定 `dt`；不再有 CFL 自适应或 `compute_dt` |
| 6  | 执行一个 SSP-RK3 整步 | `solver.py::rk3_step_stages`       | 三次计算右端项，三次组合状态；每个子步的 `rhs` 先 `execution.prepare`，再算 RHS |
| 8  | 打印诊断            | `ana.py::diagnostic_line`          | 输出时间、步数、耗时、内区平均 `rho*E` 和全场 NaN 标志              |
| 9  | 保存结果            | `io.py::save_npz`                  | 将五个内区分量及时间、步数、格式和物性元信息写入 NPZ                    |
| 10 | 重复推进            | `main.py::run`                     | 回到步骤 5，直到达到终止时间或最大步数                            |

错误分层：配置解析错误由 `runtime_config.ConfigError` 转成 `配置错误`；`init.prepare`
的失败在 `run` 边界包装成 `errors.SetupError`，由 `main` 转成 `初始化错误` 并
`parser.error`；时间推进中的异常包装成 `errors.RuntimeExecutionError`，保留原始
异常链与 `case/step/t/dt/grid` 上下文，不再被转成启动参数错误。

下面展开最耗计算量的第 5～7 步。



***

## 4. 一个时间整步内发生了什么

### 4.1 时间步长：只支持显式固定 dt

文件：`main.py::run`；`solver.py` 只保留 `rk3_step()` / `rk3_step_stages()`。

* `--dt` / JSON `time.dt` 必须显式给出，且为正的有限数；缺失或非正会在启动前报错。
* 最后一步按 `end_time` 缩短以准确到达终止时间。
* `compute_dt()`、`--cfl`、`--adaptive-dt` 和 JSON `time.cfl` 已删除；没有 CFL 自适应路径。

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

文件：`main.py`；内部函数：`rhs(q)`。

顺序固定为：



```
0. execution.prepare(q)：写边界点 → 填鬼层 → 特殊回调（见 7.2）
1. 计算无粘右端项 inviscid_rhs
2. 若算例需要，叠加 viscous_rhs
3. 叠加 cases.apply_forcing 的通用体积力
4. 若算例提供 source，再叠加专用源项（channel 体积力）
返回 du，但不在这里更新 Q
```

边界只在 `prepare` 里走一遍：鬼层、边界点、特殊回调各管一层，互不重写。
RK 外层不再重复调用 `bc_apply`（旧的重复调用已被去掉）。



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



* `boundary.py` 提供周期、出流、壁面镜像和 Ev 鬼胞；`fields.py` 只重新导出这些
  canonical 入口，实际实现只在 `boundary.py` 一处。

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

注意：Sod 初值采用常见的 `E = p/(gamma-1)` 验证约定，与粘性模块的 OCFD `Ama` 无量纲约定不同。程序保留 `sod + --viscous` 的灵活性，但会打印警告；研究粘性时应优先使用 Taylor–Green、Poiseuille 或 plate 的相应约定。



***

## 7. 边界和源项在哪里处理



| 类型        | 文件                        | 具体处理                                 |
| --------- | ------------------------- | ------------------------------------ |
| 周期边界      | `boundary.py`             | 从对侧内区环绕复制鬼胞                          |
| 出流     | `boundary.py`             | 保持已有状态鬼层，不覆盖出口边界点；缺省不开单侧带       |
| 槽道双壁      | `channel.py`              | 只写壁面层；压力 / 温度按壁面类型外推；鬼层清零 + 该侧单侧带      |
| 平板入口      | `boundary.py` + `plate.py` | 通用 INFLOW：写边界点 + 四层鬼层同一剖面，可来自 1D 剖面文件 |
| 平板出口      | `boundary.py`             | 保持已有状态鬼层；CGNS plate 的初始出口鬼层为零              |
| 无反射出流    | `boundary.py` + `schemes.py` | 原码 `NON_REFLETION` 口径：鬼层、Ev、边界带与出流相同；无粘 RHS 在边界内点只留一半分裂通量差（低侧丢正分裂、高侧丢负分裂） |
| 平板下壁 / 顶部 | `plate.py` / `boundary.py` | 下壁只写壁面点（ρ 两点外推 + 无滑移 + ρ Cv Tw）；顶部远场只写最后一层内点，鬼层清零 + 该侧单侧带 |
| 通用体积力     | `cases.py::apply_forcing` | 动量源 `rho*G`，可选能量做功源 `rho*u*G`        |
| 槽道专用体积力   | `channel.py::body_force`  | 按 `ChannelBC.F` 生成源项；是否做功由 `work` 控制 |

与原码 channel 逐项对拍时使用 `--forcing-work off`，因为原码对应内核只添加动量源。正式 `main.py` 的默认值为 `on`。

### 7.1 单块网格与边界描述（第 1 步）

文件：`mesh_data.py`。

`main.py`（`build_rhs` / `run`）和 `compare.py`（`cmd_jax_run`）在消费入口由
`from_case(case)` 即时构造只读快照 `BlockMesh`，而不是另外保存一份配置。描述包含：

* `grid`、`dxs`、`axes`、`bcs`：直接复用算例字典里的对象/值；
* 六侧 `FaceBoundary`：每侧语义（periodic/outflow/wall/inflow/farfield）、当前
  执行来源（`halo` / `callback` / `callback+halo`）和该侧的边界带开关；
* `dif_bound`、`dif_axis_map`：保留原码六标志与 JAX 轴映射（平板仍是 `(0,2,1)`）；
  但**执行用的带不再读这六个标志**，改由各面的 kind 派生（见 7.2）；
* `metrics`、`coords`：只有算例真的提供时才引用；没有就保持 `None`，不伪造。

描述现在由 `boundary.build_execution` 转成唯一一份 `BoundaryExecution`，`main.py`
和 `compare.py` 都用它决定轴级 `bcs` 与 case 回调；描述不写回算例字典，避免出现
两份可独立修改的状态。构造必须发生在创建 JIT 闭包之前；之后修改 `case` 不会回溯
已经建立的闭包。

### 7.1.1 CGNS 单块接入（第 4 步）

文件：`mesh_cgns.py`。`read_cgns_zone()` 只读 HDF5/CGNS 的 `CGNSBase_t`、`Zone_t`、
`GridCoordinates_t`、`ZoneBC_t` 和 `GridConnectivity1to1_t`；坐标转成 `(nk,nj,ni)`，
`PointRange` 的 1-based 索引转 0-based。`to_block_mesh()` 把可执行子集翻译成
`BlockMesh`（`dxs=(1,1,1)`，物理间距进入 `Ajac`/度量），`build_cgns_block()` 再补
LAP 鬼胞坐标并计算 CD6 度量。

当前执行子集：三维 `Structured`、`Vertex` 坐标、整面 `BC`、同块 identity 1to1；
支持通过 `FamilySpecified`、`FamilyName_t` 和 `FamilyBC_t` 解析边界类型。
`BCWall` 映射 `WALL`，`BCInflow` 映射 `INFLOW`，`BCExtrapolate`、`BCOutflow`、
`BCOutflowSubsonic` 映射 `OUTFLOW`，`BCFarfield` 映射单面的 `FARFIELD`。
本项目约定 `BCSymmetryPolar` 为周期，须同轴正负整面成对；`BCSymmetryPlane`
不支持；`BCOutflowSupersonic` 映射 `NONREFLECTING`（原码 `NON_REFLETION` 口径，
SIDS 字面是"全部外推"，与此不同；零梯度出流用 `outflow` 加 `boundary.outlet_ghost="copy"`，目前只有 CGNS plate
消费该开关；内置 plate 与其他算例给出时会在配置解析阶段报错。
`BCWallViscous*` 和其他 `BCInflow*` 类型不在执行子集内。CGNS 不解析 `BCDataSet`。
CGNS plate 的入口、远场原始量和壁温由 JSON 的 `boundary` 提供；
其他算例的默认来流按各自配置确定。真实 plate 的一步三阶段 RHS 与末态已与
原码对拍签收，范围与待验项见 `experiment/step49_progress_2026-11-19.md`。
`validate=False` 保留所有原始 `BCType` 供诊断，但仍拒绝
`CellCenter`、`Rind>0`、`PointList`、`Periodic_t` 和二维 zone。

### 7.1.2 多块 CGNS 接入（第 6、7 步）

文件：`multiblock.py`。给了 CGNS 网格、没有给 `--cgns-zone`、文件里不止一个 zone 时，
`main.run` 把运行交给多块路径；其余情况照旧走单块。多块路径只认一种网格：

- 一个 `CGNSBase_t`，所有 zone 都是三维 `Structured`；
- 块与块之间用 `GridConnectivity1to1_t` 相接，`Transform` 是 `(1,2,3)`（缺省即是），
  低侧面接高侧面；一个面可以分成几段，分别接不同的邻居；两个方向各写一条记录；
- 周期方向被切开时，两端露出的面用带 `Periodic_t` 的 1to1 配成周期接缝：旋转中心、
  旋转角必须有，旋转角为零；`Translation` 可以不写，写了就与实测核对，两条记录都写
  时要彼此一致。周期方向没有被切开时，沿用同轴成对的 `BCSymmetryPolar`，或同块相对
  整面的 1to1 自连接；
- 每个面元恰好被一条边条或一段连接覆盖。

启动后依次做两件事：

1. **放行**：多块路径只验收过一组配置，不在清单里的在读坐标之前报错。这一步只读
   各 zone 的名字、类型和尺寸；文件里不止一个 zone 而其中有读不出来的（不是
   `Structured`、节点残缺、尺寸解不出），也在这里按多块网格的校验报错。只有一个
   zone 的文件出了任何问题都交回单块路径，报错与没有多块路径时相同：单块路径自己
   报什么仍由它决定，残缺的文件可能在那里直接抛出原始异常。
   - 生效值：算例 `plate`、格式 `WENO7`、FP64、固定的正步长、不带特殊边界脚本。
   - 粘性项可以打开（第 7 步）。打开时粘性导数的阶数必须是 `CD8`；是 `CD6` 的在读坐标
     之前报错，消息里点出 `vis_order`。粘性关闭时阶数不参与计算，是 `CD6` 也放行。
   - 不能显式给出，取值与缺省相同也不行：`--weno-tv`、`--outlet-ghost`、`--ni`、`--nj`、
     `--nk`、`--tw`、`--inflow-profile`、`--plate-x0`、`--plate-x1`、`--plate-h`、
     `--plate-dif-bound`。
   - 入口、远场状态和壁温只从 JSON 的 `boundary` 读，壁温要大于 0，命令行没有对应的
     选项。
   - 参考尺度（`numerics.flux_scale` 或 `--flux-scale SX SY SZ`，多块必填）：WENO 的
     ε 要按等价单块的全局顶点数换算。把等价单块的全局顶点数记作 (Ni, Nj, Nk)，第 axis
     个分量是其余两轴 (N−1) 的乘积，与单块路径按网格尺寸推断的是同一个算法。
     100×40×40 对应 `1521 3861 3861`。它不是顶点数，也不是某一块的尺寸：按某一块填
     会让界面上的面值差出 2×10⁻⁵。校验只拦得住明显填小的，即某个分量小于任何一块按
     自身尺寸算出的值；各块尺寸相同时，按单块尺寸填也能通过，所以数值要按上面的算法算。
2. **校验**（`build_topology`）：读入全部 zone 后查连接、边条、覆盖表、每个面的归类和
   块的尺寸。有问题就报错，消息用原名指出是哪个 zone 的哪条连接或哪条边条；读坐标、
   校验时出的其他异常也报成这一句，消息里带原始异常的类型和文字；Base 下打不开的
   链接点名报错。通过的网格打印一份接口报告：每个接口对的两侧块、面、范围、对应
   点数；普通切口印最大坐标差，周期接缝印平移量和实测位移的最大偏离。

每个 RK 阶段的顺序是：各块从自己的阶段态做物理准备，全块同步交换接口上的守恒量
鬼胞，各块算无粘右端项，最后用准备之前的阶段态做 RK 组合。整步结束后全块再做一次
带交换的准备。接口旁的坐标和度量在装配时就换好：普通切口旁取供体的内区值，周期
接缝旁沿用单块周期端的规则。

粘性打开时，算完无粘右端项以后多出三步：各块用同一份准备且交换过的状态算自己的粘性
通量 `Ev`（`viscous.viscous_fluxes`；三个计算方向各四个分量，共 12 个平面，物理面和
块内周期面上的 `Ev` 鬼胞在这里按单块的规则填好），全块同步交换一次 `Ev`，各块对换过
的 `Ev` 求导并累加（`viscous.viscous_divergence`）。单块的 `viscous_rhs` 就是这两段的
顺序组合，结果与拆分之前逐位相同。

周期接缝的检查只用整面上的对称量：实测位移（供体坐标减接收坐标）在每个分量上的
中点当作平移量，半个跨度当作最大偏离。半个跨度不超过容差，位移才算常向量；中点
不超过容差，就算平移量约为零，报错。这样换 zone 的先后、记录怎么倒写，结论都不变。

一个面分成几段时，几段只在棱或角上相接，相接的顶点只归一段：两个切向轴上下标都
较小的那个相邻面元所属的段。T 形处它就是切向起点较小的段拥有那条棱；长棱对着两条
短棱、四段相接的角点，也只有一个写入者，结果与段的先后无关。坐标、度量、守恒量和
粘性通量 `Ev` 共用这一张表，建表后自检每个接收槽恰好被一张表写到，逐槽比对，不只是
数个数。

`Ev` 的交换只覆盖面内、另外两个下标在内区里的鬼胞，块角的鬼胞从不填。这样够用：
内区的粘性右端项只沿三个坐标方向取点，读接口外的守恒量或 `Ev` 时，读的是面内的
鬼胞；块角上的坐标只进得到鬼胞度量，而粘性项里的度量是逐点用的。
`experiment/test_step7_step.py` 在五个合成小网格上把块角的度量、守恒量、`Ev` 分别换成
NaN，内区三个阶段的粘性分项逐位不变；把块角的坐标换成 NaN 后重算度量，内区的度量逐位
不变。所以块角处的坐标不需要另外处理。

`Grid/split_cgns.py` 切网格时，在网格旁边写一份同名的 JSON（旁记），记着源网格、
等价单块的全局顶点数 `global_vertices` 和每块在全局上的起止下标。多块运行用其中的
`global_vertices` 给物理面旁的坐标鬼层选规则，与等价单块相同。没有旁记文件、或旁记
里没记这一项时，各块按自己的轴长选。文件在而读不出来，或者这一项不是三个 1 到
2³¹−1 的整数，报错。

输出每个保存步每块一个 NPZ，文件名是前缀、块名的 slug 和步号，`_meta` 多一项
`zone`。两个块的 slug 不分大小写也要不同，否则在 Windows 上会写进同一个文件。
每个显示步每块各打印一行诊断。

验收网格可以用 `Grid/split_cgns.py` 从 plate 单块网格切出来：`python Grid/split_cgns.py`
把七个预设写到 `Grid/step6_meshes/`（不入库）。五个验收网格各跑两个整步，三个阶段的
右端项、准备之前的 `q1`、`q2` 和末态与等价单块按阈值比较，见
`experiment/test_step6_acceptance.py`。阈值写在 `experiment/step6_support.py`：每个
守恒量各有相对容差（1e-11 乘自身最大值）和绝对容差（1e-13 乘参考场五个分量的最大
值），近零的守恒量由后者控制。

粘性打开的验收分两层，阈值与无粘时相同。第一层（`experiment/test_step7_acceptance.py`）：
五个验收网格各跑两个整步，比较三个阶段的总项和粘性分项、`q1`、`q2`、末态；粘性分项
的最大差 1.2e-12，总项 1.5e-10。第二层（`experiment/test_step7_linear.py`）：五个合成
小网格（两块切口、周期接缝、T 形、阶梯、一格宽的条带）上放线性场和常数来流，一步的
粘性右端项与各自的等价单块比较，最大差 4e-16。不换 `Ev`、换 `Ev` 时取错层、切口面
被套上单侧公式、RK 组合用了准备过的状态，这四种错误都被验收发现。

还没做的：多块下粘性导数取 CD6（只放行 CD8）、plate 以外算例的粘性、和原码多块结果的
对拍（这台机器没有 MPI 的编译器）、每个保存步一个 NPZ（第 8 步）、翻转和轴交换的连接
（第 10 步再定）、特殊边界脚本配多块。

### 7.2 统一边界执行与扩展入口（第 2 步）

文件：`boundary.py`、`mesh_data.py`、`main.py`。

通用边界（周期、出流、计算 j 方向壁面镜像、远场、非周期鬼层清零）只在 `boundary.py` 实现；
`fields.py` 只重新导出 canonical 名，`channel.py` / `plate.py` 保留壁面和回调的兼容入口，
壁面及平板的已有状态赋值公式由 `boundary.py` 提供。`build_execution(case, desc=None)` 在 JIT 前完成：

* 校验 `case["bcs"]`、六侧 `FaceBoundary` 与 `bc_apply` 是否同源；
* 拒绝“有回调但无专属标记”“多个专属标记同时存在”“固定六侧与 bcs 矛盾”
  “通用 WALL 不在计算 j 方向”等组合，未知组合直接报错，不静默改成出流；
* 返回 `BoundaryExecution(mode, bcs, callback, faces, inflow, band, ...)`，`mode` 为
  `generic` / `channel` / `plate` / `custom`。

### 7.2.1 三层分工：鬼层、边界点、特殊回调

`BoundaryExecution.prepare(q)` 固定三段，后段可以覆盖前段，但职责不交叉：

| 层 | 谁负责 | 做什么 |
| --- | --- | --- |
| 1. 边界点 | `write_faces` | INFLOW 写边界点 + 四层鬼层（同一剖面）；WALL 写壁面点；FARFIELD 写最后一层内点；壁面和远场只写横向内区；OUTFLOW/PERIODIC 不赋值 |
| 2. 鬼层 | `fill_faces` | 周期复制对侧内区；INFLOW 和出口保持已有值；壁面与远场清零 |
| 3. 特殊回调 | 算例 `bc_apply` | 只在常规赋值之后改局部区域（加射流、改壁面点…），继承该面的 band，自己不设 band |

三段顺序是 **写边界点 → 填鬼层 → 特殊回调**；回调最后写，避免其结果被常规填充重写。

口径依据：原码 `OCFD_bc()` 只在六个块边界面调 `pri_to_cons_kernel`（单层平面），
`commen_kernel.cu` 注明 "eyes on cells WITHOUT LAPs"，未被赋值的鬼胞保留初始化值；入口的填鬼是
入口 `do_u2d_inlet_kernel`（`for(i=0;i<=LAP;i++)`），所以 INFLOW 单列一支。壁面与远场的状态鬼层清零、边界点单独赋值；是否使用单侧带由各面配置决定。
本次原版 IBC=124 出口没有赋值函数，出口状态鬼层保留初始化值；`INIT_STAT=1` 参考中的值为零，`INIT_STAT=0` 可以不同。当前通用出口按此保持语义执行，算例负责初值，它并不等同零梯度出口。

`fill_wall_j` / `fill_halos_j(WALL)` 这一对只服务槽道壁面这种整轴同类型的算例，
按面路径不要用（`mesh_data` 给每侧单独的 `FaceBoundary`，双壁各自清零）。

### 7.2.2 边界带按面配置

构造 face 时 `dif_band(kind)` 为壁面和远场提供默认值；实际执行读取每个 `FaceBoundary.dif_band`，经 `dif_axis_map` 组成六侧 `DIF_BOUND`。CGNS plate 参考明确设置远场为 0、壁面为 1，以匹配当前 OCFD 参考配置；其他 case 可以根据自身口径设置。算例字典里的旧 `dif_bound` 只留档。

### 7.2.3 INFLOW 剖面

`write_faces` 对 INFLOW 面写“边界点 + 四层鬼层”，剖面来源按优先级：

1. `case["inflow"]`：用户给的剖面（形状 `(5, nk, nj)`），平板可由
   `--inflow-profile` / `input.json` 的 `case.inflow_profile` 指定 OCFD 1D 文件
   （`inflow.py`：首行表头，其后 `y rho u v T`，`w≡0`，与格点逐点对应、不插值，
   点数不等于 `nj` 直接 `SetupError`）；
2. 没有剖面文件时回落到 `case["farfield"]` 自由来流常数；
3. 两者都没有（旧式特殊入口）时不赋值，整段交给算例回调。


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
可设为 `OUTFLOW` halo。常规层只写该面 kind 允许的东西（INFLOW 写边界点+鬼层，壁面写
壁面点，远场写最后一层内点，出口/周期不赋值），鬼层按 kind 填：周期复制、入口保持、
出口保持、壁面与远场清零；特殊回调在之后覆盖它接管的局部区域
（例如在远场首层加激波）。鬼层由 `fill_faces` 处理，回调不需要也不应该自己清；
粘性 Ev 鬼层按面填充（周期复制，入口、出口、壁面和远场清零）；非周期鬼层清零对应当前原版参考初始化口径。
没有回调的轴只能是 `PERIODIC`/`OUTFLOW`/`WALL`/`FARFIELD`，其 halo 面的
`kind` 必须等于轴级 `bcs`；`WALL` 仍只允许计算 j 方向。`special_boundary` 不能与
`plate`/`channel_bc` 同时出现。当前这是**显式导入**入口，不做动态注册；回调只允许
数组运算，不得把 I/O 放进 JIT。槽道三点压力外推和平板两点外推在 `boundary.py` 中
分别保留原有公式。

***

（激波选择性滤波模块已于 2026-11-23 按 jisen 决定整体删除；历史行为见 git 与 OCFD_filtering.cu。）

## 9. 四个命令行算例分别怎样进入主流程

所有算例都由 `cases.py::build_case()` 统一返回 `case` 字典，后续主循环不再区分初值细节。
`channel` 已从 CLI 算例列表移除；`channel.py` 只作为壁面/初值库保留给 `plate` 和内部测试。



| 算例           | 初值和输入                                 | 求解方向与边界                  | 额外处理               | 验证辅助                              |
| ------------ | ------------------------------------- | ------------------------ | ------------------ | --------------------------------- |
| `sod`        | `make_initial_sod()` 生成左右常数态激波管       | x；x 出流，y/z 周期            | 默认无粘                    | `exact_riemann.py` 提供 Riemann 精确解 |
| `tg`         | `make_initial_tg()` 生成周期正余弦速度场        | x/y；周期边界                 | 可叠加粘性              | `tests/test_taylor_green.py`      |
| `poiseuille` | 内置 `make_initial_poiseuille()`，或 CGNS `Vertex` 坐标 | x/y；x/z 周期，y 双壁          | 始终计算粘性；x 向恒体积力；CGNS 带曲线度量 | 稳态层流剖面；Step 4 NPZ 对齐 |
| `plate`      | 均匀来流，边界层从入口向下游发展                      | x/y；入口 / 出口、下壁 / 顶部、z 周期 | 使用平板专用边界和鬼层口径      | `blasius.py` 提供相似解和 Cf 基准         |

### 9.1 CGNS 单块网格路径

`poiseuille` 可用 HDF5/CGNS 文件替代内置均匀网格：

```powershell
python -m opencfd_jax.src.main --case poiseuille `
  --mesh opencfd_jax/Grid/fixtures/poiseuille_box.cgns `
  --dt 1e-3 --end-step 1 --end-time 0.5 --kstep-save 1
```

只给 `--mesh` 时运行配置自动把 `mesh_format` 推断为 `cgns`。CGNS 的 `VertexSize`
（离散点数）与物理间距进入 `Ajac`，`dxs` 固定为索引空间 `(1,1,1)`；因此不再需要
`dt_dxs`。单块路径只支持单块、三维、整面边界；`multi_hdf5.cgns` 这类多 zone 文件要用
`--cgns-zone` 指定一个 zone，且默认严格模式会拒绝 `BCInflow*`/`BCWallViscous*` 等
未执行边界；`BCFarfield` 已执行，但来流状态需由算例提供。

#### 多块 plate

多 zone 的 CGNS 文件不给 `--cgns-zone`，在 `plate` 上走多块路径（7.1.2）。入口、
远场和壁温只有 JSON 能给，所以配一份 `input.json`，网格路径相对这个文件：

```json
{
  "case": {"name": "plate"},
  "mesh": {"format": "cgns", "file": "x2.cgns"},
  "boundary": {
    "inflow": {"rho": 1, "u": 1, "v": 0, "w": 0, "T": 1},
    "farfield": {"rho": 1, "u": 1, "v": 0, "w": 0, "T": 1},
    "wall_temperature": 1.0
  },
  "numerics": {"scheme": "WENO7", "viscous": false, "flux_scale": [1521, 3861, 3861]},
  "time": {"dt": 0.0001, "end_time": 0.0001, "end_step": 1},
  "output": {"dir": "output", "kstep_save": 1}
}
```

```powershell
python -m opencfd_jax.src.main --config 多块目录\input.json --kstep-show 1
```

`x2.cgns` 可以用 `python Grid/split_cgns.py --preset x2 --out-dir 多块目录` 从 plate 单块
网格切出来。运行时先打印接口报告，再每个显示步每块一行诊断，输出
`x2-blk-i0j0k0-istep-00000001.npz` 和 `x2-blk-i1j0k0-istep-00000001.npz`。
`flux_scale` 是等价单块 100×40×40 的参考尺度，换别的网格要按 7.1.2 的算法重算。

要打开粘性，把 `numerics` 里的 `viscous` 改成 `true`、`vis_order` 写成 `"CD8"`，再给一份
`physics`：

```json
"physics": {"Re": 1000.0, "Ama": 0.1, "ref_t": 288.15, "pr": 0.7},
"numerics": {"scheme": "WENO7", "viscous": true, "vis_order": "CD8", "flux_scale": [1521, 3861, 3861]}
```

物性与单块用同一个函数组装。启动时多打印一行 `vis-order=CD8 Re=1000.0 Ama=0.1
Ref_T=288.15 Pr=0.7`；`ref_t` 没给时取 288.15 K，并像单块一样给出提示。`vis_order` 是
`CD6` 时启动就报错。存盘的文件名和 `_meta` 与无粘时相同。

旧 `channel` 的外部文件链（`--in`、`--field`、`--mesh` OCFD dat）已删除；`infile.py`、
`mesh.read_ocfd_mesh/read_ocfd_field`、`channel.py` 只保留为内部库和旧测试依赖，
不再有正式入口。

### 9.2 平板与 Blasius 基准

`blasius.py` 不参与主求解循环。它通过打靶、二分壁面二阶导数和定步长 RK4 求解 Blasius 方程，提供：



* 任意相似变量 `eta` 上的插值；

* 平板速度剖面 `u/v`；

* 壁面摩擦系数 `Cf = 0.664/sqrt(Re_x)`。

平板算例命令示例：



```
python -m opencfd_jax.src.main --case plate `
  --ni 200 --nj 256 --nk 6 `
  --scheme OMP6_CD8 --viscous --vis-order CD8 `
  --Re 10000 --Ama 0.0955 --ref-t 1 `
  --dt 0.000009051060300912213 --end-step 1
```



***

## 10. 文件导航：想找某一步就看哪个文件



| 文件                 | 责任                                |
| ------------------ | --------------------------------- |
| `main.py`           | 命令行、正式主循环、RHS 总装、滤波 / 诊断 / 存盘节奏   |
| `init.py`           | 初始化编排：解析配置、精度、网格、数组、初场和边界组装 |
| `runtime_config.py`| `input.json` 读取 / 校验 / 合并 / 输出前缀（第 3 步） |
| `cases.py`         | 四个命令行算例的网格、物性、初值、边界与通用体积力            |
| `fields.py`        | 守恒量数组布局、`Grid`、canonical 边界入口重新导出          |
| `axes.py`          | 计算 i/j/k 词表、尺寸和六侧标签工具                  |
| `boundary.py`      | 通用鬼胞填充、壁面镜像、Ev 与统一边界执行配置       |
| `solver.py`        | 单方向 / 多方向无粘 RHS、声速、SSP-RK3；无 CFL 步长 |
| `schemes.py`       | Steger–Warming 分裂、各对流格式、边界面通量带    |
| `viscous.py`       | 物性、CD6/CD8、单侧导数、应力与粘性通量散度；`viscous_fluxes` 和 `viscous_divergence` 两段供多块在中间交换 `Ev` |
| `mesh_cgns.py`     | CGNS/HDF5 读取（单块与多 zone）、执行子集校验、BlockMesh 适配与度量（第 4、6 步） |
| `multiblock.py`    | 多块路径：入口判定与放行、拓扑校验、交换表、逐 RK 阶段的编排和运行循环（第 6 步）；粘性通量的交换和粘性分支（第 7 步） |
| `channel.py`       | 内部槽道壁面/体积力库；不再是 CLI 算例               |
| `plate.py`         | 平板形态封装：自由来流常数、入口剖面形状、`apply` = 只写壁面点 |
| `inflow.py`        | OCFD 1D 入口剖面文件读取（`y rho u v T`，`w≡0`，不插值） |
| `mesh.py`          | 原码网格 / 场读取、坐标补层、曲线网格度量（内部库）         |
| `mesh_data.py`     | 单块网格与六侧边界的只读描述（第 1 步）             |
| `infile.py`        | 原码 in 文件键值解析（内部库；无正式入口）              |
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



1. **RHS 不等于状态更新**：`main.build_rhs()` 只构造 `L(Q)`；真正更新状态的是 `solver.rk3_step*()`。

2. **滤波不在 RK 子步内**：它在整个 RK3 完成后按整步频率执行。

3. **输出不含鬼胞**：内存数组每方向多 8 层，NPZ 只有物理内区。

4. **空间数组的计算轴顺序是 `(k,j,i)`**：计算方向按 `(i,j,k)` 排列，代码通过 `spatial_array_axis()` 映射；命令行/接口的旧 `x/y/z` 标签依次对应计算 `i/j/k`。

5. **无粘和粘性是两条独立空间离散链**：前者走 “通量分裂 → 界面重构 → 通量差分”，后者走 “梯度 → 应力 / 热流通量 → 再求导”。

6. **专用边界先于每个子步的 RHS**：channel 和 plate 不只是计算结束后修边，而是在每个 RK 子步中维护边界状态。

7. **Poiseuille 的曲线度量是可选的**：内置网格走直角假设（`Ajac=1`）；给 `--mesh` 才读取 CGNS 并计算曲线度量。

8. **验证模块不参与正式求解**：`exact_riemann.py`、`blasius.py`、`compare.py` 都是基准或对拍工具。

9. **多块的参考尺度不是块的尺寸**：`flux_scale` 取等价单块的全局顶点数来算，多块文件自己不携带这个数，所以要显式给（7.1.2）。
