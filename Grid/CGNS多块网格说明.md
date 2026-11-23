# CGNS 多块结构网格：文件格式与边界条件

面向 `opencfd_jax` 的 Step 4（CGNS 单块/多块读取）。本文的正文字段名、枚举值以 CGNS 官方
SIDS 为准，**示例数字全部来自真实文件实测**：`Grid/samples/multi_hdf5.cgns`
（CFL3D 输出的 4 块结构网格，`CGNSLibraryVersion = 3.21`），用 h5py 逐节点转储得到。

- CGNS 官方组织：<https://github.com/CGNS>
- SIDS 分层结构：<https://cgns.org/standard/SIDS/hierarchy.html>
- SIDS 多块连接：<https://cgns.org/standard/SIDS/multizone.html>
- SIDS 边界条件：<https://cgns.org/standard/SIDS/boundary.html>
- SIDS 网格坐标：<https://cgns.org/standard/SIDS/grid.html>
- HDF5 映射：<https://cgns.org/standard/hdf5.html>
- 两块平板完整示例（含 1to1 与 BC 写法）：<https://cgns.org/standard/SIDS/appendix_b.html>

---

## 0. 速览（读一个多块 CGNS 只需知道这 6 条）

1. **多块 = 一个 `CGNSBase_t` 下平铺若干 `Zone_t`**，块之间没有父子关系；块间关系写在**每块自己**的
   `ZoneGridConnectivity_t` 下，且每个界面**双向各记一份**（接收侧、供体侧）。
2. **尺寸是三元组**：`Zone_t` 的数据就是 `[VertexSize, CellSize, VertexSizeBoundary]` 三行；
   `CellSize = VertexSize - 1`（结构网格逐向减一）。
3. **索引一律 1-based，i 变化最快**；`PointRange` 和块间连接里的索引指的是**顶点**，不是格心。
4. **1to1 界面**由四个东西定义：`PointRange`（本块顶点子区）、`PointRangeDonor`（供体块顶点子区）、
   `Transform`（轴映射 + 翻转符号）、节点自身的 data（**供体块名**）。
5. **边界条件**在 `ZoneBC_t/BC_t`：`BCType` 是字符串，直接存在 `BC_t` 节点的 data 里；
   配合 `PointRange`（面用顶点索引，例如 i- 面写 `[[1,1,1],[1,nj,nk]]`）；`GridLocation` 缺省为 `Vertex`。
6. **周期边界不是 BCType**，而是"一个块连到自己"的 1to1 连接，可选挂 `Periodic_t` 记录旋转/平移。

---

## 1. 样例下载（不用 git）

示例文件是 cgns.org 上的静态直链，HTTP 下载即可，不必 clone 仓库：

```powershell
# Windows PowerShell，单文件
curl.exe -L -o samples\multi_hdf5.cgns.gz `
  https://cgns.org/_downloads/f4fd85803e19a6aa9f7fc58ab110f1cd/multi_hdf5.cgns.gz
```

```bash
# 批量、可复现：直接用本目录脚本（纯标准库，支持 --list / --all）
python Grid/download_cgns_samples.py            # 默认下 5blocks + multi_hdf5
python Grid/download_cgns_samples.py --list     # 打印全部清单与链接
```

结构化样例（`Grid/samples/` 里已放好前两个）：

| 文件 | 格式 | 内容 |
| --- | --- | --- |
| `5blocks.cgns.gz` | ADF | **5 块**结构网格，最经典的多块例子，21 KB |
| `multi_hdf5.cgns.gz` | HDF5 | CFL3D 的 **4 块**网格，带 1to1、BC、流场 |
| `bump.cgns.gz` / `bump_hdf5.cgns.gz` | ADF / HDF5 | 单块轴对称凸包，格心解 |
| `delta.cgns.gz` / `delta_vertex.cgns.gz` | ADF | 单块三角翼，格心解 / 顶点解（对照 `GridLocation`） |
| `oversetnasa1.cgns.gz` | ADF | 5 块，1to1 与重叠连接混用 |
| `dlr-f6.coar.cgns.gz` | ADF | 26 块翼身组合，55 MB（不建议入库） |

两条选择原则：

- **想用 h5py / Python 直接看结构**：只下 `*_hdf5.cgns*`。
- **只想验证读取器**：`5blocks.cgns` 够小，但它是 ADF，必须走 CGNS 库（或 `cgnsconvert` 转 HDF5）。

---

## 2. 文件整体长什么样（实测转储）

`multi_hdf5.cgns` 的真实层级（省略每块的重复部分，`[ ]` 内是节点的 `label` 属性）：

```text
/                                        (HDF5 根，label = "Root Node of HDF5 File")
├─ CGNSLibraryVersion [CGNSLibraryVersion_t]  data = 3.21
└─ Base               [CGNSBase_t]            data = [CellDimension=3, PhysicalDimension=3]
   ├─ Zone   1        [Zone_t]  data = [[2,123,25],[1,122,24],[0,0,0]]
   │  ├─ ZoneType        [ZoneType_t]        data = "Structured"
   │  ├─ GridCoordinates [GridCoordinates_t]
   │  │  ├─ CoordinateX/Y/Z  [DataArray_t]   shape = (25,123,2)   ← 见 §3.4 的维度反转
   │  ├─ ZoneGridConnectivity [ZoneGridConnectivity_t]
   │  │  └─ 1to1InterfaceA   1 [GridConnectivity1to1_t]  data = "Zone   1"
   │  │     ├─ Transform       ["int[IndexDimension]"]   data = [1,-2,3]
   │  │     ├─ PointRange      [IndexRange_t]            data = [[1,1,1],[2,25,1]]
   │  │     └─ PointRangeDonor [IndexRange_t]            data = [[1,123,1],[2,99,1]]
   │  ├─ ZoneBC          [ZoneBC_t]
   │  │  └─ Ilo_Seg   1  [BC_t]  data = "BCSymmetryPlane"
   │  │     └─ PointRange [IndexRange_t]  data = [[1,1,1],[1,123,25]]
   │  └─ FlowSolution    [FlowSolution_t]  → GridLocation / Rind / Density / MomentumX/Y/Z ...
   ├─ Zone   2        [Zone_t]  data = [[2,275,45],[1,274,44],[0,0,0]]
   ├─ Zone   3        [Zone_t]  data = [[2,289,29],[1,288,28],[0,0,0]]
   ├─ Zone   4        [Zone_t]  data = [[2,321,17],[1,320,16],[0,0,0]]
   ├─ ReferenceState  [ReferenceState_t]   Mach / Reynolds / Temperature / ...
   └─ ConvergenceHistory / DataClass / BaseIterativeData ...
```

要读网格，只关心四支：

```text
CGNSBase_t → Zone_t (尺寸 + GridCoordinates) + ZoneBC_t (BCType) + ZoneGridConnectivity_t (1to1)
```

`ReferenceState`、`FlowEquationSet`、`FlowSolution`、`ConvergenceHistory` 都是后处理用的，
**生成网格时可以完全不写**（SIDS 允许 `CGNSBase_t` 之下只有 zone 和坐标）。

---

## 3. Zone_t：尺寸与索引

### 3.1 三个尺寸

`Zone_t` 节点的 data 是一个 `3 × IndexDimension` 的整型数组，三行依次是：

| 行 | 名称 | 含义 | 与相邻行的关系 |
| --- | --- | --- | --- |
| 1 | `VertexSize` | 顶点数 `(ni, nj, nk)` | — |
| 2 | `CellSize` | 单元数 | 结构网格：每个方向各减 1 |
| 3 | `VertexSizeBoundary` | 有序边界顶点数 | 一般写 `(0,0,0)`（未排序） |

实测（`multi_hdf5.cgns`，注意第一维只有 2 个平面，是准二维问题）：

| Zone | VertexSize | CellSize | 块尺寸 |
| --- | --- | --- | --- |
| `Zone   1` | (2, 123, 25) | (1, 122, 24) | 2440 单元 |
| `Zone   2` | (2, 275, 45) | (1, 274, 44) | 12056 单元 |
| `Zone   3` | (2, 289, 29) | (1, 288, 28) | 8064 单元 |
| `Zone   4` | (2, 321, 17) | (1, 320, 16) | 5120 单元 |

对接 `opencfd_jax` 时：`Grid(ni, nj, nk)` 直接用 `VertexSize`，`CellSize` 只作一致性校验
（`CellSize + 1 == VertexSize` 逐向成立才算文件自洽）。

### 3.2 索引约定

- 全部 **1-based**，`PointRange` 是 `[[i,j,k]_begin, [i,j,k]_end]`，闭区间（首尾都算）。
- **i 变化最快**，k 最慢（Fortran 序）。`PointList` 若出现，按同样顺序展开。
- 面/边/格心在 `PointRange` 里用"**最小顶点索引**"表示：i- 面写 `i = 1`，i+ 面写 `i = VertexSize_i`；
  格心 `(i,j,k)` 用构成该单元的最小顶点索引（即 `i,j,k` 本身，范围 `1..CellSize`）。

### 3.3 坐标：顶点即离散点

`GridCoordinates_t` 下的 `CoordinateX/Y/Z` 是**顶点坐标**，数组大小 = `VertexSize`。当前
`opencfd_jax` 把顶点当作**离散点**使用：`Grid(ni,nj,nk)` 直接取 `VertexSize`，坐标也按
同一组离散点进入度量；**不再做全局八点平均，也不把 CGNS 转成格心量**。因此
`VertexSize` 与 `Grid.inner()` 的内区点数一致，`CellSize` 只做 `+1` 自洽校验。

LAP=4 的鬼胞由 `mesh.pad_coords` 在读取后补，不依赖 CGNS 的 `Rind`；当前 reader 明确拒绝
`Rind>0` 和 `CellCenter` 坐标。

实测 `Zone 1` 的 `CoordinateX`：`min = 0.895`、`max = 10.0`；`CoordinateY ∈ [-1, 0]`——
坐标范围可以直接用来自检块是否接上（相邻块界面顶点应完全重合）。

### 3.4 用 h5py 读时维度是反的

HDF5 里存的是 Fortran 序数组，h5py/numpy 按 C 序解释，于是

```text
CGNS VertexSize = (ni, nj, nk) = (2, 123, 25)   →   h5py shape = (25, 123, 2)
```

读取时要 `np.transpose(arr, (2,1,0))`（或直接按 reversed index 组装），否则 x/y 会串。
`PointRange`、`Transform` 这类小数组不受影响（`IndexRange_t` 存成 `(2, IndexDimension)`）。

### 3.5 rind（鬼胞）

`Rind_t` 是可选的，挂在 `GridCoordinates_t` 或 `FlowSolution_t` 下，表示数据数组在核心区之外
多存了几层。SIDS 原文（`appendix_b`）就有一个"格心解 + 每边 2 层 rind"的例子。
对本项目最直接的价值：**原码的 4 层鬼胞可以直接对应 `Rind = [4,4,4,4]`**，省掉自己补边界层。
但注意：只有坐标/流场才带 rind 数据，`Zone_t` 的 `VertexSize` 永远只算核心区。

---

## 4. 块间连接：`ZoneGridConnectivity_t`

一块一个，下挂若干连接节点；**每个界面记两次**（本例 4 块共 16 条 1to1：
`Zone1/2/3` 各有 2 条自连接，是同一条连接的两个方向（共 6 条），其余 10 条是跨块配对，`Zone4` 没有自连接）。

### 4.1 1to1：结构网格的匹配界面（`GridConnectivity1to1_t`）

```text
GridConnectivity1to1_t:
   Transform        int[IndexDimension]   (o/d)  缺省 [1,2,3]
   PointRange       IndexRange_t          (r)    本块顶点子区
   PointRangeDonor  IndexRange_t          (r)    供体块顶点子区
   node data        C1                    (r)    供体块名（ZoneDonorName）
```

语义要点：

- 两个 `PointRange` 的 **begin 点必须是同一个物理点**，end 点也必须对应；界面必须是**逻辑矩形**
  （整面或整面的矩形子区）。
- `Transform` 是"本块索引增量的镜像"：允许 `0, ±1, ±2, ±3`，绝对值不重复，缺省 `[1,2,3]`。
  数学关系（SIDS 原文）：

  ```text
  Index2 = T · (Index1 - Begin1) + Begin2
  ```

  所以 `[1,-2,3]` 表示 j 方向翻转，`[1,2,-3]` 表示 k 方向翻转（`appendix_b` 的周期面就是 `[1,2,-3]`）。
  读进 JAX 做鬼层交换时，这个矩阵就是**轴映射 + 翻转标志**，和 `多块网格GPU.md` 里
  `FaceBC(kind="interface", neighbor, axes_swap, flip)` 需要的信息一一对应。
- 节点 data 里的块名常带空格填充（`"Zone   1"`），**匹配前先 `strip()`**。

实测 `Zone 1` 的四条：

| 节点 | 供体 | Transform | PointRange | PointRangeDonor |
| --- | --- | --- | --- | --- |
| `1to1InterfaceA   1` | Zone 1（自己） | [1,-2,3] | [1,1,1]–[2,25,1] | [1,123,1]–[2,99,1] |
| `1to1InterfaceB   1` | Zone 1（自己） | [1,-2,3] | [1,123,1]–[2,99,1] | [1,1,1]–[2,25,1] |
| `1to1InterfaceA   4` | Zone 4 | [1,2,3] | [1,1,25]–[2,57,25] | [1,1,1]–[2,57,1] |
| `1to1InterfaceA   7` | Zone 2 | [1,-2,3] | [1,57,25]–[2,123,25] | [1,67,45]–[2,1,45] |

前两条是同一条自连接的两个方向：接收区和供体区**都在 `k=1` 面上**，j 子区 `1–25` 与
`123–99` 反向对接；它不是 `k=1 ↔ k=25` 的周期。对照 Zone 1 的边条，`k=1` 面上
j=25–99 是 `BCWallViscousHeatFlux`，这条自连接接的是壁面两端的 j=1–25 与 j=99–123
两段。读索引时按 `PointRange`/`PointRangeDonor`/`Transform` 逐条算，不按"周期"推。

一个面接多个邻居在本例里就有：Zone 1 的 `k=25` 面分两段，j=1–57 接 Zone 4、
j=57–123 接 Zone 2（`[1,-2,3]`，j 反向）；Zone 2 的 `k=45` 面和 Zone 4 的 `k=1` 面
各接三个邻居。相邻两段的 `PointRange` 在交界那一条线上**重叠**（j=57 同时属于两段），
因为 CGNS 的范围是闭区间。

### 4.2 一般连接：`GridConnectivity_t`

非矩形子区、非匹配界面、重叠网格用它。`GridConnectivityType_t` 只有三个合法值：

| 值 | 含义 |
| --- | --- |
| `Abutting` | 面贴合，网格点可以不一一对应（需要插值） |
| `Abutting1to1` | 面贴合且 1to1（等价于简写形式 `GridConnectivity1to1_t`） |
| `Overset` | 重叠，靠洞（`OversetHoles_t`）+ 插值 |

子节点是 `PointRange`/`PointList` + `PointListDonor`（1to1 情形）或
`CellListDonor` + `InterpolantsDonor`（需要插值的情形）。**只有结构网格 + 矩形子区**才能用
`GridConnectivity1to1_t` 简写；其余必须用完整形式并写 `GridConnectivityType`。

### 4.3 周期连接

`ZoneGridConnectivity_t` 下的 1to1 自连接，可以再挂
`GridConnectivityProperty_t → Periodic_t`，记录 `RotationCenter` / `RotationAngle` / `Translation`
（旋转周期与平移周期）。对 `channel` 这类平移周期，只写 1to1 也够用；要不要写 `Periodic_t`
取决于下游求解器是否真的需要旋转中心这类信息。

### 4.4 两条硬约束

- **同一段面只能被"BC"或"连接"覆盖，不能兼得**（SIDS 明确的设计意图）。
- **一个面接多个邻居**：N 个邻居就要 N 条 1to1，各自描述自己那个矩形子区；
  读的时候把同一块同一面的多条连接合并成"分段边界"。

---

## 5. 边界条件

### 5.1 结构

```text
ZoneBC_t
└─ BC_t (每个 patch 一个)
   ├─ BCType            C1        (r)   ← 存在 BC_t 节点自己的 data 里
   ├─ GridLocation      GridLocation_t (o/d) 缺省 Vertex
   ├─ PointRange 或 PointList        (r:o / o:r) 二者只能有一个
   ├─ InwardNormalIndex int[IndexDimension] (o)  仅结构网格，只有一个非零分量，指向区域内部
   ├─ BCDataSet_t (0..N)
   │  ├─ BCTypeSimple_t BCTypeSimple
   │  ├─ BCData_t DirichletData → DataArray_t（变量名按 SIDS 命名，如 Temperature、VelocityX）
   │  └─ BCData_t NeumannData   → DataArray_t
   ├─ BCProperty_t → WallFunction_t / Area_t
   ├─ FamilyName_t / AdditionalFamilyName_t
   └─ ReferenceState_t (o)  不写数据时，求解器从这里取参考状态
```

要点：

- `BCType` 存的是**字符串**（C1 字符数组），不是整数枚举；解析时按字符串比较，
  遇到不认识的串要"降级处理"而不是崩。
- `PointRange` 用**顶点**索引，面用整面那两个角点。实测 i- 面：
  `[[1,1,1],[1,123,25]]`（i 固定为 1，j、k 走满）。
- `BCDataSet` 可以完全不写——SIDS 允许"只给类型，数据由求解器自行取"。
- 有 `BCDataSet` 时不写数据的变量，求解器仍自由；写了 `ReferenceState` 就以它为源。

### 5.2 BCType 全表（SIDS 9.7）

`BCType_t` 分两类。**simple**：方程固定；**compound**：按当地流动状态再选 simple 子集。
（`BCType` 字段可写 compound 值，这时必须靠 `BCDataSet` 里的 `BCTypeSimple` 匹配。）

| Simple（固定方程） | 含义 / 用途 |
| --- | --- |
| `BCTypeNull` | 未定义 |
| `BCTypeUserDefined` | 自定义 |
| `BCWall` / `BCWallInviscid` | 壁面（通用）/ 无粘滑移壁 |
| `BCWallViscous` | 粘性无滑移壁（绝热，未指定时求解器自选） |
| `BCWallViscousHeatFlux` | 粘性壁 + 给定热流 |
| `BCWallViscousIsothermal` | 粘性壁 + 给定壁温（等温壁） |
| `BCInflowSubsonic` / `BCInflowSupersonic` | 亚声速 / 超声速入口 |
| `BCOutflowSubsonic` / `BCOutflowSupersonic` | 亚声速 / 超声速出口 |
| `BCSymmetryPlane` | 对称面 |
| `BCSymmetryPolar` | 极坐标/轴对称 |
| `BCAxisymmetricWedge` | 轴对称楔（单层厚度，绕轴旋转） |
| `BCTunnelInflow` / `BCTunnelOutflow` | 风洞入口 / 出口（洞壁滑动） |
| `BCExtrapolate` | 外插（零梯度类） |
| `BCDirichlet` / `BCNeumann` | 直接给 Dirichlet / Neumann |
| `BCDegenerateLine` / `BCDegeneratePoint` | 退化线 / 退化点（块角坍缩） |
| `FamilySpecified` | 边界类型由 `Family_t` 指定（配合 `FamilyBC_t`） |
| `BCGeneral` | 通用 |

| Compound（按当地流态选子集） | 含义 |
| --- | --- |
| `BCInflow` | 任意法向马赫数入口 → 自动选 `BCInflowSubsonic` / `BCInflowSupersonic` |
| `BCOutflow` | 任意法向马赫数出口 → 自动选 `BCOutflowSubsonic` / `BCOutflowSupersonic` |
| `BCFarfield` | 远场：按法向速度 + 法向马赫数，四选一（超声/亚声 × 进/出） |
| `BCTypeNull` / `BCTypeUserDefined` | 同 simple 里的同名项 |

注意事项：

- **周期边界不在这张表里**，走 §4.3 的连接。
- `BCPeriodic` 这个名字在部分旧文档/工具里出现过，但 SIDS 的 `BCType_t` 枚举**不含**它；
  文件里若真读到，按"该面是周期面、去 `ZoneGridConnectivity` 找配对"处理。
- CGNS 4.x 后续版本在 `BCWallViscous*` 系列上还追加了催化/辐射变体（如 `...IsothermalCatalytic`）。
  解析按字符串前缀 `BCWallViscous` 归类即可，别写死全集。
- 同一块同一面被拆成多个 patch 是合法的（按段写）。

### 5.3 与 `opencfd_jax` 现有 kind 的映射

下表是读取器 `src/mesh_cgns.py` 实际执行的映射，kind 的定义见 `src/boundary.py`。
标「不执行」的 `BCType` 在严格模式下报错并打印 patch 名，不静默取默认：

| CGNS `BCType` | kind | 说明 |
| --- | --- | --- |
| `BCInflow` | `INFLOW` | 只接受精确的 `BCInflow`；`BCInflowSubsonic`、`BCInflowSupersonic` 不执行 |
| `BCFarfield` | `FARFIELD` | 单面标记，来流状态由算例侧提供；平板算例的顶面用它，入口用 `BCInflow` |
| `BCOutflowSubsonic`、`BCOutflow`、`BCExtrapolate` | `OUTFLOW` | 状态鬼层缺省保持原值；零梯度延拓要另给 `boundary.outlet_ghost=copy` |
| `BCOutflowSupersonic` | `NONREFLECTING` | 项目约定：原码 `NON_REFLETION` 口径的无反射出口；SIDS 字面是「全部外推」 |
| `BCSymmetryPolar` | `PERIODIC` | 项目约定；须同轴正负整面成对 |
| `BCSymmetryPlane` | 不执行 | 严格说要单独的滑移镜像；现有代码没有 symmetry kind |
| `BCWall` | `WALL` | 绝热无滑移；`BCWallViscous` 不执行 |
| `BCWallViscousIsothermal` + `BCDataSet/DirichletData/Temperature` | 不执行 | 等温壁的数据集尚未接入 |
| `BCWallViscousHeatFlux` | 不执行 | 定热流；现有代码没有热流入口 |
| 同块 identity 整面自连接（同轴两个面互补） | `PERIODIC` | 当前只接受 `Transform=(1,2,3)`；翻转、同面自连接、部分面见 §4，留给多块阶段 |
| 1to1 跨块连接 | 多块阶段接入 | 对应 `多块网格GPU.md` 里的 `outside` 面 |
| 平板底面的等温壁（原码 `plate`） | 不经 CGNS | 读取器不产生 `PLATE`；CGNS plate 准例的壁面是 `BCWall`→`WALL` |

### 5.4 实测：CFL3D 写出来的 BC 长什么样

`multi_hdf5.cgns` 每块 5 个 patch，命名是 `Ilo_Seg`/`Ihi_Seg`/`Jlo_Seg`/`Jhi_Seg`/`Klo_Seg`：

```text
BC 'Ilo_Seg   1'  data = "BCSymmetryPlane"      PointRange = [[1,1,1],[1,123,25]]
BC 'Ihi_Seg   1'  data = "BCExtrapolate"        PointRange = [[2,1,1],[2,123,25]]
BC 'Jlo_Seg   1'  data = "BCFarfield"           PointRange = [[1,1,1],[2,1,25]]
BC 'Klo_Seg   2'  data = "BCWallViscousHeatFlux" PointRange = [[1,25,1],[2,99,1]]
```

两个提示：

1. 名字里同样有空格填充（`"Ilo_Seg   1"`），且同一块里 patch 名可以带序号后缀。
2. 这个文件还额外写了 `CFL3DType`（`Descriptor_t`，值如 `"  1004"`）和 `CFL3DAuxData`——
   这些是 CFL3D 私有的旁注，**不在 SIDS 里**，读的时候应当忽略，只用 `BCType` + `PointRange`。

---

## 6. 你自己出网格时（Tecplot / Python）的注意事项

### 6.1 Tecplot 侧

- Tecplot 360 官方的"支持格式"清单里，CGNS 在**读取（loader）**一侧
  （<https://www.tecplot.com/getting-started-tecplot-360/>）。**写 CGNS 不要指望 Tecplot 全包**：
  即便你的版本能导出 CGNS，块间 `1to1`、`BCType`、`Periodic` 这几样通常不会自动补齐，
  需要再用 cgnslib / pyCGNS 脚本补写。
- 版本配套（Tecplot 官方给 CGNS 用户的提醒）：**CGNS 3.3 + HDF5 1.8** 或 **CGNS 4 + HDF5 1.12**；
  CGNS 3.4 是"被召回"的版本，别用；HDF5 后备（非 ADF）性能更好。
- Tecplot 里分块的方式直接决定能不能写 1to1：**块界面必须是整面或整面的矩形子区**，
  且两侧顶点一一重合；一旦有悬挂点或非匹配界面，就只能用 `GridConnectivity_t`（需要插值），
  现有 JAX 代码不支持。

### 6.2 Python 侧（本次实测结论）

| 需求 | 可行做法 | 实测结果 |
| --- | --- | --- |
| **读** CGNS/HDF5 | `h5py` 直接遍历（按 `label` 属性识别节点） | 可行，40 行就能转储全树（见附录 A） |
| **读** ADF 格式 | 必须用 CGNS 库（`cgnslib`），h5py 读不了 | ADF 是老二进制格式，`5blocks.cgns` 就是 |
| **写** CGNS | 用 `cgnslib` 的 MLL（`cg_open`/`cg_zone_write`/`cg_coord_write`/`cg_bc_write`/`cg_1to1_write`） | 官方推荐路线；本机未装 cgnslib，未实测 |
| 写 CGNS（纯 Python） | `pyCGNS.MAP` | **本机 py3.13 装不上**（构建失败），不建议 |
| 写 CGNS（纯 Python，硬来） | h5py 手写 HDF5 + 按 FMM 造 `name`/`label`/`type` 属性和 `" data"` 子节点 | 能做，但**风险最高**：属性名/维度顺序/类型串任何一处偏了，别人就读不出来 |

如果非要用 h5py 手写，至少守住三条（这是从实测文件里反推出来的结构，不是猜）：

```python
import h5py, numpy as np

with h5py.File("grid.cgns", "w") as f:
    # 1) 每个 CGNS 节点 = 一个 HDF5 group，带 name/label/type 属性
    def node(parent, name, label, type_str="MT"):
        g = parent.create_group(name)
        g.attrs["name"] = np.bytes_(name)
        g.attrs["label"] = np.bytes_(label)
        g.attrs["type"] = np.bytes_(type_str)
        return g

    # 2) 节点的"数据"是一个名字带前导空格的子 dataset（前导空格 = SLL 私有节点）
    def put_data(g, arr, type_str):
        g.attrs["type"] = np.bytes_(type_str)
        g.create_dataset(" data", data=arr)

    node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4")
    put_data(f["CGNSLibraryVersion"], np.array([3.21], dtype=np.float32), "R4")

    base = node(f, "Base", "CGNSBase_t")
    put_data(base, np.array([3, 3], dtype=np.int32), "I4")      # CellDimension, PhysicalDimension

    z = node(base, "Zone1", "Zone_t")
    # 3) 尺寸三元组；坐标按 Fortran 序，即 numpy 里 shape=(nk,nj,ni)
    put_data(z, np.array([[ni, nj, nk], [ni - 1, nj - 1, nk - 1], [0, 0, 0]], dtype=np.int32), "I4")
```

> 上面只是**层级骨架**，缺 `flags` 属性、`ZoneType`、`GridCoordinates` 等细节；
> 写成完整文件前请用 `cgnsviewer`（cgnstools）或任意 CGNS 库回读一遍。
> 结论：**能读就用 h5py 读，要写就尽量走 cgnslib。**

### 6.3 通用坑

- zone 名字不要含 `/`、不要以 `.` 开头（CGNS Name 规则）；写完记得 `strip()` 再比对。
- 坐标用 double（`R8`）；整型用 `I4`（SIDS 对整数只要 I4 精度就够）。
- `CellDimension = 2` 的准二维网格（本例 i 方向只有 2 个平面）依然按 3 维存坐标，
  `PhysicalDimension = 3`；不要试图写少一维的坐标数组。
- 多块文件不要写多个 `CGNSBase_t`：跨 base 的连接虽然在标准里允许，但绝大多数工具不处理。

---

## 7. Step 4 读取口径（已实现部分）

1. **只做 `ZoneType = Structured`**：`Unstructured` 直接报"不支持"并列出块名；二维 zone
   （任一 `VertexSize <= 1`）也拒绝。
2. **逐块读**：`VertexSize → Grid(ni,nj,nk)`，`CellSize` 检查 `+1` 关系；多 zone 文件必须
   用 `--cgns-zone` 指定，单 zone 可省略。
3. **坐标**：读顶点 `CoordinateX/Y/Z`（注意 h5py 维度反转），**顶点即离散点**，不做八点
   平均、不转格心；坐标按 `(nk,nj,ni)` 进入 `pad_coords` 和 `compute_metrics_from_coords`。
4. **鬼胞**：LAP=4 由 `mesh.pad_coords` 自己补；当前 reader 拒绝 `Rind>0`，不把 CGNS rind
   当现成鬼胞。
5. **BC**：遍历 `ZoneBC_t` 下所有 `BC_t`，按 `BCType` 字符串 + `PointRange` 建面表；
   当前执行子集是 `BCWall`→`WALL`、同轴成对的 `BCSymmetryPolar`→`PERIODIC`、
   `BCExtrapolate/BCOutflow/BCOutflowSubsonic`→`OUTFLOW`、
   `BCOutflowSupersonic`→`NONREFLECTING`（项目约定）、`BCFarfield`→`FARFIELD`、
   `BCInflow`→`INFLOW`；其余（`BCInflowSubsonic`、`BCInflowSupersonic`、`BCWallViscous*`、
   `BCSymmetryPlane` 等）在严格模式下报错并打印 patch 名，不退化。真实样例
   `multi_hdf5.cgns` 的 Zone 1 在严格模式下首先被 `Ilo_Seg 1 / BCSymmetryPlane` 拒绝。
   `BCFarfield` 是**单面**：
   只有被标记的那个面是远场，同轴对侧可以是 `BCWall`/`BCOutflow` 等不同 kind；
   本步不读 `BCDataSet`，缺省来流由算例侧提供为物理 +x 方向均匀来流。
6. **1to1**：当前只接受同块、`Transform=(1,2,3)` 的整面 identity 自连接，两范围互补时
   标 `PERIODIC`；跨块、翻转/轴交换和 `GridConnectivity_t` 留给多块阶段。
7. **自检**（写进测试 fixture）：
   - `CellSize + 1 == VertexSize`；
   - 1to1 双向存在且 `Transform`/范围互补；
   - 相邻块界面顶点坐标逐点相等（可用 `np.allclose`，容差按文件精度定）；
   - 每块外表面的 BC 面积之和 = 六个面面积之和（块内部面被 1to1 占掉，两者不重叠）。

---

## 附录 A：把 CGNS/HDF5 全树转储出来（本次实际用的脚本）

```python
import h5py

def label(o):
    v = o.attrs.get("label")
    return v.decode() if isinstance(v, bytes) else str(v)

def name(o):
    v = o.attrs.get("name")
    return v.decode() if isinstance(v, bytes) else str(v)

def data(o):
    return o[" data"][()] if " data" in o else None

def walk(o, depth=0):
    print("  " * depth + "[%s] %r" % (label(o), name(o)), data(o))
    if isinstance(o, h5py.Dataset):
        return
    for k, v in o.items():
        if k.startswith(" "):      # 私有节点（如 " data"）已经单独取过
            continue
        walk(v, depth + 1)

with h5py.File("Grid/samples/multi_hdf5.cgns", "r") as f:
    for k, v in f.items():
        if not k.startswith(" "):
            walk(v)
```

用法与产物：本目录 `samples/` 下已放 `5blocks.cgns`（ADF，5 块，仅 CGNS 库能读）与
`multi_hdf5.cgns`（HDF5，4 块，本文所有实测数字都来自它）。
