# `.tec` → 二进制 `.plt` 转换说明

**日期**：2026-11-15
**输入**：`outputest/case_iiA_Istep00054000.tec`（22,585,335 字节，ASCII）
**输出**：`outputest/case_iiA_Istep00054000.plt`（7,988,248 字节，Tecplot 旧版二进制 v112）
**转换脚本**：`tec_to_plt.py`（纯 Python，**不调用 preplot / tecplot.exe**，直接按格式规范写字节流）
**SHA256**：`4A67D54FC46905E302B0761EF598EC7A73CECA614A34C520A5F446B838E547B7`

体积降到原来的 **35.4%**（22.6 MB → 8.0 MB），变量、点数、AUXDATA、SOLUTIONTIME 全部保留。

---

## 1. 源数据与目标结构

| 项目 | 源 `.tec`（ASCII） | 写出的 `.plt`（v112） |
|---|---|---|
| TITLE | `OpenCFD-SCU Case iiA  Istep=54000` | 同 |
| 变量（10） | x y z rho u v w T p Ma | 同 |
| zone | 1 个，`Istep 54000`，ORDERED | 同 |
| 网格 | I=64, J=65, K=48（199,680 点） | 同 |
| 打包方式 | `DATAPACKING=BLOCK` | BLOCK（变量优先，i 变化最快） |
| 变量位置 | 未写 `VARLOCATION` → 全 NODAL | `SpecifyVarLocation=0`（全 NODAL） |
| 精度 | 未写 `DT` → 默认 SINGLE | 每变量 `1=Float`（float32） |
| SOLUTIONTIME | 54000.0 | FLOAT64 54000.0 |
| StrandID | 无 | -1（static） |
| AUXDATA | 7 条 | 7 条，原样写入 |

AUXDATA：`Re_tau=190.7`、`M_r=0.0955`、`gamma=1.40`、`Pr=0.70`、`Tw=1.0`、`bodyforce=rho*1.0`、`note=Case iiA, Lusher & Coleman 2023`

---

## 2. 实际写入的字节流

```
HEADER SECTION（832 B）
  "#!TDV112"                       8 B   魔数 + 版本
  INT32  1                              字节序（小端写 1）
  INT32  0                              FileType = FULL
  TITLE                                 逐字符 INT32(ASCII) + 0
  INT32  10                             变量数
  变量名 ×10                            各为 逐字符 INT32 + 0
  ── 每个 zone ──
  FLOAT32 299.0                         zone marker
  "Istep 54000"                         逐字符 INT32 + 0
  INT32 -1                              ParentZone（已废弃）
  INT32 -1                              StrandID（-1=static）
  FLOAT64 54000.0                       SolutionTime
  INT32 -1                              默认颜色（已废弃）
  INT32  0                              ZoneType = ORDERED
  INT32  0                              SpecifyVarLocation=0（全 NODAL）
  INT32  0                              raw 1-to-1 face neighbors
  INT32  0                              额外 user-defined 邻接条数
  INT32  64 / 65 / 48                   IMax, JMax, KMax
  7 × (INT32 1 + 名字 + INT32 0 + 值)   zone AUXDATA
  INT32  0                              AUX 结束
  FLOAT32 357.0                         EOHMARKER，头段结束

DATA SECTION 前缀（216 B）
  FLOAT32 299.0                         zone marker
  10 × INT32 1                          每个变量 = Float
  INT32  0                              无 passive 变量
  INT32  0                              无变量共享
  INT32 -1                              不共享 connectivity
  10 × (FLOAT64 min, FLOAT64 max)       每个变量的 min/max

数据本体（7,987,200 B）
  10 × 199,680 × float32                变量优先、i 最快（与 ASCII BLOCK 一致）
```

---

## 3. 文档写法与实际实现的三处差异（容易踩坑的地方）

1. **字符串不写长度计数字段。** 文档里 `INT32*N` 的 N 是按字符数算出来的数组长度，文件里只存“逐字符 INT32 + 0（null 结尾）”，并没有先把 N 写进去。
   （文档 2.1.3 Note 1 也写了 “All character strings are null terminated”。）
2. **ORDERED zone 头写到 `KMax` 就直接接 AUXDATA 终止符**，**不写** `NumElements` 和 `ICellDim/JCellDim/KCellDim`——那几项只在 FE zone 出现。文档 2.1.1 的条目是把两套分支混排在一起，容易被读成“所有 zone 都要写”。
3. **`EOHMARKER = 357.0` 是必需的**，头段没有它 Tecplot 无法进入数据段。

另外两点按规范固定值：`ParentZone = -1`、`默认颜色 = -1`。

---

## 4. 校验：三层独立核对，全部通过

1. **脚本自带读回**（`--info` / 转换后自动读回）：结构自检通过，尾随字节 0。
2. **第三方独立解析器**（PyPI `tecio`，纯 Python + `construct`，与本脚本无任何共用代码）读回结果：
   - title / 变量名 / zone 名 / IJK / aux / solution_time / strand 全部与源文件一致；
   - 读出的 (I,J,K) 数组按 **i 最快**（`ravel(order='F')`）与源 `.tec` **逐位相同**，10 个变量 `max|Δ| = 0`；
   - 读出的 min/max 表与数据实际极值一致。
3. **字节级核算**：文件头 `#!TDV112` + `INT32 1` + `FileType 0`；总大小 = 832 B 头段 + 216 B 数据段前缀 + 10×199,680×4 B 数据 = 7,988,248 B，**没有任何多余或缺失字节**。

各变量范围（与源文件一致）：
`x 0…12.37`、`y -1…1`、`z 0…4.1015`、`rho 0.878…1.324`、`u -4.6e-5…21.757`、`v -3.473…3.180`、`w -5.954…5.054`、`T 0.9885…1.27`、`p 77.73…109.4`、`Ma 0…1.985`

精度损失：`--precision float32` 下与源值最大相对偏差 **5.9e-8**（源 ASCII 本身只有 5 位有效数字）；`float64` 下与源值完全一致（差 0）。

> 说明：本次**没有在 Tecplot 本体里打开过这个文件**（按要求不使用 Tecplot 的可执行程序）。上面的结论来自两个互相独立的解析实现与逐字节核算。你第一次用 Tecplot 打开时，确认 zone 尺寸 64×65×48、10 个变量、AUXDATA 齐全即可。

---

## 5. 用法

```powershell
# 单个文件（默认输出同名 .plt，精度按 DT 自动判定：缺省 SINGLE → float32）
python tec_to_plt.py outputest\case_iiA_Istep00054000.tec

# 指定精度 / 指定输出路径
python tec_to_plt.py outputest\case_iiA_Istep00054000.tec --precision float64 -o D:\tmp\a.plt

# 只读回并打印 .plt 结构（不开 Tecplot 也能看里面有什么）
python tec_to_plt.py --info outputest\case_iiA_Istep00054000.plt

# 批量转换一个目录下的所有 .tec
Get-ChildItem outputest\*.tec | ForEach-Object { python tec_to_plt.py $_.FullName }
```

参数：`--precision {auto,float32,float64}`。`auto` 时若所有 `DT` 都声明为 `DOUBLE` 则用 float64，否则 float32。

---

## 6. 限制

- 只支持 **ORDERED zone + NODAL 变量**（BLOCK 或 POINT 打包都可以）。
- `VARLOCATION` 里出现 `CELLCENTERED` 会直接报错退出（Tecplot 对单元中心变量有 ghost 值补零规则，本脚本未实现，宁可不写也不写错）。
- `GEOMETRY` / `TEXT` / `CUSTOMLABELS` / `VARAUXDATA` / FE zone（`ZONETYPE=FETRIANGLE` 等）未实现：前几类会打印警告后跳过，FE 会报错。
- v112 格式用 **INT32 计数**，单个文件建议控制在 2 GB 以内（约 5 亿个 float32 值）。真到那个量级应改用 v191/v120 或分 zone 输出。
- `--info` 的读回器与写出器同源，仅用于结构自检；要做严格第三方校验可用 `tecio`（`pip install tecio`，Python 3.12+ 需给 `distutils.spawn.find_executable` 打个 shim）。
