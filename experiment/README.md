# 第 0 步：固定当前基准（回归护栏）

对应 `11-16 我建议先对齐的.md` 第六节「第 0 步：固定当前基准」。这一步**不改任何功能**，
只把当前程序的行为钉死：Sod / TG / Poiseuille / plate 各跑一小段，
连同逐文件 sha256 一起存档。之后做第 1~9 步（MeshData、boundary.py、input.json、
Gridgen 读取、case 插件化、多块串行、多块粘性、I/O、多 GPU）时，
每一步收尾都要复跑本清单并拿到 PASS，任何一位变化都能立刻定位到哪个用例。
`channel` 已从 CLI 算例列表移除，原 `.in`/场/OCFD 网格入口不再参与基准。

## 数值口径（2026-11-17 指定）

| 项 | 取值 | 说明 |
|---|---|---|
| 无粘格式 | `--scheme WENO7` | 原码 `SCHEME_INVIS=303`（7 阶 WENO-Z，LAP=4、P/M 模板 ±3）；**不用 OMP6 系列** |
| 粘性格式 | `--vis-order CD8` | 对应原码 `SCHEME_VIS=CD8` |
| 精度 | FP64（缺省） | 逐位比对的前提 |
| 激波滤波 | 关闭 | 只冻结"格式本身"，滤波属于整步后处理，另案处理 |
| dt | 显式固定 `--dt` | CFL 自适应已从入口删除；固定 dt 让逐位复现已确定 |
| 体积力做功 | `--forcing-work off`（Poiseuille 对拍时） | 原码对应内核只做 `du[1] += F·rho`；JAX 缺省 `on` 会多出 `rho*u*G` |

## 用例清单

| id | 内容 | 覆盖点 | 规模 | 耗时 |
|---|---|---|---|---|
| `sod_1step` | Sod 激波管 1 步，dt=1e-3 | 通量分裂 + WENO7 重构 + 出流/周期 | 128×4×4 | 2.7 s |
| `tg_10step` | Taylor-Green，Re=1000，10 步 | 周期双轴 + 粘性 CD8 + 2D 涡 | 64×64×4 | 4.0 s |
| `poiseuille_5step` | 层流 Poiseuille 5 步 | WALL 镜像鬼胞 + 体积力 + 粘性 | 32×64×4 | 4.2 s |
| `plate_1step` | 平板边界层 1 步 | INFLOW/PLATE 非对称边界 + 零状态鬼层 + `dif_bound` 边界带 | 64×64×4 | 5.3 s |

合计 4 条。`channel_1step` / `channel_small_100step` 已随旧 `.in` 与 OCFD 网格入口删除；
它们的历史基准文件仍保留在 `baseline/` 供旧提交对照，但不再由当前入口生成。

## 用法

在 `D:\Desktop\ocfd2jax` 下执行（基准与候选都写到 `experiment/` 里）：

```powershell
# 1) 已生成：重写基准（只在明确决定"基准该变"时才跑）
python opencfd_jax\experiment\run_step0.py --phase baseline

# 2) 每次重构后：跑候选
python opencfd_jax\experiment\run_step0.py --phase candidate

# 3) 逐点比对（缺省要求 max|diff| = 0，即逐位一致）
python opencfd_jax\experiment\check_step0.py

# 只跑某几条 / 看清单
python opencfd_jax\experiment\run_step0.py --phase candidate --only sod_1step tg_10step
python opencfd_jax\experiment\run_step0.py --list
```

`run_step0.py` **不复制 `main.py` 的参数解析**，一律子进程调用
`python -m opencfd_jax.src.main <argv>`，所以基准与真实入口永远同源；`main.py` 一旦
把参数名或调用顺序改坏，这一步会直接以非零退出码暴露。

## 判定

- `check_step0.py` 逐 npz、逐守恒量分量比较 `max|diff|` / `rms` / 位置，并核对
  `_meta` 里的 `t` / `step` / `dt`（dt 序列变了 = CFL 口径变了，同样算 FAIL）。
- 文件与字段集合、形状、dtype 必须一致；缺失或损坏 `_meta`、非有限值以及基准
  NPZ 与各用例 `case.json` 记录的 SHA-256 不一致均判 FAIL。零容差逐字节比较数组，
  因此正负零也会区分。测试可运行 `python -m unittest discover -s opencfd_jax/experiment -p test_check_step0.py`。
- 缺省 `--tol 0`：**逐位一致**。本机（Python 3.13.18 / numpy 2.5.4 / jax 0.12.0.1，
  CPU）历史 7 条用例、80 个场实测 `max|diff|=0.000e+00`；当前 CLI 清单已缩为 4 条，
  旧 `channel_*` 基准仍留在 `baseline/` 作为历史对照，不再由当前入口重跑。
  `baseline/SELFCHECK.txt` 与 `baseline/SELFCHECK.json`（该次跑完即删除 `candidate/`，
  consumer 需要重跑）。
- 只有当改动**必然**改变运算顺序（例如算子融合、规约顺序）时，才允许
  `--tol 1e-12`，并必须在改动说明里写清"哪个文件、为什么、影响哪些场"。
- 换机器或换 JAX 版本后，基准不再保证逐位可比，需要重新生成 baseline 并在
  `STEP0.md` 里保留旧版本的 git 提交号作为对照。

2026-11-17 补充检查器后，使用原基准记录的 Windows Python 和 CPU 环境重跑
7 条 JAX 用例，80 个场全部通过零容差逐字节对照；报告保存在
`source_record_20261117/check_report.json`。基准 NPZ 未重写。
`source_record_20261117/` 是事后采集的当前源码、Git 差异和输入哈希；原基准生成时
仅记录了未提交文件数量，不能据此证明当时源码与事后副本逐字相同。

## 与原版 OCFD 的单步对拍（WENO7 + CD8）

第 0 步除了"自己跟自己对"，还做了一次"跟原码对"：原码用 WSL 里的
`ocfd_p1_1/build/opencfd-dump.out`（main_dump 版）在同一套 96×97×64 网格上跑 1 步并 dump，
JAX 用 `build_case + build_rhs + rk3_step_stages` 从原码 step0 出发走同一步。

```powershell
# 原码（WSL，约 40 s）
wsl -d Ubuntu2404 -- bash -lc "cd /mnt/d/Desktop/ocfd2jax/opencfd_jax/experiment/ocfd/channel_weno7 && mkdir -p dump && /mnt/d/Desktop/ocfd2jax/ocfd_p1_1/build/opencfd-dump.out"
# JAX 侧解析 + 比对（约 15 s）
python opencfd_jax\experiment\compare_ocfd_jax.py
```

结果（**2026-11-17 修正 WENO7 epsilon 之后**，详见 `ocfd/channel_weno7/对拍_单步.md`）：

| 区域 | du1 最坏 | 一步末态最坏 |
|---|---:|---:|
| 内区 j=12..84 | 1.8e-12（rho，相对 1e-13） | 4.4e-16（rho） |
| 贴壁带 j<12 / j>84 | 4.1e-12（rho，相对 7e-14） | 2.0e-15（rho） |
| 全場最坏 | 2.5e-09（du2 的 rhoE，相对 7e-14） | 4.0e-13（rhoE，相对 7e-16） |

结论：**WENO7 无粘 + CD8 粘性 + 度量链 + 壁面带 + 壁面 BC 已全部对齐到 1e-9 以内**
（绝大多数场到 1e-11~1e-15）。找 bug 的关键一步是 UP7 逐层排查：原码与 JAX 都换成
`UP7`（壁面宏与 WENO7 共用，只改内区重构）后残差是 1e-12~1e-9，说明分裂/差分/度量/BC
本来就对；残差全在 WENO7 重构里，根因是 `_weno7_face_P` 把 `3.456e-4` 当 eps 又平方了一次
（有效加性项 1.19e-7，原码是 3.456e-4，差 2900 倍）。在通量很小（u→0）的贴壁带，
`smoothness ~ eps` 恰好由 epsilon 主导，所以这个口径错误表现为"只在壁面附近、随距壁衰减"
的假象，看起来像边界格式问题。修正后 7 条基准已重新生成，自检仍是 7/7 逐位 PASS。

## 当前基准的已知事实（别当成 bug 重跑）

1. **7 条用例的 16 个 npz 文件、共 80 个守恒量场里没有 NaN/Inf**，rho 范围分别是：
   sod `0.125~1`、tg `0.9996~1.0006`、poiseuille `1~1`、plate `0.8625~1.0048`、
   channel_1step `0.9757~1.3700`、channel_small `1~1`。
2. `channel_small_100step` 的 rho 恒为 1 是**一致性解**而不是冻结错误：封闭槽道 +
   沿流向均匀体积力 + 等温壁 ⇒ 压力近似均匀、ρT=1，所以密度不变；
   它在这里的作用是"多步稳定性 + 实现一致性"，不是物理验证。
3. **Poiseuille 长跑仍会发散**（`tests/test_wall_poiseuille.py::test_poiseuille_steady`
   目前是 skip：主格式模板在壁面邻格产生质量通量泄漏）。第 0 步只冻结前 5 步的有限演化。
   等 `viscous.py` / `schemes.py` 的边界带单侧壁面公式落地、这条测试转绿时，
   **必须显式重新生成 baseline** 并在提交说明里写明"基准因修复 x 而更新"，而不是
   悄悄放宽阈值。
4. **不重跑** `channel_case_iiA/main_128x129x96` 的 20000 步原版长跑（那是原版程序
   的外部参考，第 0 步不碰）；JAX 与原版的逐点对拍属于 `src/compare.py` 的职责，
   与本目录的"自身回归"分开。

## 第 1 步验收：单块网格与边界描述

`test_step1_mesh.py` 冻结五个内置算例的网格/步长/方向，并验收 `src/mesh_data.py`
的六侧语义、执行来源、只读性、非法输入报错，以及"配置先改、描述后建"的入口时序。
真网格用例直接读 `channel_case_iiA/quick` 的 in / `OCFD3d-Mesh.dat`；该文件不存在时
该条自动 skip。

```powershell
cd D:\Desktop\ocfd2jax\opencfd_jax
python -m pytest experiment/test_step1_mesh.py -q
```

仓库现有的 `tests/` 已迁移为从 `opencfd_jax.src.*` 导入，不再依赖
`opencfd_jax = opencfd_jax.src` 的旧包别名；各测试文件自身插入仓库根路径，
可直接从仓库根目录运行。真网格用例依赖 `channel_case_iiA/quick` 的本地数据，
换机缺文件时只跳过该条。

## 第 2 步验收：统一边界与执行配置

`test_step2_boundary.py` 先冻结旧行为，再验收 `src/boundary.py` 的统一实现和
`BoundaryExecution`：

- 计算 i/j/k 周期与出流、四层鬼胞、角点覆盖次序；
- 槽道三点壁面外推、不同壁温、单侧绝热；平板入口/出口/下壁/顶部及平板两点外推；
- 状态壁面镜像、派生温度、Ev 的计算 j 符号与平板顶部复制；
- 滤波启用时的周期/出流前后值，以及现有不支持的滤波边界报错；
- 正式路径初态、三个 RHS 和末态的回调次数；`build_execution` 的冲突拒绝；
- case 显式 `special_boundary` 对象 + `bc_apply` 的注入、六侧校验和阶段衔接。

```powershell
cd D:\Desktop\ocfd2jax\opencfd_jax
python -m pytest experiment/test_step2_boundary.py -q
```

验收结果（2026-11-17，本机）：`26 passed`。Step 0 七例候选重跑 + `check_step0.py`
总判定 PASS，七例全部 `max|diff|=0.000e+00`；全量套件用别名启动器运行，结果为
`143 passed, 2 skipped, 1 xfailed`。

滤波只支持周期/出流；通用 `WALL` 只支持 y 方向；槽道三点外推和平板两点外推仍留在
`channel.py` / `plate.py`。添加通用边界和 case 特殊边界的位置、参数与约束见
`src/README.md` 第 7.2 节；边界带公式的核对材料见 `docs/boundary_scheme_notes.md`。
`special_boundary` 是显式导入入口，不做动态注册，也不允许 I/O 进入 JIT。

## 第 3 步验收：统一运行配置（input.json）

`test_step3_config.py` 覆盖：

- 默认值 / JSON / 显式命令行的合并优先级，显式判定不靠与默认值比较；
- 未知字段、算例不适用字段（如 Sod 的 `p_grad`）、CGNS 缺 `mesh.file` / 非 `poiseuille` 报错；
- `--no-*` 显式关闭、`time.cfl` / `--adaptive-dt` / `--cfl` 已拒绝、`--fp32/--fp64`；
- 输出前缀取网格文件名，以及在临时目录里 CLI 与 JSON 两条路径的 NPZ 逐位一致；
- 输出文件名为 `<前缀>-istep-00000001.npz` 形式，步号至少八位；`_meta.N` 记录实际 `grid.ni`，
  `_meta.Re/Ama` 记录实际物性；
- 四条 Step 0 用例的 JSON 参数与 CLI 参数逐项相等。

```powershell
cd D:\Desktop\ocfd2jax
python -m pytest opencfd_jax/experiment/test_step3_config.py -q
```

七例 JSON 配置仍放在 `experiment/cases/*.json`，但当前 `step0_cases.py` 只保留 4 条 CLI 用例；
`channel_1step.json` / `channel_small_100step.json` / `sod_cfl_5step.json` 是历史遗留配置，不再被清单引用。
网格维数仍由命令行提供（JSON 明确不承载维数/边界开关）。用 JSON 重跑候选并比对：

```powershell
python opencfd_jax\experiment\run_step0.py --phase candidate_json --arg-source json
python opencfd_jax\experiment\check_step0.py --candidate opencfd_jax\experiment\candidate_json
```

验收结果（2026-11-18，本机）：`test_step3_config.py` 全部通过；`experiment/` 全量 100 passed。
旧 `channel_1step` 的输出名 `OCFD3d-Mesh-istep-00000001.npz` 仍由 `check_step0.py` 的
旧基准映射读取，但当前入口不再生成它。

## 第 3.5 步验收：计算方向命名

`test_step3_5_axes.py` 覆盖：

- `i/j/k`、`0/1/2` 与六侧 `i±/j±/k±` 的规范映射；旧轴名/面名被拒绝；
- `Grid` 只接受 `ni/nj/nk`；旧 `nx/ny/nz` 关键字、属性和只读视图已移除；
- `FaceBoundary` 只接受规范面名，`BlockMesh.face("x-")` 直接报错；
- 只保留 `fill_halos_i/j/k`、`fill_wall_j`、`fill_ev_wall_j`/`fill_ev_plate_j`；
  旧填充函数名和 `spatial_array_axis` 不再导出；
- `infile.grid_dims_ijk` 是唯一的网格尺寸读取入口；
- CLI `--ni/--nj/--nk` 入口；旧 `--N/--ny/--nz` 已移除。

```powershell
cd D:\Desktop\ocfd2jax\opencfd_jax
python -m pytest experiment/test_step3_5_axes.py -q
```

验收结果（2026-11-17，本机）：`test_step3_5_axes.py` 12 passed；
`test_step1_mesh.py + test_step2_boundary.py + test_step3_5_axes.py` 共 57 passed；
仓库 `tests/` 全量 99 passed、2 skipped、1 xfailed。`tests/` 目录按 `.gitignore`
不入库，导入迁移只保留在本地工作区。

新入口尺寸验证：

```powershell
cd D:\Desktop\ocfd2jax
python -m opencfd_jax.src.main --case sod --ni 16 --nj 4 --nk 4 --scheme UP7 `
  --dt 5e-4 --end-step 3 --end-time 0.01 --kstep-show 1 --kstep-save 1 --out-dir tmp_new
```

`Ak/Ai/As`、`Axx/Ayy/Azz` 仍是原码/物理度量键，不随计算轴改名；`dif_axis_map`
（平板 `(0,2,1)`）和 `DIF_BOUND` 六标志顺序未改。输出八位步号属于第 3 步，
不并入本步结论。

## 第 4 步验收：单块 CGNS/HDF5 结构网格

`test_step4_cgns_reader.py`、`test_step4_cgns_adapter.py`、`test_step4_cgns_run.py` 覆盖：

- fixture `Grid/fixtures/poiseuille_box.cgns` 的 `VertexSize=(16,16,8)`、`CellSize=(15,15,7)`、
  `Vertex` 坐标形状 `(nk,nj,ni)` 和四组同块 identity 1to1；
- `read_cgns_zone(validate=True)` 的执行子集校验，zone 名空格归一化，以及真实样例
  `Grid/samples/multi_hdf5.cgns` Zone 1 的 `validate=False` 原始读取；
- `BCFarfield` 在严格模式被接受并映射为通用 `FARFIELD`；真实样例
  `multi_hdf5.cgns` 仍在 `Klo_Seg 2` 的非整面 patch 处被拒绝；
- `to_block_mesh()` 的 `(i,j,k)` 顺序、`(periodic,wall,periodic)`、六侧 `halo`；
  `build_cgns_block()` 的 `Ajac` 形状与均匀间距解析值；
- `poiseuille` 用 CGNS 和内置网格各跑一步，五个守恒分量在 `rtol=1e-12, atol=1e-14` 内一致。

`test_step4n_farfield.py` 另覆盖通用 `FARFIELD` 的边界内点/鬼层写入、`Ev` 零梯度、
`build_execution` 的来流状态校验，以及 CGNS `BCFarfield` 到通用远场的适配。

```powershell
cd D:\Desktop\ocfd2jax
python -m pytest opencfd_jax/experiment/test_step4_cgns_reader.py `
  opencfd_jax/experiment/test_step4_cgns_adapter.py `
  opencfd_jax/experiment/test_step4_cgns_run.py `
  opencfd_jax/experiment/test_step4n_farfield.py -q
```

验收结果（2026-11-18，本机）：Step 4 + 4.n 四个文件 22 passed；`experiment/` 全量
110 passed。CGNS 一步与内置一步的逐分量最大绝对差在 `2.27e-13` 以内；两边 `_meta`
同为 `N=16`、`dt=0.001`、`step=1`。

当前未覆盖：非均匀 CGNS 的 `poiseuille` 解析初值、`BCInflow*`/粘性壁面的执行、
CGNS `BCFarfield` 的 `BCDataSet` 来流参数、多块连接与翻转/轴交换、二维 zone、
`Rind`/`PointList`、以及仓库级 `tests/` 全量套件。
已知回归差异：`plate_1step` 的旧基准仍按“特殊回调后清零非周期鬼层”生成，新
“特殊回调最后覆盖”口径下 `check_step0 --only plate_1step` 为 FAIL（最坏
`cons_4 max=2.61e1`）；`poiseuille_5step`、`sod_1step`、`tg_10step` 仍逐位 PASS。
仓库级 `tests/` 仍有 8 个 collection error，均为旧测试导入 `solver.compute_dt`。
`Grid/samples/` 被 `.gitignore` 排除，真实样例测试在缺文件时自动 skip。

## 目录结构

```text
experiment/
  step0_cases.py          用例清单（argv 与 JSON 配置入口）
  run_step0.py            批量跑 + sha256/统计 + manifest.json + STEP0.md
  check_step0.py          逐点比对 + check_report.json + PASS/FAIL 退出码
  test_step1_mesh.py      第 1 步单块网格/边界描述验收测试
  test_step2_boundary.py  第 2 步统一边界/执行配置验收测试
  test_step3_config.py    第 3 步 input.json 合并/前缀/等价验收测试
  test_step3_5_axes.py    第 3.5 步计算方向命名/兼容迁移验收测试
  test_step4_cgns_reader.py  第 4 步 CGNS reader 验收测试
  test_step4_cgns_adapter.py 第 4 步 CGNS -> BlockMesh 适配/度量测试
  test_step4_cgns_run.py     第 4 步 CGNS 与内置 Poiseuille 一步 NPZ 对齐
  test_step4n_farfield.py    第 4.n 步通用远场/单面 BCFarfield/特殊覆盖顺序
  compare_ocfd_jax.py     原码 dump vs JAX 的单步对拍（--dir/--scheme 可换口径）
  cases/channel_small.in  小网格槽道 in 文件（历史遗留，当前入口不再读取）
  cases/*.json            七条历史 Step 0 JSON；当前清单只引用 4 条
  baseline/               基准数据（7 条用例、28.6 MB、可复现）
    manifest.json         全局清单：环境、git、逐用例命令与哈希
    STEP0.md              自动生成的摘要表
    SELFCHECK.txt/json    生成基准当天"自比对 PASS"的证据
  ocfd/channel_weno7/     原码 WENO7 口径的运行目录 + 对拍_单步.md
  ocfd/channel_up7/       原码 UP7 口径（逐层排查用，壁面带与 WENO7 共用）
```

两个 `ocfd/channel_*` 目录里只保留 `dump/parsed.npz`（各约 100 MB），
原始文本 dump（各 724 MB）已删除，删掉 npz 会触发重新解析。

`candidate/` 与 `candidate_json/` 是临时目录，已被 `.gitignore` 排除；`baseline/` 需要随仓库保管
（去掉它，第 1~9 步就没有护栏了）。如果不想把 23 MB 的二进制入库，最低要求是
**`manifest.json` 必须入库**（它含逐文件 sha256，换机后可用它判断"数据是否被动过"），
数据本体可以留在本地磁盘或用 LFS。
