---
title: 补尾巴（第九步之前） - Plan
type: feat
date: 2026-11-26
topic: tails-before-step9
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
---

# 补尾巴（第九步之前） - Plan

## Goal Capsule

- 目标：把第九步之前留下的尾巴里这次说好要补的补掉；让程序在 WSL 的 GPU 上跑通，并有验收的办法。
- 依据：2026-11-26 对路线图、各步方案、README、CGNS 说明和 `experiment/step49_待闭合事项_2026-11-19.md` 的清点，每一条都到代码和测试里核了现状；jisen 对每一条的答复（见 Key Decisions 和 Sources）。
- 约束：CPU 上单块路径的数值逐位不变；第六步到第八步的结果不变；原码树不动。
- 未决：B、D 两组各有几个要 grilling 的问题，轮到时再定，不挡 A、E、C1 开工。

---

## Product Contract

### Summary

这份方案管五组活，分三份详细程度。A、E、C1 三组写到能开工；B、D 两组只写范围和待定的问题，轮到时另出方案。末尾附数值验证的建议，不在这次实施的范围内。

### Problem Frame

第九步、第十步先不做。清点下来，不算这两步，有 19 条尾巴（第 13 条里其实是两件事，总表里拆成 13a 和 13b），另加两条新发现的：WSL 里还没有 JAX 的 GPU 环境；`src/` 下没有一份标准的 `input.json`。总表里一共 22 行。它们的来路不一样：有的是当时定了“以后做”，有的是验收没收口，有的是功能没做，有的是记录和代码上的小尾巴。逐条核下来，有的已经做完了只是记录没更新，有的做不了，要等外部条件。

### 清点总表

| 编号 | 尾巴 | jisen 的决定 | 归属 |
|---|---|---|---|
| 1 | 滚动可视文件 | 这次补 | 新计划（D） |
| 2 | 报错要写清是哪里的错 | 这次补 | A |
| 3 | 对称面边界 | 拿掉，有需要再做（jisen 2026-11-27：以后的网格用不到） | — |
| 4 | WENO7-SYMBO 自身对拍 | 不做，要等 SYMBO 口径的原码参考数据 | — |
| 5 | 特殊边界脚本配多块 | 并进“库接口”：先做单块，多块以后再管（jisen 2026-11-27） | 新计划 |
| 6 | 超声速层流平板的定量验收（壁面摩擦系数比相似解低约一半，来源没定位） | 数值验证，单独排 | — |
| 7 | 真实的多块网格没验过 | 网格已到（`Grid/Multi/Multiblock.cgns`），现在读不了，见下 | 新计划 |
| 8 | 续算的两个边角（连续多次续算的累积；带出口的 CGNS Poiseuille） | 数值验证，单独排 | — |
| 9 | 和原码多进程分块运行的对拍 | 不做，没有这样的工况 | — |
| 10 | `BCWallViscous*` 系列不认成壁面；网格文件里的壁温不读 | 拿掉，有需要再做（jisen 2026-11-27：以后的网格用不到） | — |
| 11 | 单块路径只支持整面边条 | 拿掉，有需要再做（jisen 2026-11-27：以后的网格用不到） | — |
| 12 | 特殊边界脚本不能带参数，不能放在仓库外 | 维持现状 | — |
| 13a | 多块粘性只放行 CD8 | 这次补 | C1 |
| 13b | 多块“不能只支持 plate 这一种构型” | 不加体积力，不要“认所有算例”那一组；去掉写死 plate 的地方 | 新计划 |
| 14 | 接口旁四层鬼胞里的粘性通量白算 | 记录，性能优化时做 | — |
| 15 | RK 接口里没人用的 `bc_apply` | 这次补 | A |
| 16 | 存盘那一步没有错误分层 | 这次补，和第 2 条一起 | A |
| 17 | 续算的两处不一致 | 这次补 | A |
| 18 | 路线图里几处进度没更新；旧待办没标哪些已关闭 | 这次补 | A |
| 19 | `opencfd_jax/tests/` 下的测试不入库 | 维持现状 | — |
| 20 | WSL 里没有 JAX 的 GPU 环境（新发现） | 环境已建好；验收的办法这次补 | E |
| 21 | `src/` 下没有标准的 `input.json`（新发现） | 这次补 | A |

### Key Decisions

- **数值验证单独排。** 以后验证只用平板这一类边界（入口、出口、壁面、远场、周期）；sod、tg、poiseuille 不再当验证工况。
- **以后的计算都不加体积力。** 多块“不能只支持 plate 这一种构型”，指不该把多块写死成 plate 这个测试算例的样子，不是要支持别的算例。
- **顺序：A、E、C1、B、D。** A 小而确定，先做；E 是后面所有验收的运行环境；C1 小；B 是边条，量中等；D 的文件要在 Tecplot 本体里验证。
- **GPU 上的验收用 CPU 存下的数据加阈值。** GPU 和 CPU 的浮点运算顺序不同，不逐位相同。
- **只检验 FP64。** 状态是 FP32 的调用方式在 GPU 上不和 CPU 存下的值比（jisen 2026-11-27 的决定）。
- **性能优化以后再做。** 记下一条：多块算粘性通量时，接口旁四层鬼胞里算出来的值紧接着被邻块内区的值换掉，等于白算，占各块这部分计算量的 5% 到 40%。要省它，得让每块只算一个去掉这四层的小盒子。
- **特殊边界脚本维持现状。** JSON 里只填脚本名，不带参数；脚本只能放在 `src/special_boundaries/` 下。
- **`opencfd_jax/tests/` 维持不入库。**
- **以后都用 CGNS 网格，不用内置的算例**（jisen 2026-11-27）。以后会用多块：同向切块、点一一对上，周期方向可能被切开。
- **把程序做成库**（jisen 2026-11-27）：正式的入口 `load` 和 `run`，能 pip 安装，`run` 里直接传用户写的边界函数，把网格信息交给它。第一版不带时间（接口留好位置）、只管单块 CGNS、跑完返回最终流场（原始变量）也能一步一步取、检查点照常写。写进新计划，排在收尾之后。
- **B 组的三条边条类型拿掉，有需要再做**（jisen 2026-11-27）：对称面、`BCWallViscous*`、单块一个面分几段，以后的网格都用不到。
- **壁温从 JSON 读。** `BCWallViscous*` 系列认成壁面；网格文件里写了壁温时提示一句没读。

### Requirements

**A 组：小尾巴**

- RA1. 路线图里第 4、4.n、4.9 步的进度行改成现状；第 4.n 步的旧方案和 `experiment/step49_待闭合事项_2026-11-19.md` 里过时的状态段也改，历史记录留着，在旁边写现状。旧待办每一节现在算关闭还是转去了哪里，按 KTD11 的对照表写。
- RA2. 启动时的两类错误各只往 stderr 写一行，退出码 2：`配置错误：<出处>：<原因>`、`初始化错误：<出处>：<原因>`。不再跟着打印命令行的用法说明。命令行本身解析不了（不认识的选项、缺值）时照旧由 argparse 报。时间推进开始以前出的错，都归这两类，没有例外。
- RA3. 出处只有一种写法。由某个参数引起的，写 `<参数名>（命令行 <选项>，JSON <键>）`，例 `kstep_save（命令行 --kstep-save，JSON output.kstep_save）`；不是参数引起的，写阶段：新加的阶段名只有读 JSON 配置、建输出目录两个，其余沿用消息里原有的开头（初始化失败、多块模式不放行、多块网格没有通过校验、续算的检查点）。路径、文件名写在原因里，并写它是从命令行来的还是从 JSON 来的。原因的说法不改，只在前面加出处，原来消息里的关键词保留。
- RA4. 运行期错误的消息统一成 `运行期错误（<阶段>）：case=… step=… t=… dt=… grid=… 原始异常=…`。阶段四个：装配 RHS、RHS 预热、时间推进、存盘。存盘是新加的：写不出检查点（目录不可写、磁盘满）时报它，带输出路径。多块原来只有“时间推进”一个名目，补上另外三个：进循环以前先用起点状态空跑一步、结果丢弃，算“RHS 预热”，这样编译出的错和第一步跑出来的错分得开。traceback 照旧保留。
- RA5. `solver.rk3_step_stages` 和 `rk3_step` 不再收 `bc_apply`，函数体里相应的四个分支删掉。现在没有任何调用方传它。
- RA6. JSON 里 `time.restart` 写空字符串，等于没给，与命令行的 `--restart ""` 一致。
- RA7. 命令行给了 `--restart` 时，JSON 里的 `time.restart` 即使已经失效也不挡：`time.restart` 在读 JSON 时只解析相对路径，不核对存在；合并以后核对最终生效的那一个。其他路径字段（`mesh.file`、`case.inflow_profile`）不变。四种情形：只有一边给了，按那一边；两边都给了，命令行的为准，JSON 的那个不核对；命令行给了空字符串，等于没给，即使 JSON 里有路径也从头算；最终生效的那个不是普通的文件（不存在，或者是个目录），报配置错误。
- RA8. `src/input.json` 是一份能直接跑的配置：单块的 plate，网格用仓库里的 `Grid/testcgnstest.cgns`。键就是 A5 列出的那些，一个不多一个不少。README 的选项表说明每个键；多块比它多什么、改什么，指向 README 的多块例子。
- RA9. 单块和多块的每一个异常出口都归到某一类、某个阶段，见 KTD10 的矩阵。矩阵里的每一种情形各有一条测试。

**E 组：GPU 环境**

- RE1. 入库的测试在 Linux 上也能跑。两处写法只适用于 Windows 的改掉：一处用反斜杠当路径分隔符；一处靠“嵌套 3000 层的 JSON 会让解码器出错”造一个读不出来的旁记，Linux 上这个深度解码器扛得住。
- RE2. 和 CPU 存下的数据比的那些断言，在 GPU 上改成按第六步验收的阈值比；参考值还是存下来的 CPU 数据。在同一台设备上算出来的两份结果，照旧逐位比，不管这台设备是 CPU 还是 GPU。CPU 上一切照旧。
- RE3. 比较的判别力不丢，两头都保：同一台设备上改坏一个位，比较必须失败；跨设备时改坏一个明显超出阈值的量，比较必须失败。
- RE4. README 写明在 WSL 的 GPU 上怎么建环境、怎么跑、跑测试要关显存预分配，以及 GPU 和 CPU 不逐位相同。

**C1 组：CD6 放行**

- RC1. 多块运行粘性打开时，粘性导数的阶数取 CD6 或 CD8 都放行。
- RC2. 第七步的两层验收对 CD6 再跑一遍：五个验收网格各跑两个整步，对照等价单块；五个合成小网格上的线性场。单块的 CD6 参考用正式入口的装配。每个阶段分三块区域比：接口邻域、物理边界带、其余。切口面不套物理边界的单侧模板，有测试盯着。
- RC3. 第七步故意注错的四条测试（少换一次粘性通量、取错供体层、切口面套上单侧公式、RK 组合用了准备过的状态）按阶数参数化，CD6 下也必须抓得住。
- RC4. 正式入口的多块 CD6 运行，读回的末态与打开粘性的单块 CD6 参考在阈值内一致；检查点的 `_meta.vis_order` 记 `CD6`。
- RC5. 原来断言“CD6 被拒”的测试改掉，共五条（第六步配置测试里两条，第七步配置测试里三条）。

**B、D 两组**：只列范围和待定的问题，见 Implementation Units。

**文档**

- RD1. 每一组做完，路线图和 README 跟着改。

### Key Flows

**启动出错**（A 组以后）

1. 命令行解析不了：argparse 报错，带用法说明，退出码 2（不变）。
2. 读 JSON、合并配置、核对输入文件、建输出目录：哪一步出错，往 stderr 写一行 `配置错误：<出处>：<原因>`，退出码 2。
3. 初始化、多块的放行校验和网格校验、读续算的检查点、多块的装配：出错写一行 `初始化错误：<出处>：<原因>`，退出码 2。单块的“装配 RHS”归在运行期错误里，不变。
4. 进入计算以后出错：`RuntimeExecutionError`，traceback 的最后一行是 `运行期错误（<阶段>）：…`，阶段是装配 RHS、RHS 预热、时间推进、存盘之一。

**在 GPU 上验证**（E 组以后）

1. CPU（Windows 的 `.venv`）跑全量：逐位的护栏照旧。
2. GPU（WSL 的 `/root/venv-jax`，关掉显存预分配）跑全量：和 CPU 存下的数据按阈值比。

### Acceptance Examples

- AE1. 给一个不存在的 `--restart`：stderr 只有一行 `配置错误：restart（命令行 --restart，JSON time.restart）：命令行给的文件不存在或不可读：<路径>`，退出码 2，没有用法说明。
- AE2. 存盘时输出目录不可写：traceback 的最后一行是 `运行期错误（存盘）：case=sod step=1 … 输出路径=… 原始异常=PermissionError: …`。
- AE3. JSON 里 `"restart": ""`：从头算，启动信息里没有 `restart:` 那一行。命令行给 `--restart ""`、JSON 里有路径，也一样从头算。
- AE4. JSON 里的 `time.restart` 指向一个已经挪走的文件，命令行给了存在的新文件：按命令行的续算，启动时不报错。
- AE5. `python -m opencfd_jax.src.main --config opencfd_jax/src/input.json` 能跑完，产物在工作区根目录的 `output/` 下。
- AE6. 在 WSL 的 GPU 上跑全量测试：没有失败。
- AE7. 多块 x2 粘性打开、`vis_order` 给 CD6：走通正式入口，读回的末态与单块的 CD6 参考在阈值内一致，`_meta.vis_order` 是 `CD6`。
- AE8. 配置文件在，但没有读的权限：stderr 只有一行 `配置错误：读 JSON 配置：<路径>：<原因>`，退出码 2。
- AE9. `--viscous` 和 `--no-viscous` 同时给：stderr 只有一行 `配置错误：viscous（命令行 --viscous 和 --no-viscous，JSON numerics.viscous）：不能同时给出`。
- AE10. 输出目录建不出来（它的上一级是一个文件）：stderr 只有一行 `配置错误：建输出目录：<路径>（命令行 --out-dir，JSON output.dir）：<原因>`。
- AE11. 多块的三种启动错误各一行：`--vis-order` 给 CD8 以外的阶数名时是配置错误（本来就有）；粘性打开而 `vis_order` 给 CD6 时放行（C1 组以后）；进循环以前空跑那一步出错时是 `运行期错误（RHS 预热）`。
- AE12. `exact` 的判别力：同一台设备上改坏一个位，报错；跨设备差 1e-6，报错；跨设备差 1e-16，通过。

### Scope Boundaries

不在这次：

- 数值验证（第 6、8 条）；真实多块网格的验证（第 7 条）。
- WENO7-SYMBO 自身对拍（第 4 条）；和原码多进程分块运行的对拍（第 9 条）。
- 性能优化（第 14 条）。GPU 上每次编译比 CPU 慢得多，全量测试单独跑要 23 分钟上下，编译缓存也归到这一条。
- 特殊边界脚本的参数和存放位置（第 12 条）；`tests/` 入库（第 19 条）。
- 一个面上混着不同类型的边条：单块和多块现在都不支持，等有了具体的网格再说。

### Dependencies / Assumptions

- 第八步已提交并推送（`4c25367`）。
- WSL 的 GPU 环境已建好（见 Sources），A、E、C1 的验收在 CPU 和 GPU 上各跑一遍全量。

**需要 jisen 知道的默认处理**

1. 已分类的启动错误不再打印用法说明。只有命令行解析本身出错时才打印。
2. 运行期错误保留 traceback，退出码不变。
3. `src/input.json` 的输出目录是 `../../output`（相对 JSON 所在目录，落在工作区根目录），和命令行缺省的 `output/` 同一个地方。
4. 只有 `time.restart` 的空字符串等于没给；别的路径字段写空字符串照旧报错。
5. `tests/` 下有六个只在本机的测试文件写死了 `D:\Desktop\ocfd2jax\channel_case_iiA` 的路径：三个直接挂，三个靠“缺数据就跳过”在 Linux 上被跳过（所以 GPU 上跳过的是 8 条，CPU 上是 2 条）。改成相对工作区根目录拼出来，它们才能在 Linux 上跑。这些文件不入库，改动只在本机。改了以后跑了却挂的，如实报告。
6. 跨设备的比较里（GPU 上算的和 CPU 存下的数据比），“改坏一个值必须失败”改坏的是一个远大于阈值的量；同一台设备上的比较照旧翻最低位。
7. 多块进循环以前先用起点状态空跑一步、结果丢弃，算“RHS 预热”。多出来的是一步的计算量。
8. 两个互相冲突的开关、输入文件不存在、建输出目录失败、配置文件读不了：从 argparse 的报错（带用法说明）或裸抛，改成一行配置错误。

### Outstanding Questions

**Resolve Before Planning**：没有。

**Deferred to Implementation**

- A2 里“逐条过启动错误的消息”改到什么程度：原则是只加出处、不改原因；实施时如果发现某条原因本身说不清，再单独提。
- E2 里 `exact` 的确切名字和参数。

**B、D 两组的待定问题**（轮到时 grilling）

- B3 对称面：只支持 j 方向，还是三个方向都支持？粘性通量在对称面的鬼层和边界带怎么取？CGNS 的 `BCSymmetryPlane` 和 `BCSymmetryGeneral` 各怎么映射？
- B4 特殊边界配多块：脚本按绝对坐标施加，不用知道块号（第六步的方案定的）。待定的是回调怎么拿到块的坐标、先放行哪一类脚本。
- **jisen 的多块网格**（`Grid/Multi/Multiblock.cgns`，Pointwise 2023.2.2 导出，2026-11-27 只读看过，没有入库）：沿 x 切成四段，第二段又沿 y 切成两块，共五块；`blk-1` 32×50×50，`blk-2-split-1` 91×46×50，`blk-2-split-2` 91×5×50，`blk-3` 104×50×50，`blk-4` 76×50×50；连接全是同向的 1to1；周期在 k 方向、没有被切开（各族 `BCSymmetryPolar`）；壁面 `BCWall` 在整个下壁；出口 `BCOutflow` 在 x 最大处；入口（x 最小）和顶面用的是同一族 `BCFarfield`。程序的元数据读取五块都过，`build_topology` 拒绝：`blk-2-split-2 的 j+ 面是物理面，j 向只有 5 个顶点，至少 7 个`。另外没有旁记文件，参考尺度要按等价单块 300×50×50 手填（2401、14651、14651）。
- **库接口（jisen 2026-11-27 新定，要做的）。** 把程序做成库：正式的入口 `load` 和 `run`、能 pip 安装、`run` 里直接传用户写的边界函数、把网格信息交给这个函数。这条代替了原来“放开脚本目录”的做法。实测现状：在自己的 Python 里 import、读 `input.json`（连带网格）、建初场和边界、不带边界函数算一步并拿到结果数组，这四样现在就通；挂用户写的边界函数被两处拒绝：`init.attach_specificboundary` 按算例名 `plate` 拒绝；`boundary.validate_special_sides` 要求没有回调面的方向上不能有入口，CGNS plate 的 i 方向正好有入口（老的按方向的模型）。另外：边界函数现在拿不到时间（它被编译进右端项）；程序只能在那个目录下 import，没有打包配置。轮到它时另写方案。还没定的细节逐条问 jisen：第一版带不带时间；第一版只管单块还是也管多块；结果怎么交还；排在 B 组之前还是之后。
- 单块的 CGNS plate 挂不了脚本这一条，随库接口一起解决（上面一条）。
- B5 去掉写死 plate 的地方：算例名 `plate` 保留，只去掉写死的假设（Astra 第 31 轮的意见，实现方采纳）。
- D 滚动可视文件：文件名、变量表（要不要压力和马赫数）、精度；多块是不是一个文件多个 zone。

### Sources / Research

**清点**（实现方读过，每条都在代码或测试里核过）

- 路线图第 4、4.n、4.9 步的进度行停在 2026-11-18、19；第 4.n 步的补记里写着 4.9 的对拍在 2026-11-22 收口到机器精度。
- `solver.py:280`、`:311`：`rk3_step_stages`、`rk3_step` 仍收 `bc_apply`；入库的调用方没有一处传它；源码归档 `experiment/source_record_20261117/` 里有，那是历史记录。本机不入库的 `tests/test_plate_boundaries.py` 里有一条玩具测试用它，写方案时漏查了，见实施记录第 17 条。
- `main.py`：存盘的调用在三个 `try` 之外；启动错误走 `argparse` 的 `error`，会打印用法说明。实测给一个不存在的 `--restart`，stderr 一共 17 行，前 16 行是用法。
- `mesh_cgns.py`：`_BC_WALL` 只有 `BCWall`；单块的 `_validate_executable` 要求每个轴两侧都有整面边条；多块的 `_classify` 要求同一个面的边条是同一种类型，允许一个面由几段拼成。
- `boundary.py`：没有对称面这种类型。原码 `OCFD_boundary_Liftbody3d.cu` 的 `do_symmetry_kernel_m`、`_p`（第 84 到 114 行）：只在 j 方向，密度、切向速度、温度取镜像点的值，法向速度取相反数。
- `multiblock.py`：`build_blocks` 从头到尾按 plate 写死；`_check_release` 只放行名为 `plate` 的算例，粘性打开时只放行 CD8。
- 第七步方案的决定是“放行 plate 加 CD8，CD6 只加一条小网格测试”；那条测试在 `experiment/test_step7_linear.py`，与等价单块差 1.4e-16。

**WSL 的 GPU 环境**（2026-11-26 建好，实现方操作）

- Ubuntu 24.04.5，WSL 2；显卡 RTX 5070 Ti Laptop，显存 12 GB，Windows 驱动 596.49；WSL 里有 CUDA 12.8。
- 装了 `python3.12-venv`；虚拟环境 `/root/venv-jax`（Python 3.12.3）：`jax[cuda13]==0.12.1`（带 `jax-cuda13-plugin`、`jax-cuda13-pjrt`）、`numpy==2.5.5`、`h5py==3.16.0`、`scipy==1.18.1`、`pytest==9.2.0`，与 Windows 的 `.venv` 同版本。
- `jax.devices()` 给 `[CudaDevice(id=0)]`。求解器在 GPU 上跑通：sod、plate 各出了检查点。
- 跑测试要设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`：入库的测试里有 16 处另起进程跑命令行入口，父进程把显存占满以后子进程起不来。

**实测**

- plate 的 CGNS 网格 100×40×40，粘性 CD8，float64，各跑 5 步：GPU 第一次编译约 15 秒，编译完以后每步约 0.05 秒；CPU 第一次编译约 3 秒，每步约 0.29 秒。
- 一步的结果，GPU 与 CPU 比：密度的最大相对差 4.4e-16（160000 个值里 1890 个不同），温度 5.6e-16，速度分量差在 1e-16 上下。第六步验收的阈值是相对 1e-11、绝对 1e-13。
- E 组改之前，GPU 上的全量：32 failed、768 passed、8 skipped、1 xfailed，23 分钟。32 条分两类：27 条是和 CPU 存下的数据逐位比（冻结测试 2 条，第七步的基线抽样 19 条，第七步的交换 6 条）；5 条是测试的写法只适用于 Windows（本机 `tests/` 里 3 条写死了 `D:\…` 的路径，`test_step3_config.py` 一条用反斜杠当分隔符，`test_step6_run.py` 一条靠嵌套 3000 层的 JSON）。

**决定**

jisen 在 2026-11-26 的答复：数值验证单独排；以后的计算都不加体积力，多块不能只支持 plate 这一种构型；壁温从 JSON 读，`src/` 下要有标准的 `input.json`；不要“认所有算例”那一组；GPU 上的验收用 CPU 数据也可以；以后的程序都在 WSL 的 JAX GPU 上跑；验证方案按实现方建议的三段来。

jisen 在 2026-11-27 又定：只检验 FP64（FP32 的调用方式在 GPU 上不和存下的值比）；以后的审查用 6.2-sol，不再用 Astra。

jisen 在 2026-11-27 逐条定的：坏检查点清单加的两条（`_meta`、块清单不是对象）保留；`--plate-dif-bound` 暂时保留现在的做法；以后都用 CGNS 网格，不用内置的算例；把程序做成库（见 Outstanding Questions 的“库接口”）。

**Astra 第 31 轮**（2026-11-26，只读审查，没有跑测试，没有改仓库里的文件）

结论：A、E、C1 三组不能按第一版直接开工，A2、A4、E2 和 C1 的验收要先补清；C1 的实现路径本身明确。实现方把每条意见到方案和源码里核了一遍，都成立，必须改的 5 条和建议改的 3 条全部采纳，已经改进这份方案。

| 意见 | 级别 | 改在哪里 |
|---|---|---|
| 总表把 13 编了两次，条数不符；旧待办里多条“待验证”没对应到总表 | 必须 | 总表改成 13a、13b，共 22 行；KTD11 加旧待办的对照表，A1 照它写 |
| A2 漏了读 JSON 的 `OSError`、`resolve_cli` 里冲突的开关、`os.makedirs`、多块的入口和装配、多块的存盘；多块没有“装配 RHS”“RHS 预热”的名目 | 必须 | RA9 和 KTD10 的异常出口矩阵；A2 逐情形补验收；KTD12 多块进循环前空跑一步 |
| A4 的 `--restart ""` 没有测试；AE1 写死两个入口名，RA7 又要求说明最终值从哪边来，出处格式不一致 | 必须 | RA3 出处只有一种写法；A4 补命令行空字符串、两边都给、是目录这几条 |
| E2 的 `exact` 按全局 backend 选比较方式，同一台 GPU 上的两份也只做阈值比较；转成 NumPy 以后看不出原来的设备 | 必须 | KTD6 改成按两份输入各自所在的设备决定，设备在转 NumPy 以前看；KTD7、AE12 加判别力测试 |
| C1 没要求 CD6 下分别检查接口切口、物理边界带和其余 | 必须 | RC2 分三块区域比；RC3 注错测试按阶数参数化；切口面不套单侧模板写明 |
| A5 的键集说得不确定 | 建议 | A5 列出确切内容，测试核对整个字典相等 |
| 路线图和旧 Step 4.n 方案里过时的状态段要一起改，保留历史 | 建议 | RA1、A1 |
| `src/errors.py` 的说明同步 | 建议 | KTD1 |

B、D 两组和数值验证建议的意见，分别记在 Outstanding Questions、Implementation Units 和文末的附录里。

---

## Planning Contract

### Key Technical Decisions

- **KTD1. 启动错误从一个口出。** `main.main` 里不再调 `argparse` 的 `error`，改成往 stderr 写一行再 `sys.exit(2)`。`_validate_input_paths`、`resolve_cli` 里互相冲突的开关、建输出目录，都从“调 `parser.error`”或裸抛改成抛 `ConfigError`；`load_config` 里读文件的 `OSError` 也转成 `ConfigError`；`run` 里那句没分类的 `ValueError`（dt 不合格）改成 `ConfigError`。argparse 自己的解析错误不受影响。`src/errors.py` 的说明同步：`RuntimeExecutionError` 覆盖装配、预热、时间推进、存盘四个阶段。
- **KTD2. 出处由一个函数生成，不手写。** `runtime_config` 里加一个函数：给参数名，返回 `<参数名>（命令行 <选项>，JSON <键>）`，从 `_SCHEMA`（JSON 键）和解析器的选项表（命令行选项）反查。`main.py`、`multiblock.py` 里启动时抛的错，消息前加它。`runtime_config.py` 里 JSON 校验的消息已经带 JSON 键名，补上命令行选项名。
- **KTD3. 存盘的错误分层只包写文件那一步。** 把存盘的调用包进 `try`，抛 `RuntimeExecutionError`，消息里多一项输出路径。不改存盘本身。
- **KTD4. `time.restart` 用一种新的字段类型。** 空字符串给 `None`；其余按 `path` 解析相对路径，但不核对存在。存在性合并以后在 `_validate_input_paths` 里核对，原因里写它是从命令行来的还是从 JSON 来的。
- **KTD5. `src/input.json` 不含 `numerics.flux_scale`。** 单块的右端项自己算参考尺度（`flux_scales_for`），不读这一项；多块才要它。放进去会让人以为单块也用它。
- **KTD6. 比较只有一个入口，按输入所在的设备决定。** `experiment/step6_support.py` 加 `exact(got, want, label)`：两份输入在同一台设备上算出来，就 `same_bits` 逐位比，不管这台设备是 CPU 还是 GPU；一份是存下来的数据（读出来是 NumPy，按 CPU 算），另一份在 GPU 上算的，用 `compare` 按阈值比。设备在转成 NumPy 以前看。要改的断言都换成它，不在各处写判断后端的分支。
- **KTD7. 判别力两头都保。** 同一台设备上改坏一个位，`exact` 必须报错。跨设备时翻转最低位在阈值比较下看不出来，那是阈值比较的本性，所以改坏一个远大于阈值的量：冻结测试在 GPU 上把 `b0` 里一个值乘以 `(1 + 1e-6)`，比较必须失败。
- **KTD8. 旁记“读不出来”的测试用不合 UTF-8 的字节。** 任何平台上 `read_text(encoding="utf-8")` 都读不出来，不依赖递归上限。
- **KTD9. CD6 的验收复用第七步的装配。** `compile_blocks` 的阶数是参数，装配不读阶数。验收和故意注错的测试按阶数参数化，单块的参考多建一份 CD6 的（`init.prepare(viscous_args(网格, "CD6"))`）。比的时候分接口邻域、物理边界带、其余三块，沿用第六步验收的分法。
- **KTD10. 异常出口矩阵。** 单块和多块的每个出口归到一类、一个阶段。凡是现在还会打印用法说明或裸抛的，改成矩阵里的样子。

  | 出口 | 单块在哪 | 多块在哪 | 类别：阶段 |
  |---|---|---|---|
  | 命令行解析不了 | `argparse` | 同 | argparse 自己报，带用法说明，退出码 2（不变） |
  | 两个互相冲突的开关 | `resolve_cli` | 同 | 配置错误：出处是这两个开关 |
  | 配置文件不存在、读不了、不是合法 JSON、顶层不是对象 | `load_config` | 同 | 配置错误：读 JSON 配置：路径加原因 |
  | JSON 的字段不认识、类型不对、超出范围 | `load_config` | 同 | 配置错误：出处是字段 |
  | 合并后的校验（适用范围、网格尺寸、plate 的参数、`--plate-dif-bound` 的写法、参考尺度只给多块） | `resolve_args` | `resolve_args`、`enter` | 配置错误 |
  | 输入文件不存在（网格、续算的检查点、入口剖面），dt 不合格 | `_validate_input_paths` | 同 | 配置错误：出处是参数，原因里有路径和来源 |
  | 建输出目录失败 | `main` | 同 | 配置错误：建输出目录：路径加出处加原因 |
  | 多块的放行校验、zone 的元数据 | — | `enter` | 初始化错误：多块模式不放行：… |
  | 多块的拓扑校验、读坐标 | — | `enter` | 初始化错误：多块网格没有通过校验：… |
  | 建算例、网格描述、边界执行器 | `init.prepare` | — | 初始化错误：初始化失败：case=… 原始异常=…（原有消息不变）。`init.prepare` 出的任何异常都算，特殊边界脚本自己抛的也算，消息里点名脚本 |
  | 读续算的检查点 | `_restart` | `_restart` | 初始化错误或配置错误（第八步定的分法）。检查点的 `_meta`、块清单是合法的 JSON 但不是对象，也算检查点不可用 |
  | 装配 | `build_rhs` | `build_blocks` | 单块：运行期错误（装配 RHS）；多块：初始化错误（多块网格装配失败）。两边的归类不变 |
  | RHS 预热 | `rhs(cons)` | 空跑一步、结果丢弃（KTD12） | 运行期错误（RHS 预热） |
  | 时间推进 | 循环 | 循环 | 运行期错误（时间推进） |
  | 存盘 | 存盘 | 存盘 | 运行期错误（存盘） |

- **KTD11. 旧待办的对照表。** `experiment/step49_待闭合事项_2026-11-19.md` 每一节的现状，A1 照这张表写进文件头。只写核过的；没有逐项复核的如实写。

  | 那份文件的节 | 现状（2026-11-26） |
  |---|---|
  | 参考来源必须先修正；干净 HEAD 的后续执行 | 已关闭。干净 HEAD 的一次末态与工作区参考逐位相同（文件里已记）；逐步对拍 2026-11-22 收口到机器精度（路线图第 4.n 步的补记） |
  | 只读登记的那张表 | 不再维护 |
  | 通用出口影响：CGNS plate | 已关闭。4.9 的对拍到机器精度；出口的行为有 `experiment/test_step49_plate.py` 的 `test_cgns_plate_outlet_uses_ocfd_ghosts`、`test_generic_outflow_preserves_state_ghosts_and_clears_ev` |
  | 通用出口影响：内置 plate（交界、重复 prepare） | 没有专门的测试。鬼层的分工有 `experiment/test_step4p_halo_contract.py`。原待办的这一项没有逐项复核 |
  | 通用出口影响：Sod 等用 OUTFLOW 的调用方 | 已有行为测试：上面那条 `test_generic_outflow_…`，`experiment/test_outlet_ghost.py` |
  | 通用出口影响：通用 CGNS、自定义 case（初始化、重启、非零鬼层） | 重启有第八步的测试（以 `cons0` 为底，鬼胞留初场的）。其余没有逐项复核 |
  | 通用出口影响：滤波 | 已关闭。激波选择性滤波 2026-11-23 从正式入口删除 |
  | 周期与出口的交界、RK 下不同的初始鬼层 | 滤波的那一半已不适用；其余没有逐项复核 |
  | 后续验证顺序 1：固定来源 | 已关闭，同第一行 |
  | 2：用合成数据独立检验转换 | 已关闭。`experiment/test_step49_plate.py` 里转换器的测试（ni、nj、nk 取 9、8、7，期望值直接按 (k, j, i) 写） |
  | 3：周期逐层、对侧端点 | 状态与 Ev 有 `test_periodic_state_and_ev_preserve_inner_endpoints`（三个轴）；各块的坐标拼回来与源网格一致有 `experiment/test_step6_meshes.py` |
  | 4：逐变量、逐区域比较 | 已关闭。4.9 收口；第六步、第七步的验收按块、接口邻域、其余分开比 |
  | 5：失败路径 | 缺 Family、跨轴的周期、重复的和不完整的 patch 有 `experiment/test_step49_plate.py`；连接的失败路径有 `experiment/test_step6_topology.py`。分段边条没有做，归 B 组 |
  | `_collect_sides()` 把同块的 identity 1to1 当周期 | 第六步定了：按块内周期环绕处理，不和 Polar 混配。有测试 |

- **KTD12. 多块进循环以前空跑一步。** `multiblock.run` 在循环以前用起点状态调一次 `solve_step`，结果丢掉，包在“RHS 预热”的错误分层里。它是纯函数，不改变状态；多出来的是一步的计算量（GPU 上约 0.05 秒），换来的是所有编译都在预热这一步完成。

### 触及清单

| 组 | 文件 | 改动 |
|---|---|---|
| A | `11-16 我建议先对齐的.md`、`experiment/step49_待闭合事项_2026-11-19.md`、`docs/plans/2026-11-18-002-feat-step4n-boundary-execution-plan.md` | 进度和现状 |
| A | `src/main.py`、`src/multiblock.py`、`src/runtime_config.py`、`src/solver.py` | 错误出处、存盘分层、`time.restart`、删 `bc_apply` |
| A | `src/input.json`、`src/README.md` | 标准配置和说明 |
| A | `experiment/test_step3_config.py`、`experiment/test_step6_config.py`、`experiment/test_step8_restart.py`、`experiment/test_step8_config.py`，新建 `experiment/test_tails_a.py` | 测试 |
| E | `experiment/step6_support.py`、`experiment/test_step49_frozen.py`、`experiment/test_step7_baseline.py`、`experiment/test_step7_exchange.py`、`experiment/test_step3_config.py`、`experiment/test_step6_run.py` | `exact`、可移植的写法 |
| E | 本机 `tests/` 下六个文件（不入库） | 路径 |
| C1 | `src/multiblock.py`、`src/README.md`、第七步的方案 | 放行 |
| C1 | `experiment/test_step6_config.py`、`experiment/test_step7_config.py`、`experiment/test_step7_acceptance.py`、`experiment/test_step7_linear.py`、`experiment/step7_support.py` | 按阶数参数化 |

### Risks & Dependencies

| 风险 | 应对 |
|---|---|
| 改启动错误的格式，带走一批靠消息里的字找错误的测试 | 只加出处、不改原因，原来的关键词都留着；全量测试两遍（CPU、GPU） |
| GPU 上的阈值比较让某个真错误溜过去 | 阈值用第六步验收的那两个数，实测 GPU 与 CPU 的差在 1e-16 量级，隔四五个数量级；KTD7 保住判别力 |
| 删 `bc_apply` 以后有没查到的调用方 | 搜过入库的文件和本机的 `tests/`，只有源码归档和一条玩具测试用它，那条测试已经改掉；全量测试两遍 |
| CD6 的验收暴露出新问题（接口旁的差分模板、边界带） | 验收分两层，和 CD8 一样；出现问题如实报告，不放行 |
| GPU 上的全量单独跑要 23 分钟上下，每组都跑两遍很费时间 | 改动只涉及测试的比较方式时，GPU 上只跑相关的文件；提交前跑一遍全量 |
| 本机 `tests/` 的改动不入库，换机器要重做 | 如实告诉 jisen；他定了维持不入库 |

---

## Implementation Units

### A 组：小尾巴

**A1. 记录**

- Files: 路线图、`experiment/step49_待闭合事项_2026-11-19.md`、第 4.n 步的方案。
- Approach:
  - 路线图第 4 步：单块 CGNS 读取已落地；plate 的边条和正确性验收由第 4.9 步在 2026-11-22 收口到机器精度；拒绝清单里被指出有漏洞的两项（`Periodic_t` 的层级漏检、两种周期写法混配）已修，有测试；`BCWallViscous*`、`BCSymmetryPlane`、子区边条归补尾巴 B 组，翻转和轴交换的连接归第十步。
  - 第 4.n 步：`specificboundary` 的入口已实现；R7（删 RK 的 `bc_apply`）和 R19 里存盘那一步，当时没做完，补尾巴 A 组做。
  - 第 4.9 步：改成已收口，写明日期和口径（三阶段总 RHS 与一步末态到机器精度）。
  - 待闭合事项的文件头加“现状（2026-11-26）”，照 KTD11 的对照表逐节写：已关闭的、转到哪里的、没有逐项复核的。
  - 第 4.n 步的旧方案（`docs/plans/2026-11-18-002-feat-step4n-boundary-execution-plan.md`）文件头加一段现状：`specificboundary` 的入口已实现；R7 和 R19 里存盘那一步由补尾巴 A 组补上；验收以 4.9 的收口为准。历史记录留着。
- Verification: 文档里每句话都能在代码、测试或路线图里找到对应。

**A2. 报错写清出处，连同存盘的错误分层**

- Requirements: RA2、RA3、RA4、RA9。
- Files: `src/main.py`、`src/multiblock.py`、`src/runtime_config.py`、`src/errors.py`；测试 `experiment/test_tails_a.py`，必要时改现有测试里匹配消息的断言。
- Approach: 按 KTD1 到 KTD3、KTD10、KTD12。对着矩阵逐个出口过：还在打印用法说明或裸抛的改掉；启动时抛的错给原因加出处；单块和多块的存盘包进错误分层；多块补上空跑的那一步。
- Execution note: 先把矩阵里每一种情形的测试写出来（“一行、退出码 2、没有用法说明、带出处”，或者运行期错误的阶段名），看到它们在旧代码上挂，再改。存盘的测试用把输出目录设成不可写来造失败；建目录失败的测试用“上一级是一个文件”来造。
- Test scenarios:
  - Covers AE1. 不存在的 `--restart`：一行，退出码 2。
  - `--kstep-save 0`：`配置错误：kstep_save（命令行 --kstep-save，JSON output.kstep_save）：输出间隔 … 必须是正数，得到 0`。
  - JSON 里 `output.kstep_save` 给 0：出处同上；原因里说明值是 JSON 给的。
  - 网格对不上、检查点坏了：初始化错误和配置错误各一条，出处是“读续算的检查点”，原因里有文件名。
  - 多块不放行的错：`初始化错误：多块模式不放行：<参数的出处>：…`。
  - 多块网格校验的错：`初始化错误：多块网格没有通过校验：…`（现有的一条，保留）。
  - Covers AE2. 存盘失败：`运行期错误（存盘）`，单块、多块各一条。
  - 多块空跑那一步出错：`运行期错误（RHS 预热）`。
  - Covers AE8. 配置文件读不了：一行，退出码 2。
  - Covers AE9. 互相冲突的开关：一行，退出码 2。
  - Covers AE10. 建输出目录失败：一行，退出码 2。
  - 命令行给一个不认识的选项：argparse 照旧报，带用法说明。
- Verification: 测试通过；全量两遍。

**A3. 删 RK 接口的 `bc_apply`**

- Requirements: RA5。
- Files: `src/solver.py`、`src/README.md`、第 4.n 步的方案。
- Approach: 删两个函数的参数和分支；README 里“RK 外层不再重复调用 `bc_apply`”那句改成“RK 接口不收回调”。
- Test scenarios: 两个函数的签名里没有 `bc_apply`；第七步和第六步的基线逐位不变（现有测试）。

**A4. 续算的两处不一致**

- Requirements: RA6、RA7。
- Files: `src/runtime_config.py`、`src/main.py`、`src/multiblock.py`；测试 `experiment/test_step8_restart.py`。
- Approach: 按 KTD4。
- Test scenarios:
  - Covers AE3. JSON 里 `"restart": ""`：从头算。
  - 命令行 `--restart ""`：从头算（原来就有的一条，保留）；JSON 里有路径、命令行给空字符串：也从头算。
  - Covers AE4. JSON 的路径失效、命令行给了新文件：按命令行的续算，启动时不报错。
  - 两边都没给：从头算。
  - 最终生效的那个不存在：`配置错误：restart（命令行 --restart，JSON time.restart）：命令行给的文件不存在或不可读：<路径>`；换成 JSON 给的，原因里写“JSON 给的”。
  - 最终生效的是个目录：配置错误，原因里写“不是普通文件”。
  - `mesh.file` 的行为不变：JSON 里失效的路径在读 JSON 时就报错。

**A5. 标准的 `src/input.json`**

- Requirements: RA8。
- Files: `src/input.json`、`src/README.md`；测试 `experiment/test_tails_a.py`。
- Approach: 按 KTD5。内容定死：

  ```json
  {
    "case": {"name": "plate"},
    "mesh": {"format": "cgns", "file": "../Grid/testcgnstest.cgns"},
    "boundary": {
      "inflow": {"rho": 1, "u": 1, "v": 0, "w": 0, "T": 1},
      "farfield": {"rho": 1, "u": 1, "v": 0, "w": 0, "T": 1},
      "wall_temperature": 1.0,
      "outlet_ghost": "keep"
    },
    "physics": {"Re": 1000.0, "Ama": 0.1, "ref_t": 288.15, "pr": 0.7},
    "numerics": {"scheme": "WENO7", "viscous": true, "vis_order": "CD8"},
    "time": {"dt": 0.0001, "end_time": 0.0001, "end_step": 1},
    "output": {"dir": "../../output", "kstep_show": 1, "kstep_save": 1}
  }
  ```

  有意不写的：`numerics.flux_scale`（只有多块要）；`numerics.weno_tv`（只有 WENO7-SYMBO 读它）；`time.restart`（不给就是从头算）；`physics.tw`、`case.inflow_profile`、`case.plate_x0`、`case.plate_x1`、`case.plate_h`、`case.p_grad`、`case.forcing_work`（内置网格的 plate，或 poiseuille 才用；多块也不放行它们）。壁温放在 `boundary.wall_temperature`，CGNS 网格的壁面用它；`physics.tw` 是内置网格的 plate 用的，不放进来。
- Test scenarios:
  - Covers AE5. 从仓库根目录用 `--config opencfd_jax/src/input.json` 跑，退出码 0，产物在 `output/`。
  - 读进来的 JSON，和上面一字不差（整个字典相等）。

**A6. README**：报错的格式、`time.restart` 的两条规则、标准 `input.json` 怎么用。

### E 组：GPU 环境

**E1. 入库的测试在 Linux 上能跑**

- Requirements: RE1。
- Files: `experiment/test_step3_config.py`、`experiment/test_step6_run.py`。
- Approach: 前缀的测试用 `os.path.join` 拼路径；旁记的测试按 KTD8 换掉“嵌套太深”的那一条。
- Test scenarios: 两条在 Linux 和 Windows 上都通过。

**E2. 跨设备的比较**

- Requirements: RE2、RE3。
- Files: `experiment/step6_support.py`、`experiment/test_step49_frozen.py`、`experiment/test_step7_baseline.py`、`experiment/test_step7_exchange.py`；新建 `experiment/test_tails_e.py`。
- Approach: 按 KTD6、KTD7。和 CPU 存下的数据逐位比的 27 条断言换成 `exact`；在同一台设备上比的断言不动。
- Test scenarios:
  - Covers AE6. WSL 的 GPU 上全量没有失败。
  - Covers AE12. `exact` 的判别力：同一台设备上改坏一个位，报错；跨设备差 1e-6，报错；差 1e-16，通过。
  - CPU 上这些测试照旧逐位。
  - 冻结测试：GPU 上改坏一个远大于阈值的量，必须失败。

**E3. README**：在 WSL 的 GPU 上运行一节（RE4）。

### C1 组：多块粘性放行 CD6

- Requirements: RC1 到 RC5。
- Files: `src/multiblock.py`、`experiment/step7_support.py`、第七步的四个测试文件、第六步和第七步的配置测试、`src/README.md`、第七步的方案。
- Approach: 去掉 `_check_release` 里“只放行 CD8”那一条；验收按 KTD9 参数化。
- Execution note: 先把验收按阶数参数化，在 CD6 上看结果，再改放行。哪一层在 CD6 上超出阈值、哪一条注错的测试抓不住，先查清原因，不放行。
- Test scenarios:
  - Covers AE7. 正式入口的 x2，CD6。
  - Covers RC2. 两层验收，CD6 和 CD8 各一遍；每个阶段分接口邻域、物理边界带、其余三块比。
  - Covers RC3. 故意注错的四条，CD6 和 CD8 各一遍，都必须抓得住。
  - Covers RC5. 原来断言“CD6 被拒”的五条改成“CD6 放行”。
  - 多块的 CD6 一步与等价单块的 CD6 在阈值内一致（原来就有的一条，保留）。
- Verification: 测试通过；CPU 和 GPU 上全量各一遍。

### B 组：CGNS 边条（只列范围，轮到时另出方案）

- B1. `BCWallViscous*` 系列认成壁面。网格文件里壁面边条带了温度的数据集时，提示一句没读。单块和多块的映射一起改。
- B2. 单块路径支持一个面由几段同类型的边条拼成，对齐多块的规则。不同类型混在一个面上不做。
- B3. 对称面边界。对照原码 `do_symmetry_kernel_m`、`_p`。待定的问题见 Outstanding Questions。
- B4. 特殊边界脚本配多块。脚本按绝对坐标施加，不需要知道块号，第六步的方案里已经这样定。要做的是把同一个回调接到每一块上，让脚本拿得到这一块的坐标。待定的是接法和先放行哪一类脚本。
- B5. 去掉多块里写死 plate 的地方：`build_blocks` 里按 plate 取的入口和远场状态、初场、`plate_faces`，以及网格和边界处理里按 plate 几何写死的假设。算例名 `plate` 保留，它是唯一验收的算例；`_check_release` 里对算例名的放行不删。不碰体积力。

### D 组：滚动可视文件（只列范围，轮到时另出方案）

- 每个保存步写一个固定名字的 Tecplot 文件，先写临时文件再改名。写文件的工具从 `outputest/snapshot_to_plt.py` 搬进 `src/`。
- 做完要 jisen 在 Tecplot 本体里打开一次。现在的写文件工具没在 Tecplot 里打开过。

---

## Verification Contract

两遍全量。第一遍在 Windows 的 `.venv` 上，第二遍在 WSL 的 `/root/venv-jax` 上：

```bash
# Windows，工作区根目录
.venv/Scripts/python.exe -m pytest opencfd_jax/tests/ opencfd_jax/experiment/ -q -p no:cacheprovider
# WSL，同一个目录
XLA_PYTHON_CLIENT_PREALLOCATE=false /root/venv-jax/bin/python -m pytest opencfd_jax/tests/ opencfd_jax/experiment/ -q -p no:cacheprovider
```

| 门槛 | 通过标准 |
|---|---|
| CPU 全量 | 没有失败。基线是 806 passed、2 skipped、1 xfailed；passed 只增 |
| CPU 上逐位不变 | 冻结测试、第六步和第七步的基线照旧逐位 |
| GPU 全量 | 没有失败（E 组改之前是 32 failed） |
| 单块和多块的数值 | 第六步、第七步、第八步的验收不变 |

---

## 实施记录（实施时定下的决定）

**A 组（2026-11-27 做完）**

A2 报错：

1. 类别（`配置错误`、`初始化错误`）只在出口 `main.main` 里加，出处加在异常消息里。直接调 `main.run`、`multiblock.enter` 的测试看到的消息只是多了出处。
2. 阶段的出处沿用消息里原有的开头：`初始化失败`、`多块模式不放行`、`多块网格没有通过校验`、`多块网格装配失败`、`续算的检查点…`。不另起名字，免得出现“初始化错误：初始化：初始化失败”。RA3、KTD10 里相应的说法已经按这个改过。
3. 运行期的阶段名用代码里原有的 `RHS 预热`。方案第一版里的“右端项预热”是同一个阶段，正文已经统一。
4. 注明来源的写法是 `（命令行给的）`、`（JSON 给的）`；值取的是缺省值、或者查不到来源时不注明。
5. 出处由 `runtime_config` 里的四个函数生成：`origins(name)` 列出入口，`where(name)` 拼成出处，`given_by` 说出是哪一边给的，`blame(sources, name, reason)` 拼出“出处：原因（来源）”。命令行选项从 `main.build_parser()` 的选项表反查。argparse 没有公开的列举接口，用的是它的 `_actions`。`main` 引用 `runtime_config`，所以反查时才导入 `main`。
6. 不加出处的两类：不认识的字段（没有参数名可写）；一次涉及三个参数的“缺少 boundary 参数”（单块、多块各一条）。
7. 多块装配那一段的 `try` 把 `compile_blocks` 也包进去；不是 `CgnsError` 的一律算“多块网格装配失败”，原来只接 `ValueError`。
8. 出口把消息里的换行换成空格，保证只有一行。
9. `_NOT_GIVEN` 改成参数名的元组，`FLUX_SCALE` 常量删掉，原来手写的出处都由反查生成。几条原因里重复的主语随之去掉，例如“`numerics.flux_scale` 在 i 向的分量…”改成“在 i 向的分量…”。
10. `resolve_cli(parsed)`、`_validate_input_paths(args)` 不再收解析器。入库测试里五处调用跟着改；`test_step3_config.py` 里断言冲突开关走 argparse 的那一条，改成断言 `ConfigError`。
11. 测试在 `experiment/test_tails_a.py`，A2 的部分 47 条，在进程里调 `main.main(argv)`；另起进程的只留存盘那一条。配置文件没有读的权限用猴子补丁造，Windows 上改权限位挡不住文件的属主。读续算的检查点那几行由 `experiment/test_step8_restart.py` 原有的测试守着。

A3：`rk3_step_stages`、`rk3_step` 删掉 `bc_apply` 和四个分支，测试查签名。冻结测试和第七步的线性小网格测试照旧通过。

    17. 写方案时说“全仓库的调用方没有一处传 `bc_apply`”，这句不对：本机不入库的 `tests/test_plate_boundaries.py::test_rk3_applies_boundary_to_each_intermediate_state` 给 RK 传了它。漏查的原因是当时用的搜索会跳过被 `.gitignore` 排除的 `tests/` 目录，而 `experiment/step49_progress_2026-11-19.md` 的 2026-11-23 追记里明明写着这条测试。A 组做完后的 CPU 全量抓到了它。这条测试验的是“每个 RK 中间态先过一遍边界”，改成了现在的口径：边界由右端项函数自己做，整步做完以后调用方再做一次。它在 `tests/` 下，改动只在本机。

A4：

12. `time.restart` 用新的字段类型 `restart`：空字符串给 `None`；其余按 `path` 解析相对路径，不核对存在。测试在 `experiment/test_step8_restart.py` 的末尾，6 条。

A5：

13. `src/input.json` 的 `output.dir` 是 `../../output`，方案第一版写的是 `../output`。正式入口要从工作区根目录 `D:\Desktop\ocfd2jax` 运行，从仓库根目录运行找不到 `opencfd_jax` 这个包，所以命令行缺省的 `output/` 在工作区根目录下。`../output` 落在 `opencfd_jax/output/`，是另一个地方，还会往仓库里写不入库的文件。AE5 和 Assumptions 第 3 条已经按这个更正。
14. 标准文件不含 `numerics.weno_tv`。它只在 `WENO7-SYMBO` 下被读到，标准文件用的是 WENO7，写上去会让人以为它有作用。
15. A5 的测试把输出目录改到临时目录，不往工作区里写文件；缺省的输出目录另用解析结果核对。按方案里的原命令从工作区根目录实跑过一次，退出码 0。

A6（README）：

16. 1.1 节的键表原来写“只分六组”，漏了 `boundary`、`time.restart` 和顶层的 `specificboundary`，补成七组加两个顶层字段。示例里的 `--config experiment\cases\sod_1step.json` 是相对仓库根目录的路径，从那里运行不了，改成 `opencfd_jax/experiment/cases/sod_1step.json`。

    17. jisen 要 README 写明白怎么用、边条（特别是特殊边条）怎么设。原来这些信息散在 1.1、7.1.1、7.1.2、7.2 几节，是按“程序怎么实现的”写的，没有一处能照着做。加了 1.3 节“用 CGNS 网格时边条怎么设”：CGNS 边条类型和 `input.json` 里对应要给的项的对照表、出口的两种处理、无反射出口、入口剖面文件的格式、特殊边界脚本的写法和例子、现在设不了的。例子走真实的命令行验过：`sod`、16×8×4、挂上示例脚本跑一步，和不挂相比内区密度最大差 7e-5。顺带改了 7.1.2 里“单块路径可能直接抛出原始异常”的旧话。

    18. 写 1.3 时实测到一件事：特殊边界脚本和 `plate` 算例互斥，所以拿 CGNS 网格跑 plate 时挂不了脚本。`poiseuille` 配 CGNS 网格不被拦、能挂上，但没有验收过。要在 plate 上挂脚本，得改互斥规则，归 B 组，做不做请 jisen 定。

---

**E 组（2026-11-27 做完）**

1. `experiment/step6_support.py` 加了 `computed_on(value)` 和 `exact(got, want, label, got_on, want_on)`。设备从 jax 数组的 `devices()` 看，jax 0.12.1 里没取值的数组也看得到；平台名在 CPU 上是 `cpu`，在显卡上是 `gpu`。NumPy 数组当作 CPU 上的数据。
2. 一维的抽样没有分量可分（第七步交换测试的抽样是整场打平以后隔 379 个取一个），跨设备时整条按一个分量算容差，用的是 `compare(..., count=1)`。没有重新生成存下来的抽样文件。
3. 只检验 FP64（jisen 2026-11-27 的决定）。状态是 FP32 的那一个调用方式 `cube_mixed_cd8`，GPU 与 CPU 存下的值差 2.4e-7，FP64 的阈值是 3.1e-13，按 FP64 的阈值比不过。它在 GPU 上不和存下的值比，跳过并写明原因；在 CPU 上算的仍然逐位比。实现方先做的是按机器精度的比值放大阈值，jisen 定了以后撤掉，`compare` 和 `exact` 里没有留下放大的参数。
4. 冻结测试的两条：`test_one_step_matches_frozen_file` 改用 `exact`；`test_comparison_detects_one_flipped_bit` 改名为 `test_comparison_notices_damage`，同一台设备上翻一个最低位，跨设备时把一个值乘 (1 + 1e-6)，两种情况比较都必须失败。
5. 判别力的测试在 `experiment/test_tails_e.py`（5 条，没有 GPU 的机器上跳过 1 条）。改过的三个文件在 CPU 上 28 条逐位照旧，在 GPU 上 27 条通过、1 条按第 3 条跳过。
6. E1 的两处：`test_step3_config.py` 的路径改用 `os.path.join` 拼；`test_step6_run.py` 里“旁记读不出来”的那一个子情况改成写进不合 UTF-8 的字节。两个平台上都过。
7. 本机不入库的 `tests/` 里六个文件写死的 `D:\Desktop\ocfd2jax\channel_case_iiA` 路径，改成相对本文件拼工作区根目录。Windows 的 CPU 上 37 条通过，WSL 的 GPU 上 37 条通过；E 组改之前它们在 Linux 上是 3 条挂、3 条跳过。这些改动只在本机。
8. README 加了 1.2 节“在 WSL 的 GPU 上运行”。

---

**C1 组（2026-11-27 做完）**

1. 验收按阶数参数化。`step7.reference(vis_order)`、`periodic_reference(vis_order)` 每种阶数各建一份单块参考，走正式入口的装配。`test_step7_acceptance.py` 的单块链和多块运行从 fixture 改成模块级的缓存，测试按（预设，阶数）参数化；`test_step7_linear.py` 和 `test_step7_exchange.py` 里“一块的多块对正式路径”那条同样。
2. 新加“物理边界带”这一层：`step7.physical_band(built, note)`，跟着各块的 `execution.band` 走，是紧挨着打开了单侧公式的那一面的 4 层内区点。验收网格上只有壁面（j−）开着单侧公式，入口、远场、出口、周期面都不开；合成小网格里只有带壁面的才有。`test_step6_acceptance._check` 多了一个可选的参数，第六步的无粘验收不传，行为不变。报告行变成四层：全域、接口邻域、物理边界带、其余（其余不含前两层）。
3. 合成小网格上，一块区域非空、并且不满的时候才单独比。一格宽的条带网格上，物理边界带可以盖满全局。
4. 新加两条测试：掩码跟着各块的标志走；切口两侧的面上的带标志都是 0。四条注错的测试对两种阶数各验一遍，都抓得住。写掩码函数时把 `enumerate` 给出的两个量写反了，验收照样通过，是前一条测试抓出来的。
5. 放行：删掉 `_check_release` 里“粘性打开时只放行 CD8”那一条。
6. 原来断言“CD6 被拒”的测试：第六步配置测试的不放行清单里两行删掉，命令行那一条改用 UP7 做例子；第七步配置测试里，拒绝的那条改成“放行校验过了、拦在后面的读坐标上”，命令行的那条改成跑完并核对 `_meta.vis_order`，粘性关闭的那条保留；走通入口的那条按阶数参数化，并核对 `_meta.vis_order`。
7. 两种阶数下，五个验收网格各阶段最大的差用到容差的 2% 以内（x2 第 2 步的粘性分项 3）。
8. `src/README.md` 和第七步的方案（`docs/plans/2026-11-25-001-feat-step7-multiblock-viscous-plan.md`）里“只放行 CD8”的说法改掉了，第七步方案的文件头加了现状。

---

**sol 第 1 轮（2026-11-27，代替 Astra；只读审查，跑了定点测试，没有重跑全量）**

结论：这批暂时不能提交。定点测试通过，但启动错误还有漏网的出口。实现方逐条复现，都成立，全部采纳：

| sol 的意见 | 级别 | 实现方的处理 |
|---|---|---|
| 特殊边界脚本的 `build_special_boundary()` 抛 `RuntimeError` 时裸抛，退出码 1 | 必须 | `main.run` 里建算例那一步出的任何异常都包成初始化错误，已经是 `SetupError`、`ConfigError` 的原样往上走；脚本载入、装配时出的异常换成 `ValueError` 并点名脚本（`src/init.py`） |
| 检查点的 `_meta` 是合法的 JSON 但不是对象时，绕过 `CheckpointError` 裸抛 | 必须 | `io.load_checkpoint` 核对 `_meta` 和块清单都是对象，不是就报检查点不可用。这一条超出第八步定下的坏文件清单，加了两条测试，请 jisen 过目 |
| `--plate-dif-bound` 写得不对时报错没有出处 | 必须 | 在配置阶段校验（六个 0 或 1，空格或逗号分开），报带出处的配置错误（`runtime_config._check_dif_bound`）。比原来严：`0.5` 这种写法原来悄悄当成 0，现在报错 |
| `--inflow-profile` 指向不存在的文件时报错没有出处 | 必须 | 纳入 `_validate_input_paths`，和网格、续算的检查点一样报带出处的配置错误 |
| 待闭合事项的文件头写着“工作区里现在没有未提交的源码改动”，和当时的工作区不符 | 必须 | 改成只说历史：那份文件里说还没入库的实现，后来都入库了 |
| 方案里有一行还写着 `exact` 带 `scale` 参数 | 建议 | 改掉 |
| README 里 GPU 全量的用时（23 分钟）和方案里的（29 分 25 秒）对不上 | 建议 | 实测单独跑是 22 分半，和 CPU 那一遍同时跑才到 30 分钟上下。README 和方案都写明“单独跑约 23 分钟，不要同时跑” |

顺带改了一条旧测试的断言：`test_step6_config.py` 的 `test_damaged_mesh_files_never_raise_raw_exceptions` 原来拿“直接调 `init.prepare` 抛的类型”去要求 `main.run` 抛同一种类型，在“建算例出的异常一律包成初始化错误”的规则下不稳，改成按规则断言。

新增的测试：`experiment/test_tails_a.py` 的第 13 节（脚本抛异常 1 条、`--plate-dif-bound` 4 条、入口剖面 1 条），`experiment/test_step8_restart.py` 的两条坏检查点测试各加一个子情况。

特殊边界脚本自己的配置错误（文件名不合规、脚本不存在、缺入口、缺回调、与 plate 互斥、脚本载入或装配时出的错）：jisen 2026-11-27 定了现在就写成正式的出处。`init.attach_specificboundary` 一律报 `SetupError`，消息是 `specificboundary（命令行 --specificboundary，JSON specificboundary）：原因`，最终那一行是 `初始化错误：specificboundary（…）：原因（哪一边给的）`。`experiment/test_specificboundary.py` 里六处断言期待的异常类型跟着改；`experiment/test_tails_a.py` 加了三条走正式入口的测试，原来用“脚本不存在”当例子的那一条换成读不出来的网格文件。

**全量的结果（2026-11-27，按 sol 第 1 轮的意见和 jisen 以后的决定改完，两遍先后单独跑的）**

| | 通过 | 跳过 | 预期失败 | 用时 |
|---|---|---|---|---|
| CPU，Windows 的 `.venv` | 906 | 3 | 1 | 10 分 29 秒 |
| GPU，WSL 的 `/root/venv-jax` | 906 | 3 | 1 | 24 分 03 秒 |

- Verification Contract 的四条门槛：CPU 全量没有失败；CPU 上逐位的护栏不变（冻结测试、第六步和第七步的基线照旧逐位）；GPU 全量没有失败；第六步、第七步、第八步的验收不变。

---

## Definition of Done

**全局**

- A、E、C1 三组做完，Verification Contract 的四条门槛都满足；B、D 两组各有出完的方案和进度记录。
- 没有留下试验性的代码、注释掉的分支或一次性脚本。
- 没有引入任何哈希值。
- 路线图和 README 与代码一致；实施时定下的决定记进这份方案。

**逐组**

- A：启动错误一行、带出处；KTD10 矩阵里的每一个出口都归了类、有测试盯着；存盘有错误分层；RK 接口不收 `bc_apply`；续算的两处不一致改掉；有标准的 `src/input.json`；路线图的进度是最新的。
- E：WSL 的 GPU 上全量没有失败；CPU 上逐位的护栏不变；同一台设备上（GPU 也算）比的两份结果照旧逐位。
- C1：多块粘性打开时 CD6、CD8 都放行，两层验收对 CD6 也过。

---

## 附：数值验证的建议（不在这次实施的范围内，Astra 第 31 轮看过）

**现状**

- 第 5 步留下的那一条：超声速层流平板（Ama=2，等温壁），壁面摩擦系数比相似解 CLBL 低约一半，网格加密也不敏感，来源没定位。速度剖面本身在加密后与参考差 −1.5% 到 −4.7%。
- 以后的程序都在 WSL 的 GPU 上跑，float64；GPU 与 CPU 的差在舍入量级。
- 以后验证只用平板这一类边界。

**建议分三段，从快到慢**（Astra 第 31 轮看过，认为顺序合适）

1. **先回答“代码写得对不对”，不需要外部参考数据。**
   - 制造解：选一个光滑的解析解，代进方程算出源项，让程序带着源项算，看误差随网格加密按几阶下降。离散方程、粘性项、边界实现哪里写错，这里会直接暴露。JAX 能自动求导，源项不用手推。
     - 源项只给验证用，不进正式算例的配置：JSON 不加键，正式入口不读。
     - 现在的源项接口是 `case["source"]`，一个回调 `source(q)`，`build_rhs` 在每个 RK 子步把它的返回值加到右端项上。它只拿得到守恒量，拿不到坐标和时间。坐标由验证脚本自己备好、闭包进回调：CGNS 网格的内区坐标在 `desc.coords`，内置的直角网格由 `desc.dxs` 和下标算出来。与时间有关的制造解，验证脚本自己写推进循环，每一步带着当时的 t 重新装配右端项，不走 `main.run`。
     - 收敛阶分两块区域各报一个：内部区域，和物理边界带里的点（单侧差分模板盖到的范围）。只报整体一个阶数，分不出是内部格式还是单侧模板的问题。
   - 低马赫数层流平板对 Blasius 解（代码里已有 `blasius.py`）。Astra 指出：这一个对照单独证明不了“问题只在摩擦系数的提取上”。所以先把摩擦系数拆开，分四样与 Blasius 的对应量比：壁面剪切（从速度剖面差分出来的）、壁面的黏度、无量纲化用的参考量（ρ∞、U∞）、最后的系数。差在哪一样，来源就缩小到哪一层。然后做低马赫数和可压缩平板（Ama=2）两个对照：四样在低马赫数下都对，问题在可压缩这一侧（黏度随温度的变化、温度的约定）；低马赫数下壁面剪切就低了，问题在壁面附近的边界实现和差分。
2. **层流可压缩。** 回到超声速平板，把摩擦系数对上 CLBL。放在第一段之后，因为第一段很可能直接指出原因。
3. **带 recycling 的超声速湍流边界层。** 值得做，但不建议现在上：底下压着一条没定位的层流问题，湍流结果出了偏差分不清是谁的；recycling 自己还带进一批要调的东西（回收站的位置、重新标度、平均的时间）。Astra 认为放在最后合适。

**当时问 Astra 的三件事**

- 顺序和切分有没有更省的办法；“摩擦系数低约一半、加密不敏感、剖面却在收敛”最可能是哪几类原因。Astra 回了前一半：顺序合适，Blasius 的对照要先拆开再比。后一半没有回答。
- 制造解在这套代码上（WENO7 加单侧的边界带、CD8 粘性、等温壁、周期面）有没有已知的坑。Astra 没有回答，只要求分区域报收敛阶。
- recycling 在单卡（12 GB 显存）、float64 下的规模估算和更省的替代办法。Astra 没有回答。
