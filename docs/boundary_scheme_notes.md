# 边界带公式核对笔记

本文件保存原先写在 `src/boundary.py` 模块 docstring 里的原码核对材料。第 2 步把
`boundary.py` 改成实际实现后，这些历史笔记迁移到这里；运行时边界实现见
`src/boundary.py`，实施状态见 `experiment/README.md` 的第 2 步验收节。

当前求解链中的边界带实现位于 `schemes.flux_band_faces` 和
`viscous.band_one_sided`；本文件保留原码公式、索引推导和待核对事项，不参与
主循环调用。阅读或修改边界处理时，应同时对照上述两个实现和本文件的说明。

历史核对记录（2026-11-14）：公式材料逐行核实完毕，但 coords（边界层偏移）
与面编号的精确对应依赖原码 outer kernel 的 job 范围与 block 重叠技巧
（put_du_p 的 `if(threadIdx.x != 0)` 跳过每 warp 首线程、由相邻 block 的
重叠线程补写）。周期/出流算例不经过这些物理边界带；出流边界使用零梯度
鬼胞加主格式模板，边界四格精度受限。

已核实的公式材料（OCFD_bound_Scheme.cu，全部逐行核对）：

**WENO5 变体 8 个（489-667 行，边界层专用，实参为原码 stencil 的连续 5 点
切片，内 s0..s4 = 实参起 5 点；ε=1e-6、12ε 进权）**：

- P_lift（基 s2，面 s2+1/2）：单候选
    face = (2*s2 + 5*s3 - s4)/6
- P_lift_plus（基 s2，两候选 a0=3/(12ε+S0)²、a1=6/(12ε+S1)²，
  S0 用 s2..s4、S1 用 s1..s3）：
    q03 = 2*s2+5*s3-s4;  q13 = -s1+5*s2+2*s3
    face = (a0*q03 + a1*q13) / (6*(a0+a1))
- P_right（基 s1，面 s1+1/2）：单候选
    face = (2*s0 + 7*(-s1) + 11*s2)/6   # 原码 q23 = 2*s0-7*s1+11*s2
- P_right_plus（基 s1，两候选 a0=3/(12ε+S0)²、a1=6/(12ε+S1)²，
  S0 用 s0..s2、S1 用 s1..s3）：
    q03 = 2*s2+5*s1-s0;  q13 = -s3+5*s2+2*s1
    face = (a0*q03 + a1*q13) / (6*(a0+a1))
- WENO5_P（完整三候选，同 schemes._weno5_face_P 的单面形式）
- M_lift / M_lift_plus / M_right / M_right_plus：P 系列的镜像
  （基 s2、面 s2-1/2；q 系数下标翻转，如 M_lift = (2*s4-7*s3+11*s2)/6）
- WENO5_M（完整三候选镜像）

**边界层结构**（boundp1/boundp2/boundm1/boundm2，671-876 行）：

- 低端 P（boundp1，`coords <= -ka1` 时替换）：coords = 0,1,2,...,|ka1|
    coords=0:   face = f[cell coords+1]            （零阶直接值）
    coords=1:   face = P_lift
    coords=2:   face = P_lift_plus
    coords=3:   face = WENO5_P
    （层数 = |ka1|+1；UP7 |ka1|=3 → 4 层；WENO5 |ka1|=2 → 3 层）
- 高端 P（boundp2，`coords > -kb1`）：coords' = -1..-kb1
    coords'=-1:  face = f[cell nx-1] + 0.5*(f[nx-1]-f[nx-2])   （NND 外推；
                 原码 minmod2(Δ,Δ)=Δ）
    coords'=-2:  face = P_right_plus
    coords'=-3:  face = WENO5_P          （完整 5 点，模板 cells n-5..n-1）
    coords'=-4 及更低: 不覆盖（bound 函数回落主格式，见 schemes.py hi 分支）
    （层数 = kb1；注意高端公式**只用 cell ≤ n-1**，不读上界鬼胞——2026-11-15 用
      `diag_top_band_verify.py` 与 `diag_top_ghost_dependence.py` 逐面实测确认：
      面 n/n-1/n-2 读 0 层鬼胞，n-3 的 P 侧走正常 OMP6 才读 1 层）
- 低端 M（boundm1，`coords < -ka1_M`）：
    coords=0:   face = M_lift                （(2f2-7f1+11f0)/6 形式）
    coords=1:   face = M_lift_plus
    coords>=2:  face = WENO5_M               （完整 5 点，域内）
- 高端 M（boundm2，`coords >= -kb1_M-1`）：
    coords'=-1:  face = f[cell nx-1]          （零阶）
    coords'=-2:  face = f[nx-1] - 0.5*(f[nx-1]-f[nx-2])   （NND 外推）
    coords'=-3:  face = M_right_plus
    coords'=-4:  face = WENO5_M

**各格式的 ka1/kb1**（从各 kernel 的 ia1/ib1；P/M 分别取）：

    UP7:    P(-3,4)  M(-4,3)     WENO5:  P(-2,2)  M(-2,2)
    WENO7:  P(-3,3)  M(-4,3)     NND2:   P(-2,2)  M(-2,2)
    OMP6:   P(-3,4)  M(-4,3)

**开放问题（对拍锚定点）**：

1. coords 与面编号 j 的映射（原码 P 线程产面 x+1/2 但 put_du_p 由
   threadIdx.x!=0 与 block 重叠共同决定实际写入位置）；
2. 物理面 0 的 P 通量与物理面 nx 的 M 通量的产生机制（outer kernel）；
3. 每格式的边界层数与 bound 宏条件（coords <= -ka1 vs coords < -ka1）的
   精确边界（boundp1 用 <=、boundm1 用 <）。

实现时验证：均匀流+单点扰动场，比对边界面通量 vs 原码 dump（方案 9.1 节）。

## 与 Step 2 统一边界的关系

- `src/boundary.py` 现在负责**通用鬼胞**：周期、出口鬼层（缺省保持原值）、y 方向无滑移壁面镜像、
  Ev 镜像、非周期鬼层清零，以及 JIT 前的 `BoundaryExecution` 校验。
- `src/schemes.py` 的 `flux_band_faces` 和 `src/viscous.py` 的单侧导数仍负责**边界带
  公式**；本文件的公式材料继续对应这两处，不与通用鬼胞混为一谈。
- 槽道三点压力/温度外推、平板入口/顶部 Dirichlet 和两点压力外推仍留在
  `channel.py` / `plate.py`，没有为了统一而改写。
- 当前限制：通用 `WALL` 只支持 y 方向；滤波只支持周期/出流；`special_boundary`
  采用 case 显式对象 + `bc_apply`，不做动态注册，也不允许 I/O 进入 JIT。
