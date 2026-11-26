# Step 4.9 分阶段参考

本目录保存针对 OCFD Git HEAD `5eefe8fc032cda224e1e1f6ab7e6f940d7a1c8a4` 的诊断补丁与 JAX RHS 对照工具。诊断只用于独立源码副本，禁止在用户原有 OCFD 工作区直接应用。

`ocfd_diagnostic.patch` 在 `ana_Jac` 处保存带鬼层的 `Ajac`，在每个 RK 阶段的 `du_comput` 后保存总 RHS 与五个原始量，在无粘计算之后保存无粘 RHS，并在各方向 Ev 交换后保存四个 Ev 分量、在分裂后保存六组分裂通量。原始量和 Ev 文件大小为 `(nz+8)*(ny+8)*(nx+8)` 个 float64，布局为 `(z,y,x)`；RHS 布局为 `(5,nz,ny,nx)`，分裂通量为 `(5,nz+8,ny+8,nx+8)`。此插桩仅覆盖 `STREAM=0` 路径，不导出重构面通量。新增插桩后的五个末态原始量与未插桩干净 HEAD 逐位一致。

干净 HEAD 建议先用 `git archive` 导出到新的空目录，再在该目录执行 `git apply` 或 `patch -p1 < ocfd_diagnostic.patch`。先分别编译未插桩版本和诊断版本；在独立输出目录用相同输入运行，并逐字段确认最终结果一致。当前验证使用 CUDA 12.8、MPICH 和以下 make 覆盖参数，未改 OCFD 数值源码：

```sh
mkdir -p obj
make default -j4 DEV_PATH=/usr/local/cuda \
  'DEV=nvcc -gencode arch=compute_120,code=sm_120' \
  'OPT_Commen=-O3' \
  'OPT_Dev=-dc -I /usr/include/x86_64-linux-gnu/mpich -I /usr/include/mpich -O3'
```

这套参数与 Git HEAD 自带的 CUDA 11.6、sm_75、`-G` 设置不同，应记录为编译差异。已有输入、结果与哈希见 `../step49_待闭合事项_2026-11-19.md`。

`prepare_reference.py` 从指定 CGNS 坐标生成本例网格、均匀入口初值、`.in` 与输入哈希；输出目录必须不存在，避免覆盖既有结果。默认只生成本次单位来流、固定步长、IBC=124 的参考，不支持将任意 JSON 参数隐式转换。重建的网格与初场文件和原输入 SHA-256 一致；以新 `.in` 从干净 HEAD 运行一步，五个原始量与先前干净 HEAD 输出逐位一致。

`compare_rhs.py` 以指定 JSON 运行 JAX 三个阶段，并读取 `diagnostic_du_*.bin` 和 `diagnostic_invis_*.bin`，分别比较总项、无粘项以及两者相减得到的粘性项。结果按物理变量映射、计算轴 `(k,j,i)` 和 all、入口、出口、壁面、远场、深内区统计；边界区域有重叠。相对误差分母为对应区域参考场最大绝对值，为零时记为 null；非有限值单独计数，绝对误差位置使用全局内区索引。默认路径针对当前本地样例，可用 `--config`、`--reference`、`--output` 覆盖。

工具还读取三个阶段的原始量，比较物理内区、纯法向鬼层、周期层与两个出口角区；原始量报告的位置是**含四层鬼胞的数组索引**。`--ocfd-farfield-band`、`--interior-face-writes` 和 `--rescale-weno-epsilon` 仅用于进程内诊断，报告记录开关，不修改正式代码。epsilon 诊断只适用于当前 JAX 步长为 1、原版为 `1/(N-1)` 且 WENO7 的参考，不能当作通用修正。

`compare_ev.py` 对照每阶段、每方向的 Ev，物理分量映射并将原版结果除以另外两方向 `(N-1)` 的乘积后，按内区及纯法向鬼层统计。原版复用 Ev 工作数组，横向鬼层可能遗留其他方向数据，应针对求导方向判断其影响；相关归一化限于本参考的步长设置。报告不覆盖分裂/重构通量和完整参考坐标度量。

`compare_split.py` 使用相同换算，逐阶段比较三个计算方向的正负分裂通量，含物理内区及纯法向/周期鬼层。此缩放只适用于当前参考步长设置；重构面通量尚未导出。
