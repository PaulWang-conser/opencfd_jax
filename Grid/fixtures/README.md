# CGNS fixtures

`poiseuille_box.cgns` 是第 4 步的仓库内契约 fixture：

- HDF5/CGNS，单个 `Zone 1`，`VertexSize=(16,16,8)`，`CellSize=(15,15,7)`；
- i/k 两面用同块整面 identity 1to1 表示周期，j 两面是 `BCWall`；
- 坐标与内置 `poiseuille` 的均匀网格同口径：`x=(i+0.5)*dx`、
  `y=(j+0.5)*dy`、`z=k*dz`，`dx=dz=2π/ni`、`dy=1/nj`；
- 生成命令：`python Grid/make_cgns_fixture.py`。

该文件由 `Grid/make_cgns_fixture.py` 用 h5py 按
`Grid/samples/multi_hdf5.cgns` 的 `label`/节点布局写出。h5py 能写
HDF5-CGNS 结构，但不是 SIDS 校验器；fixture 只用于仓库内 reader/adapter
契约，不宣称可被任意 CGNS 工具接受。

`nk=8` 是度量计算的硬下限：`compute_metrics_from_coords` 的
`jac_bound=(1,1,1)` 需要每轴内区至少 7 个点，`band_one_sided` 才能取到 7 点
单侧公式。更小的 `nk` 会在度量阶段报错，不是 fixture 写错。
