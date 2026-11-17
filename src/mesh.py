"""曲线坐标度量：从节点/中心坐标场预计算雅可比度量，缓存为 npz。

与 OCFD 同款路线（OCFD_Comput_Jacobian3d.cu）：正式计算前算好度量写入
文件，计算时直接读取。公式照搬 comput_Jac3d_kernal（245-292 行）：

    J = xi*yj*zk + yi*zj*xk + zi*xj*yk - zi*yj*xk - yi*xj*zk - xi*zj*yk
    Ajac = 1/J
    Akx = yj*zk - zj*yk,  Aky = zj*xk - xj*zk,  Akz = xj*yk - yj*xk
    Aix = yk*zi - zk*yi,  Aiy = zk*xi - xk*zi,  Aiz = xk*yi - yk*xi
    Asx = yi*zj - zi*yj,  Asy = zi*xj - xi*zj,  Asz = xi*yj - yi*xj

其中 xi = d(x)/dξ，xj = d(x)/dη，xk = d(x)/dζ；计算方向
(i,j,k) 对应 (ξ,η,ζ)，物理坐标是 (x,y,z)。直角单位度量时
Akx=Aiy=Asz=1。偏导用 CD6/CD8
（复用 viscous.cd_along_axis），鬼胞复制最近内点（等价原码 kernel 的
边界复制，此处用 OUTFLOW 填充实现）。

度量键沿用原码：Akx/Aky/Akz 对应计算 i（ξ）方向，Aix/Aiy/Aiz 对应
计算 j（η）方向，Asx/Asy/Asz 对应计算 k（ζ）方向；末尾 x/y/z 是
物理坐标分量。这些键中的 k/i/s 是原码名称，不是这里的 i/j/k 索引。
"""

import numpy as np

import jax.numpy as jnp

from .axes import resolve_dimensions
from .fields import LAP
from .viscous import cd_along_axis

METRIC_KEYS = ("Akx", "Aky", "Akz", "Aix", "Aiy", "Aiz",
               "Asx", "Asy", "Asz", "Ajac")


# 原码 comput_Jac3d_kernal 会把内区边界值复制进鬼胞；JAX 版不需要——
# 度量场由 cd_along_axis 输出（与坐标场同形、鬼胞随坐标场鬼胞的解析
# 延拓决定），鬼胞度量值不参与内区结果。


def compute_metrics_from_coords(coords: dict, dxs, order=6,
                                jac_bound=(1, 1, 1),
                                periodic=(True, False, True)):
    """由坐标场（含鬼胞）计算度量。coords: {"Axx","Ayy","Azz"}，计算轴序 (k,j,i)。

    dxs: 三轴参考步长 (hx, hy, hz)——注意是计算空间步长（均匀网格索引间距），
    不是物理 x 方向步长。返回 METRIC_KEYS 的度量场 dict（含鬼胞）。

    jac_bound: 原码 `JAC_BOUND`（计算 i/j/k 三标志）。为 1 的轴，每侧 LAP 层
    坐标导数覆盖成单侧公式（`OCFD_Dx0_bound/Dy0_bound/Dz0_bound`，
    OCFD_bound_Scheme.cu:232-269 / viscous.band_one_sided）。
    原码缺省 `JAC_BOUND=1 1 1`，槽道 in 文件也是 `1 1 1`；只有全 0 时才
    与旧的"全场中心 CD6"逐位相同。

    periodic: 原码 `IPERIODIC`（计算 i/j/k），按数组轴 (k,j,i) 使用。周期轴的
    坐标鬼胞按**环绕**填（原码 MPI Sendrecv 自收自发），非周期轴边缘复制。
    注意：jac_bound 为 1 时边界带本来就被单侧覆盖，鬼胞填法只影响
    jac_bound=0 的轴，但为了与原码一致仍按 IPERIODIC 填。
    """
    # 坐标场可能只在一个轴上有变化（如 y=y(j) 给成 (1,ny+2L,1)），先广播成同形
    shape = np.broadcast_shapes(*(np.shape(v) for v in coords.values()))
    Axx, Ayy, Azz = (jnp.broadcast_to(jnp.asarray(coords[k]), shape)
                     for k in ("Axx", "Ayy", "Azz"))
    # 标量数组轴 (k,j,i)=(0,1,2)；计算方向 (i,j,k) 对应 dxs[0..2]
    # 每项 (array_axis, dxs_index, jac_bound_index)
    combos = ((2, 0, 0), (1, 1, 1), (0, 2, 2))

    def deriv(f, axis, hidx, jbidx):
        """计算一个坐标导数，并按对应方向的 JAC_BOUND 标志切换边界带公式。"""
        b = int(jac_bound[jbidx]) if jac_bound else 0
        return cd_along_axis(f, axis, dxs[hidx], order, bound=(b, b))

    xi, xj, xk = (deriv(Axx, *c) for c in combos)
    yi, yj, yk = (deriv(Ayy, *c) for c in combos)
    zi, zj, zk = (deriv(Azz, *c) for c in combos)

    J = (xi * yj * zk + yi * zj * xk + zi * xj * yk
         - zi * yj * xk - yi * xj * zk - xi * zj * yk)
    metrics = {
        "Ajac": 1.0 / J,
        "Akx": yj * zk - zj * yk,
        "Aky": zj * xk - xj * zk,
        "Akz": xj * yk - yj * xk,
        "Aix": yk * zi - zk * yi,
        "Aiy": zk * xi - xk * zi,
        "Aiz": xk * yi - yk * xi,
        "Asx": yi * zj - zi * yj,
        "Asy": zi * xj - xi * zj,
        "Asz": xi * yj - yi * xj,
    }
    return {k: _edge_ghosts(v) for k, v in metrics.items()}


def _edge_ghosts(a, lap=None):
    """把含鬼胞的度量场在鬼胞区复制最近内点。

    原码的度量只在**内区**算（`calcul_Jac3d` 的 job 覆盖内区），鬼胞度量
    从不被读（`put_du_p_kernel`/`YF_Pe_XF` 的 job 也只覆盖内区）。JAX 这边
    整个数组一起乘 `Ajac`，若鬼胞留 `inf`/`nan`，`0*inf` 会把鬼胞 du 污染成
    nan，再借 BC/填充渗回内区。因此这里的鬼胞复制是**纯粹的安全措施**，
    不改变任何内区数值。
    """
    lap = LAP if lap is None else lap
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        n = a.shape[ax]
        lo[ax] = slice(0, lap)
        hi[ax] = slice(n - lap, n)
        src_lo = jnp.expand_dims(jnp.take(a, lap, axis=ax), ax)
        src_hi = jnp.expand_dims(jnp.take(a, n - lap - 1, axis=ax), ax)
        a = a.at[tuple(lo)].set(src_lo)
        a = a.at[tuple(hi)].set(src_hi)
    return a


def pad_coords(arr, periodic=(True, False, True)):
    """按 IPERIODIC 给坐标场补 LAP 层鬼胞（与 `pad_to_lap` 的边缘复制不同）。

    periodic 按计算方向 (i, j, k) 给出，数组轴是 (k, j, i)；周期轴取**对侧内区**
    （原码 `exchange_boundary_x/y/z_packed_dev` 单块自收自发等价于环绕），
    非周期轴边缘复制。
    """
    out = arr
    for axis, per in zip((2, 1, 0), periodic):
        n = out.shape[axis]
        if LAP >= n:
            raise ValueError(f"axis {axis} 长 {n} < LAP={LAP}，无法环绕填鬼胞")
        if per:
            lo = np.take(out, np.arange(n - LAP, n), axis=axis)
            hi = np.take(out, np.arange(0, LAP), axis=axis)
        else:
            lo = np.take(out, np.zeros(LAP, dtype=int), axis=axis)
            hi = np.take(out, np.full(LAP, n - 1), axis=axis)
        out = np.concatenate([lo, out, hi], axis=axis)
    return out


def make_metrics_cartesian(grid, dxs=(1.0, 1.0, 1.0)):
    """均匀笛卡尔度量：Ajac=1、Akx=Aiy=Asz=1、其余 0（与原码初始化一致）。"""
    shape = (grid.nk + 2 * LAP, grid.nj + 2 * LAP, grid.ni + 2 * LAP)
    one = np.ones(shape)
    zero = np.zeros(shape)
    return {
        "Akx": one.copy(), "Aky": zero.copy(), "Akz": zero.copy(),
        "Aix": zero.copy(), "Aiy": one.copy(), "Aiz": zero.copy(),
        "Asx": zero.copy(), "Asy": zero.copy(), "Asz": one.copy(),
        "Ajac": one.copy(),
    }


def read_ocfd_mesh(path, ni, nj, nk):
    """读原码的 `OCFD3d-Mesh.dat`（X/Y/Z 三个坐标场，内区 (nk, nj, ni)）。

    尺寸参数只接受规范 ``ni/nj/nk``。格式（由 channel_case_iiA/gen_case.py 写、
    原码 OCFD_Comput_Jacobian3d.cu:126-130 + read_3d1 读，两边一致）：
      - 无文件头；三个场依次是 X、Y、Z；
      - 每个场按 k 面一条记录：`int32 长度标记 + ni*nj 个 float64 + int32 长度标记`
        （长度标记 = ni*nj*8 字节），数组内 i 最快、j 次之（C 序）；
      - 长度标记只用作完整性校验（原码的 offset 步长
        `(2*sizeof(int) + NX*NY*sizeof(REAL)) * NZ` 就来自它）。

    尺寸由 in 文件的 GRID_3D 给出（原码也是从 in 文件读，网格文件里没有维数）。
    返回 {"Axx": x, "Ayy": y, "Azz": z}，均为 (nk, nj, ni)，**不含鬼胞**。
    """
    ni, nj, nk = resolve_dimensions(ni, nj, nk)
    n = ni * nj
    payload = 8 * n
    rec = 8 + payload
    nbytes = 3 * nk * rec
    import os
    size = os.path.getsize(path)
    if size != nbytes:
        raise ValueError(
            f"{path} 大小 {size} 与 ni,nj,nk=({ni},{nj},{nk}) 推出的 {nbytes} 不符"
            f"（每面 {rec} 字节 = 2×int32 + {n}×float64）——in 文件的 GRID_3D 与"
            f"网格文件不是同一套，或该目录的 Mesh.dat 是空文件（0 字节）")
    buf = np.fromfile(path, dtype=np.uint8)
    out = {}
    for f, key in enumerate(("Axx", "Ayy", "Azz")):
        blk = buf[f * nk * rec:(f + 1) * nk * rec].reshape(nk, rec)
        head = blk[:, :4].copy().view("<i4").ravel()
        tail = blk[:, 4 + payload:].copy().view("<i4").ravel()
        if not np.all(head == payload) or not np.all(tail == payload):
            raise ValueError(f"{path} 记录标记不是 {payload}，不是原码网格格式")
        arr = blk[:, 4:4 + payload].copy().view("<f8").reshape(nk, nj, ni)
        out[key] = arr
    return out


def pad_to_lap(arr):
    """把内区 (nk, nj, ni) 场按 LAP 鬼胞扩展，鬼胞复制最近内点（原码做法）。"""
    return np.pad(arr, ((LAP, LAP), (LAP, LAP), (LAP, LAP)), mode="edge")


def read_ocfd_field(path, ni, nj, nk, nvars=5):
    """读原码的场文件（`opencfd.dat` / `OCFD00056000.dat`），返回内区数组列表。

    尺寸参数只接受规范 ``ni/nj/nk``。格式（OCFD_IO.c read_file + gen_case.py write）：
      - 文件头 20 字节：`int32(12) + int32(Istep) + float64(tt) + int32(12)`；
      - 之后 nvars 个场依次排列，每个场按 k 面一条记录
        `int32 长度标记 + ni*nj float64 + int32 长度标记`；
      - 场顺序 **rho, U, V, W, T**（原始变量）。
    """
    import os
    ni, nj, nk = resolve_dimensions(ni, nj, nk)
    payload = 8 * ni * nj
    rec = 8 + payload
    size = os.path.getsize(path)
    nbytes = 20 + nvars * nk * rec
    if size != nbytes:
        raise ValueError(
            f"{path} 大小 {size} 与 ni,nj,nk=({ni},{nj},{nk})、{nvars} 个场的"
            f" {nbytes} 不符（每面 {rec} 字节 + 20 字节文件头）")
    buf = np.fromfile(path, dtype=np.uint8)
    head = buf[:20].copy().view("<i4")
    if head[0] != 12 or head[-1] != 12:
        raise ValueError(f"{path} 文件头标记不是 12/12（不是原码场文件）")
    out = []
    for f in range(nvars):
        blk = buf[20 + f * nk * rec:20 + (f + 1) * nk * rec].reshape(nk, rec)
        marks = blk[:, :4].copy().view("<i4").ravel()
        if not np.all(marks == payload):
            raise ValueError(f"{path} 第 {f} 个场的记录标记不是 {payload}")
        out.append(blk[:, 4:4 + payload].copy().view("<f8").reshape(nk, nj, ni))
    return out




def make_or_load_metrics(cache_path, grid, coords_fn=None, dxs=(1.0, 1.0, 1.0),
                         order=6, cartesian=True, jac_bound=(1, 1, 1),
                         periodic=(True, False, True)):
    """度量获取入口：有缓存读缓存，没有则计算并存。

    cartesian=True（默认）直接返回常数度量，不落盘；曲线网格传
    coords_fn() 返回 {"Axx","Ayy","Azz"} 坐标场（含鬼胞）。
    缓存元信息带 jac_bound/periodic，与当前请求不符视为缓存失效。
    """
    if cache_path is not None:
        try:
            from . import io
            fields, meta = io.load_npz(cache_path)
            # 校验缓存与当前 grid 匹配（shape/LAP 不符视为缓存缺失，重算覆写）
            if meta is not None and (meta.get("shape") != list(grid.shape)
                                     or meta.get("LAP") != LAP
                                     or meta.get("jac_bound") != list(jac_bound)):
                fields = None
            if fields is not None:
                return fields
        except (FileNotFoundError, EOFError, ValueError):
            pass   # npz 损坏类异常视为缓存缺失；其余异常让它抛
    if cartesian:
        metrics = make_metrics_cartesian(grid)
    else:
        assert coords_fn is not None, "curvilinear mesh requires coords_fn"
        metrics = compute_metrics_from_coords(coords_fn(), dxs, order,
                                              jac_bound=jac_bound,
                                              periodic=periodic)
    if cache_path is not None:
        from . import io
        io.save_npz(cache_path, metrics,
                    meta={"shape": list(grid.shape), "LAP": LAP,
                          "dxs": list(dxs), "order": order,
                          "jac_bound": list(jac_bound)})
    return metrics
