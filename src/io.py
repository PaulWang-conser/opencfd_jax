"""IO：npz 快速读写（自用缓存/存档）+ 原版 OCFD dat reader（对拍用）。

快速格式：未压缩 npz（numpy 原生、零依赖、顺序读取快）。度量缓存、
检查点存档都走这里。字段以 key 存取，附 `_meta` json 记录 shape/LAP/dtype。

原版 dat 格式（OCFD3d-Jacobi.dat / OCFD3d-Mesh.dat，OCFD_Comput_Jacobian3d.cu:119-175
与 OCFD_IO.c 的 read_3d1）：字段序列逐个存放，每场 = 逐 z 层堆叠，
每层前置 2 个 int（MPI 并行写的层尺寸头），随后 NX_GLOBAL*NY_GLOBAL 个 REAL
（double，C 序 x 最快）。字段顺序（Jacobi.dat）：Axx, Ayy, Azz, Akx, Aky,
Akz, Aix, Aiy, Aiz, Asx, Asy, Asz, Ajac（末尾字段以基准文件实测为准）。
"""

import json

import numpy as np


def save_npz(path, fields: dict, meta: dict | None = None):
    """未压缩 npz 存档。fields: {name: ndarray}; meta 任意可 json 化对象。"""
    arrays = {k: np.asarray(v) for k, v in fields.items()}
    if meta is not None:
        arrays["_meta"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)


def load_npz(path) -> tuple[dict, dict | None]:
    """读 npz，返回 (fields, meta)。_meta 若存在则解析为 dict 并从 fields 剔除。"""
    data = np.load(path)
    fields = {}
    meta = None
    for k in data.files:
        if k == "_meta":
            meta = json.loads(data[k].tobytes().decode("utf-8"))
        else:
            fields[k] = data[k]
    return fields, meta


def read_ocfd_dat(path, shape: tuple[int, int, int], n_fields: int,
                  dtype=np.float64, offset: int = 0):
    """读原版 OCFD dat 文件（MPI 并行写格式）为全场数组列表。

    布局：逐场、逐 z 层，每层 = 2 个 int 头 + nx*ny 个 dtype（x 最快）。
    shape 为全局 (nx, ny, nz)。返回 [n_fields 个 (nz, ny, nx) numpy 数组]。
    注意：2 int 头的确切含义（层尺寸/哨兵）待首个基准文件核对后补充断言；
    层头大小 8 字节由 OCFD_Comput_Jacobian3d.cu:127 的 offset 步进公式
    (2*sizeof(int) + NX*NY*sizeof(REAL))*NZ 得出。
    """
    nx, ny, nz = shape
    data_bytes = nx * ny * np.dtype(dtype).itemsize
    arrays = []
    with open(path, "rb") as fh:
        fh.seek(offset)
        for _ in range(n_fields):
            field = np.empty((nz, ny, nx), dtype=dtype)
            for k in range(nz):
                fh.seek(8, 1)                          # 跳过层头（2 int）
                layer = np.fromfile(fh, dtype=dtype, count=nx * ny)
                field[k] = layer.reshape(ny, nx)
            arrays.append(field)
    return arrays
