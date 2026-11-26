"""对 CGNS 网格做 y 向单侧指数拉伸（壁面加密），拓扑与边条原样保留。

用法：
    python experiment/stretch_cgns_y.py --src Grid/testcgnstest.cgns \
        --out Grid/testcgnstest_ystr.cgns --beta 3.0

y(j) = H·(exp(β·j/(nj-1)) - 1)/(exp(β) - 1)：壁面斜率 = H·β/((nj-1)(e^β-1))
（β=3、H=4、nj=40 时 dy_w≈0.016，均匀的 1/6.4）；顶部变粗（远场可接受）。
"""
import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()

    with h5py.File(args.src, "r") as src, h5py.File(args.out, "w") as dst:
        src.copy("Base", dst)  # 结构、边条、维数原样
        zone_path = None
        base = dst["Base"]
        for name, obj in base.items():
            if obj.attrs.get("label") == np.bytes_("Zone_t"):
                zone_path = f"Base/{name}"
                break
        if zone_path is None:
            raise SystemExit("找不到 Zone_t")
        y = np.asarray(src[f"{zone_path}/GridCoordinates/CoordinateY/ data"])
        shape = y.shape                      # (ni, nj, nk)（CGNS 原始轴序）
        nj = shape[1]
        H = float(y[:, -1, :].max() - y[:, 0, :].min())
        j = np.arange(nj, dtype=np.float64)
        new_y = H * (np.exp(args.beta * j / (nj - 1)) - 1.0) / (np.expm1(args.beta))
        stretched = new_y[None, :, None] + 0.0 * y
        dst[f"{zone_path}/GridCoordinates/CoordinateY/ data"][...] = stretched
        dy_w = new_y[1] - new_y[0]
        dy_top = new_y[-1] - new_y[-2]
        print(f"nj={nj} H={H:.3f} β={args.beta}: dy_wall={dy_w:.4f} "
              f"dy_top={dy_top:.4f}（均匀时 {H/(nj-1):.4f}）")


if __name__ == "__main__":
    main()
