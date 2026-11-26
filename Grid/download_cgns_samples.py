# -*- coding: utf-8 -*-
"""不用 git 从 CGNS 官方站点下载结构化 CGNS 样例网格。

背景
----
CGNS 官方库在 GitHub（https://github.com/CGNS），但样例网格（.cgns / .cgns.gz）
是挂在 cgns.org 的 ``_downloads`` 目录下的静态文件，用 HTTP 直链就能下，
不需要 clone 仓库、也不需要 git 客户端。

用法
----
    python download_cgns_samples.py            # 下载 DEFAULT 里的结构化样例
    python download_cgns_samples.py --list     # 只打印清单与链接
    python download_cgns_samples.py 5blocks.cgns.gz bump_hdf5.cgns.gz
    python download_cgns_samples.py --all

输出目录：脚本所在目录下的 ``samples/``（已存在则复用）。
下载后自动 gunzip；``.gz`` 保留一份，方便核对来源。

注意
----
* 站点会换版本号目录，链接里的 hash 段（如 ``_downloads/ac7fa3af...``）是固定的，
  但若官方改版，脚本里的 URL 可能失效，届时打开
  https://cgns.org/current/examples.html 重新抄链接即可。
* ADF 格式（老文件，无 ``_hdf5`` 后缀）只能被 CGNS 库读，h5py 读不了；
  想用 h5py 直接检查请下 ``*_hdf5.cgns.gz``。
"""

import argparse
import gzip
import os
import shutil
import ssl
import sys
import urllib.request

BASE = "https://cgns.org/_downloads"

# name -> (hash 段, 说明)
CATALOG = {
    # ---- 结构化，单块 ----
    "bump.cgns.gz": ("f535ec98e2eef06feb6202e171045d6b", "ADF｜单块｜轴对称凸包，格心解"),
    "bump_hdf5.cgns.gz": ("34baf05667ffeecd7203fa4a1eeac93d", "HDF5｜单块｜同上"),
    "delta.cgns.gz": ("3e3b85068444186453786bb99305f597", "ADF｜单块｜三维三角翼，格心解"),
    "delta_vertex.cgns.gz": ("22e92c277b1a35489bf29cd7c7aae475", "ADF｜单块｜同上，顶点解"),
    # ---- 结构化，多块 ----
    "5blocks.cgns.gz": ("ac7fa3af74eef5849010991c56cfb026", "ADF｜5 块｜最经典的多块结构化样例，21 KB"),
    "multi.cgns.gz": ("12d778c06cc3396538e91a661e2954a0", "ADF｜多块｜CFL3D 输出的 4 块网格"),
    "multi_hdf5.cgns.gz": ("f4fd85803e19a6aa9f7fc58ab110f1cd", "HDF5｜多块｜同上，可用 h5py 直接看结构"),
    "oversetnasa1.cgns.gz": ("04cc9bb3357c797e70221d4f817ecdcd", "ADF｜5 块｜1to1 + 重叠连接混合，二维机翼"),
    "oversetnasa2.cgns.gz": ("1f8b964176f01d8517e25df84efc40d6", "ADF｜多块｜重叠网格"),
    "dlr-f6.coar.cgns.gz": ("6b2988b92a91e261e7dfbaae1f2a92c5", "ADF｜26 块｜DLR-F6 翼身组合，55 MB"),
    # ---- 非结构（仅作对照）----
    "yf17_hdf5.cgns.gz": ("65ab7b41b4913156194114239631fee5", "HDF5｜非结构｜YF-17，带边界条件"),
}

DEFAULT = ["5blocks.cgns.gz", "multi_hdf5.cgns.gz"]


def url_of(name):
    return "%s/%s/%s" % (BASE, CATALOG[name][0], name)


def download(name, outdir, keep_gz=True):
    url = url_of(name)
    gz_path = os.path.join(outdir, name)
    req = urllib.request.Request(url, headers={"User-Agent": "cgns-samples-downloader"})
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as r:
        data = r.read()
    with open(gz_path, "wb") as f:
        f.write(data)

    out_path = gz_path[:-3] if name.endswith(".gz") else gz_path
    if name.endswith(".gz"):
        with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        if not keep_gz:
            os.remove(gz_path)
    print("  %-24s %8d B -> %s" % (name, len(data), os.path.basename(out_path)))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="下载 CGNS 官方样例网格（HTTP 直链，不用 git）")
    ap.add_argument("files", nargs="*", help="要下载的文件名，见 --list")
    ap.add_argument("--list", action="store_true", help="只打印清单和链接")
    ap.add_argument("--all", action="store_true", help="下载 CATALOG 里全部条目")
    ap.add_argument("--outdir", default=None, help="输出目录，默认脚本旁的 samples/")
    args = ap.parse_args(argv)

    if args.list:
        print("可用样例（%d 个）：" % len(CATALOG))
        for k, (h, note) in sorted(CATALOG.items()):
            print("  %-24s %-42s %s" % (k, note, url_of(k)))
        return 0

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(outdir, exist_ok=True)

    want = list(CATALOG) if args.all else (args.files or DEFAULT)
    unknown = [w for w in want if w not in CATALOG]
    if unknown:
        print("未知文件名：%s（用 --list 查看可用清单）" % ", ".join(unknown), file=sys.stderr)
        return 2

    print("输出目录：%s" % outdir)
    failed = []
    for name in want:
        try:
            download(name, outdir)
        except Exception as exc:  # 网络/链接失效都归到这里，最后统一报告
            failed.append((name, repr(exc)))
            print("  下载失败 %s: %r" % (name, exc))
    if failed:
        print("\n%d 个失败：" % len(failed))
        for n, e in failed:
            print("  %s %s" % (n, e))
        return 1
    print("\n完成。校验建议：python -c \"import h5py;f=h5py.File(r'%s','r');print(list(f.keys()))\"" % os.path.join(outdir, "multi_hdf5.cgns"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
