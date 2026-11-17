#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tecplot ASCII (.tec/.dat) -> Tecplot 二进制 (.plt, 旧版 v112) 转换器

不依赖 Tecplot 安装（不调用 preplot / tecplot.exe），完全按二进制格式规范手写字节流。
参照 tecplot_data_format.pdf《Tecplot Data 格式说明文档》第 2 章（Binary Data）：

    HEADER SECTION
        "#!TDVnnn"                     8 字节魔数 + 版本号（v112）
        INT32  1                       字节序标志（小端写入 1）
        INT32  FileType                0=FULL 1=GRID 2=SOLUTION
        TITLE                          逐字符 INT32 + 结尾 0（不写长度计数字段）
        INT32  NumVars
        变量名                         逐字符 INT32 + 结尾 0
        每个 zone：
            FLOAT32 299.0              zone marker
            zone 名                    逐字符 INT32 + 结尾 0
            INT32   ParentZone         （已废弃，写 -1）
            INT32   StrandID           -1=static
            FLOAT64 SolutionTime
            INT32   默认颜色           （已废弃，写 -1）
            INT32   ZoneType           0=ORDERED
            INT32   SpecifyVarLocation 0=全部在节点上(NODAL)
            INT32   是否存在 1-to-1 邻接
            INT32   额外 user-defined 邻接条数
            INT32*3 IMax JMax KMax     （ORDERED）
            AUXDATA 循环：INT32 1 + 名字 + INT32 0 + 值 ... INT32 0 结束
        FLOAT32 357.0                  EOHMARKER，头段结束

    DATA SECTION（每个 zone 一段）
        FLOAT32 299.0
        INT32*NV 每个变量的数据类型     1=Float 2=Double
        INT32  是否有 passive 变量
        INT32  是否有变量共享
        INT32  共享 connectivity 的 zone 号（-1=不共享）
        FLOAT64*2*NV 每个变量的 min/max
        数据本体（BLOCK：变量优先，i 变化最快）

用法：
    python tec_to_plt.py outputest/case_iiA_Istep00054000.tec
    python tec_to_plt.py in.tec -o out.plt --precision float64
    python tec_to_plt.py --info out.plt
"""

import argparse
import os
import re
import struct
import sys
import time

import numpy as np

ZONE_MARKER = 299.0     # v112 的 zone marker
EOH_MARKER = 357.0      # header section 结束标记
VERSION = 112

TECPLOT_DT = {1: "Float", 2: "Double", 3: "LongInt", 4: "ShortInt", 5: "Byte", 6: "Bit"}

# ---------------------------------------------------------------- ASCII 解析


def _is_numeric_line(s):
    s = s.strip()
    if not s or "=" in s or s.startswith("#"):
        return False
    return bool(re.fullmatch(r"[0-9eE+\-.,\s]+", s)) and any(c.isdigit() for c in s)


def _take_record(lines, i):
    """Tecplot 允许一条记录用行尾的逗号或反斜杠续行。"""
    parts = [lines[i].rstrip()]
    while parts[-1].rstrip().endswith((",", "\\")):
        parts[-1] = parts[-1].rstrip()[:-1]
        i += 1
        if i >= len(lines):
            break
        parts.append(lines[i].rstrip())
    return " ".join(parts).strip(), i + 1


def _parse_numbers(text):
    text = text.strip()
    if not text:
        return np.empty(0, dtype=np.float64)
    try:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = np.fromstring(text, dtype=np.float64, sep=" ")
        if a.size:
            return a
    except Exception:
        pass
    return np.array(text.split(), dtype=np.float64)


_KW_RE = re.compile(r"^\s*(ZONE|TITLE|VARIABLES|FILETYPE|DATASETAUXDATA|VARAUXDATA|"
                    r"TEXT|GEOMETRY|CUSTOMLABELS|VARSHARELIST|CONNECTIVITYSHAREZONE)\b", re.I)


def parse_tec(path):
    """解析 ASCII .tec，返回 {title, filetype, variables, dataset_aux, zones[...]}"""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    out = {"title": "", "filetype": 0, "variables": [], "dataset_aux": [], "zones": []}
    n_lines = len(lines)
    i = 0
    while i < n_lines:
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        m = _KW_RE.match(s)
        if not m:
            raise ValueError("第 %d 行无法识别的记录：%r" % (i + 1, s[:80]))
        kw = m.group(1).upper()

        if kw == "ZONE":
            rec, i = _take_record(lines, i)
            zone = _parse_zone_header(rec)
            # 收集紧随其后的数据行
            buf = []
            while i < n_lines:
                ln = lines[i]
                if ln.strip() and not _is_numeric_line(ln):
                    break
                if ln.strip():
                    buf.append(ln.strip())
                i += 1
            # 若上一条数字行的行末把数字截断了（如 "...E" 换行 "+000"），拼回去
            text = ""
            for ln in buf:
                if text and text[-1:] in ("e", "E", "+", "-"):
                    text += ln
                else:
                    text += " " + ln
            vals = _parse_numbers(text)
            need = len(out["variables"]) * zone["npts"]
            if vals.size != need:
                raise ValueError("zone %r 数据个数不对：读到 %d，应为 %d"
                                 % (zone["name"], vals.size, need))
            zone["data"] = vals
            out["zones"].append(zone)
            continue

        rec, i = _take_record(lines, i)
        if kw == "TITLE":
            mm = re.search(r'"([^"]*)"', rec)
            out["title"] = mm.group(1) if mm else rec.split("=", 1)[-1].strip()
        elif kw == "VARIABLES":
            body = rec.split("=", 1)[1] if "=" in rec else rec
            out["variables"] = re.findall(r'"([^"]*)"', body) or body.replace(",", " ").split()
        elif kw == "FILETYPE":
            ft = rec.split("=", 1)[-1].strip().upper()
            out["filetype"] = {"FULL": 0, "GRID": 1, "SOLUTION": 2}.get(ft, 0)
        elif kw == "DATASETAUXDATA":
            out["dataset_aux"].extend(_parse_aux(rec))
        elif kw == "VARAUXDATA":
            print("  [warn] 忽略 VARAUXDATA（未实现）: %s" % rec[:80])
        else:
            print("  [warn] 忽略 %s 段（本转换器只处理数据）: %s" % (kw, rec[:60]))

    if not out["variables"]:
        raise ValueError("没有解析到 VARIABLES")
    if not out["zones"]:
        raise ValueError("没有解析到 ZONE")
    return out


_AUX_RE = re.compile(r'(?:\bAUXDATA|\bDATASETAUXDATA)\s+([A-Za-z_][\w.]*)\s*=\s*'
                     r'(?:"([^"]*)"|([^,\s]+))', re.I)


def _parse_aux(rec):
    """AUXDATA name="value" / name=812 -> [(name, value_str), ...]"""
    pairs = []
    for m in _AUX_RE.finditer(rec):
        val = m.group(2) if m.group(2) is not None else m.group(3)
        pairs.append((m.group(1), val))
    return pairs


def _parse_zone_header(rec):
    z = {"name": "ZONE", "zonetype": "ORDERED", "datapacking": "BLOCK",
         "i": 1, "j": 1, "k": 1, "npts": 0, "nelem": 0,
         "solution_time": 0.0, "strand_id": -1, "aux": [], "varlocation": None,
         "dt": None, "nfaces": 0}

    mm = re.search(r'T\s*=\s*"([^"]*)"', rec, re.I)
    if mm:
        z["name"] = mm.group(1)

    def num(key):
        mm = re.search(r'(?<![A-Za-z])%s\s*=\s*([0-9]+)' % key, rec, re.I)
        return int(mm.group(1)) if mm else None

    zt = re.search(r'ZONETYPE\s*=\s*([A-Za-z]+)', rec, re.I)
    if zt:
        z["zonetype"] = zt.group(1).upper()
    dp = re.search(r'DATAPACKING\s*=\s*([A-Za-z]+)', rec, re.I)
    if dp:
        z["datapacking"] = dp.group(1).upper()

    imax, jmax, kmax = num("I"), num("J"), num("K")
    npts, nelem = num("NODES"), num("ELEMENTS")
    if imax is not None:
        z["i"], z["j"], z["k"] = imax, jmax or 1, kmax or 1
        z["npts"] = z["i"] * z["j"] * z["k"]
    elif npts is not None:
        z["npts"], z["nelem"] = npts, nelem or 0

    st = re.search(r'SOLUTIONTIME\s*=\s*([-+0-9.eE]+)', rec, re.I)
    if st:
        z["solution_time"] = float(st.group(1))
    sid = re.search(r'STRANDID\s*=\s*(-?\d+)', rec, re.I)
    if sid:
        z["strand_id"] = int(sid.group(1))
    z["aux"] = _parse_aux(rec)

    dt = re.search(r'DT\s*=\s*\(([^)]*)\)', rec, re.I)
    if dt:
        z["dt"] = [t.strip().upper() for t in dt.group(1).replace(",", " ").split() if t.strip()]

    vl = re.search(r'VARLOCATION\s*=\s*\(\(([^)]*)\)\s*=\s*([A-Za-z]+)\)', rec, re.I)
    if vl:
        z["varlocation"] = (vl.group(1), vl.group(2).upper())
        if z["varlocation"][1] != "NODAL":
            raise NotImplementedError("暂不支持 CELLCENTERED 变量（需要按 ghost 值规则补零），"
                                      "zone %r: %s" % (z["name"], vl.group(0)))
    return z


# ---------------------------------------------------------------- 二进制写出


def _w_i32(buf, v):
    buf.extend(struct.pack("<i", int(v)))


def _w_f32(buf, v):
    buf.extend(struct.pack("<f", float(v)))


def _w_f64(buf, v):
    buf.extend(struct.pack("<d", float(v)))


def _w_str(buf, s):
    """字符串：逐字符 INT32(ASCII) + 结尾 0，返回写入的字符数 N。"""
    arr = np.array([ord(c) for c in s] + [0], dtype="<i4")
    buf.extend(arr.tobytes())
    return arr.size


def write_plt(tec, path, precision="auto"):
    """把 parse_tec 的结果写成 v112 二进制 .plt"""
    variables = tec["variables"]
    nvar = len(variables)
    zones = tec["zones"]

    if precision == "auto":
        # ASCII 里 DT=(SINGLE ...) 或缺省 -> 单精度；全 DOUBLE -> 双精度
        single = True
        for z in zones:
            dt = z.get("dt")
            if dt and all(t.startswith("DOUBLE") for t in dt):
                single = False
        precision = "float32" if single else "float64"
    fmt_code = 1 if precision == "float32" else 2
    np_dt = np.dtype("<f4" if precision == "float32" else "<f8")

    buf = bytearray()
    t0 = time.time()

    # ---------- HEADER SECTION ----------
    buf.extend(("#!TDV%03d" % VERSION).encode("ascii"))   # 8 字节
    _w_i32(buf, 1)                                        # 字节序
    _w_i32(buf, tec["filetype"])                          # FileType
    _w_str(buf, tec["title"])
    _w_i32(buf, nvar)
    for v in variables:
        _w_str(buf, v)

    for z in zones:
        if z["zonetype"] != "ORDERED":
            raise NotImplementedError("目前只支持 ORDERED zone，遇到 %s" % z["zonetype"])
        _w_f32(buf, ZONE_MARKER)
        _w_str(buf, z["name"])
        _w_i32(buf, -1)                     # ParentZone（已废弃）
        _w_i32(buf, z["strand_id"])         # -1 = static strand
        _w_f64(buf, z["solution_time"])
        _w_i32(buf, -1)                     # 默认颜色（已废弃）
        _w_i32(buf, 0)                      # ZoneType = ORDERED
        _w_i32(buf, 0)                      # SpecifyVarLocation = 0 -> 全部 NODAL
        _w_i32(buf, 0)                      # raw local 1-to-1 neighbors
        _w_i32(buf, 0)                      # misc user-defined neighbor 条数
        _w_i32(buf, z["i"])
        _w_i32(buf, z["j"])
        _w_i32(buf, z["k"])
        for nm, val in z["aux"]:
            _w_i32(buf, 1)                  # 还有 auxiliary
            _w_str(buf, nm)
            _w_i32(buf, 0)                  # 值格式：0 = 字符串
            _w_str(buf, val)
        _w_i32(buf, 0)                      # aux 结束

    if tec["dataset_aux"]:
        _w_f32(buf, 799.0)                  # DataSetAux marker
        for nm, val in tec["dataset_aux"]:
            _w_str(buf, nm)
            _w_i32(buf, 0)
            _w_str(buf, val)

    _w_f32(buf, EOH_MARKER)                 # 357.0，头段结束

    # ---------- DATA SECTION ----------
    for z in zones:
        if z["datapacking"] == "POINT":
            arr = np.ascontiguousarray(z["data"].reshape(z["npts"], nvar).T)
        else:
            arr = z["data"].reshape(nvar, z["npts"])      # BLOCK：变量优先，i 最快

        data_out = np.array(arr, dtype=np_dt, order="C", copy=True)

        _w_f32(buf, ZONE_MARKER)
        for _ in range(nvar):
            _w_i32(buf, fmt_code)
        _w_i32(buf, 0)                      # 无 passive 变量
        _w_i32(buf, 0)                      # 无变量共享
        _w_i32(buf, -1)                     # 不共享 connectivity
        for k in range(nvar):
            col = data_out[k].astype(np.float64)
            _w_f64(buf, col.min())
            _w_f64(buf, col.max())
        buf.extend(data_out.tobytes(order="C"))

    with open(path, "wb") as fh:
        fh.write(buf)

    return {"precision": precision, "fmt_code": fmt_code, "np_dtype": np_dt,
            "bytes": len(buf), "seconds": time.time() - t0}


# ---------------------------------------------------------------- 二进制读回


class _Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def i32(self):
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def f64(self):
        v = struct.unpack_from("<d", self.d, self.p)[0]
        self.p += 8
        return v

    def string(self):
        codes = []
        while True:
            c = self.i32()
            if c == 0:
                break
            codes.append(c)
        return "".join(chr(c) for c in codes)

    def peek_f32(self):
        v = struct.unpack_from("<f", self.d, self.p)[0]
        return v

    def eof(self):
        return self.p >= len(self.d)


def read_plt(path):
    """严格按 v112 规范读回 .plt，用于自校验/结构打印"""
    with open(path, "rb") as fh:
        data = fh.read()
    r = _Reader(data)

    magic = data[0:8].decode("ascii", "replace")
    r.p = 8
    version = int(magic[5:8])
    assert magic.startswith("#!TDV"), "魔数不对：%r" % magic
    byte_order = r.i32()
    assert byte_order == 1, "字节序标志=%d" % byte_order
    filetype = r.i32()
    title = r.string()
    nvar = r.i32()
    varnames = [r.string() for _ in range(nvar)]

    info = {"magic": magic, "version": version, "filetype": filetype,
            "title": title, "nvar": nvar, "variables": varnames, "zones": []}

    # zone headers
    while not r.eof() and abs(r.peek_f32() - ZONE_MARKER) < 1e-6:
        r.f32()
        z = {"name": r.string(), "marker": ZONE_MARKER}
        z["parent"] = r.i32()
        z["strand"] = r.i32()
        z["solution_time"] = r.f64()
        z["color"] = r.i32()
        z["zonetype"] = r.i32()
        specify_loc = r.i32()
        if specify_loc == 1:
            z["varlocation"] = [r.i32() for _ in range(nvar)]
        z["raw_neighbors"] = r.i32()
        nmisc = r.i32()
        if nmisc != 0:
            z["misc_mode"] = r.i32()
        if z["zonetype"] != 0:
            r.i32()                       # FE: completely specified flag
            npts = r.i32()
            if z["zonetype"] in (6, 7):
                raise NotImplementedError("FE polygon/polyhedron 未实现")
            z["npts"] = npts
            z["nelem"] = r.i32()
            z["cell_dims"] = [r.i32(), r.i32(), r.i32()]
        else:
            z["i"], z["j"], z["k"] = r.i32(), r.i32(), r.i32()
            z["npts"] = z["i"] * z["j"] * z["k"]
        z["aux"] = []
        while r.i32() == 1:
            nm = r.string()
            vfmt = r.i32()
            z["aux"].append((nm, r.string(), vfmt))
        z["nvars"] = nvar
        info["zones"].append(z)

    # optional sections before EOH
    while not r.eof():
        marker = r.peek_f32()
        if abs(marker - EOH_MARKER) < 1e-6:
            r.f32()
            info["eoh"] = EOH_MARKER
            break
        if abs(marker - 799.0) < 1e-6:
            r.f32()
            info["dataset_aux"] = []
            # 规范：重复 INT32 1 + name + fmt + value
            while r.i32() == 1:
                info["dataset_aux"].append((r.string(), r.i32(), r.string()))
        elif abs(marker - 899.0) < 1e-6:
            r.f32()
            info["var_aux"] = []
            while r.i32() == 1:
                info["var_aux"].append((r.i32(), r.string(), r.i32(), r.string()))
        elif abs(marker - 599.0) < 1e-6:
            r.f32()
            nl = r.i32()
            info["custom_labels"] = [r.string() for _ in range(nl)]
        elif abs(marker - 699.0) < 1e-6:
            r.f32()
            info["user_rec"] = r.string()
        else:
            raise ValueError("头段出现未知标记 %.3f（offset=%d）" % (marker, r.p))

    # data sections
    for k, z in enumerate(info["zones"]):
        marker = r.f32()
        assert abs(marker - ZONE_MARKER) < 1e-6, "data section marker=%.3f" % marker
        fmts = [r.i32() for _ in range(nvar)]
        z["data_formats"] = fmts
        has_passive = r.i32()
        if has_passive:
            z["passive"] = [r.i32() for _ in range(nvar)]
        has_sharing = r.i32()
        if has_sharing:
            z["sharing"] = [r.i32() for _ in range(nvar)]
        z["connectivity_shared_with"] = r.i32()
        z["minmax"] = [(r.f64(), r.f64()) for _ in range(nvar)]

        npts = z["npts"]
        arr = np.empty((nvar, npts), dtype="<f4" if fmts[0] == 1 else "<f8")
        for v in range(nvar):
            dt = "<f4" if fmts[v] == 1 else "<f8" if fmts[v] == 2 else None
            if dt is None:
                raise NotImplementedError("data format %d 未实现" % fmts[v])
            nbytes = npts * (4 if fmts[v] == 1 else 8)
            arr[v] = np.frombuffer(data, dtype=dt, count=npts, offset=r.p)
            r.p += nbytes
        z["data"] = arr

    info["trailing_bytes"] = len(data) - r.p
    info["file_bytes"] = len(data)
    return info


# ---------------------------------------------------------------- 校验与打印


def verify(tec, plt_path, zp):
    res = read_plt(plt_path)

    problems = []
    if res["title"] != tec["title"]:
        problems.append("TITLE 不一致: %r vs %r" % (res["title"], tec["title"]))
    if res["variables"] != tec["variables"]:
        problems.append("变量名不一致")
    if len(res["zones"]) != len(tec["zones"]):
        problems.append("zone 数不一致")

    print("  文件大小 : %.3f MB (%d 字节)" % (res["file_bytes"] / 1e6, res["file_bytes"]))
    print("  版本/尾随 : %s, 尾随字节 %d" % (res["magic"], res["trailing_bytes"]))
    if res["trailing_bytes"] != 0:
        problems.append("数据后有 %d 字节未消费" % res["trailing_bytes"])

    for zs, zt in zip(res["zones"], tec["zones"]):
        src = zt["data"].reshape(len(tec["variables"]), zt["npts"])
        got = zs["data"]
        diff = np.abs(got.astype(np.float64) - src)
        rel = diff / np.maximum(1.0, np.abs(src))
        exact = np.array_equal(got, np.array(src, dtype=got.dtype))
        print("  zone %-14r IJK=%d/%d/%d npts=%d  aux=%d 条"
              % (zs["name"], zs["i"], zs["j"], zs["k"], zs["npts"], len(zs["aux"])))
        print("           类型=%s  长度=%s  min/max 表=%d 对"
              % (TECPLOT_DT.get(zs["data_formats"][0], zs["data_formats"][0]),
                 got.dtype, len(zs["minmax"])))
        print("           与源数据比对: max|Δ|=%.3e, max rel=%.3e, 逐位一致=%s"
              % (diff.max(), rel.max(), exact))
        if not exact:
            problems.append("zone %r 数值与源文件不完全一致" % zs["name"])
        # min/max 表是否与真实数据一致
        for v in range(len(tec["variables"])):
            col = got[v].astype(np.float64)
            if abs(zs["minmax"][v][0] - col.min()) > 0 or abs(zs["minmax"][v][1] - col.max()) > 0:
                problems.append("zone %r 变量 %d 的 min/max 表不符" % (zs["name"], v))
        if zs["aux"] != [(nm, val, 0) for nm, val in zt["aux"]]:
            problems.append("zone %r aux 不一致: %r" % (zs["name"], zs["aux"]))
        if abs(zs["solution_time"] - zt["solution_time"]) > 0:
            problems.append("zone %r SOLUTIONTIME 不一致" % zs["name"])

    # 结构体量自洽：文件大小 = 已消费字节
    print("  结构自检 : %s" % ("通过" if not problems else "失败"))
    for p in problems:
        print("    !! " + p)
    return res, problems


def print_info(plt_path):
    info = read_plt(plt_path)
    print("文件      : %s (%.3f MB)" % (plt_path, info["file_bytes"] / 1e6))
    print("魔数/版本 : %s / %d" % (info["magic"], info["version"]))
    print("FileType  : %d (0=FULL 1=GRID 2=SOLUTION)" % info["filetype"])
    print("TITLE     : %s" % info["title"])
    print("变量(%d)  : %s" % (info["nvar"], " ".join(info["variables"])))
    print("EOHMARKER : %s" % info.get("eoh"))
    print("数据集 aux: %s" % info.get("dataset_aux"))
    for z in info["zones"]:
        print("zone %r: %s IJK=%s npts=%d aux=%s soltime=%g strand=%d"
              % (z["name"], "ORDERED" if z["zonetype"] == 0 else "FE%d" % z["zonetype"],
                 (z["i"], z["j"], z["k"]) if z["zonetype"] == 0 else z["npts"],
                 z["npts"], z["aux"], z["solution_time"], z["strand"]))
        print("   data format: %s  minmax 前 3: %s"
              % ([TECPLOT_DT.get(f, f) for f in z.get("data_formats", [])][:6],
                 z.get("minmax", [])[:3]))
    print("尾随字节  : %d" % info["trailing_bytes"])


def main():
    ap = argparse.ArgumentParser(description="Tecplot ASCII .tec -> 二进制 .plt (v112)")
    ap.add_argument("input", help="输入 .tec/.dat（或配合 --info 传入 .plt）")
    ap.add_argument("-o", "--output", default=None, help="输出 .plt 路径")
    ap.add_argument("--precision", choices=["auto", "float32", "float64"], default="auto",
                    help="auto=按 DT 判定（缺省 SINGLE -> float32）")
    ap.add_argument("--info", action="store_true", help="只读回 .plt 打印结构")
    args = ap.parse_args()

    if args.info:
        print_info(args.input)
        return

    src = os.path.abspath(args.input)
    dst = os.path.abspath(args.output) if args.output else os.path.splitext(src)[0] + ".plt"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    print("读取 %s ..." % src)
    t0 = time.time()
    tec = parse_tec(src)
    print("  TITLE: %s" % tec["title"])
    print("  变量 %d 个: %s" % (len(tec["variables"]), " ".join(tec["variables"])))
    for z in tec["zones"]:
        print("  zone %r: I=%d J=%d K=%d, %s, %s, 点数 %d, aux %d 条, t=%g"
              % (z["name"], z["i"], z["j"], z["k"], z["zonetype"], z["datapacking"],
                 z["npts"], len(z["aux"]), z["solution_time"]))
    print("  解析 %.1f s" % (time.time() - t0))

    t0 = time.time()
    wp = write_plt(tec, dst, args.precision)
    print("写出 %s  (%.3f MB, %s, %.1f s)"
          % (dst, wp["bytes"] / 1e6, wp["precision"], wp["seconds"]))

    print("读回校验:")
    verify(tec, dst, wp)


if __name__ == "__main__":
    sys.exit(main())
