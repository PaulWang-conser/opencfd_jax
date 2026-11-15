"""对流通量分裂与界面重构格式。

包含 Steger–Warming 正/负通量分裂，以及 UP7、WENO7-SYMBO、NND2、WENO5、
WENO7 和 OMP6 系列格式。各格式先把格点分裂通量重构到单元界面，再由
``solver.py`` 对相邻界面通量作差，形成守恒形式的无粘右端项。

面通量约定：面 j 位于 cell j-1 与 cell j 之间（cell 为含鬼胞的全局索引，
cell c 对应数组索引 c）。重构以面左邻 cell j-1 为基准，模板偏移 k=-3..3：
    F[j] = sum_k coeff[k] * f[(j-1)+k]
UP7 正/负通量共用该约定，权重互为镜像（推导见 tests/test_up7.py）。
"""

import jax.numpy as jnp

from .fields import LAP

# OCFD_Schemes.cu:870（__device__ REAL OCFD_UP7_kernel_P）
# 模板 f[i-3..i+3]（相对面 i+1/2 左邻 cell i 的偏移 k=-3..3），重构 F+_{i+1/2}
UP7_P_COEFFS = (-3.0, 25.0, -101.0, 319.0, 214.0, -38.0, 4.0)   # /420
UP7_P_OFF_LO = -3

# OCFD_Schemes.cu:905（__device__ REAL OCFD_UP7_kernel_M）
# 权重作用于 stencil[1..7] 即 f[i-3..i+3]：从左到右 (4,-38,214,319,-101,25,-3)/420，
# 相对面 j-1/2 左邻 cell j-1 的偏移 k=-2..4，重构 F-_{j-1/2}
UP7_M_COEFFS = (4.0, -38.0, 214.0, 319.0, -101.0, 25.0, -3.0)   # /420
UP7_M_OFF_LO = -2

_UP7_DENOM = 420.0


def _face_flux_along_last_axis(g, coeffs, off_lo=-3):
    """沿 axis=-1 的 7 点迎风面通量。

    g: (..., nx+2*LAP)，含鬼胞。coeffs 从左到右对应模板偏移
    off_lo..off_lo+6（相对面左邻 cell）。
    返回 (..., nx+1)：紧凑面数组，索引 j 对应物理面 j（cell j-1|j 之间），
    j=0..nx 覆盖内区全部左右面。模板 cell j-1+k 的数组索引为
    j-1+k+LAP，P（k=-3..3）与 M（k=-2..4）的需求恰好落在 LAP=4 鬼胞内
    （原码内区 kernel 靠 job 缩进 [2LAP, nx-LAP) 达到同样效果，单卡版
    不需要缩进）。
    """
    nx = g.shape[-1] - 2 * LAP
    lo, hi = 0, nx + 1                  # 输出面窗口（长 nx+1，紧凑索引）
    acc = 0.0
    for k, c in enumerate(coeffs):
        off = off_lo + k
        acc = acc + c * g[..., lo - 1 + off + LAP: hi - 1 + off + LAP]
    return acc / _UP7_DENOM


def _du_from_faces(F):
    """由面通量做内区差分。

    F 是紧凑面数组（长 nx+1，索引 0 对应全局面 LAP，即内区第一个 cell 的
    左面）。du[c] = -(F[c+1] - F[c])，c=0..nx-1 对应内区 cell。
    """
    return -(F[..., 1:] - F[..., :-1])


def up7_rhs_along_last_axis(fp, fm, h=1.0, bound=None):
    """单方向 UP7 通量差分。

    fp/fm: (..., nx+2*LAP) 已填好鬼胞的正/负分裂通量场。
    F_{j-1/2} = P 重构(fp)[j] + M 重构(fm)[j]；du = -(F_{j+1/2}-F_{j-1/2})/h
    （符号与 1/h 因子对应 OCFD_Schemes.cu:306 的
    -Ajac*(tmp_r-tmp_l)/hx_d，直角均匀网格 Ajac=1）。
    bound: None 或 (lo, hi)，两侧是否按 DIF_BOUND 走原码边界带。
    返回 (..., nx)。
    """
    Fp = _face_flux_along_last_axis(fp, UP7_P_COEFFS, UP7_P_OFF_LO)
    Fm = _face_flux_along_last_axis(fm, UP7_M_COEFFS, UP7_M_OFF_LO)
    if bound is not None and (bound[0] or bound[1]):
        Fp, Fm = flux_band_faces(Fp, Fm, fp, fm, lo=bool(bound[0]), hi=bool(bound[1]))
    return _du_from_faces(Fp + Fm) / h


def steger_warming(cons, gamma, epsl_sw=0.0, axis=0, metric=None):
    """Steger-Warming 通量分裂。

    照搬 OCFD_split.cu:92-142（Stager_Warming_ker）与 OCFD_init.cu:388
    （tmp0 = 1/(2*gamma), split_C1 = 2*(gamma-1), split_C3 = (3-gamma)/(2*(gamma-1))）。

    metric: None → 直角坐标下沿 axis 的单位向量（ss=1，与旧行为逐位相同）；
            或 (A1, A2, A3) 三元组（该方向的**余子式场**，即原码传进
            `Stager_Warming_ker` 的 `Akx/Aky/Akz`、`Aix/Aiy/Aiz`、`Asx/Asy/Asz`）。
            此时 `ss = |A|` 进入 E1/E2/E3，等于把**面面积**装进分裂通量——
            原码 `E1 = A1u+A2v+A3w`、`fp = tmp0(C1·E1P + E2P + E3P)` 都带 ss，
            所以面通量是"面积 × 单位法向通量"。缺这一步则 du 会小 ss 倍
            （见 patches/main_dump.c 与 ocfd_p1_1 的线性场定标）。

    cons: (5, nz+2L, ny+2L, nx+2L) 守恒量，axis: 0=x, 1=y, 2=z。
    返回 (fp, fm)，各 (5, ...)，cell 中心的分裂通量（鬼胞为任意值，
    调用方负责分裂后的鬼胞填充）。
    """
    d, rhou, rhov, rhow, rhoE = cons
    # 平板物理边界存在全零状态鬼层；原码从派生量计算，零状态的通量为零。
    empty = (d == 0.0) & (rhou == 0.0) & (rhov == 0.0) & (rhow == 0.0) & (rhoE == 0.0)
    safe_d = jnp.where(empty, 1.0, d)
    u = rhou / safe_d
    v = rhov / safe_d
    w = rhow / safe_d
    E = rhoE / safe_d
    vv2 = u * u + v * v + w * w
    p = (gamma - 1.0) * d * (E - 0.5 * vv2)
    cc = jnp.sqrt(gamma * p / safe_d)

    if metric is None:
        # 度量方向单位向量 A（直角坐标）
        if axis == 0:
            A1, A2, A3 = 1.0, 0.0, 0.0
        elif axis == 1:
            A1, A2, A3 = 0.0, 1.0, 0.0
        else:
            A1, A2, A3 = 0.0, 0.0, 1.0
    else:
        A1, A2, A3 = metric

    ss = jnp.sqrt(A1 * A1 + A2 * A2 + A3 * A3)
    E1 = A1 * u + A2 * v + A3 * w
    E2 = E1 - cc * ss
    E3 = E1 + cc * ss
    ss_inv = 1.0 / ss
    A1m, A2m, A3m = A1 * ss_inv, A2 * ss_inv, A3 * ss_inv

    eps2 = epsl_sw * epsl_sw
    E1P = 0.5 * (E1 + jnp.sqrt(E1 * E1 + eps2))
    E2P = 0.5 * (E2 + jnp.sqrt(E2 * E2 + eps2))
    E3P = 0.5 * (E3 + jnp.sqrt(E3 * E3 + eps2))
    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    uc1 = u - cc * A1m
    uc2 = u + cc * A1m
    vc1 = v - cc * A2m
    vc2 = v + cc * A2m
    wc1 = w - cc * A3m
    wc2 = w + cc * A3m
    vvc1 = 0.5 * (uc1 * uc1 + vc1 * vc1 + wc1 * wc1)
    vvc2 = 0.5 * (uc2 * uc2 + vc2 * vc2 + wc2 * wc2)
    vvm = (gamma - 1.0) * vv2
    W2 = (3.0 - gamma) / (2.0 * (gamma - 1.0)) * cc * cc   # split_C3 * cc^2
    C1 = 2.0 * (gamma - 1.0)                               # split_C1
    tmp0 = 1.0 / (2.0 * gamma)
    tmp0 = d * tmp0

    fp = jnp.stack([
        tmp0 * (C1 * E1P + E2P + E3P),
        tmp0 * (C1 * E1P * u + E2P * uc1 + E3P * uc2),
        tmp0 * (C1 * E1P * v + E2P * vc1 + E3P * vc2),
        tmp0 * (C1 * E1P * w + E2P * wc1 + E3P * wc2),
        tmp0 * (E1P * vvm + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)),
    ])
    fm = jnp.stack([
        tmp0 * (C1 * E1M + E2M + E3M),
        tmp0 * (C1 * E1M * u + E2M * uc1 + E3M * uc2),
        tmp0 * (C1 * E1M * v + E2M * vc1 + E3M * vc2),
        tmp0 * (C1 * E1M * w + E2M * wc1 + E3M * wc2),
        tmp0 * (E1M * vvm + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M)),
    ])
    return fp, fm


# ============================ WENO7-SYMBO ==================================
# Martin et al, JCP 220 (2006) 带宽优化对称 WENO（照搬 OCFD_Schemes.cu:387-592）。
# P：8 点模板 f[i-3..i+4]（off -3..4 相对面左邻 cell i，面位置 i+0.5）；
# M 为镜像（模板 off -4..3），通过对称翻转复用同一实现。
# TV 快速路径（WENO_LMT_FLAG=1）：光滑区（TVmax < Limiter*TVmin 且 TVmax < TV_MAX）
# 直接用线性权 dr；否则完整 WENO 权。eps = 2.592e-6（原码值）。

WENO7_DR = (0.0401954833730, 0.2493800006710, 0.4802686256260,
            0.2009775476730, 0.0291783426580)
WENO7_EPS = 2.592e-6
WENO7_TV_LIMITER_DEFAULT = 5.0
WENO7_TV_MAX_DEFAULT = 0.2


def _weno7_symbo_face_P(g, tv_limiter=None, tv_max_cap=None):
    """WENO7-SYMBO P 面重构（沿末轴）。g: (..., nx+2*LAP)。

    返回 (..., nx+1) 紧凑面数组（索引 j = 物理面 j，模板 cell j-1+k, k=-3..4）。
    tv_limiter/tv_max_cap 为 None 时不启用快速路径（等价原码 LMT_FLAG=0）。
    """
    nx = g.shape[-1] - 2 * LAP
    lo, hi = 0, nx + 1
    off_lo = -3
    s = [g[..., lo - 1 + off_lo + i + LAP: hi - 1 + off_lo + i + LAP]
         for i in range(8)]
    s0, s1, s2, s3, s4, s5, s6, s7 = s

    # 光滑度指标（原码 425-443 行：1st/2nd/3rd 差分平方 × 720/780/781）
    S0 = (720.0 * (-2.0 * s0 + 9.0 * s1 - 18.0 * s2 + 11.0 * s3) ** 2
          + 780.0 * (-6.0 * s0 + 24.0 * s1 - 30.0 * s2 + 12.0 * s3) ** 2
          + 781.0 * (-6.0 * s0 + 18.0 * (s1 - s2) + 6.0 * s3) ** 2)
    S1 = (720.0 * (s1 - 6.0 * s2 + 3.0 * s3 + 2.0 * s4) ** 2
          + 780.0 * (6.0 * s2 - 12.0 * s3 + 6.0 * s4) ** 2
          + 781.0 * (-6.0 * s1 + 18.0 * (s2 - s3) + 6.0 * s4) ** 2)
    S2 = (720.0 * (-2.0 * s2 - 3.0 * s3 + 6.0 * s4 - s5) ** 2
          + 780.0 * (6.0 * s3 - 12.0 * s4 + 6.0 * s5) ** 2
          + 781.0 * (-6.0 * s2 + 18.0 * (s3 - s4) + 6.0 * s5) ** 2)
    S3 = (720.0 * (-11.0 * s3 + 18.0 * s4 - 9.0 * s5 + 2.0 * s6) ** 2
          + 780.0 * (12.0 * s3 - 30.0 * s4 + 24.0 * s5 - 6.0 * s6) ** 2
          + 781.0 * (-6.0 * s3 + 18.0 * (s4 - s5) + 6.0 * s6) ** 2)
    S4 = (720.0 * (-26.0 * s4 + 57.0 * s5 - 42.0 * s6 + 11.0 * s7) ** 2
          + 780.0 * (18.0 * s4 - 48.0 * s5 + 42.0 * s6 - 12.0 * s7) ** 2
          + 781.0 * (-6.0 * s4 + 18.0 * (s5 - s6) + 6.0 * s7) ** 2)

    # TV 快速路径开关（原码 396-414 行）
    if tv_limiter is not None:
        tv_a = jnp.abs(s1 - s0) + jnp.abs(s2 - s1) + jnp.abs(s3 - s2)
        tv_b = jnp.abs(s2 - s1) + jnp.abs(s3 - s2) + jnp.abs(s4 - s3)
        tv_c = jnp.abs(s3 - s2) + jnp.abs(s4 - s3) + jnp.abs(s5 - s4)
        tv_d = jnp.abs(s4 - s3) + jnp.abs(s5 - s4) + jnp.abs(s6 - s5)
        tv_e = jnp.abs(s5 - s4) + jnp.abs(s6 - s5) + jnp.abs(s7 - s6)
        tv_min = jnp.minimum(jnp.minimum(jnp.minimum(tv_a, tv_b),
                                         jnp.minimum(tv_c, tv_d)), tv_e)
        tv_max = jnp.maximum(jnp.maximum(jnp.maximum(tv_a, tv_b),
                                         jnp.maximum(tv_c, tv_d)), tv_e)
        use_linear = (tv_max < tv_limiter * tv_min) & (tv_max < tv_max_cap)
    else:
        use_linear = jnp.zeros_like(S0, dtype=bool)

    # 完整 WENO 权（原码 452-463 行）
    e = WENO7_EPS
    d0, d1, d2, d3, d4 = WENO7_DR
    a0 = d0 * (e + S1) * (e + S2) * (e + S3) * (e + S4)
    a1 = d1 * (e + S0) * (e + S2) * (e + S3) * (e + S4)
    a2 = d2 * (e + S0) * (e + S1) * (e + S3) * (e + S4)
    a3 = d3 * (e + S0) * (e + S1) * (e + S2) * (e + S4)
    a4 = d4 * (e + S0) * (e + S1) * (e + S2) * (e + S3)
    am = a0 + a1 + a2 + a3 + a4

    # 候选多项式 ×4（原码 469-473 行，最后统一 /12）
    p = ((-3.0 * s0 + 13.0 * s1 - 23.0 * s2 + 25.0 * s3) * a0
         + (s1 - 5.0 * s2 + 13.0 * s3 + 3.0 * s4) * a1
         + (-s2 + 7.0 * s3 + 7.0 * s4 - s5) * a2
         + (3.0 * s3 + 13.0 * s4 - 5.0 * s5 + s6) * a3
         + (25.0 * s4 - 23.0 * s5 + 13.0 * s6 - 3.0 * s7) * a4)
    face_weno = p / (12.0 * am)

    # 线性权面值（光滑区快速路径）
    face_linear = ((-3.0 * s0 + 13.0 * s1 - 23.0 * s2 + 25.0 * s3) * d0
                   + (s1 - 5.0 * s2 + 13.0 * s3 + 3.0 * s4) * d1
                   + (-s2 + 7.0 * s3 + 7.0 * s4 - s5) * d2
                   + (3.0 * s3 + 13.0 * s4 - 5.0 * s5 + s6) * d3
                   + (25.0 * s4 - 23.0 * s5 + 13.0 * s6 - 3.0 * s7) * d4) \
        / (12.0 * (d0 + d1 + d2 + d3 + d4))

    return jnp.where(use_linear, face_linear, face_weno)


def weno7_symbo_face_M(g, tv_limiter=None, tv_max_cap=None):
    """WENO7-SYMBO M 面重构：对称翻转复用 P 实现（镜像关系 M_k = P_{1-k}）。"""
    flipped = jnp.flip(g, axis=-1)
    face = _weno7_symbo_face_P(flipped, tv_limiter, tv_max_cap)
    return jnp.flip(face, axis=-1)


def weno7_symbo_rhs_along_last_axis(fp, fm, h=1.0,
                                    tv_limiter=None, tv_max_cap=None):
    """单方向 WENO7-SYMBO 通量差分（fp 走 P、fm 走 M，M 为镜像实现）。"""
    face = (_weno7_symbo_face_P(fp, tv_limiter, tv_max_cap)
            + weno7_symbo_face_M(fm, tv_limiter, tv_max_cap))
    return _du_from_faces(face) / h


# ============================ 格式族扩展（P2） ==============================
# NND2 / WENO5 / WENO7 / OMP6，系数逐条照搬 OCFD_Schemes.cu。
# 统一模式：每个格式实现 face_P（沿末轴，紧凑面数组），face_M 用轴翻转复用
# （镜像关系已在 WENO7-SYMBO 数值验证）。minmod 系列 jnp 向量化。

def minmod2(x1, x2):
    """OCFD_Schemes.cu:777：0.5(sign(x1)+sign(x2))*min(|x1|,|x2|)。"""
    return 0.5 * (jnp.sign(x1) + jnp.sign(x2)) * jnp.minimum(jnp.abs(x1), jnp.abs(x2))


def minmod4(x1, x2, x3, x4):
    """OCFD_Schemes.cu:784：四参数同号 minmod（首因子不取绝对值）。"""
    m = 0.5 * (jnp.sign(x1) + jnp.sign(x2))
    m = m * jnp.abs(0.5 * (jnp.sign(x1) + jnp.sign(x3)))
    m = m * jnp.abs(0.5 * (jnp.sign(x1) + jnp.sign(x4)))
    return m * jnp.minimum(jnp.minimum(x1, x2), jnp.minimum(x3, x4))


def _slice8(g, off_lo):
    """返回 s0..s7：g[(j-1)+off_lo+i+LAP]，j = 0..nx（紧凑面窗口）。"""
    nx = g.shape[-1] - 2 * LAP
    lo, hi = 0, nx + 1
    return tuple(g[..., lo - 1 + off_lo + i + LAP: hi - 1 + off_lo + i + LAP]
                 for i in range(8))


# ---- NND2（306，OCFD_Schemes.cu:801-838；P off -1..1，M 镜像 off 0..2）----
def _nnd2_face_P(g):
    """用三点 NND2/minmod 限制器重构正通量的末轴界面值。"""
    fm1, f0, fp1 = _slice8(g, -1)[:3]
    return f0 + 0.5 * minmod2(fp1 - f0, f0 - fm1)


# ---- WENO5（302，OCFD_Schemes.cu:1119-1152；P off -2..2，WENO-Z 权）----
_WENO5_EPS = 1e-6


def _weno5_face_P(g):
    """用五点和 WENO-Z 非线性权重构正通量的末轴界面值。"""
    s0, s1, s2, s3, s4 = _slice8(g, -2)[:5]
    S0 = 13.0 * (s2 - 2.0 * s3 + s4) ** 2 + 3.0 * (3.0 * s2 - 4.0 * s3 + s4) ** 2
    S1 = 13.0 * (s1 - 2.0 * s2 + s3) ** 2 + 3.0 * (s1 - s3) ** 2
    S2 = 13.0 * (s0 - 2.0 * s1 + s2) ** 2 + 3.0 * (s0 - 4.0 * s1 + 3.0 * s2) ** 2
    q03 = 2.0 * s2 + 5.0 * s3 - s4
    q13 = -s1 + 5.0 * s2 + 2.0 * s3
    q23 = 2.0 * s0 - 7.0 * s1 + 11.0 * s2
    e12 = 12.0 * _WENO5_EPS
    a0 = 3.0 * (e12 + S1) ** 2 * (e12 + S2) ** 2
    a1 = 6.0 * (e12 + S0) ** 2 * (e12 + S2) ** 2
    a2 = (e12 + S0) ** 2 * (e12 + S1) ** 2
    return (a0 * q03 + a1 * q13 + a2 * q23) / (6.0 * (a0 + a1 + a2))


# ---- WENO7（303，OCFD_Schemes.cu:617-664；P off -3..3，4 候选 WENO-Z 权）----
_WENO7_EPS = 3.456e-4


def _weno7_face_P(g):
    """用七点、四候选模板和 WENO-Z 权重构正通量的末轴界面值。"""
    s0, s1, s2, s3, s4, s5, s6 = _slice8(g, -3)[:7]
    t0 = -2.0 * s0 + 9.0 * s1 - 18.0 * s2 + 11.0 * s3
    t1 = -6.0 * s0 + 24.0 * s1 - 30.0 * s2 + 12.0 * s3
    t2 = -6.0 * s0 + 18.0 * (s1 - s2) + 6.0 * s3
    S0 = 960.0 * t0 ** 2 + 1040.0 * t1 ** 2 + 1043.0 * t2 ** 2 + 80.0 * t0 * t2
    t0 = s1 - 6.0 * s2 + 3.0 * s3 + 2.0 * s4
    t1 = 6.0 * s2 - 12.0 * s3 + 6.0 * s4
    t2 = -6.0 * s1 + 18.0 * (s2 - s3) + 6.0 * s4
    S1 = 960.0 * t0 ** 2 + 1040.0 * t1 ** 2 + 1043.0 * t2 ** 2 + 80.0 * t0 * t2
    t0 = -2.0 * s2 - 3.0 * s3 + 6.0 * s4 - s5
    t1 = 6.0 * s3 - 12.0 * s4 + 6.0 * s5
    t2 = -6.0 * s2 + 18.0 * (s3 - s4) + 6.0 * s5
    S2 = 960.0 * t0 ** 2 + 1040.0 * t1 ** 2 + 1043.0 * t2 ** 2 + 80.0 * t0 * t2
    t0 = -11.0 * s3 + 18.0 * s4 - 9.0 * s5 + 2.0 * s6
    t1 = 12.0 * s3 - 30.0 * s4 + 24.0 * s5 - 6.0 * s6
    t2 = -6.0 * s3 + 18.0 * (s4 - s5) + 6.0 * s6
    S3 = 960.0 * t0 ** 2 + 1040.0 * t1 ** 2 + 1043.0 * t2 ** 2 + 80.0 * t0 * t2
    e2 = _WENO7_EPS * _WENO7_EPS
    a0 = (e2 + S1) * (e2 + S1) * (e2 + S2) * (e2 + S2) * (e2 + S3) * (e2 + S3)
    a1 = 12.0 * (e2 + S0) * (e2 + S0) * (e2 + S2) * (e2 + S2) * (e2 + S3) * (e2 + S3)
    a2 = 18.0 * (e2 + S0) * (e2 + S0) * (e2 + S1) * (e2 + S1) * (e2 + S3) * (e2 + S3)
    a3 = 4.0 * (e2 + S0) * (e2 + S0) * (e2 + S1) * (e2 + S1) * (e2 + S2) * (e2 + S2)
    am = a0 + a1 + a2 + a3
    p = ((-3.0 * s0 + 13.0 * s1 - 23.0 * s2 + 25.0 * s3) * a0
         + (s1 - 5.0 * s2 + 13.0 * s3 + 3.0 * s4) * a1
         + (-s2 + 7.0 * s3 + 7.0 * s4 - s5) * a2
         + (3.0 * s3 + 13.0 * s4 - 5.0 * s5 + s6) * a3)
    return p / (12.0 * am)


# ---- OMP6（307-309，OCFD_Schemes.cu:937-1002；P off -3..4，8 点）----
def _omp6_face_P(g, omp6_flag):
    """按指定 OMP6 参数档重构正通量界面值，并在检测到外插极值时施加限制器。"""
    s = _slice8(g, -3)
    s0, s1, s2, s3, s4, s5, s6, s7 = s
    if omp6_flag == 1:
        m, n = 0.001, 0.0
    elif omp6_flag == 2:
        m, n = 0.0, -1.0 / 140.0
    else:
        m, n = 0.015, 0.0
    mid = 0.5 * (m + n)
    m = 0.5 * (m - n)
    mid_nf = (60.0 * mid * s7
              + (1.0 - 60.0 * m - 360.0 * mid) * s6
              + (-8.0 + 360.0 * m + 900.0 * mid) * s5
              + (37.0 - 900.0 * m - 1200.0 * mid) * s4
              + (37.0 + 1200.0 * m + 900.0 * mid) * s3
              + (-8.0 - 900.0 * m - 360.0 * mid) * s2
              + (1.0 + 360.0 * m + 60.0 * mid) * s1
              - 60.0 * m * s0) / 60.0
    m_nnd = s3 + minmod2(s4 - s3, s3 - s2)
    # 极值点检测：限制器仅在 mid_nf 落在 s3 与 NND 值之外时启用
    cond = (mid_nf - s3) * (mid_nf - m_nnd) >= 1.0e-10
    m_ = s2 + s4 - 2.0 * s3
    n_ = s3 + s5 - 2.0 * s4
    tmp_ = 4.0 * m_ - n_
    tmp1_ = 4.0 * n_ - m_
    tmp = 0.5 * (s3 + s4) - 0.5 * minmod4(tmp_, tmp1_, n_, m_)
    n_2 = s1 + s3 - 2.0 * s2
    tmp1 = s3 + 0.5 * (s3 - s2) + 4.0 * minmod4(4.0 * n_2 - m_, 4.0 * m_ - n_2, m_, n_2) / 3.0
    lb = jnp.minimum(jnp.minimum(s3, s4), tmp)
    na = s3 + 4.0 * (s3 - s2)
    na = jnp.minimum(jnp.minimum(s3, na), tmp1)
    lb = jnp.maximum(lb, na)
    ub = jnp.maximum(jnp.maximum(s3, tmp), s4)
    na = s3 + 4.0 * (s3 - s2)
    na = jnp.maximum(jnp.maximum(s3, na), tmp1)
    ub = jnp.minimum(ub, na)
    mid_limited = mid_nf + minmod2(ub - mid_nf, lb - mid_nf)
    return jnp.where(cond, mid_limited, mid_nf)


def _omp6_face_M(g, omp6_flag):
    """OMP6 M 面重构（OCFD_Schemes.cu:1028-1093，模板 off -4..3）。

    不能用翻转复用 P：minmod4 与 fmin/fmax 夹逼链对翻转不对称。
    本函数逐行照搬原码 OMP6_kernel_M。
    模板 cell p-3..p+4（p 为面编号，off -3..4 相对面左邻 cell p-1），
    基点 s4 = f[cell p]（面右邻）。窗口 p=0..nx 全物理面，LAP=4 鬼胞恰好
    覆盖（与 UP7/WENO5 的 max off=4 同界）。
    """
    s = _slice8(g, -3)
    s0, s1, s2, s3, s4, s5, s6, s7 = s
    if omp6_flag == 1:
        m, n = 0.001, 0.0
    elif omp6_flag == 2:
        m, n = 0.0, -1.0 / 140.0
    else:
        m, n = 0.015, 0.0
    mid = 0.5 * (m + n)
    m = 0.5 * (m - n)
    mid_nf = (60.0 * mid * s0
              + (1.0 - 60.0 * m - 360.0 * mid) * s1
              + (-8.0 + 360.0 * m + 900.0 * mid) * s2
              + (37.0 - 900.0 * m - 1200.0 * mid) * s3
              + (37.0 + 1200.0 * m + 900.0 * mid) * s4
              + (-8.0 - 900.0 * m - 360.0 * mid) * s5
              + (1.0 + 360.0 * m + 60.0 * mid) * s6
              - 60.0 * m * s7) / 60.0
    m_nnd = s4 + minmod2(s3 - s4, s4 - s5)
    cond = (mid_nf - s4) * (mid_nf - m_nnd) >= 1.0e-10
    m_ = s5 + s3 - 2.0 * s4
    n_ = s4 + s2 - 2.0 * s3
    tmp_ = 4.0 * m_ - n_
    tmp1_ = 4.0 * n_ - m_
    tmp = 0.5 * (s4 + s3) - 0.5 * minmod4(tmp_, tmp1_, n_, m_)
    n_2 = s6 + s4 - 2.0 * s5
    tmp1 = s4 + 0.5 * (s4 - s3) + 4.0 * minmod4(4.0 * n_2 - m_, 4.0 * m_ - n_2, m_, n_2) / 3.0
    lb = jnp.minimum(jnp.minimum(s5, s3), tmp)
    na = s4 + 4.0 * (s4 - s5)
    na = jnp.minimum(jnp.minimum(s4, na), tmp1)
    lb = jnp.maximum(lb, na)
    ub = jnp.maximum(jnp.maximum(s4, tmp), s3)
    na = s4 + 4.0 * (s4 - s5)
    na = jnp.maximum(jnp.maximum(s4, na), tmp1)
    ub = jnp.minimum(ub, na)
    mid_limited = mid_nf + minmod2(ub - mid_nf, lb - mid_nf)
    return jnp.where(cond, mid_limited, mid_nf)


# ---- 面通量带：OCFD boundp1/boundp2 + boundm1/boundm2 ----
#
# 原码在物理边界那侧把 job 往外扩一行（OCFD_Schemes_Choose.cu:201-213 / ~547-550
# `job_in.start.y -= 1` / `job_in.end.y += 1`，blockdim.y += 1），所以宏里的局部坐标
# coords = 内区节点号 ∓ 1（`get_data_kernel` 返回的是**作业内局部下标**，
# job.start 的偏移在 get_SoA_LAP 的 offset 里加，见 OCFD_Schemes.cu:159-232）。
# 据此把 OCFD_bound_Scheme.cu:689-701（P 侧 boundp2）与 :789-804（M 侧 boundm2）
# 翻译成面数组的覆盖值。上界作业的几何是
#     P 内核：节点 y = ny-4 .. ny+2，blockdim.y = 8  ⇒ tmp1 = coords - (2LAP+1) ∈ [-9,-1]
#     M 内核：节点 y = ny-3 .. ny+3（最上一行是上界鬼胞）
# 层数由各格式的 ka1/kb1 定：OMP6 P kb1=4 ⇒ 守卫 coords > -4 ⇒ 命中 -1,-2,-3；
# OMP6 M kb1=3 ⇒ 守卫 coords >= -4 ⇒ 命中 -1,-2,-3,-4；tmp1 ≤ -5 时 bound 函数
# 不返回 0、代码落到末尾 `return 1`，调用方于是跑正常重构（静默回落，无未定义行为）。
#
#   P 侧（fp，boundp2）       面号 → 公式
#       face n   = P[n-1] + 0.5(P[n-1]-P[n-2])           (tmp1=-1，2 阶单侧外推)
#       face n-1 = P_right_plus，s = P[n-4..n-1]         (tmp1=-2)
#       face n-2 = WENO5_P，s = P[n-5..n-1]              (tmp1=-3)
#       face n-3 / 更低  正常 OMP6（tmp1=-4 越界 ⇒ 回落）
#   M 侧（fm，boundm2）
#       face n   = M[n-1]                                (tmp1=-1，壁面 cell 的 M 拷贝)
#       face n-1 = M[n-1] - 0.5(M[n-1]-M[n-2])           (tmp1=-2)
#       face n-2 = M_right_plus，s = M[n-4..n-1]         (tmp1=-3)
#       face n-3 = WENO5_M，s = M[n-5..n-1]              (tmp1=-4)
#
# 关键点：上面 7 条**全部只用 compact cell ≤ n-1**（n-1 就是壁面那一层 cell，由 BC
# 写成 u=v=w=0 / T=Tw / ρ=pw/T），一个鬼胞都不读。这不是巧合：M 内核 end+1 的偏移
# 让它的"节点 n"正好落在鬼胞上，公式于是改用 cell n-1 的拷贝；壁面面通量由壁面状态
# 决定（(P+M)[n-1] 数值上就是 (0, 0, p_w, 0, 0)），用镜像鬼胞做插值拿到的是反射状态
# 的分裂通量，会把唯一非零的压力项也搅掉。
#
# 低端（boundp1/boundm1）照旧保留：P 3 面 + M 1 面，M 侧那一处原码确实借了 f_{-1}，
# 这里照抄。整体公式按 OMP6 的 ka1/kb1 硬编码（槽道算例用 OMP6_CD8）。
_BAND_EP = 1e-6


def _band_S_right(f0, f1, f2):
    """P_right/M_lift 型（q23 候选）光顺因子：13(f0-2f1+f2)^2 + 3(f0-4f1+3f2)^2。"""
    return 13.0 * (f0 - 2.0 * f1 + f2) ** 2 + 3.0 * (f0 - 4.0 * f1 + 3.0 * f2) ** 2


def _band_S_sym(f0, f1, f2):
    """q13 候选光顺因子：13(f0-2f1+f2)^2 + 3(f0-f2)^2。"""
    return 13.0 * (f0 - 2.0 * f1 + f2) ** 2 + 3.0 * (f0 - f2) ** 2


def _band_S_lift(f0, f1, f2):
    """q03 候选光顺因子：13(f0-2f1+f2)^2 + 3(3f0-4f1+f2)^2。"""
    return 13.0 * (f0 - 2.0 * f1 + f2) ** 2 + 3.0 * (3.0 * f0 - 4.0 * f1 + f2) ** 2


def flux_band_faces(Fp, Fm, fp, fm, lo=True, hi=True):
    """把原码的边界带覆盖到 P/M 面数组上（只改边界那几个面，其余原样）。

    Fp/Fm: (..., n+1) 正常重构出来的 P/M 面值；fp/fm: (..., n+2LAP) 分裂通量场。
    lo: 该侧是否 DIF_BOUND=1（下界，原码走 boundp1/boundm1）。
    hi: 上界，原码走 boundp2/boundm2（P 3 层 + M 4 层，全部只用 cell ≤ n-1）。

    当前低端带的面与窗口映射尚需逐面和原码核对；调用方应根据边界状态计算
    分裂通量鬼层，不能假定它们都为零。
    """
    n = fp.shape[-1] - 2 * LAP
    if n < 6:
        return Fp, Fm                       # 太小，跳过（原码也不会有这么小的块）
    P = fp[..., LAP:LAP + n]                # 内区节点（紧凑索引 = 节点号）
    M = fm[..., LAP:LAP + n]
    e = 12.0 * _BAND_EP

    if lo:
        # ---------------- P 侧：boundp1 ----------------
        # 面 0 → stencil[-ka1+1]（源码就是"拷贝偏移 +1 那格"，er 已简化为直接拷）
        # 面 1 → P_lift 窗口 (0,1,2)  ⇒ (2f0+5f1−f2)/6
        # 面 2 → P_lift_plus 窗口 (−1,0,1,2)
        Fp = Fp.at[..., 0].set(P[..., 0])
        Fp = Fp.at[..., 1].set((2.0 * P[..., 0] + 5.0 * P[..., 1] - P[..., 2]) / 6.0)
        q03 = 2.0 * P[..., 1] + 5.0 * P[..., 2] - P[..., 3]
        q13 = -P[..., 0] + 5.0 * P[..., 1] + 2.0 * P[..., 2]
        a0 = 3.0 / (e + _band_S_lift(P[..., 1], P[..., 2], P[..., 3])) ** 2
        a1 = 6.0 / (e + _band_S_sym(P[..., 0], P[..., 1], P[..., 2])) ** 2
        Fp = Fp.at[..., 2].set((a0 * q03 + a1 * q13) / (6.0 * (a0 + a1)))

        # ---------------- M 侧：boundm1 ----------------
        # coords==0 → OCFD_weno5_kernel_M_lift(&stencil[-ka1-2])：**单候选**，
        # 结果就是纯 q23/6（窗口 (0,1,2)，不含鬼胞）。之前这里写成了 q13/q23 的加权混合，
        # 壁面 k0 因此差 48.8%；改成纯 q23/6 后差 0.45%。
        q23 = 2.0 * M[..., 2] - 7.0 * M[..., 1] + 11.0 * M[..., 0]
        Fm = Fm.at[..., 0].set(q23 / 6.0)

    if hi:
        # ---------------- P 侧：boundp2（tmp1=-1,-2,-3 三层；-4 越界回落正常）--------
        # 面 n：2 阶单侧外推（minmod2(d,d)=d），锚在壁面 cell n-1
        Fp = Fp.at[..., n].set(P[..., n - 1] + 0.5 * (P[..., n - 1] - P[..., n - 2]))
        # 面 n-1：P_right_plus，s = (P[n-4], P[n-3], P[n-2], P[n-1])
        s0, s1, s2, s3 = P[..., n - 4], P[..., n - 3], P[..., n - 2], P[..., n - 1]
        a1 = 6.0 / (e + _band_S_sym(s1, s2, s3)) ** 2      # S1 = 13(s1-2s2+s3)^2 + 3(s1-s3)^2
        a2 = 1.0 / (e + _band_S_right(s0, s1, s2)) ** 2    # S2 = 13(s0-2s1+s2)^2 + 3(s0-4s1+3s2)^2
        q13 = -s1 + 5.0 * s2 + 2.0 * s3
        q23 = 2.0 * s0 - 7.0 * s1 + 11.0 * s2
        Fp = Fp.at[..., n - 1].set((a1 * q13 + a2 * q23) / (6.0 * (a1 + a2)))
        # 面 n-2：完整 WENO5_P，s = (P[n-5], ..., P[n-1])
        t0, t1, t2, t3, t4 = (P[..., n - 5], P[..., n - 4], P[..., n - 3],
                              P[..., n - 2], P[..., n - 1])
        S0 = _band_S_lift(t2, t3, t4)                      # 13(t2-2t3+t4)^2 + 3(3t2-4t3+t4)^2
        S1 = _band_S_sym(t1, t2, t3)                       # 13(t1-2t2+t3)^2 + 3(t1-t3)^2
        S2 = _band_S_right(t0, t1, t2)                     # 13(t0-2t1+t2)^2 + 3(t0-4t1+3t2)^2
        q03 = 2.0 * t2 + 5.0 * t3 - t4
        q13 = -t1 + 5.0 * t2 + 2.0 * t3
        q23 = 2.0 * t0 - 7.0 * t1 + 11.0 * t2
        A0 = 3.0 * (e + S1) ** 2 * (e + S2) ** 2
        A1 = 6.0 * (e + S0) ** 2 * (e + S2) ** 2
        A2 = (e + S0) ** 2 * (e + S1) ** 2
        Fp = Fp.at[..., n - 2].set((A0 * q03 + A1 * q13 + A2 * q23)
                                   / (6.0 * (A0 + A1 + A2)))

        # ---------------- M 侧：boundm2（tmp1=-1..-4 四层）--------
        # 面 n：M 内核 end+1 的偏移让"节点 n"落在鬼胞上，公式取 cell n-1 的拷贝
        Fm = Fm.at[..., n].set(M[..., n - 1])
        # 面 n-1：M_lift 型单侧外推，锚在壁面 cell n-1
        Fm = Fm.at[..., n - 1].set(M[..., n - 1] - 0.5 * (M[..., n - 1] - M[..., n - 2]))
        # 面 n-2：M_right_plus，s = (M[n-4], M[n-3], M[n-2], M[n-1])
        u0, u1, u2, u3 = M[..., n - 4], M[..., n - 3], M[..., n - 2], M[..., n - 1]
        b0 = 3.0 / (e + _band_S_lift(u2, u1, u0)) ** 2     # S0 = 13(u2-2u1+u0)^2 + 3(3u2-4u1+u0)^2
        b1 = 6.0 / (e + _band_S_sym(u3, u2, u1)) ** 2      # S1 = 13(u3-2u2+u1)^2 + 3(u3-u1)^2
        p03 = 2.0 * u2 + 5.0 * u1 - u0
        p13 = -u3 + 5.0 * u2 + 2.0 * u1
        Fm = Fm.at[..., n - 2].set((b0 * p03 + b1 * p13) / (6.0 * (b0 + b1)))
        # 面 n-3：完整 WENO5_M，s = (M[n-5], ..., M[n-1])
        v0, v1, v2, v3, v4 = (M[..., n - 5], M[..., n - 4], M[..., n - 3],
                              M[..., n - 2], M[..., n - 1])
        T0 = _band_S_lift(v2, v1, v0)                      # 13(v2-2v1+v0)^2 + 3(3v2-4v1+v0)^2
        T1 = _band_S_sym(v3, v2, v1)                       # 13(v3-2v2+v1)^2 + 3(v3-v1)^2
        T2 = _band_S_right(v4, v3, v2)                     # 13(v4-2v3+v2)^2 + 3(v4-4v3+3v2)^2
        r03 = 2.0 * v2 + 5.0 * v1 - v0
        r13 = -v3 + 5.0 * v2 + 2.0 * v1
        r23 = 2.0 * v4 - 7.0 * v3 + 11.0 * v2
        B0 = 3.0 * (e + T1) ** 2 * (e + T2) ** 2
        B1 = 6.0 * (e + T0) ** 2 * (e + T2) ** 2
        B2 = (e + T0) ** 2 * (e + T1) ** 2
        Fm = Fm.at[..., n - 3].set((B0 * r03 + B1 * r13 + B2 * r23)
                                   / (6.0 * (B0 + B1 + B2)))

    return Fp, Fm


def _face_M_from_P(face_P_fn):
    """由正通量界面函数生成镜像的负通量界面函数。"""
    def face_M(g, *args, **kwargs):
        """翻转末轴后调用 P 重构，再把界面结果翻回原方向。"""
        return jnp.flip(face_P_fn(jnp.flip(g, axis=-1), *args, **kwargs), axis=-1)
    return face_M


def _rhs_from_faces(face):
    """把已求和的界面通量转换为未除网格步长的内区通量差分。"""
    return _du_from_faces(face)


# ---- 格式注册表：scheme 名 → (face_P, face_M) ----
def _make_rhs(face_P, face_M):
    """把一对 P/M 界面重构函数包装成统一的单方向 RHS 接口。"""
    def rhs_along_last_axis(fp, fm, h=1.0, bound=None, **kwargs):
        """重构 P/M 面通量，按需覆盖边界带，并返回除以步长后的通量散度。"""
        Fp = face_P(fp, **kwargs)
        Fm = face_M(fm, **kwargs)
        if bound is not None and (bound[0] or bound[1]):
            Fp, Fm = flux_band_faces(Fp, Fm, fp, fm,
                                     lo=bool(bound[0]), hi=bool(bound[1]))
        return _du_from_faces(Fp + Fm) / h
    return rhs_along_last_axis


_nnd2_face_M = _face_M_from_P(_nnd2_face_P)
_weno5_face_M = _face_M_from_P(_weno5_face_P)
_weno7_face_M = _face_M_from_P(_weno7_face_P)

SOLVE_SCHEMES = {
    "NND2": _make_rhs(_nnd2_face_P, _nnd2_face_M),
    "WENO5": _make_rhs(_weno5_face_P, _weno5_face_M),
    "WENO7": _make_rhs(_weno7_face_P, _weno7_face_M),
    "OMP6-1": _make_rhs(lambda g: _omp6_face_P(g, 1), lambda g: _omp6_face_M(g, 1)),
    "OMP6-2": _make_rhs(lambda g: _omp6_face_P(g, 2), lambda g: _omp6_face_M(g, 2)),
    "OMP6-3": _make_rhs(lambda g: _omp6_face_P(g, 3), lambda g: _omp6_face_M(g, 3)),
}

# 原码 in 文件的 SCHEME_INVIS 名字 → 本模块的键。
# OCFD_Schemes_Choose.cu:128-138 把 307/308/309 分别传 OMP6_FLAG 0/1/2；
# OCFD_Schemes.cu:941-947 中 FLAG=1 → (m=0.001,n=0)、FLAG=2 → (0,n=-1/140)、
# 其余（含 0 与 3）→ (m=0.015,n=0)。本模块的 "OMP6-k" 直接用旗标号，所以：
#   OMP6_HR (307, FLAG 0) == "OMP6-3"（走 else 分支）
#   OMP6_LD (308, FLAG 1) == "OMP6-1"
#   OMP6_CD8(309, FLAG 2) == "OMP6-2"      ← 槽道算例用的就是这个
# 寄存器同一函数对象，别名只是让 in 文件名可以直接照抄进 --scheme。
SOLVE_SCHEMES.update({
    "OMP6_HR": SOLVE_SCHEMES["OMP6-3"],
    "OMP6_LD": SOLVE_SCHEMES["OMP6-1"],
    "OMP6_CD8": SOLVE_SCHEMES["OMP6-2"],
})
