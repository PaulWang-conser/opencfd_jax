"""Blasius 相似解（不可压平板层流边界层）——平板算例的初值与验收基准。

方程（标准形式）：
    2 f''' + f f'' = 0,   f(0) = f'(0) = 0,   f'(∞) = 1
打靶 + 定步长 RK4 + 二分求 f''(0)。参考值 f''(0) = 0.3320573362。

物理量（与 OCFD 无量纲口径一致，见 OCFD_init.cu:381-386）：
    η = y * sqrt(U / (ν x)),   u = U f'(η)
    v = 0.5 * sqrt(ν U / x) * (η f' - f)
    Cf = 0.664 / sqrt(Re_x)   （τ_w 用 0.5ρU² 无量纲化）

注意 μ(T=1) = 1/Re（`amu_C0/(Tsb+1)` 里 Tsb=110.4/Ref_T 恰好相消），所以
T 均匀为 1 时 ν = 1/Re，Re_x = U x Re，与 Ref_T 无关。
"""

import functools

import numpy as np

_H = 2.0e-3          # η 步长
_ETA_MAX = 12.0      # 覆盖到 f'=1 的远场
_FPP0_REF = 0.33205733621519630


def _deriv(y):
    """y = (f, f', f'')，返回 dy/dη。"""
    f, fp, fpp = y
    return (fp, fpp, -0.5 * f * fpp)


def _rk4(s, h=_H, eta_max=_ETA_MAX):
    """给定 f''(0)=s，积分到 eta_max，返回 (eta[], f[], f'[], f'') 的 numpy 数组。"""
    n = int(round(eta_max / h)) + 1
    eta = np.empty(n)
    f = np.empty(n)
    fp = np.empty(n)
    fpp = np.empty(n)
    y = (0.0, 0.0, s)
    f[0], fp[0], fpp[0] = 0.0, 0.0, s
    eta[0] = 0.0
    for i in range(n - 1):
        k1 = _deriv(y)
        k2 = _deriv(tuple(y[j] + 0.5 * h * k1[j] for j in range(3)))
        k3 = _deriv(tuple(y[j] + 0.5 * h * k2[j] for j in range(3)))
        k4 = _deriv(tuple(y[j] + h * k3[j] for j in range(3)))
        y = tuple(y[j] + h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                  for j in range(3))
        eta[i + 1] = eta[i] + h
        f[i + 1], fp[i + 1], fpp[i + 1] = y
    return eta, f, fp, fpp


@functools.lru_cache(maxsize=1)
def table():
    """返回 (eta, f, f', f'') 表（默认精度，二分到 f'(eta_max)-1 < 1e-13）。"""
    lo, hi = 0.2, 0.5
    tab = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        eta, f, fp, fpp = _rk4(mid)
        tab = (eta, f, fp, fpp)
        if abs(fp[-1] - 1.0) < 1e-13:
            break
        if fp[-1] < 1.0:
            lo = mid
        else:
            hi = mid
    eta, f, fp, fpp = tab
    return eta, f, fp, fpp


def fpp0():
    """f''(0) 的数值解（用来核对 0.3320573）。"""
    eta, f, fp, fpp = table()
    return float(fpp[0])


def interp(eta):
    """给定 η（任意形状）返回 (f, f', f'')，超出表范围按远场值夹住。"""
    eta_t, f_t, fp_t, fpp_t = table()
    eta = np.asarray(eta, dtype=np.float64)
    e = np.clip(eta, 0.0, eta_t[-1])
    f = np.interp(e, eta_t, f_t)
    fp = np.interp(e, eta_t, fp_t)
    fpp = np.interp(e, eta_t, fpp_t)
    # 远场：f'=1、f''=0、f 继续线性增长（η f'-f 的差才有意义）
    big = eta > eta_t[-1]
    if np.any(big):
        f = np.where(big, eta - (eta_t[-1] - f_t[-1]), f)
        fp = np.where(big, 1.0, fp)
        fpp = np.where(big, 0.0, fpp)
    return f, fp, fpp


def field(y, x, U=1.0, nu=1.0):
    """平板边界层剖面：返回 u, v（形状 = broadcast(y, x)）。

    y: (ny,) 壁法向坐标（壁面 y=0）；x: (nx,) 流向坐标（x>0）。
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    eta = np.sqrt(U / (nu * x[:, None])) * y[None, :]      # (nx, ny)
    f, fp, _ = interp(eta)
    u = U * fp
    v = 0.5 * np.sqrt(nu * U / x[:, None]) * (eta * fp - f)
    return u, v


def cf(x, U=1.0, nu=1.0):
    """Blasius 壁面摩擦系数 Cf(x) = 0.664 / sqrt(Re_x)，Re_x = U x / ν。"""
    x = np.asarray(x, dtype=np.float64)
    return 0.664 / np.sqrt(U * x / nu)


if __name__ == "__main__":
    print("f''(0) = %.16f  (参考 %.16f)" % (fpp0(), _FPP0_REF))
    eta, f, fp, fpp = table()
    for e in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        i = int(round(e / _H))
        print("eta=%4.1f  f=%.6f  f'=%.6f  f''=%.6f" % (eta[i], f[i], fp[i], fpp[i]))
    print("δ99/√(νx/U) ≈ %.4f" % eta[np.argmax(fp > 0.99)])
