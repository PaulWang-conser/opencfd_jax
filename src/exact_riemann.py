"""Sod 激波管问题的 Riemann 精确解（Toro, "Riemann Solvers and Numerical
Methods for Fluid Dynamics", 第 4 章。只用于验证，不参与求解）。

初值：左 (1, 0, 1)、右 (0.125, 0, 0.1)，gamma=1.4，间断在 x=0.5，t=0.2。
标准结构：左稀疏波 + 接触间断 + 右激波。实现按通用两波结构写，
支持左/右波各自为激波或稀疏波。
"""

import numpy as np


def _f_press(p, rho_K, p_K, gamma):
    """波函数 f_K(p)：p>p_K 走激波支，否则稀疏波支。"""
    c_K = np.sqrt(gamma * p_K / rho_K)
    if p > p_K:
        A = 2.0 / (rho_K * (gamma + 1.0))
        B = (gamma - 1.0) / (gamma + 1.0) * p_K
        return (p - p_K) * np.sqrt(A / (rho_K * p + B))
    return (2.0 * c_K / (gamma - 1.0)) * ((p / p_K) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _df_press(p, rho_K, p_K, gamma):
    """f_K'(p)。"""
    c_K = np.sqrt(gamma * p_K / rho_K)
    if p > p_K:
        A = 2.0 / (rho_K * (gamma + 1.0))
        B = (gamma - 1.0) / (gamma + 1.0) * p_K
        s = np.sqrt(A / (rho_K * p + B))
        return s + (p - p_K) * A * s / (2.0 * (rho_K * p + B))
    return (1.0 / (rho_K * c_K)) * (p / p_K) ** (-(gamma + 1.0) / (2.0 * gamma))


def _star_state(rhoL, uL, pL, rhoR, uR, pR, gamma):
    """牛顿迭代求星域压力 p*，返回 (p*, u*)。"""
    p = 0.5 * (pL + pR)
    for _ in range(200):
        f = (_f_press(p, rhoL, pL, gamma)
             + _f_press(p, rhoR, pR, gamma) + (uR - uL))
        df = (_df_press(p, rhoL, pL, gamma)
              + _df_press(p, rhoR, pR, gamma))
        dp = -f / df
        p_new = p + dp
        p = max(p_new, 1e-10 * (pL + pR))
        if abs(dp) < 1e-13 * (pL + pR):
            break
    u_star = 0.5 * (uL + uR) + 0.5 * (_f_press(p, rhoR, pR, gamma)
                                      - _f_press(p, rhoL, pL, gamma))
    return p, u_star


def sod_exact(x, t, gamma=1.4,
              rhoL=1.0, uL=0.0, pL=1.0,
              rhoR=0.125, uR=0.0, pR=0.1, x_diaphragm=0.5):
    """返回 (rho, u, p) 剖面（numpy 数组，与 x 同长）。"""
    x = np.asarray(x, dtype=np.float64)
    xi = (x - x_diaphragm) / t

    p_star, u_star = _star_state(rhoL, uL, pL, rhoR, uR, pR, gamma)
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    # ---- 左波结构 ----
    if p_star > pL:   # 左激波
        rho_starL = rhoL * ((p_star / pL + (gamma - 1.0) / (gamma + 1.0))
                            / ((gamma - 1.0) / (gamma + 1.0) * p_star / pL + 1.0))
        S_L = uL - cL * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pL
                                + (gamma - 1.0) / (2.0 * gamma))
        # 分区边界：s < S_L 原左态；s > S_L 星域
        left_head = left_tail = None
        L_kind = "shock"
        L_speed = S_L
    else:             # 左稀疏波
        rho_starL = rhoL * (p_star / pL) ** (1.0 / gamma)
        c_starL = cL * (p_star / pL) ** ((gamma - 1.0) / (2.0 * gamma))
        L_kind = "fan"
        left_head = uL - cL            # 头（远离间断）
        left_tail = u_star - c_starL   # 尾（靠近间断）

    # ---- 右波结构 ----
    if p_star > pR:   # 右激波
        rho_starR = rhoR * ((p_star / pR + (gamma - 1.0) / (gamma + 1.0))
                            / ((gamma - 1.0) / (gamma + 1.0) * p_star / pR + 1.0))
        S_R = uR + cR * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pR
                                + (gamma - 1.0) / (2.0 * gamma))
        R_kind = "shock"
        R_speed = S_R
    else:             # 右稀疏波
        rho_starR = rhoR * (p_star / pR) ** (1.0 / gamma)
        c_starR = cR * (p_star / pR) ** ((gamma - 1.0) / (2.0 * gamma))
        R_kind = "fan"
        right_tail = u_star + c_starR  # 尾（靠近间断）
        right_head = uR + cR           # 头（远离间断）

    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    for i, s in enumerate(xi):
        # 1) 左波之外：原左态
        if L_kind == "shock" and s < L_speed:
            rho[i], u[i], p[i] = rhoL, uL, pL
            continue
        if L_kind == "fan" and s < left_head:
            rho[i], u[i], p[i] = rhoL, uL, pL
            continue
        # 2) 左稀疏波扇形内部
        if L_kind == "fan" and s < left_tail:
            uu = (2.0 / (gamma + 1.0)) * (cL + (gamma - 1.0) / 2.0 * uL + s)
            cc = u_du = uu - s                       # 特征关系 c = u - s
            rr = rhoL * (cc / cL) ** (2.0 / (gamma - 1.0))
            rho[i], u[i], p[i] = rr, uu, pL * (rr / rhoL) ** gamma
            continue
        # 3) 星域（接触间断两侧，p、u 相同，rho 由左/右波后态决定）
        if s < u_star:
            rho[i], u[i], p[i] = rho_starL, u_star, p_star
            continue
        # 4) 右稀疏波扇形内部
        if R_kind == "fan" and s < right_head:
            if s > right_tail:
                uu = (2.0 / (gamma + 1.0)) * (s - cR + (gamma - 1.0) / 2.0 * uR)
                cc = s - uu                          # 特征关系 c = s - u
                rr = rhoR * (cc / cR) ** (2.0 / (gamma - 1.0))
                rho[i], u[i], p[i] = rr, uu, pR * (rr / rhoR) ** gamma
                continue
        # 5) 右波之外：原右态；右激波之前的星域
        if R_kind == "shock":
            if s < R_speed:
                rho[i], u[i], p[i] = rho_starR, u_star, p_star
            else:
                rho[i], u[i], p[i] = rhoR, uR, pR
        else:  # 右稀疏波：s <= right_tail 为星域
            rho[i], u[i], p[i] = rho_starR, u_star, p_star

    return rho, u, p
