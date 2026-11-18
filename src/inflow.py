"""OCFD 1D 入口剖面（``flow1d-inlet.dat`` 格式）读取。

原码格式见 ``benchmark_comp/init3d-inlet1d.c:151-160``：首行表头，其后每行

    y  rho  u  v  T

``%lf%lf%lf%lf%lf``，``w`` 不在文件里，原码 ``Creat_flow2d()`` 固定写 0。
这里的口径：

* 剖面沿壁法向（计算 j）给出，与格点**逐点对应**，不做插值；
* 点数和网格 ``nj`` 必须一致，否则报 ``SetupError``（用户已明确不插值）；
* 返回守恒量 ``(5, nj)``，``w`` 分量为 0，``rhoE = rho*Cv*T + 0.5*rho*(u^2+v^2)``；
* 值按壁法向逐行给出，沿展向（k）和入口鬼层方向都按原码复用同一剖面。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .errors import SetupError

__all__ = ["read_inflow_profile", "profile_to_cons"]


def profile_to_cons(y: np.ndarray, rho: np.ndarray, u: np.ndarray,
                    v: np.ndarray, T: np.ndarray, cv: float) -> np.ndarray:
    """把剖面原始量装配成 5 分量守恒量 ``(5, n)``（``w`` 取 0）。"""
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    kinetic = 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, np.zeros_like(rho),
                     rho * cv * T + kinetic], axis=0)


def read_inflow_profile(path, nj: Optional[int], cv: float) -> np.ndarray:
    """读取 1D 入口剖面，返回守恒量 ``(5, nj)``。

    ``nj`` 给出时要求剖面行数与网格 j 点数严格相同（不插值、不外推）。
    """
    p = Path(path)
    if not p.is_file():
        raise SetupError(f"入口剖面文件不存在：{p}")
    rows = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if lineno == 1:
                continue                    # 原码 fgets 跳过表头
            text = line.strip()
            if not text:
                continue
            parts = text.replace(",", " ").split()
            if len(parts) < 5:
                raise SetupError(
                    f"入口剖面 {p} 第 {lineno} 行需要 5 列（y rho u v T），"
                    f"实际 {len(parts)} 列")
            try:
                rows.append([float(t) for t in parts[:5]])
            except ValueError as exc:
                raise SetupError(
                    f"入口剖面 {p} 第 {lineno} 行无法解析为浮点数：{text}") from exc
    if not rows:
        raise SetupError(f"入口剖面 {p} 没有数据行（首行应为表头）")
    data = np.asarray(rows, dtype=np.float64)
    if nj is not None and data.shape[0] != int(nj):
        raise SetupError(
            f"入口剖面 {p} 有 {data.shape[0]} 个点，网格 j 方向是 {int(nj)} 个点；"
            "剖面必须与格点逐点对应（程序不做插值）")
    y, rho, u, v, T = (data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4])
    if not np.all(np.isfinite(data)):
        raise SetupError(f"入口剖面 {p} 含非有限值")
    if np.any(rho <= 0.0) or np.any(T <= 0.0):
        raise SetupError(f"入口剖面 {p} 的 rho/T 必须为正")
    return profile_to_cons(y, rho, u, v, T, cv)
