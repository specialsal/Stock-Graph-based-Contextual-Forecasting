# coding: utf-8
"""
optim_l2qp.py
L2 风险 + L2 调仓 的组合优化器（QP），可选 L2 换手预算约束。
- 无行业/风格暴露
- 支持候选池裁剪、单票上限、现金可选（sum(w) <= 1）

依赖: cvxpy, numpy, pandas, scipy (可选), scikit-learn (可选用于shrinkage)
"""

from typing import Dict, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import cvxpy as cp


def build_diag_cov(returns: pd.DataFrame,
                   min_var: float = 1e-6,
                   ann_factor: float = 252.0,
                   half_life: Optional[int] = None) -> pd.Series:
    """
    从历史日收益构造对角协方差（仅特定风险），返回每只股票的年化方差（对角项）。
    - returns: DataFrame(index=date, columns=stock) 的日收益
    - half_life: 若给定，按指数加权（EWMA）估计波动
    """
    R = returns.copy()
    R = R.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if R.empty:
        raise ValueError("returns is empty for covariance estimation")
    if half_life is None:
        # 简单样本方差
        var = R.var(axis=0, ddof=1).fillna(0.0)
    else:
        # EWMA 方差
        lam = np.exp(np.log(0.5) / half_life)
        w = np.array([lam ** (len(R)-1 - i) for i in range(len(R))], dtype=float)
        w /= w.sum()
        mu = (R.values * w[:, None]).sum(axis=0)
        var = ((R.values - mu) ** 2 * w[:, None]).sum(axis=0)
        var = pd.Series(var, index=R.columns)
    var = (var * ann_factor).clip(lower=min_var)
    return var


def align_and_select(universe: Sequence[str],
                     z: pd.Series,
                     w_prev: pd.Series,
                     ub: Optional[pd.Series] = None,
                     top_n: Optional[int] = None) -> Tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """
    对齐索引并可选择 top_n 进入优化
    - universe: 底池列表（字符串）
    - z: 预测收益或打分（index=stock）
    - w_prev: 上期权重（index=stock），缺失视为0
    - ub: 单票上限（index=stock），缺失用默认1.0
    - 返回：idx, z_vec, wprev_vec, ub_vec
    """
    uni = pd.Index(universe).unique()
    z = z.reindex(uni).astype(float)
    if top_n is not None and top_n < len(uni):
        # 仅保留最高的 top_n（按 z 降序）
        keep = z.sort_values(ascending=False).head(top_n).index
        uni = keep
        z = z.reindex(uni)
    w_prev = w_prev.reindex(uni).fillna(0.0).astype(float)
    if ub is None:
        ub_vec = np.ones(len(uni), dtype=float)
    else:
        ub_vec = ub.reindex(uni).fillna(1.0).astype(float).values
    return uni, z.values, w_prev.values, ub_vec


def solve_portfolio_l2qp(
    z: np.ndarray,                     # shape (n,)
    sigma_diag: np.ndarray,            # 对角协方差 shape (n,)
    w_prev: np.ndarray,                # shape (n,)
    ub: Optional[np.ndarray] = None,   # 单票上限 shape (n,), 默认1.0
    lambda_risk: float = 5.0,          # 风险厌恶（越大越保守）
    gamma_tc: float = 10.0,            # L2 调仓惩罚系数（越大越少换手）
    full_invest: bool = False,         # True: sum(w)=1；False: sum(w)<=1 允许现金
    tau2_budget: Optional[float] = None,  # 可选 L2 换手预算（对 ||Δw||_2 的约束），单位为绝对权重
    solver: str = "OSQP",
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    解如下 QP：
    maximize  z^T w − (λ/2)·w^T diag(sigma_diag) w − γ · ||w − w_prev||_2^2
    s.t.      sum(w) = 1 (或 ≤1)； 0 ≤ w ≤ ub

    返回 (w_opt, diag)
    """
    n = len(z)
    assert z.shape == (n,)
    assert sigma_diag.shape == (n,)
    assert w_prev.shape == (n,)
    if ub is None:
        ub = np.ones(n, dtype=float)
    assert ub.shape == (n,)

    # 变量
    w = cp.Variable(n)

    # 风险项：0.5*λ * w^T Σ w，Σ=diag(sigma_diag)
    risk_term = 0.5 * lambda_risk * cp.quad_form(w, cp.diag(sigma_diag))

    # 调仓项：γ * ||w - w_prev||_2^2
    tc_term = gamma_tc * cp.sum_squares(w - w_prev)

    # 目标：最大化 z^T w - 风险 - 调仓
    obj = cp.Maximize(z @ w - risk_term - tc_term)

    cons = []
    if full_invest:
        cons.append(cp.sum(w) == 1.0)
    else:
        cons.append(cp.sum(w) <= 1.0)  # 允许留现金
        cons.append(cp.sum(w) >= 0.0)  # 防负杠杆（可按需删除）
    cons.append(w >= 0.0)
    cons.append(w <= ub)

    if tau2_budget is not None:
        # 约束 0.5 * ||Δw||_2 <= τ2_budget  =>  ||Δw||_2 <= 2*tau2_budget
        cons.append(cp.norm2(w - w_prev) <= 2.0 * float(tau2_budget))

    prob = cp.Problem(obj, cons)

    # 求解
    if solver.upper() == "OSQP":
        prob.solve(solver=cp.OSQP, verbose=verbose, eps_abs=1e-6, eps_rel=1e-6, max_iter=100000)
    elif solver.upper() == "ECOS":
        prob.solve(solver=cp.ECOS, verbose=verbose, abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=100000)
    else:
        prob.solve(verbose=verbose)  # 让 cvxpy 选择

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP not solved to optimality: status={prob.status}")

    w_opt = w.value.astype(float)
    w_opt = np.clip(w_opt, 0.0, ub)  # 数值稳健
    # 若 full_invest=True，做一次微调归一；若允许现金则不需
    if full_invest:
        s = w_opt.sum()
        if s > 0:
            w_opt = w_opt / s

    # 诊断
    diag = {
        "status": prob.status,
        "objective": float(prob.value),
        "alpha": float(z @ w_opt),
        "risk": float(0.5 * lambda_risk * (w_opt * sigma_diag @ w_opt)),
        "tc": float(gamma_tc * np.sum((w_opt - w_prev) ** 2)),
        "sum_w": float(w_opt.sum()),
        "turnover_l2_half": float(0.5 * np.linalg.norm(w_opt - w_prev)),  # 仅诊断（与预算保持一致单位）
        "turnover_l1_half": float(0.5 * np.abs(w_opt - w_prev).sum()),
    }
    return w_opt, diag