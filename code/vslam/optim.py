"""Nonlinear least squares on manifolds (Chapter 4).

A problem describes its unknowns as an opaque state x living on a
manifold (a vector, a pose, a tuple of both...). It provides

    linearize(x) -> (r, J)     whitened residual r (m,) and its Jacobian
                               J (m, n) with respect to a tangent
                               perturbation delta at x
    retract(x, delta) -> x'    the "plus" of Chapter 2, x (+) delta

and, for robust kernels, an attribute `block_size`: the residual vector
is a stack of blocks of that size (2 for a pixel), and the kernel acts on
the squared norm of each block.

The solvers minimise F(x) = 1/2 sum_i rho(|r_i(x)|^2) (rho(s) = s for
plain least squares) by Gauss-Newton or Levenberg-Marquardt, both with
iteratively reweighted least squares for robust kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .robust import Trivial


class Problem(Protocol):
    block_size: int

    def linearize(self, x: Any) -> tuple[np.ndarray, np.ndarray]: ...

    def retract(self, x: Any, delta: np.ndarray) -> Any: ...


@dataclass
class Result:
    """Final state, convergence flag and per-iteration history; each entry
    holds the cost and the current state "x" when it was recorded."""

    x: Any
    converged: bool
    iterations: int
    history: list[dict] = field(default_factory=list)

    @property
    def costs(self) -> np.ndarray:
        return np.array([h["cost"] for h in self.history])


def _weighted(problem: Problem, x: Any, kernel
              ) -> tuple[float, np.ndarray, np.ndarray]:
    """Robust cost, and residual/Jacobian scaled by sqrt of the IRLS
    weights (the weights are frozen at x)."""
    r, J = problem.linearize(x)
    m = getattr(problem, "block_size", 1)
    s = np.sum(r.reshape(-1, m) ** 2, axis=1)
    cost = 0.5 * float(np.sum(kernel.rho(s)))
    sw = np.repeat(np.sqrt(kernel.weight(s)), m)
    return cost, sw * r, sw[:, None] * J


def robust_cost(problem: Problem, x: Any, kernel=None) -> float:
    """F(x) = 1/2 sum_i rho(|r_i|^2)."""
    kernel = kernel or Trivial()
    r, _ = problem.linearize(x)
    m = getattr(problem, "block_size", 1)
    s = np.sum(r.reshape(-1, m) ** 2, axis=1)
    return 0.5 * float(np.sum(kernel.rho(s)))


def gauss_newton(problem: Problem, x0: Any, kernel=None,
                 max_iters: int = 50, tol: float = 1e-10) -> Result:
    """Undamped Gauss-Newton: solve J^T J delta = -J^T r and retract.

    No step control: the cost may increase. Stops when the relative cost
    decrease, or the step, falls below `tol`.
    """
    kernel = kernel or Trivial()
    x = x0
    history = []
    for it in range(max_iters):
        cost, r, J = _weighted(problem, x, kernel)
        g = J.T @ r
        H = J.T @ J
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -r, rcond=None)[0]
        history.append({"cost": cost, "grad": float(np.abs(g).max()),
                        "step": float(np.linalg.norm(delta)), "x": x})
        x_new = problem.retract(x, delta)
        if (np.linalg.norm(delta) < tol
                or (it > 0 and abs(history[-2]["cost"] - cost)
                    <= tol * max(cost, 1e-300))):
            return Result(x_new, True, it + 1, history)
        x = x_new
    history.append({"cost": robust_cost(problem, x, kernel), "x": x})
    return Result(x, False, max_iters, history)


def levenberg_marquardt(problem: Problem, x0: Any, kernel=None,
                        max_iters: int = 100, tau: float = 1e-3,
                        marquardt: bool = True, tol_grad: float = 1e-10,
                        tol_step: float = 1e-12,
                        tol_cost: float = 1e-10) -> Result:
    """Levenberg-Marquardt with Nielsen's damping update.

    Solves (J^T J + lambda D) delta = -g with D = I (Levenberg) or
    D = diag(J^T J) (Marquardt scaling), and accepts the step when the gain
    ratio rho = (F(x) - F(x+)) / (L(0) - L(delta)) is positive, where
    L(0) - L(delta) = 1/2 delta^T (lambda D delta - g).
    """
    kernel = kernel or Trivial()
    x = x0
    cost, r, J = _weighted(problem, x, kernel)
    g, H = J.T @ r, J.T @ J
    # With Marquardt scaling lambda is dimensionless (relative to diag(H));
    # with Levenberg's D = I it has the units of H.
    lam = tau if marquardt else tau * float(np.max(np.diag(H)))
    nu = 2.0
    history = [{"cost": cost, "lambda": lam, "accepted": True,
                "grad": float(np.abs(g).max()), "x": x}]
    for it in range(max_iters):
        if np.abs(g).max() < tol_grad:
            return Result(x, True, it, history)
        D = np.maximum(np.diag(H), 1e-12) if marquardt else np.ones(len(g))
        delta = np.linalg.solve(H + lam * np.diag(D), -g)
        if np.linalg.norm(delta) < tol_step:
            return Result(x, True, it, history)
        x_new = problem.retract(x, delta)
        cost_new, r_new, J_new = _weighted(problem, x_new, kernel)
        predicted = 0.5 * delta @ (lam * D * delta - g)
        gain = (cost - cost_new) / predicted if predicted > 0 else -1.0
        if gain > 0:  # accept, decrease damping
            small = cost - cost_new <= tol_cost * cost
            x, cost, r, J = x_new, cost_new, r_new, J_new
            g, H = J.T @ r, J.T @ J
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3)
            nu = 2.0
        else:  # reject, increase damping
            lam *= nu
            nu *= 2.0
        history.append({"cost": cost, "lambda": lam, "accepted": gain > 0,
                        "gain": gain, "grad": float(np.abs(g).max()),
                        "x": x})
        if gain > 0 and small:
            return Result(x, True, it + 1, history)
    return Result(x, False, max_iters, history)


def covariance(problem: Problem, x: Any) -> np.ndarray:
    """Laplace approximation of the covariance of the tangent perturbation
    at an optimum: (J^T J)^{-1}, pseudo-inverse if singular (gauge)."""
    _, J = problem.linearize(x)
    return np.linalg.pinv(J.T @ J)
