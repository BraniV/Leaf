"""Logscale ordinates and the uncalibrated Abry--Veitch estimator.

For coefficients d_{j,1..n_j} at level j:

  log-energy   Y_j = log2( mean_k d_{j,k}^2 )
  log-median   Y_j = log2( median_k d_{j,k}^2 / m1 ),   m1 = median of chi^2_1 = 0.4549...
  avg-log      Y_j = mean_k log2( d_{j,k}^2 )                          (LEAF with lambda = 0)
  LEAF         Y_j = mean_k log2( d_{j,k}^2 + lambda * s_j^2 ),         s_j^2 = mean or median pilot

H is recovered from the slope beta of Y_j on j as H = (beta - 1) / 2.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import chi2

from .dwt import details, interior_masks

MEDIAN_CHI2_1 = chi2.ppf(0.5, 1)          # 0.4549364..., Gaussian calibration of the median pilot


@dataclass(frozen=True)
class Ordinate:
    """Specification of a logscale ordinate.

    kind  : "log-energy", "log-median", "avg-log" or "leaf"
    pilot : "mean" or "median" (LEAF only)
    lam   : floor parameter lambda > 0 (LEAF only)
    """
    kind: str = "log-energy"
    pilot: str = "mean"
    lam: float = 0.0

    @property
    def name(self):
        return f"LEAF {self.pilot} {self.lam:g}" if self.kind == "leaf" else self.kind

    def __call__(self, D):
        """Ordinate matrix (M, K) from a list of K detail arrays, each of shape (M, n_j)."""
        if self.kind == "log-energy":
            return np.column_stack([np.log2(np.mean(d ** 2, 1)) for d in D])
        if self.kind == "log-median":
            return np.column_stack([np.log2(np.median(d ** 2, 1) / MEDIAN_CHI2_1) for d in D])
        if self.kind == "avg-log":
            return np.column_stack([np.mean(np.log2(d ** 2), 1) for d in D])
        if self.kind == "leaf":
            cols = []
            for d in D:
                s2 = np.mean(d ** 2, 1) if self.pilot == "mean" else np.median(d ** 2, 1) / MEDIAN_CHI2_1
                cols.append(np.mean(np.log2(d ** 2 + self.lam * s2[:, None]), 1))
            return np.column_stack(cols)
        raise ValueError(self.kind)


# The ordinates reported in the paper (Sections 5 and 6)
PAPER_ORDINATES = ([Ordinate("log-energy"), Ordinate("log-median"), Ordinate("avg-log")]
                   + [Ordinate("leaf", "mean", l) for l in (0.1, 0.25, 0.5, 1, 2)]
                   + [Ordinate("leaf", "median", l) for l in (0.05, 0.1, 0.25, 0.5, 1, 2)])


def level_details(paths, levels, interior=False):
    """Detail coefficients at the fitted levels; optionally only boundary-free coefficients."""
    N = np.atleast_2d(paths).shape[1]
    ds = details(paths, max(levels))
    if interior:
        masks = interior_masks(N, max(levels))
        return [ds[j - 1][:, masks[j - 1]] for j in levels]
    return [ds[j - 1] for j in levels]


def ordinates(paths, levels, ords=(Ordinate(),), interior=False):
    """Dictionary {ordinate name: (M, K) array} for the paths (rows) and fitted levels."""
    D = level_details(paths, levels, interior)
    return {o.name: o(D) for o in ords}


def level_counts(N, levels, interior=False):
    """Number of coefficients n_j at each fitted level."""
    return np.array([d.shape[1] for d in level_details(np.zeros((1, N)), levels, interior)])


def av_correction(n):
    """Abry--Veitch bias of log2 of a mean of n independent chi^2_1/1 variables:
    g(n) = psi(n/2)/log 2 - log2(n/2)."""
    return digamma(n / 2) / np.log(2) - np.log2(n / 2)


def av_weights(n):
    """Abry--Veitch WLS weights, the inverse of Var(Y_j) = psi'(n_j/2)/(log 2)^2."""
    return 1.0 / (polygamma(1, n / 2) / np.log(2) ** 2)


def slope_vectors(levels, n):
    """OLS and AV-weighted (WLS) slope row vectors q with q 1 = 0 and q j = 1."""
    js = np.asarray(levels, float)
    X = np.column_stack([np.ones_like(js), js])
    w = av_weights(n)
    q_ols = np.linalg.solve(X.T @ X, X.T)[1]
    q_wls = np.linalg.solve(X.T @ (w[:, None] * X), X.T * w)[1]
    return q_ols, q_wls


def av_estimate(Y_log_energy, levels, n):
    """Abry--Veitch estimate of H from log-energy ordinates (M, K): digamma-corrected WLS."""
    _, q_wls = slope_vectors(levels, n)
    return ((Y_log_energy - av_correction(n)) @ q_wls - 1) / 2
