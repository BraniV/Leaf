"""Gaussian theory for LEAF ordinates (Section 3, Table 1).

All quantities are for one level with independent N(0, sigma^2) coefficients; Z ~ N(0,1).

  b(rho)      = E log2(Z^2 + rho)                         calibration function (Definition 2)
  w(lambda)   = lambda E (Z^2 + lambda)^-1                 pilot weight in the expansion
  IF(z)       = [log(z^2+lambda) - E log(Z^2+lambda) + w(lambda) phi(z)] / log 2
  V(lambda)   = E IF(Z)^2                                  asymptotic variance (Theorem 1)

with phi(z) = z^2 - 1 for the mean pilot and
phi(z) = (1/2 - 1{z^2 <= m1}) / (m1 f1(m1)) for the median pilot (m1, f1: median and
density of chi^2_1). The log-energy bound is V = 2 / (log 2)^2 (Proposition 2).
"""
import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm, chi2

LOG2 = np.log(2)
M1 = chi2.ppf(0.5, 1)
F1 = chi2.pdf(M1, 1)
V_LOG_ENERGY = 2 / LOG2 ** 2


def _expect(f):
    """E f(Z) for Z ~ N(0,1) and even f, by adaptive quadrature on (0, inf).

    The range is split at sqrt(m1), where the median-pilot influence function jumps, and
    near 0, where log z^2 is singular.
    """
    g = lambda z: f(z) * norm.pdf(z)
    r = np.sqrt(M1)
    parts = [(0.0, 1e-3), (1e-3, r), (r, 8.0), (8.0, np.inf)]
    return 2 * sum(integrate.quad(g, a, b, limit=400)[0] for a, b in parts)


def calibration_b(rho):
    """b(rho) = E log2(Z^2 + rho)."""
    return _expect(lambda z: np.log(z * z + rho)) / LOG2


def pilot_weight(lam):
    """w(lambda) = lambda E (Z^2 + lambda)^-1, increasing from 0 to 1."""
    if lam == 0:
        return 0.0
    if np.isinf(lam):
        return 1.0
    return lam * _expect(lambda z: 1.0 / (z * z + lam))


def _phi(pilot):
    if pilot == "mean":
        return lambda z: z * z - 1
    if pilot == "median":
        return lambda z: (0.5 - (z * z <= M1)) / (M1 * F1)
    raise ValueError(pilot)


def asymptotic_variance(lam, pilot="mean"):
    """V(lambda) per coefficient, in log2 units (Theorem 1; lambda = 0 and inf included)."""
    if lam == 0:
        return (np.pi ** 2 / 2) / LOG2 ** 2                       # Var log Z^2 = psi'(1/2)
    phi = _phi(pilot)
    if np.isinf(lam):
        return _expect(lambda z: phi(z) ** 2) / LOG2 ** 2
    c = pilot_weight(lam)
    m = _expect(lambda z: np.log(z * z + lam))
    return _expect(lambda z: (np.log(z * z + lam) - m + c * phi(z)) ** 2) / LOG2 ** 2


def efficiency(lam, pilot="mean"):
    """Asymptotic efficiency relative to log-energy, V(inf, mean) / V(lambda, pilot)."""
    return V_LOG_ENERGY / asymptotic_variance(lam, pilot)


def optimal_median_floor():
    """Variance-minimizing floor for the median pilot (about 0.096; efficiency about 0.597)."""
    r = optimize.minimize_scalar(lambda t: asymptotic_variance(np.exp(t), "median"),
                                 bounds=(np.log(0.01), np.log(1.0)), method="bounded")
    return float(np.exp(r.x))
