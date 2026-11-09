"""Exact Gaussian maximum likelihood estimation of H from fGn increments (benchmark).

The scale is profiled out: for unit-variance fGn correlation matrix R(h) and increments x,
    l(h) = -(N/2) log(x' R(h)^-1 x / N) - (1/2) log det R(h).
l is evaluated on a grid of step 0.005 in [0.005, 0.995] via Cholesky factors of the
Toeplitz matrix R(h), and the maximizer is refined by a quadratic fit through the three
grid points around the grid maximum. Cost: O(N^3) per grid point for the factorization
(shared by all paths) and O(N^2) per path.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular, toeplitz

from .fbm import fgn_autocovariance

MLE_GRID = np.round(np.arange(0.005, 0.996, 0.005), 3)


def _factors(N, grid):
    out = []
    for h in grid:
        L = cholesky(toeplitz(fgn_autocovariance(N - 1, h)), lower=True)
        out.append((L, np.sum(np.log(np.diag(L)))))
    return out


def profile_loglik(increments, grid=MLE_GRID, factors=None):
    """Profile log-likelihood on the grid; increments has shape (M, N). Returns (len(grid), M)."""
    x = np.atleast_2d(increments).T
    N, M = x.shape
    factors = factors or _factors(N, grid)
    return np.array([-0.5 * N * np.log(np.sum(solve_triangular(L, x, lower=True) ** 2, 0) / N) - logdet
                     for L, logdet in factors])


def fgn_mle(increments, grid=MLE_GRID, factors=None):
    """Exact Gaussian MLE of H for each row of `increments` (fGn model, scale profiled out).

    For an fBm path B(1..N) use increments = diff of [0, B(1), ..., B(N)].
    Pass `factors=leaf.mle._factors(N, grid)` to reuse Cholesky factors across calls.
    """
    ll = profile_loglik(increments, grid, factors)
    M = ll.shape[1]
    j = np.clip(ll.argmax(0), 1, len(grid) - 2)
    a, b, c = ll[j - 1, range(M)], ll[j, range(M)], ll[j + 1, range(M)]
    den = a - 2 * b + c
    off = np.where(den < 0, 0.5 * (a - c) / den, 0.0)
    return grid[j] + (grid[1] - grid[0]) * np.clip(off, -1, 1)
