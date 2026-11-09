"""Mean- and covariance-calibrated estimation of H (Section 4).

A CalibrationBank stores, for one ordinate, one signal length N and one set of fitted
levels, the finite-sample mean mu_N(h) and covariance Sigma_N(h) of the ordinate vector Y
under unit-scale fBm, estimated from independent simulations on the grid GRID, interpolated
linearly to the fine grid FINE, and the GLS slope vectors q_N(h) on FINE.

The calibrated estimator is the root of  G(h) = q(h) {Y - mu(h)} = 0  (full calibration)
or of  q_W {Y - mu(h)} = 0  with fixed AV weights q_W (mean-only calibration).
Root-search rules (Section 4.5): the first downcrossing of G on FINE, interpolated
linearly; h_min if G(h_min) < 0; h_max if G > 0 on the whole grid.
"""
import numpy as np

from .fbm import fbm
from .ordinates import Ordinate, level_details, level_counts, slope_vectors

GRID = np.round(np.arange(0.01, 0.991, 0.02), 2)          # calibration grid (50 points)
FINE = np.round(np.arange(0.01, 0.99 + 1e-9, 0.001), 3)    # interpolation / root-search grid
RIDGE = 1e-6

# diagnostic codes returned by the root search
INTERIOR, MULTIPLE, BELOW_GRID, ABOVE_GRID = 0, 1, 2, 3


def gls_slope(X, S):
    """GLS slope row vector e_2' (X' S^-1 X)^-1 X' S^-1."""
    Si = np.linalg.inv(S)
    return np.linalg.solve(X.T @ Si @ X, X.T @ Si)[1]


class CalibrationBank:
    """Simulated calibration of one ordinate for fixed N, levels, wavelet and boundary rule."""

    def __init__(self, ordinate, N, levels, interior, mu, Q):
        self.ordinate, self.N, self.levels, self.interior = ordinate, N, np.asarray(levels), interior
        self.mu, self.Q = mu, Q                        # arrays (len(FINE), K)
        self.n = level_counts(N, self.levels, interior)
        self.q_wls = slope_vectors(self.levels, self.n)[1]

    # ------------------------------------------------------------------ building
    @classmethod
    def build_many(cls, ordinates_, N, levels, n_paths=2000, interior=False, seed=7, verbose=False):
        """Build banks for several ordinates from the same simulated paths.

        At grid point index g, n_paths unit-scale fBm paths are drawn with
        np.random.default_rng([seed, N, g]); the paper uses seed = 7 and n_paths = 2000.
        """
        levels = np.asarray(levels)
        K = len(levels)
        X = np.column_stack([np.ones(K), levels])
        mus = {o.name: [] for o in ordinates_}
        Ss = {o.name: [] for o in ordinates_}
        for g, h in enumerate(GRID):
            P = fbm(N, h, np.random.default_rng([seed, N, g]), n_paths)
            D = level_details(P, levels, interior)
            for o in ordinates_:
                Y = o(D)
                S = np.cov(Y.T)
                S = S + RIDGE * np.trace(S) / K * np.eye(K)
                mus[o.name].append(Y.mean(0))
                Ss[o.name].append(S)
            if verbose and g % 10 == 0:
                print(f"  bank N={N}: grid point {g + 1}/{len(GRID)}", flush=True)
        banks = {}
        for o in ordinates_:
            mu_g, S_g = np.array(mus[o.name]), np.array(Ss[o.name])
            MU = np.empty((len(FINE), K))
            Q = np.empty((len(FINE), K))
            for i, h in enumerate(FINE):                 # linear interpolation of mu and Sigma
                t = np.interp(h, GRID, np.arange(len(GRID)))
                a = min(int(t), len(GRID) - 2)
                w = t - a
                MU[i] = (1 - w) * mu_g[a] + w * mu_g[a + 1]
                Q[i] = gls_slope(X, (1 - w) * S_g[a] + w * S_g[a + 1])
            banks[o.name] = cls(o, N, levels, interior, MU, Q)
        return banks

    @classmethod
    def build(cls, ordinate=Ordinate(), N=1024, levels=None, n_paths=2000, interior=False, seed=7):
        """Bank for a single ordinate. Default levels are 2..J-3 for N = 2^J."""
        if levels is None:
            levels = np.arange(2, int(np.log2(N)) - 2)
        return cls.build_many([ordinate], N, levels, n_paths, interior, seed)[ordinate.name]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        o = self.ordinate
        np.savez_compressed(path, kind=o.kind, pilot=o.pilot, lam=o.lam, N=self.N, levels=self.levels,
                            interior=self.interior, mu=self.mu, Q=self.Q)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        o = Ordinate(str(z["kind"]), str(z["pilot"]), float(z["lam"]))
        return cls(o, int(z["N"]), z["levels"], bool(z["interior"]), z["mu"], z["Q"])

    # ------------------------------------------------------------------ estimation
    def ordinate_matrix(self, paths):
        """Ordinate vectors (M, K) of the given paths, with this bank's settings."""
        return self.ordinate(level_details(np.atleast_2d(paths), self.levels, self.interior))

    def _root(self, G):
        """Apply the root-search rules to G (M, len(FINE)); returns estimates and codes."""
        neg = G < 0
        n_changes = np.sum(neg[:, 1:] != neg[:, :-1], axis=1)
        first = np.where(neg.any(1), neg.argmax(1), len(FINE) - 1)
        est = np.empty(len(G))
        code = np.full(len(G), INTERIOR)
        for m, i in enumerate(first):
            if not neg[m, i]:                       # G > 0 everywhere
                est[m], code[m] = FINE[-1], ABOVE_GRID
            elif i == 0:                            # G(h_min) < 0
                est[m], code[m] = FINE[0], BELOW_GRID
            else:                                   # first downcrossing, linear interpolation
                g0, g1 = G[m, i - 1], G[m, i]
                est[m] = FINE[i - 1] + (FINE[i] - FINE[i - 1]) * g0 / (g0 - g1)
                code[m] = MULTIPLE if n_changes[m] > 1 else INTERIOR
        return est, code

    def estimate_from_ordinates(self, Y, mean_only=False, return_codes=False):
        """Calibrated estimate(s) of H from ordinate vectors Y (M, K) or (K,).

        mean_only=False : root of q(h){Y - mu(h)} with the calibrated GLS slope q(h)
        mean_only=True  : root of q_W{Y - mu(h)} with fixed AV weights (mean-only calibration)
        """
        Y = np.atleast_2d(Y)
        if mean_only:
            G = (Y @ self.q_wls)[:, None] - (self.mu @ self.q_wls)[None, :]
        else:
            G = np.einsum("hk,mk->mh", self.Q, Y) - np.einsum("hk,hk->h", self.Q, self.mu)[None, :]
        est, code = self._root(G)
        return (est, code) if return_codes else est

    def estimate(self, paths, mean_only=False, return_codes=False):
        """Calibrated estimate(s) of H from observed path(s) of length N (rows)."""
        return self.estimate_from_ordinates(self.ordinate_matrix(paths), mean_only, return_codes)

    def one_step(self, Y, h1):
        """One-step update h1 + q(h1){Y - mu(h1)}/2 (Section 4.3).

        Starting values outside [0.01, 0.99] are projected onto the grid and the update is
        evaluated there; the returned one-step estimate itself is not projected.
        """
        Y = np.atleast_2d(Y)
        i = np.clip(np.rint((np.clip(h1, FINE[0], FINE[-1]) - FINE[0]) / 0.001).astype(int), 0, len(FINE) - 1)
        return FINE[i] + 0.5 * np.einsum("mk,mk->m", self.Q[i], Y - self.mu[i])

    def covariance_only(self, Y, h1):
        """GLS slope estimate with the covariance calibrated at h1 but no mean correction."""
        Y = np.atleast_2d(Y)
        i = np.clip(np.rint((np.clip(h1, FINE[0], FINE[-1]) - FINE[0]) / 0.001).astype(int), 0, len(FINE) - 1)
        return (np.einsum("mk,mk->m", self.Q[i], Y) - 1) / 2
