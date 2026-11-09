"""Simulation of fractional Gaussian noise / fractional Brownian motion and of the
contamination scenarios of Section 6.

All generators take a numpy Generator and return arrays of shape (M, N), one path per row.
The random-number streams are consumed exactly as in the paper, so that the seeds used
in the reproduction scripts regenerate the same paths.
"""
import numpy as np
from scipy.stats import norm, t as tdist


def fgn_autocovariance(N, H):
    """Autocovariance gamma(k), k = 0..N, of unit-variance fractional Gaussian noise."""
    k = np.arange(N + 1)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fbm(N, H, rng, M=1, chunk=500):
    """Exact fBm paths B(1), ..., B(N) by Davies--Harte circulant embedding of fGn.

    The fGn autocovariance is embedded in a circulant of length 2N, whose eigenvalues are
    nonnegative for every H in (0, 1); the paths are cumulative sums of the fGn.
    Paths are generated in chunks of `chunk` rows (this fixes the random stream; keep the
    default to reproduce the paper).
    """
    g = fgn_autocovariance(N, H)
    c = np.concatenate([g, g[-2:0:-1]])                 # first row of the 2N circulant
    lam = np.maximum(np.fft.fft(c).real, 0.0)          # its eigenvalues (clip round-off)
    L = len(c)
    out = np.empty((M, N))
    for s in range(0, M, chunk):
        m = min(chunk, M - s)
        Z = rng.standard_normal((m, L)) + 1j * rng.standard_normal((m, L))
        out[s:s + m] = np.cumsum(np.fft.fft(np.sqrt(lam / L) * Z, axis=1).real[:, :N], axis=1)
    return out


def fgn(N, H, rng, M=1):
    """Exact fGn (increments of `fbm`), shape (M, N)."""
    b = fbm(N, H, rng, M)
    return np.diff(np.concatenate([np.zeros((M, 1)), b], axis=1), axis=1)


def arfima_increments(N, d, rng, M=1, innovations="gauss", burn=8192):
    """ARFIMA(0, d, 0) series by a truncated MA(infinity) filter with a burn-in.

    innovations: "gauss" or an integer nu (Student t with nu degrees of freedom, scaled to
    unit variance). The output is standardized; the scale is irrelevant for all estimators.
    H = d + 1/2 for the integrated series.
    """
    L = N + burn
    k = np.arange(1, L)
    psi = np.concatenate([[1.0], np.cumprod((k - 1 + d) / k)])   # MA weights of (1-B)^(-d)
    if innovations == "gauss":
        e = rng.standard_normal((M, L))
    else:
        nu = innovations
        e = rng.standard_t(nu, (M, L)) / np.sqrt(nu / (nu - 2))
    n2 = 1 << int(np.ceil(np.log2(2 * L)))
    x = np.fft.irfft(np.fft.rfft(e, n2) * np.fft.rfft(psi, n2), n2)[:, :L]
    x = x[:, burn:]
    return x / np.std(x)


SCENARIOS = ["clean", "AO p=0.5% k=10", "AO p=1% k=10", "AO p=2% k=10", "AO p=1% k=5",
             "Jumps p=1% k=10", "ARFIMA gauss", "ARFIMA t5", "ARFIMA t3", "fGn->t3 marginals"]
_AO = {"AO p=0.5% k=10": (0.005, 10), "AO p=1% k=10": (0.01, 10),
       "AO p=2% k=10": (0.02, 10), "AO p=1% k=5": (0.01, 5)}


def scenario(name, H, M, seed, N=1024):
    """Observed paths (M, N) for the scenarios of Section 6.

    clean              fBm
    AO p=..% k=..      fBm + additive outliers +-kappa (in increment SDs) w.p. p at each time
    Jumps p=1% k=10    outliers of the same form added to the increments (level shifts)
    ARFIMA ...         integrated ARFIMA(0, H-1/2, 0), Gaussian / t5 / t3 innovations
    fGn->t3 marginals  fGn transformed monotonically to standardized t3 marginals (H is the
                       latent Gaussian parameter; for H < 1/2 antipersistence is not preserved)
    `seed` drives the Gaussian part; contamination uses the stream seed + 1.
    """
    rng = np.random.default_rng(seed)
    if name.startswith("ARFIMA"):
        innov = {"ARFIMA gauss": "gauss", "ARFIMA t5": 5, "ARFIMA t3": 3}[name]
        return np.cumsum(arfima_increments(N, H - 0.5, rng, M, innov), axis=1)
    x = fgn(N, H, rng, M)
    r2 = np.random.default_rng(seed + 1)
    if name == "clean":
        return np.cumsum(x, axis=1)
    if name in _AO:
        p, kap = _AO[name]
        y = np.cumsum(x, axis=1)
        mask = r2.random((M, N)) < p
        return y + mask * kap * r2.choice([-1, 1], (M, N))
    if name == "Jumps p=1% k=10":
        mask = r2.random((M, N)) < 0.01
        return np.cumsum(x + mask * 10 * r2.choice([-1, 1], (M, N)), axis=1)
    if name == "fGn->t3 marginals":
        nu = 3
        z = tdist.ppf(norm.cdf(x), nu) / np.sqrt(nu / (nu - 2))
        return np.cumsum(z, axis=1)
    raise ValueError(f"unknown scenario {name!r}")
