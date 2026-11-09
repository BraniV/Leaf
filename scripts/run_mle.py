"""Exact Gaussian MLE benchmark (Tables 8, 9 and 11).

Clean fBm: the first 1000 test paths of Section 5 (N = 1024) at H in {0.1,0.2,0.3,0.5,0.7,0.9}.
Contamination: the first 1000 paths of selected Section 6 scenarios at H in {0.2, 0.5, 0.8}.
Output: results/mle.pkl
"""
import time
import numpy as np
from _common import leaf, test_seed, robust_seed, save
from leaf.mle import MLE_GRID, _factors

N, MSUB = 1024, 1000
t0 = time.time()
F = _factors(N, MLE_GRID)                       # Cholesky factors shared by all paths
print(f"Cholesky factors: {time.time() - t0:.0f}s", flush=True)
incr = lambda P: np.diff(np.concatenate([np.zeros((len(P), 1)), P], axis=1), axis=1)
out = {}
for H in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    P = leaf.fbm(N, H, np.random.default_rng(test_seed(N, H)), 5000)[:MSUB]   # same stream as Section 5
    out[("clean", H)] = leaf.fgn_mle(incr(P), factors=F)
    print("clean", H, flush=True)
for sc in ["AO p=0.5% k=10", "AO p=1% k=10", "AO p=2% k=10", "AO p=1% k=5", "ARFIMA gauss", "ARFIMA t3"]:
    for H in [0.2, 0.5, 0.8]:
        P = leaf.scenario(sc, H, 5000, robust_seed(H), N)[:MSUB]
        out[(sc, H)] = leaf.fgn_mle(incr(P), factors=F)
    print(sc, flush=True)
    save(out, "mle.pkl")
save(out, "mle.pkl")
print("done")
