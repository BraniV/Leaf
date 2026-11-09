"""Section 5 (Tables 2-6): Gaussian fBm, N in {512, 1024, 4096}, H = 0.1..0.9, M = 5000.

For every path and every ordinate of the paper this computes the uncorrected WLS slope
(AV weights), the covariance-only GLS slope at the AV estimate, the one-step update from AV,
and the calibrated root; plus AV and the OLS log-energy slope. All estimators use the same
paths (common random numbers). Output: results/gaussian.pkl

Usage: python scripts/run_gaussian.py [--N 1024] [--M 5000]
Runtime: about 20-30 minutes for all three N on a laptop.
"""
import argparse, time
import numpy as np
from _common import leaf, H_GRID, M_TEST, test_seed, default_levels, banks_for, save

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="*", default=[1024, 512, 4096])
ap.add_argument("--M", type=int, default=M_TEST)
args = ap.parse_args()

out = {}
for N in args.N:
    levels = default_levels(N)
    banks = banks_for(N)
    n = leaf.level_counts(N, levels)
    q_ols, q_wls = leaf.slope_vectors(levels, n)
    for H in H_GRID:
        t0 = time.time()
        P = leaf.fbm(N, H, np.random.default_rng(test_seed(N, H)), args.M)
        O = leaf.ordinates(P, levels, leaf.PAPER_ORDINATES)
        av = leaf.av_estimate(O["log-energy"], levels, n)
        est = {"AV": av, "Classical OLS": (O["log-energy"] @ q_ols - 1) / 2}
        for name, Y in O.items():
            b = banks[name]
            est[f"{name}|WLS"] = (Y @ q_wls - 1) / 2
            est[f"{name}|GLS"] = b.covariance_only(Y, av)
            est[f"{name}|one-step"] = b.one_step(Y, av)
            est[f"{name}|root"] = b.estimate_from_ordinates(Y)
        out[(N, H)] = {k: v.astype(np.float32) for k, v in est.items()}
        print(f"N={N} H={H}: {time.time() - t0:.0f}s", flush=True)
    save(out, "gaussian.pkl")
print("done")
