"""Section 6 (Table 10): contaminated and non-Gaussian scenarios, N = 1024, H in {0.2, 0.5, 0.8}.

All estimators are calibrated with the clean Gaussian fBm banks of Section 5 (levels 2-7).
The scenarios share their Gaussian draws (seed 5000 + 10 H; contamination uses seed + 1).
The "Clean fBm" rows of Table 10 come from results/gaussian.pkl.
Output: results/contamination.pkl

Usage: python scripts/run_contamination.py [--M 5000]
"""
import argparse, time
import numpy as np
from _common import leaf, H_ROBUST, M_TEST, robust_seed, default_levels, banks_for, save

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=M_TEST)
args = ap.parse_args()

N = 1024
levels = default_levels(N)
banks = banks_for(N)
n = leaf.level_counts(N, levels)
out = {}
for sc in leaf.SCENARIOS[1:]:
    t0 = time.time()
    for H in H_ROBUST:
        P = leaf.scenario(sc, H, args.M, robust_seed(H), N)
        O = leaf.ordinates(P, levels, leaf.PAPER_ORDINATES)
        est = {"AV": leaf.av_estimate(O["log-energy"], levels, n)}
        for name, Y in O.items():
            est[f"{name}|root"] = banks[name].estimate_from_ordinates(Y)
        out[(sc, H)] = {k: v.astype(np.float32) for k, v in est.items()}
    print(f"{sc}: {time.time() - t0:.0f}s", flush=True)
    save(out, "contamination.pkl")
print("done")
