"""Section 5.4 (Tables 7-9 and the root-search diagnostics): where the gain over AV comes from.

Log-energy ordinate, N = 1024, H = 0.1..0.9, M = 5000, same test paths as Section 5, in three
variants, each with its own calibration bank:
  base      levels 2-7, all coefficients
  drop1     levels 3-7 (finest fitted level omitted)
  interior  levels 2-7, only coefficients whose support avoids the boundary extension
Estimators: AV, AV restricted to [0.01, 0.99], uncorrected WLS, mean-only calibrated root
(AV weights), fully calibrated root (GLS), one-step update from AV. For the base variant the
bias of the WLS slope is decomposed into a structural part (slope of log2 of the expected level
energies) and a logarithmic part. Output: results/sources.pkl
"""
import argparse, time
import numpy as np
from _common import leaf, H_GRID, M_TEST, test_seed, banks_for, save
from leaf.ordinates import level_details

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=M_TEST)
args = ap.parse_args()

N = 1024
LE = leaf.Ordinate("log-energy")
VARIANTS = {"base": (np.arange(2, 8), False), "drop1": (np.arange(3, 8), False), "interior": (np.arange(2, 8), True)}
out = {}
for var, (levels, interior) in VARIANTS.items():
    t0 = time.time()
    bank = banks_for(N, levels, interior, [LE])["log-energy"]
    n = bank.n
    q_wls = bank.q_wls
    g = leaf.av_correction(n)
    for H in H_GRID:
        P = leaf.fbm(N, H, np.random.default_rng(test_seed(N, H)), args.M)
        D = level_details(P, levels, interior)
        E = np.column_stack([np.mean(d ** 2, 1) for d in D])          # level energies
        Y = np.log2(E)
        av = ((Y - g) @ q_wls - 1) / 2
        mean_only, c_mean = bank.estimate_from_ordinates(Y, mean_only=True, return_codes=True)
        full, c_full = bank.estimate_from_ordinates(Y, return_codes=True)
        est = {"AV": av, "AV restricted": np.clip(av, 0.01, 0.99), "LE-WLS": (Y @ q_wls - 1) / 2,
               "mean-only": mean_only, "full": full, "one-step": bank.one_step(Y, av)}
        rec = {"est": {k: v.astype(np.float32) for k, v in est.items()},
               "codes_mean_only": np.bincount(c_mean, minlength=4),       # [interior, multiple, below, above]
               "codes_full": np.bincount(c_full, minlength=4),
               "AV_outside_grid": float(np.mean((av < 0.01) | (av > 0.99)))}
        if var == "base":
            rec["H_structural"] = float((np.log2(E.mean(0)) @ q_wls - 1) / 2)   # slope of log2 E(energy)
            rec["H_meanlog"] = float((Y.mean(0) @ q_wls - 1) / 2)               # slope of E log2(energy)
            rec["AV_correction"] = float(-(g @ q_wls) / 2)
        out[(var, H)] = rec
    print(f"{var}: {time.time() - t0:.0f}s", flush=True)
    save(out, "sources.pkl")
print("done")
