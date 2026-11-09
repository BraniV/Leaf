"""Quick start: estimate H for one fBm path, with and without outliers.

Builds two small calibration banks (log-energy and median-pilot LEAF, lambda = 0.1) for
N = 1024 with 500 paths per grid point (about a minute), saves them for reuse, and compares
AV, calibrated log-energy, and calibrated median-pilot LEAF on a clean and a contaminated path.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import leaf

N, H, levels = 1024, 0.3, np.arange(2, 8)
ords = [leaf.Ordinate("log-energy"), leaf.Ordinate("leaf", "median", 0.1)]

os.makedirs("banks", exist_ok=True)
files = {o.name: os.path.join("banks", f"quickstart_{o.name.replace(' ', '_')}.npz") for o in ords}
if all(os.path.exists(p) for p in files.values()):
    banks = {k: leaf.CalibrationBank.load(p) for k, p in files.items()}
else:
    banks = leaf.CalibrationBank.build_many(ords, N, levels, n_paths=500)
    for k, b in banks.items():
        b.save(files[k])

rng = np.random.default_rng(2026)
x = leaf.fbm(N, H, rng)                                     # one clean path, shape (1, N)
y = x.copy()
idx = rng.choice(N, 10, replace=False)                      # 1% additive outliers of size 10
y[0, idx] += 10 * rng.choice([-1, 1], 10)

n = leaf.level_counts(N, levels)
for label, path in [("clean", x), ("1% outliers", y)]:
    Y = leaf.ordinates(path, levels, ords)
    av = leaf.av_estimate(Y["log-energy"], levels, n)[0]
    le = banks["log-energy"].estimate(path)[0]
    le_mean_only = banks["log-energy"].estimate(path, mean_only=True)[0]
    lf = banks["LEAF median 0.1"].estimate(path)[0]
    print(f"{label:12s} true H={H}:  AV {av:.3f} | calibrated log-energy {le:.3f} "
          f"(mean-only {le_mean_only:.3f}) | calibrated LEAF median 0.1 {lf:.3f}")
