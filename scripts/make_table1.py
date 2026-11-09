"""Table 1: asymptotic variance V(lambda), efficiency relative to log-energy, and pilot weight
w(lambda), by numerical quadrature. Also prints the variance-minimizing median-pilot floor.
Output: results/tables/table1_vlam.tex
"""
import os
import numpy as np
from _common import RESULTS
from leaf.theory import asymptotic_variance, pilot_weight, V_LOG_ENERGY, optimal_median_floor

lam_star = optimal_median_floor()
rows = []
for lam in [0, 0.05, lam_star, 0.25, 0.5, 1, 2, 5, np.inf]:
    vm, vd = asymptotic_variance(lam, "mean"), asymptotic_variance(lam, "median")
    lab = r"$\infty$" if np.isinf(lam) else (f"{lam:.3f}" if lam == lam_star else f"{lam:g}")
    rows.append(f"{lab} & {vm:.3f} & {V_LOG_ENERGY / vm:.3f} & {vd:.3f} & {V_LOG_ENERGY / vd:.3f} & {pilot_weight(lam):.3g}\\\\")
open(os.path.join(RESULTS, "tables", "table1_vlam.tex"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
print(f"optimal median-pilot floor: {lam_star:.4f}, efficiency {V_LOG_ENERGY / asymptotic_variance(lam_star, 'median'):.4f}")
