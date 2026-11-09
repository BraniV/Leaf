"""Figure 1: the binding curve Hbar_N(h) and the sensitivity s_N(h) = q_N(h) mu_N'(h) / 2
for the log-energy ordinate, N = 1024, db2, levels 2-7.

Derivatives of the simulated mean mu_N(h) are taken from smoothing splines fitted to the
bank means on the calibration grid (one spline per level). Output: results/figure1.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from _common import leaf, RESULTS, banks_for

bank = banks_for(1024, np.arange(2, 8), False, [leaf.Ordinate("log-energy")])["log-energy"]
F, G = leaf.FINE, leaf.GRID
gi = np.rint((G - F[0]) / 0.001).astype(int)               # grid points inside FINE
MU, Q, qW = bank.mu, bank.Q, bank.q_wls
K = MU.shape[1]

# lightly smoothed mean (for the curves) ...
MUs = np.column_stack([UnivariateSpline(G, MU[gi, k], k=4, s=0.6e-4 * len(G))(F) for k in range(K)])
H_wls = (MUs @ qW - 1) / 2                                  # expected WLS log-energy estimate
H_gls = (np.einsum("hk,hk->h", Q, MUs) - 1) / 2             # expected covariance-only GLS estimate
dH_gls = UnivariateSpline(G, H_gls[gi], k=4, s=len(G) * 2e-5).derivative()(F)
# ... and a smoothing level matched to the Monte Carlo noise for the derivative mu'(h)
dMU = []
for k in range(K):
    y = MU[gi, k]
    noise = np.var(np.diff(y, 2)) / 6                       # second differences estimate the noise variance
    dMU.append(UnivariateSpline(G, y, k=4, s=3 * len(G) * noise).derivative()(F))
s_N = 0.5 * np.einsum("hk,hk->h", Q, np.column_stack(dMU))

for h in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9):
    i = int(round((h - F[0]) / 0.001))
    print(f"h={h:.2f}  s_N={s_N[i]:.2f}  Hbar'={dH_gls[i]:.2f}")

m = (F >= 0.03) & (F <= 0.97)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(F[m], H_wls[m], color="#1f4e79", lw=2, label="WLS log-energy")
ax[0].plot(F[m], H_gls[m], color="#c55a11", lw=1.5, ls="-.", label="GLS log-energy")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="identity")
ax[0].set_xlabel("true $h$"); ax[0].set_ylabel(r"$\overline{H}_N(h)$"); ax[0].legend(frameon=False)
ax[1].plot(F[m], s_N[m], color="#c55a11", lw=2, label=r"$s_N(h)=\frac{1}{2}q_N(h)\mu_N^{\prime}(h)$")
ax[1].plot(F[m], dH_gls[m], color="gray", lw=1.2, ls="--", label=r"$\overline{H}_N^{\prime}(h)$ (GLS)")
ax[1].axhline(1, color="k", ls=":", lw=1); ax[1].axhline(2, color="k", lw=0.6, alpha=0.5)
ax[1].set_ylim(0, 5); ax[1].set_xlabel("true $h$"); ax[1].set_ylabel("sensitivity"); ax[1].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "figure1.pdf"))
print("wrote results/figure1.pdf")
