# LEAF: calibrated and robust wavelet estimation of the Hurst exponent

Companion code for

> V. Lakmini and B. Vidakovic. *Calibrated Log-Energy with Adaptive Floor: Bias-Corrected and Robust Wavelet Estimation of the Hurst Exponent.* Department of Statistics, Texas A&M University.

The package estimates the Hurst exponent $H$ of fractional Brownian motion (fBm) from wavelet logscale diagrams. It combines two ideas.

1. **LEAF ordinates (Log-Energy with Adaptive Floor).** At wavelet level $j$, with coefficients $d_{j,1},\dots,d_{j,n_j}$ and a pilot estimate $\widehat\sigma_j^2$ of the level variance (mean or median energy),

   $$Y_j(\lambda)=\frac{1}{n_j}\sum_{k=1}^{n_j}\log_2\left(d_{j,k}^2+\lambda\,\widehat\sigma_j^2\right).$$

   The floor is proportional to the level variance, so it does not bias the scaling slope. It stabilizes near-zero energies. With a median pilot the influence of a single outlying coefficient grows only logarithmically. $\lambda=0$ gives the average-log ordinate. As $\lambda\to\infty$ the ordinate reduces to the pilot ordinate: log-energy for the mean pilot, log-median for the median pilot.

2. **Calibration.** For any ordinate vector $\boldsymbol Y=(Y_{j_1},\dots,Y_{j_K})$, the finite-sample mean $\boldsymbol\mu_N(h)$ and covariance $\boldsymbol\Sigma_N(h)$ under fBm are simulated once (a *calibration bank*). $H$ is then estimated as the root of

   $$q_N(h)\{\boldsymbol Y-\boldsymbol\mu_N(h)\}=0 .$$

   Here $q_N(h)$ is the GLS slope vector built from $\boldsymbol\Sigma_N(h)$. In the *mean-only* version, $q_N(h)$ is replaced by fixed Abry–Veitch weights. Calibration corrects the structural finite-sample bias of sampled fBm at fine scales, which the Abry–Veitch correction does not address.

## Installation

The code needs Python 3.9 or later with NumPy, SciPy and Matplotlib.

```bash
git clone <repository-url> leaf-hurst
cd leaf-hurst
pip install -r requirements.txt
```

The package is the folder `leaf/`. Scripts and examples add the repository root to the Python path, so no installation step is needed.

## Quick start

```python
import numpy as np
import leaf

N, levels = 1024, np.arange(2, 8)                         # db2, levels 2-7 (paper's main design)
ordinate = leaf.Ordinate("leaf", pilot="median", lam=0.1)  # robust LEAF ordinate

# One-time calibration bank (2000 paths per grid point reproduces the paper; 500 is fine for a test)
bank = leaf.CalibrationBank.build(ordinate, N, levels, n_paths=2000)
bank.save("bank_leaf_median_0.1_N1024.npz")              # reload with leaf.CalibrationBank.load(...)

x = leaf.fbm(N, 0.3, np.random.default_rng(1))           # an fBm path, shape (1, N); use your data here
print(bank.estimate(x))                                  # fully calibrated root
print(bank.estimate(x, mean_only=True))                  # mean-only calibration (practical default)

# Abry-Veitch estimate for comparison
n = leaf.level_counts(N, levels)
Y = leaf.ordinates(x, levels, [leaf.Ordinate("log-energy")])["log-energy"]
print(leaf.av_estimate(Y, levels, n))
```

`examples/quickstart.py` compares AV, calibrated log-energy and calibrated median-pilot LEAF on a clean path and on the same path with 1% additive outliers.

**Which ordinate to use.** If the data are believed to be Gaussian, use the mean pilot with $\lambda\in[0.5,2]$; it is practically equivalent to log-energy. If heavy tails are suspected, use the mean pilot with $\lambda\approx0.1$–$0.25$. If isolated outliers are suspected, use the median pilot with $\lambda\approx0.1$. A calibration bank is specific to $N$, the wavelet, the boundary rule, the fitted levels, the ordinate and $\lambda$, so build one bank per configuration.

**Data requirements.** Paths must be the fBm (cumulative) series, not the increments, with length $N$ equal to that of the bank. The calibration assumes fBm. Under a different model, such as ARFIMA at small $H$, it can be less accurate than AV; see Section 6 of the paper.

## Repository layout

```
leaf/                      the package
  fbm.py                   Davies-Harte fGn/fBm, ARFIMA, contamination scenarios (Section 6)
  dwt.py                   db2 DWT with half-point symmetric extension; boundary-free coefficient masks
  ordinates.py             log-energy, LEAF, average-log, log-median ordinates; AV estimator and weights
  calibration.py           CalibrationBank: simulated mean/covariance, root search, one-step update
  theory.py                b(rho), w(lambda), asymptotic variance V(lambda), efficiency (Table 1)
  mle.py                   exact Gaussian maximum likelihood benchmark (fGn, scale profiled out)
scripts/                   reproduction of the paper (results are written to results/)
  run_gaussian.py          Section 5: Tables 2-6
  run_sources.py           Section 5.4: Tables 7-9 and root-search diagnostics
  run_mle.py               Gaussian MLE columns of Tables 8-9 and Table 11
  run_contamination.py     Section 6: Table 10
  make_table1.py           Table 1 (theory) and the optimal median-pilot floor
  make_figure1.py          Figure 1 (binding curve and sensitivity s_N)
  make_tables.py           LaTeX rows of Tables 2-11 -> results/tables/
examples/quickstart.py
tests/test_leaf.py
```

## Reproducing the paper

```bash
cd scripts
python make_table1.py          # Table 1                                    (seconds)
python run_sources.py          # Tables 7-9, root diagnostics              (~2 min)
python run_mle.py              # MLE benchmark for Tables 8, 9, 11         (~5 min)
python run_gaussian.py         # Tables 2-6, N = 1024, 512, 4096           (~30 min)
python run_contamination.py    # Table 10                                  (~10 min)
python make_figure1.py         # Figure 1
python make_tables.py          # writes results/tables/table*.tex
```

The times are for a single laptop core. Calibration banks are cached in `results/banks/` and reused by later scripts.

### Exact settings

| Item | Setting |
|---|---|
| fBm generator | Davies–Harte circulant embedding of fGn (length-$2N$ circulant), cumulative sum; complex normals drawn in blocks of 500 paths |
| Wavelet | Daubechies db2, half-point symmetric extension (MATLAB `dwt(...,'sym')`), detail coefficients at odd positions of the filtered extended signal |
| Levels | $j=2,\dots,J-3$ for $N=2^J$ (levels 2–7 for $N=1024$); $j=1$ is the finest level |
| Test paths, Section 5 | $M=5000$ per $(N,H)$, $H=0.1,\dots,0.9$, seed `default_rng([99, N, round(100 H)])`; the same paths for every estimator |
| Calibration bank | grid $h=0.01,0.03,\dots,0.99$ (50 points), $M_{\mathrm{cal}}=2000$ paths per point with seed `default_rng([7, N, g])`, $g$ = grid index; ridge $10^{-6}\,\mathrm{tr}(\widehat\Sigma)/K$; $\widehat\mu$ and $\widehat\Sigma$ interpolated linearly to the fine grid $0.010,0.011,\dots,0.990$, on which the GLS slope vectors are tabulated |
| Root search | $\widehat G(h)=\widehat q(h)\{Y-\widehat\mu(h)\}$ on the fine grid; first downcrossing, from positive to negative, interpolated linearly; $0.01$ if $\widehat G(0.01)<0$; $0.99$ if $\widehat G>0$ everywhere; multiple sign changes are recorded (codes: 0 interior, 1 multiple, 2 below grid, 3 above grid) |
| One-step update | started from AV; a start outside $[0.01,0.99]$ is projected onto the grid and the update is evaluated there; the one-step estimate itself is not projected |
| AV estimator | log-energy ordinates minus $g(n_j)=\psi(n_j/2)/\log 2-\log_2(n_j/2)$, WLS with weights $(\log 2)^2/\psi'(n_j/2)$; "restricted AV" clips to $[0.01,0.99]$ |
| Boundary-free coefficients | the indicator of extended samples is propagated through the filter cascade; a coefficient is dropped if any input in its window is affected (254, 126, 62, 30, 14, 6 coefficients at levels 2–7 for $N=1024$) |
| Section 6 scenarios | $N=1024$, $H\in\{0.2,0.5,0.8\}$, $M=5000$, seed `5000 + 10 H` for the Gaussian part and seed + 1 for the contamination; calibration with the clean fBm banks |
| Gaussian MLE | exact fGn likelihood with the scale profiled out, evaluated on $h=0.005,0.010,\dots,0.995$ by Cholesky factors of the Toeplitz correlation matrix, with quadratic refinement around the grid maximum; first 1000 paths |
| Figure 1 | derivatives of the bank means from quartic smoothing splines (one per level), with the smoothing level set from second differences of the bank means |

Results are written as Python pickles of dictionaries of float32 arrays, one estimate per path. Every table entry is computed from these arrays together with its Monte Carlo standard error.

## Tests

```bash
python -m pytest tests        # or: python tests/test_leaf.py
```

The tests check four things:
- the vectorized DWT against `np.convolve`;
- the coefficient counts, including boundary-free coefficients;
- the Table 1 values and the scale equivariance of the ordinates;
- the large-floor equivalence of mean-pilot LEAF and log-energy, and that a small calibration bank removes most of the bias.

## Citation

If you use this code, please cite the paper above. A BibTeX entry will be added on publication.

## License

To be chosen by the authors (for example MIT or BSD-3-Clause); add a `LICENSE` file before making the repository public.
