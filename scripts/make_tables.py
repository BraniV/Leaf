"""LaTeX bodies of Tables 2-11 from the results of the run_*.py scripts.

Writes results/tables/table{2..11}_*.tex (rows only; the table environments are in the paper).
Every entry carries its Monte Carlo standard error: for a mean squared error the SE of the
mean of squared errors; for a root mean squared error the delta-method SE.
"""
import os
import numpy as np
from _common import RESULTS, H_GRID, load

T = {}
f = lambda v: np.asarray(v, float)


def mse(v, H, d=2):
    e = (f(v) - H) ** 2
    return f"{1e3 * e.mean():.{d}f} ({1e3 * e.std(ddof=1) / np.sqrt(len(e)):.{d}f})"


def rmse(v, H, bold=False):
    e = (f(v) - H) ** 2
    m = e.mean()
    se = e.std(ddof=1) / np.sqrt(len(e)) / (2 * np.sqrt(m))
    return f"{100 * np.sqrt(m):.2f} ({100 * se:.2f})"


def bias_sd(v, H):
    v = f(v)
    return f"{1e3 * (v.mean() - H):.1f} & {1e3 * v.std(ddof=1):.1f}"


def write(name, rows):
    T[name] = "\n".join(rows)
    open(os.path.join(RESULTS, "tables", name + ".tex"), "w").write(T[name] + "\n")


have = lambda fn: os.path.exists(os.path.join(RESULTS, fn))

# ---------------------------------------------------------------- Section 5
if have("gaussian.pkl"):
    R = load("gaussian.pkl")
    cols = ["AV", "log-energy|WLS", "log-energy|one-step", "log-energy|root", "LEAF mean 0.5|root",
            "LEAF median 0.1|root", "avg-log|root", "log-median|root"]
    write("table2_gaussian_mse", [f"{H:.1f} & " + " & ".join(mse(R[(1024, H)][k], H) for k in cols) + r" \\" for H in H_GRID])
    cb = ["AV", "log-energy|WLS", "log-energy|one-step", "log-energy|root", "LEAF median 0.1|root"]
    rows = []
    for H in H_GRID:
        e = R[(1024, H)]
        rows.append(f"{H:.1f} & " + " & ".join(f"{1e3 * (f(e[k]).mean() - H):.1f} ({1e3 * f(e[k]).std(ddof=1) / np.sqrt(len(e[k])):.1f})" for k in cb) + r" \\")
    write("table3_gaussian_bias", rows)
    # Table 4: efficiency relative to calibrated log-energy
    from leaf.theory import efficiency
    rows = []
    for pilot, lams in [("mean", [0.1, 0.25, 0.5, 1, 2]), ("median", [0.05, 0.1, 0.25, 0.5, 1, 2])]:
        for l in lams:
            k = f"LEAF {pilot} {l:g}|root"
            r = [((f(R[(1024, H)]["log-energy|root"]) - H) ** 2).mean() / ((f(R[(1024, H)][k]) - H) ** 2).mean() for H in H_GRID]
            e = R[(1024, 0.5)]
            d = 1e3 * ((f(e[k]) - .5) ** 2 - (f(e["log-energy|root"]) - .5) ** 2)
            rows.append(f"{pilot} & {l:g} & {efficiency(l, pilot):.3f} & {np.mean(r):.3f} & [{min(r):.3f}, {max(r):.3f}] & {d.mean():+.3f} ({d.std(ddof=1) / np.sqrt(len(d)):.3f})" + r" \\")
        if pilot == "mean":
            rows.append(r"\midrule")
    for k, lab, lam in [("avg-log", "average-log", 0), ("log-median", "log-median", np.inf)]:
        r = [((f(R[(1024, H)]["log-energy|root"]) - H) ** 2).mean() / ((f(R[(1024, H)][k + "|root"]) - H) ** 2).mean() for H in H_GRID]
        e = R[(1024, 0.5)]
        d = 1e3 * ((f(e[k + "|root"]) - .5) ** 2 - (f(e["log-energy|root"]) - .5) ** 2)
        eff = efficiency(lam, "median")
        rows.append(f"\\multicolumn{{2}}{{l}}{{{lab}}} & {eff:.3f} & {np.mean(r):.3f} & [{min(r):.3f}, {max(r):.3f}] & {d.mean():+.3f} ({d.std(ddof=1) / np.sqrt(len(d)):.3f})" + r" \\")
    write("table4_efficiency", rows)
    # Table 5: paired relative differences, mean-pilot LEAF minus log-energy
    rows = []
    for H in H_GRID:
        e = R[(1024, H)]
        le = (f(e["log-energy|root"]) - H) ** 2
        c = []
        for l in [0.5, 1, 2]:
            d = (f(e[f"LEAF mean {l:g}|root"]) - H) ** 2 - le
            c.append(f"{100 * d.mean() / le.mean():+.2f} ({100 * d.std(ddof=1) / np.sqrt(len(d)) / le.mean():.2f})")
        rows.append(f"{H:.1f} & " + " & ".join(c) + r" \\")
    write("table5_paired", rows)
    # Table 6: N = 512 and 4096
    cN = ["AV", "log-energy|root", "LEAF mean 0.5|root", "LEAF median 0.1|root"]
    rows = []
    for H in H_GRID:
        cells = [mse(R[(512, H)][k], H) for k in cN] + [mse(R[(4096, H)][k], H, 3) for k in cN] if (512, H) in R and (4096, H) in R else []
        if cells:
            rows.append(f"{H:.1f} & " + " & ".join(cells) + r" \\")
    if rows:
        write("table6_other_N", rows)

# ---------------------------------------------------------------- Section 5.4
if have("sources.pkl"):
    S = load("sources.pkl")
    ML = load("mle.pkl") if have("mle.pkl") else {}
    rows = []
    for H in H_GRID:
        r = S[("base", H)]
        e = r["est"]
        wls, av = f(e["LE-WLS"]).mean() - H, f(e["AV"]).mean() - H
        st, lg, dg = r["H_structural"] - H, r["H_meanlog"] - r["H_structural"], r["AV_correction"]
        rows.append(f"{H:.1f} & {1e3 * wls:.1f} & {1e3 * st:.1f} & {1e3 * lg:.1f} & {1e3 * dg:.1f} & {1e3 * (lg + dg):.1f} & {1e3 * av:.1f}" + r" \\")
    write("table7_bias_decomposition", rows)
    H6 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for H in H6:
        e = S[("base", H)]["est"]
        cells = [bias_sd(e[k], H) for k in ["AV", "AV restricted", "mean-only", "full"]]
        cells.append(bias_sd(ML[("clean", H)], H) if ("clean", H) in ML else "-- & --")
        rows.append(f"{H:.1f} & " + " & ".join(cells) + r" \\")
    write("table8_bias_sd", rows)
    rows = []
    for H in H6:
        c = [mse(S[("base", H)]["est"][k], H) for k in ["AV restricted", "mean-only", "full"]]
        c += [mse(S[("drop1", H)]["est"][k], H) for k in ["AV restricted", "full"]]
        c += [mse(S[("interior", H)]["est"][k], H) for k in ["AV restricted", "full"]]
        c.append(mse(ML[("clean", H)], H, 3) if ("clean", H) in ML else "--")
        rows.append(f"{H:.1f} & " + " & ".join(c) + r" \\")
    write("table9_variants", rows)
    print("root-search codes [interior, multiple, below grid, above grid]:")
    for var in ["base", "drop1", "interior"]:
        for H in H_GRID:
            r = S[(var, H)]
            print(f"  {var:8s} H={H}: mean-only {r['codes_mean_only']}  full {r['codes_full']}  AV outside grid {r['AV_outside_grid']:.3f}")

# ---------------------------------------------------------------- Section 6
NAMES = {"AO p=0.5% k=10": r"AO, $p=0.5\%$, $\kappa=10$", "AO p=1% k=10": r"AO, $p=1\%$, $\kappa=10$",
         "AO p=2% k=10": r"AO, $p=2\%$, $\kappa=10$", "AO p=1% k=5": r"AO, $p=1\%$, $\kappa=5$",
         "Jumps p=1% k=10": r"Jumps, $p=1\%$, $\kappa=10$", "ARFIMA gauss": "ARFIMA, Gaussian",
         "ARFIMA t5": r"ARFIMA, $t_5$", "ARFIMA t3": r"ARFIMA, $t_3$", "fGn->t3 marginals": r"fGn $\to t_3$ marginals"}
if have("contamination.pkl") and have("gaussian.pkl"):
    C, R = load("contamination.pkl"), load("gaussian.pkl")
    c2 = ["AV", "log-energy|root", "LEAF mean 0.1|root", "LEAF mean 0.5|root", "LEAF median 0.1|root", "avg-log|root", "log-median|root"]
    rows = []
    for sc in ["clean"] + list(NAMES):
        for H in [0.2, 0.5, 0.8]:
            e = R[(1024, H)] if sc == "clean" else C[(sc, H)]
            vals = [rmse(e[k], H) for k in c2]
            best = int(np.argmin([float(v.split()[0]) for v in vals]))
            vals[best] = r"\textbf{" + vals[best] + "}"
            lab = ("Clean fBm" if sc == "clean" else NAMES[sc]) if H == 0.2 else ""
            rows.append(f"{lab} & {H} & " + " & ".join(vals) + r" \\")
        rows.append(r"\addlinespace")
    write("table10_contamination", rows[:-1])
if have("contamination.pkl") and have("mle.pkl"):
    C, ML = load("contamination.pkl"), load("mle.pkl")
    sub = lambda v: f(v)[:1000]
    rows = []
    for sc in ["AO p=0.5% k=10", "AO p=1% k=10", "AO p=2% k=10", "AO p=1% k=5", "ARFIMA gauss", "ARFIMA t3"]:
        for H in [0.2, 0.5, 0.8]:
            e = C[(sc, H)]
            cells = [rmse(ML[(sc, H)], H)] + [rmse(sub(e[k]), H) for k in ["AV", "log-energy|root", "LEAF mean 0.5|root", "LEAF median 0.1|root"]]
            rows.append(f"{NAMES[sc] if H == 0.2 else ''} & {H} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    write("table11_mle_contamination", rows[:-1])

print("wrote:", ", ".join(sorted(T)))
