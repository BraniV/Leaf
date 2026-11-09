"""Shared settings for the reproduction scripts (exact seeds and designs of the paper)."""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import leaf  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(os.path.join(RESULTS, "banks"), exist_ok=True)
os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)

H_GRID = np.round(np.arange(0.1, 0.91, 0.1), 1)       # true H values, Section 5
H_ROBUST = [0.2, 0.5, 0.8]                              # true H values, Section 6
M_TEST = 5000                                           # test paths per (N, H)
M_CAL = 2000                                            # calibration paths per grid point
BANK_SEED = 7                                           # bank: default_rng([7, N, g]), g = grid index


def test_seed(N, H):
    """Seed of the Gaussian test paths in Section 5: default_rng([99, N, round(100 H)])."""
    return [99, N, int(round(H * 100))]


def robust_seed(H):
    """Seed of the scenario paths in Section 6 (contamination uses seed + 1)."""
    return 5000 + int(round(H * 10))


def default_levels(N):
    """Fitted levels 2..J-3 for N = 2^J (levels 2-7 for N = 1024)."""
    return np.arange(2, int(np.log2(N)) - 2)


def banks_for(N, levels=None, interior=False, ordinates=None, tag=""):
    """Load or build (and cache) calibration banks for the paper's ordinates."""
    ordinates = ordinates or leaf.PAPER_ORDINATES
    levels = default_levels(N) if levels is None else np.asarray(levels)
    lv = f"{levels[0]}-{levels[-1]}" + ("-interior" if interior else "")
    paths = {o.name: os.path.join(RESULTS, "banks", f"bank_N{N}_L{lv}_{o.name.replace(' ', '_')}{tag}.npz")
             for o in ordinates}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: leaf.CalibrationBank.load(p) for k, p in paths.items()}
    print(f"building calibration banks: N={N}, levels {lv}, {len(ordinates)} ordinates, M_cal={M_CAL}", flush=True)
    banks = leaf.CalibrationBank.build_many(ordinates, N, levels, M_CAL, interior, BANK_SEED, verbose=True)
    for k, b in banks.items():
        b.save(paths[k])
    return banks


def save(obj, name):
    with open(os.path.join(RESULTS, name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(os.path.join(RESULTS, name), "rb") as f:
        return pickle.load(f)
