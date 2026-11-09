"""Basic checks; run with `python -m pytest tests` (or `python tests/test_leaf.py`)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import leaf
from leaf.dwt import dwt_sym, DB2_LO, DB2_HI


def test_dwt_matches_convolution():
    """Vectorized filtering equals np.convolve on the symmetric extension."""
    x = np.random.default_rng(0).standard_normal((2, 64))
    a, d = dwt_sym(x)
    xe = np.concatenate([x[:, 2::-1], x, x[:, :-4:-1]], axis=1)
    for r in range(2):
        assert np.allclose(d[r], np.convolve(xe[r], DB2_HI, "valid")[1::2])
        assert np.allclose(a[r], np.convolve(xe[r], DB2_LO, "valid")[1::2])


def test_level_counts_and_interior():
    assert list(leaf.level_counts(1024, range(2, 8))) == [258, 130, 66, 34, 18, 10]
    assert list(leaf.level_counts(1024, range(2, 8), interior=True)) == [254, 126, 62, 30, 14, 6]


def test_fgn_variance():
    x = leaf.fgn(4096, 0.7, np.random.default_rng(1), 200)
    assert abs(x.var() - 1) < 0.05


def test_theory_values():
    """Table 1 values."""
    from leaf.theory import asymptotic_variance as V, pilot_weight
    assert abs(V(np.inf) - 2 / np.log(2) ** 2) < 1e-6
    assert abs(V(2, "mean") - 4.1811) < 1e-3
    assert abs(V(0.1, "median") - 6.977) < 1e-3
    assert abs(V(0) - 10.2711) < 1e-3
    assert abs(pilot_weight(0.5) - 0.5456) < 1e-3


def test_scale_equivariance():
    """Multiplying the data by a shifts every LEAF ordinate by 2 log2|a|."""
    P = leaf.fbm(1024, 0.4, np.random.default_rng(3), 3)
    o = [leaf.Ordinate("leaf", "median", 0.1), leaf.Ordinate("leaf", "mean", 2), leaf.Ordinate("log-energy")]
    Y1, Y2 = leaf.ordinates(P, range(2, 8), o), leaf.ordinates(-5 * P, range(2, 8), o)
    for k in Y1:
        assert np.allclose(Y2[k] - Y1[k], 2 * np.log2(5))


def test_large_floor_equivalence():
    """Mean-pilot LEAF minus log-energy is constant across levels up to O(lambda^-2)."""
    P = leaf.fbm(1024, 0.6, np.random.default_rng(4), 5)
    Y = leaf.ordinates(P, range(2, 8), [leaf.Ordinate("log-energy"), leaf.Ordinate("leaf", "mean", 1e4)])
    d = Y["LEAF mean 10000"] - Y["log-energy"]
    assert np.max(np.ptp(d, axis=1)) < 1e-4


def test_calibrated_estimator_small_bank():
    """A coarse bank already removes most of the bias at H = 0.2."""
    b = leaf.CalibrationBank.build(leaf.Ordinate("log-energy"), 512, np.arange(2, 7), n_paths=60)
    P = leaf.fbm(512, 0.2, np.random.default_rng(5), 200)
    h = b.estimate(P)
    assert abs(h.mean() - 0.2) < 0.03


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
