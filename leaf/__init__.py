"""LEAF: Log-Energy with Adaptive Floor, and calibrated wavelet estimation of the Hurst exponent.

Companion code for
    V. Lakmini and B. Vidakovic, "Calibrated Log-Energy with Adaptive Floor:
    Bias-Corrected and Robust Wavelet Estimation of the Hurst Exponent".

Main entry points
-----------------
fbm, fgn                    exact Davies--Harte simulation (fbm.py)
details                     db2 DWT with half-point symmetric extension (dwt.py)
Ordinate, ordinates         log-energy, LEAF, average-log, log-median ordinates (ordinates.py)
av_estimate, slope_vectors  Abry--Veitch estimator and regression weights (ordinates.py)
CalibrationBank             simulated mean/covariance bank and calibrated root (calibration.py)
asymptotic_variance, ...    Gaussian theory for LEAF ordinates (theory.py)
fgn_mle                     exact Gaussian maximum likelihood benchmark (mle.py)
"""
from .fbm import fgn, fbm, arfima_increments, scenario, SCENARIOS
from .dwt import details, interior_masks, DB2_LO, DB2_HI
from .ordinates import (Ordinate, ordinates, level_details, level_counts, av_correction, av_weights,
                        slope_vectors, av_estimate, PAPER_ORDINATES)
from .calibration import CalibrationBank, GRID, FINE
from .theory import calibration_b, pilot_weight, asymptotic_variance, efficiency
from .mle import fgn_mle

__all__ = ["fgn", "fbm", "arfima_increments", "scenario", "SCENARIOS", "details", "interior_masks",
           "DB2_LO", "DB2_HI", "Ordinate", "ordinates", "level_details", "level_counts", "av_correction", "av_weights",
           "slope_vectors", "av_estimate", "PAPER_ORDINATES", "CalibrationBank", "GRID", "FINE",
           "calibration_b", "pilot_weight", "asymptotic_variance", "efficiency", "fgn_mle"]
__version__ = "1.0.0"
