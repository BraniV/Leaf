"""Discrete wavelet transform used in the paper: Daubechies db2 filters, half-point
symmetric extension ("sym" mode, identical to MATLAB's dwt with 'sym'), detail coefficients
at the odd positions of the filtered extended signal.

Level convention: j = 1 is the finest level; resolution decreases as j increases.
"""
import numpy as np

DB2_LO = np.array([-0.129409522550921, 0.224143868041857, 0.836516303737469, 0.482962913144690])
DB2_HI = np.array([-0.482962913144690, 0.836516303737469, -0.224143868041857, -0.129409522550921])


def dwt_sym(x, lo=DB2_LO, hi=DB2_HI):
    """One analysis step for each row of x. Returns (approximation, detail)."""
    lf = len(lo)
    ext = lf - 1
    xe = np.concatenate([x[:, ext - 1::-1], x, x[:, :-ext - 1:-1]], axis=1)   # half-point symmetric
    n = xe.shape[1] - lf + 1

    def conv_valid(h):                                   # == np.convolve(row, h, 'valid'), vectorized
        return sum(h[k] * xe[:, lf - 1 - k:lf - 1 - k + n] for k in range(lf))
    return conv_valid(lo)[:, 1::2], conv_valid(hi)[:, 1::2]


def details(x, J):
    """Detail coefficients for levels 1..J; returns a list whose element j-1 is level j."""
    out, a = [], np.atleast_2d(x)
    for _ in range(J):
        a, d = dwt_sym(a)
        out.append(d)
    return out


def interior_masks(N, J, lf=4):
    """Boolean masks (one per level 1..J) of coefficients whose support does not reach the
    symmetric boundary extension at any stage of the cascade (Section 5.4).

    The indicator of extended samples is propagated through the same filtering and
    downsampling; a coefficient is boundary-affected if any input in its window is.
    For N = 1024 and db2 this keeps 254, 126, 62, 30, 14, 6 coefficients at levels 2-7.
    """
    f, masks, ext = np.zeros((1, N)), [], lf - 1
    for _ in range(J):
        fe = np.concatenate([np.ones((1, ext)), f, np.ones((1, ext))], axis=1)
        n = fe.shape[1] - lf + 1
        aff = (sum(fe[:, lf - 1 - k:lf - 1 - k + n] for k in range(lf)) > 0)[:, 1::2]
        masks.append(~aff[0])
        f = aff.astype(float)
    return masks
