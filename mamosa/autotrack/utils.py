import numpy as np
import scipy.linalg
from typing import Tuple


def deconvolve_seismic(
    seismic: np.ndarray, wavelet: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-probabilistic deconvolution of seismic data.

    Assumes symmetric wavelet.

    Args:
        seismic: Seismic data geovolume (3d) or line (2d).
        wavelet: 1d seismic wavelet.

    Returns:
        Two-tuple; first element is the array of estimated reflection coefficients,
        second element is the convolution matrix (wavelet matrix).

    """
    if seismic.ndim == 2:
        # Convert to 3d
        seismic = seismic[np.newaxis]
    else:
        assert seismic.ndim == 3
    I, X, T = seismic.shape
    s = wavelet.size
    assert s <= T
    first_col = np.zeros(T)
    first_col[: s // 2] = wavelet[s // 2 :]
    W = scipy.linalg.toeplitz(first_col)
    R = np.apply_along_axis(lambda t: np.linalg.lstsq(W, t)[0], axis=-1, arr=seismic)
    return R, W
