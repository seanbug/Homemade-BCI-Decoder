import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

fsTarget = 250
bandpassFilter = (8, 30)

# Designed once, at the target rate. Second-order sections rather than
# transfer-function form: butter(4, [8, 30], fs=48000) puts poles outside the
# unit circle and filtfilt then returns all NaN.
_sos = butter(4, bandpassFilter, btype="band", fs=fsTarget, output="sos")


def extract_features(x, fs_in=48000):
    """Log mean band power in 8-30 Hz. Used by both training and the live decoder.

    Raises ValueError rather than returning a fallback, so a broken signal path
    fails loudly instead of producing a plausible-looking constant.
    """
    x = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("input contains non-finite samples")

    n_out = int(round(x.shape[-1] * fsTarget / fs_in))
    x = resample(x, n_out)

    x = sosfiltfilt(_sos, x)

    if not np.all(np.isfinite(x)):
        raise ValueError("filter produced non-finite output")

    power = np.mean(x ** 2)

    if power <= 0:
        raise ValueError(f"non-positive band power: {power!r}")

    return np.array([np.log(power)])
