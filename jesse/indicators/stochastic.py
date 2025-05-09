from collections import namedtuple

import numpy as np

from jesse.helpers import slice_candles
from jesse.indicators.ma import ma

Stochastic = namedtuple('Stochastic', ['k', 'd'])


def stoch(candles: np.ndarray, fastk_period: int = 14, slowk_period: int = 3, slowk_matype: int = 0,
          slowd_period: int = 3, slowd_matype: int = 0, sequential: bool = False) -> Stochastic:
    """
    The Stochastic Oscillator

    :param candles: np.ndarray
    :param fastk_period: int - default: 14
    :param slowk_period: int - default: 3
    :param slowk_matype: int - default: 0
    :param slowd_period: int - default: 3
    :param slowd_matype: int - default: 0
    :param sequential: bool - default: False

    :return: Stochastic(k, d)
    """
    if any(matype in (24, 29) for matype in (slowk_matype, slowd_matype)):
        raise ValueError("VWMA (matype 24) and VWAP (matype 29) cannot be used in stochastic indicator.")

    candles = slice_candles(candles, sequential)

    candles_close = candles[:, 2]
    candles_high = candles[:, 3]
    candles_low = candles[:, 4]
    
    hh = _rolling_max(candles_high, fastk_period)
    ll = _rolling_min(candles_low, fastk_period)

    stoch_val = 100 * (candles_close - ll) / (hh - ll)
    k = ma(stoch_val, period=slowk_period, matype=slowk_matype, sequential=True)
    d = ma(k, period=slowd_period, matype=slowd_matype, sequential=True)

    if sequential:
        return Stochastic(k, d)
    else:
        return Stochastic(k[-1], d[-1])

def _rolling_max(x, window):
    if len(x) < window:
        return np.full(x.shape, np.nan, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=window)
    result = np.full(x.shape, np.nan, dtype=np.float64)
    result[window - 1:] = np.max(windows, axis=1)
    return result

def _rolling_min(x, window):
    if len(x) < window:
        return np.full(x.shape, np.nan, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=window)
    result = np.full(x.shape, np.nan, dtype=np.float64)
    result[window - 1:] = np.min(windows, axis=1)
    return result
