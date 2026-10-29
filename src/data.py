"""Price loading. Real data comes from yfinance; a synthetic generator is
included so the pipeline can be tested without internet access."""
from pathlib import Path

import numpy as np
import pandas as pd

# 30 large US financial stocks (one sector, so pairs are plausible).
# NOTE: this is today's list of survivors, which adds survivorship bias.
TICKERS = [
    "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "COF",
    "SCHW", "BNY", "STT", "AXP", "BLK", "MET", "PRU", "AIG", "ALL", "TRV",
    "CB", "PGR", "AFL", "HIG", "FITB", "KEY", "RF", "HBAN", "CFG", "MTB",
]
BENCHMARK = "SPY"


def load_prices(start, end, cache="data/prices.csv"):
    """Adjusted close prices for TICKERS + SPY, cached to CSV."""
    path = Path(cache)
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    df = yf.download(
        TICKERS + [BENCHMARK], start=start, end=end,
        auto_adjust=True, progress=False,
    )["Close"]
    df = df.dropna(axis=1, thresh=int(0.95 * len(df))).ffill().dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def make_synthetic_prices(start, end, seed=0):
    """Fake prices: some pairs whose spread mean-reverts, and a few of those
    pairs stop mean-reverting partway through (like real relationships do).
    Only for testing the code. Results on this data mean nothing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    market = np.cumsum(rng.normal(0.0003, 0.010, n))
    cols = {}
    break_at = int(n * 0.65)  # some relationships break after this point
    for k in range(0, 24, 2):
        base = market + np.cumsum(rng.normal(0.0001, 0.011, n))
        beta = rng.uniform(0.7, 1.3)
        spread = np.zeros(n)
        for t in range(1, n):
            phi = 0.94 if (t < break_at or k >= 16) else 1.0  # 4 pairs break
            spread[t] = phi * spread[t - 1] + rng.normal(0, 0.012)
        cols[TICKERS[k]] = np.log(50) + base
        cols[TICKERS[k + 1]] = np.log(50) + (base - spread) / beta
    for name in TICKERS[24:]:
        cols[name] = np.log(50) + market + np.cumsum(rng.normal(0, 0.013, n))
    cols[BENCHMARK] = np.log(300) + market
    return np.exp(pd.DataFrame(cols, index=idx))
