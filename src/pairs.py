"""Pair selection. Uses ONLY the formation period."""
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Pair:
    a: str
    b: str
    beta: float   # slope of the line of best fit: log(a) = alpha + beta * log(b)
    alpha: float  # intercept of that line
    corr: float   # correlation of daily returns in the formation period
    pvalue: Optional[float] = None  # only filled in if method="coint"

    @property
    def name(self):
        return f"{self.a}/{self.b}"


def select_pairs(log_prices, n_pairs=10, min_corr=0.6, method="corr", max_p=0.05):
    """Pick the most correlated pairs of stocks.

    corr:  rank by correlation of daily returns (simple, the default)
    coint: also require a cointegration test to pass (optional, stricter)

    Each stock is used at most once so the portfolio is not concentrated.
    Caveat: high correlation does not guarantee the gap between two stocks
    stays stable. That is why results are judged on later data."""
    corr = log_prices.diff().dropna().corr()
    found = []
    for a, b in combinations(log_prices.columns, 2):
        c = float(corr.loc[a, b])
        if c < min_corr:
            continue
        p = None
        if method == "coint":
            from statsmodels.tsa.stattools import coint
            p = float(coint(log_prices[a], log_prices[b])[1])
            if p >= max_p:
                continue
        found.append((c, p, a, b))
    found.sort(key=lambda t: t[1] if method == "coint" else -t[0])

    used, pairs = set(), []
    for c, p, a, b in found:
        if a in used or b in used:
            continue
        # line of best fit through the two log-price series
        beta, alpha = np.polyfit(log_prices[b], log_prices[a], 1)
        pairs.append(Pair(a, b, float(beta), float(alpha), c, p))
        used.update([a, b])
        if len(pairs) == n_pairs:
            break
    return pairs


def build_spreads(log_prices, pairs):
    """Spread = actual log(a) minus what the best-fit line predicts from log(b).
    The line is fitted in the formation period and then frozen."""
    return pd.DataFrame({p.name: log_prices[p.a] - p.beta * log_prices[p.b] - p.alpha for p in pairs})


def zscores(spreads, window=60, ma="sma"):
    """How many standard deviations the spread is from its moving average.
    ma="sma": simple moving average. ma="ema": exponential moving average.
    Uses only data up to and including day t."""
    if ma == "ema":
        e = spreads.ewm(span=window, min_periods=window)
        return (spreads - e.mean()) / e.std()
    roll = spreads.rolling(window)
    return (spreads - roll.mean()) / roll.std()
