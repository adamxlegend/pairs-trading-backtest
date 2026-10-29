import numpy as np
import pandas as pd

from src.backtest import make_positions, pair_returns
from src.pairs import Pair, zscores


def test_zscore_ignores_the_future():
    rng = np.random.default_rng(1)
    s = pd.DataFrame({"x": np.cumsum(rng.normal(size=300))})
    z1 = zscores(s, 60)
    s2 = s.copy()
    s2.iloc[200:] += 100  # change the future
    z2 = zscores(s2, 60)
    assert np.allclose(z1.iloc[:200].fillna(0), z2.iloc[:200].fillna(0))


def test_position_earns_next_day_return_only():
    idx = pd.date_range("2020-01-01", periods=5)
    logp = pd.DataFrame({"A": [0, 0, 0.1, 0.1, 0.1], "B": 0.0}, index=idx)
    pair = Pair("A", "B", 1.0, 0.0, 0.01)
    pos = pd.DataFrame({"A/B": [0, 1, 1, 1, 1]}, index=idx, dtype=float)
    r = pair_returns(logp, [pair], pos, cost_bps=0)["A/B"]
    # jump happens on day 2; position was opened on day 1 close -> earns it
    assert r.iloc[2] > 0 and r.iloc[1] == 0


def test_stop_blocks_reentry():
    z = pd.DataFrame({"x": [0, -2.5, -5, -5, -2.5, 0]})
    pos = make_positions(z, entry=2, exit_=0.5, stop=4)["x"].tolist()
    assert pos == [0, 1, 0, 0, 1, 0]
