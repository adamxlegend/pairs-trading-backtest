"""Signals, PnL, and performance metrics."""
import numpy as np
import pandas as pd


def make_positions(z, entry=2.0, exit_=0.5, stop=4.0, max_hold=40, approve=None):
    """Position decided at the close of day t from z_t.
    +1 = long the spread (long A, short B), -1 = short the spread.
    approve: optional boolean DataFrame; entries only happen where True."""
    pos = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    for col in z.columns:
        zc = z[col].to_numpy()
        ok = approve[col].to_numpy() if approve is not None else np.ones(len(zc), bool)
        out = np.zeros(len(zc))
        p, held = 0, 0
        for i, zi in enumerate(zc):
            if np.isnan(zi):
                p, held = 0, 0
            elif p == 0:
                if ok[i] and entry < zi < stop:
                    p, held = -1, 0
                elif ok[i] and -stop < zi < -entry:
                    p, held = 1, 0
            else:
                held += 1
                if abs(zi) < exit_ or abs(zi) > stop or held >= max_hold:
                    p = 0
            out[i] = p
        pos[col] = out
    return pos


def pair_returns(log_prices, pairs, positions, cost_bps=5.0):
    """Daily return per pair, per $1 of capital allocated to that pair.
    Yesterday's position earns today's return (no lookahead).
    Cost is charged on every change in position (both legs, incl. slippage)."""
    rets = log_prices.diff()
    out = {}
    for p in pairs:
        gross = 1 + abs(p.beta)
        spread_ret = (rets[p.a] - p.beta * rets[p.b]) / gross
        pos = positions[p.name]
        pnl = pos.shift(1) * spread_ret
        cost = pos.diff().abs() * cost_bps / 1e4
        out[p.name] = (pnl - cost).fillna(0.0)
    return pd.DataFrame(out)


def portfolio_returns(per_pair):
    """Equal capital in every pair; idle capital earns 0."""
    return per_pair.mean(axis=1)


def metrics(r, label, positions=None):
    r = r.dropna()
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    m = {
        "strategy": label,
        "ann_return": r.mean() * 252,
        "ann_vol": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "max_drawdown": dd.min(),
        "total_return": equity.iloc[-1] - 1,
    }
    if positions is not None:
        entered = (positions.diff() != 0) & (positions != 0)
        m["trades"] = int(entered.sum().sum())
    return m
