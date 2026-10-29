"""Run the full pipeline:  python run.py"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import backtest as bt
from src import ml
from src.data import BENCHMARK, TICKERS, load_prices, make_synthetic_prices
from src.pairs import build_spreads, select_pairs, zscores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--formation-end", default="2018-12-31")
    ap.add_argument("--train-end", default="2020-12-31")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=4.0)
    ap.add_argument("--method", choices=["corr", "coint"], default="corr",
                    help="how to pick pairs (corr = correlation, simple)")
    ap.add_argument("--min-corr", type=float, default=0.6)
    ap.add_argument("--ma", choices=["sma", "ema"], default="sma",
                    help="moving average used for the z-score")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="approval cutoff; default = win rate in the training data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.synthetic:
        prices = make_synthetic_prices(a.start, a.end)
    else:
        prices = load_prices(a.start, a.end)
    stocks = [c for c in prices.columns if c != BENCHMARK]
    logp = np.log(prices)

    #   formation: pick pairs + hedge ratios
    #   train:     fit the logistic regression
    #   test:      everything is judged here
    formation = logp.loc[: a.formation_end, stocks]
    train_start = formation.index[-1] + pd.Timedelta(days=1)
    test_start = pd.Timestamp(a.train_end) + pd.Timedelta(days=1)

    pairs = select_pairs(formation, n_pairs=a.pairs, min_corr=a.min_corr, method=a.method)
    if not pairs:
        raise SystemExit("No pairs found. Try a lower --min-corr.")
    print(f"Selected {len(pairs)} pairs from the formation period ({a.method})")
    pd.DataFrame([p.__dict__ for p in pairs]).to_csv(out / "pairs.csv", index=False)

    spreads = build_spreads(logp, pairs)
    z = zscores(spreads, a.window, a.ma)

    pos_base = bt.make_positions(z, a.entry, a.exit, a.stop)

    data = ml.build_dataset(spreads, z, a.horizon)
    model, n_rows, best_c, base_rate = ml.train_model(data, train_start, a.train_end, a.horizon)
    threshold = a.threshold if a.threshold is not None else base_rate
    approve = ml.approval_table(model, data, z.index, z.columns, threshold)
    pos_ml = bt.make_positions(z, a.entry, a.exit, a.stop, approve=approve)
    ev = ml.evaluate(model, data, test_start, a.end, threshold, a.horizon, a.entry)

    def test_ret(pos):
        r = bt.portfolio_returns(bt.pair_returns(logp, pairs, pos, a.cost_bps))
        return r.loc[test_start:]

    r_base, r_ml = test_ret(pos_base), test_ret(pos_ml)
    r_spy = logp[BENCHMARK].diff().loc[test_start:]

    table = pd.DataFrame([
        bt.metrics(r_base, "Pairs (z-score rules)", pos_base.loc[test_start:]),
        bt.metrics(r_ml, "Pairs + logistic regression", pos_ml.loc[test_start:]),
        bt.metrics(r_spy, "SPY buy and hold"),
    ]).set_index("strategy")
    table["trades"] = table["trades"].astype("Int64")
    table.to_csv(out / "metrics.csv")

    fmt = table.copy()
    for c in ["ann_return", "ann_vol", "max_drawdown", "total_return"]:
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    fmt["sharpe"] = fmt["sharpe"].round(2)
    lines = [
        f"Test period: {test_start.date()} to {r_base.index[-1].date()}",
        f"Cost assumption: {a.cost_bps} bps per unit of position change",
        "", fmt.to_string(), "",
        f"Logistic regression: trained on {n_rows} rows, best C = {best_c}, "
        f"approval cutoff = {threshold:.3f}",
        f"Test-period candidates (|z| >= {a.entry}): n={ev['n']}, "
        f"AUC={ev['auc']:.3f}, accuracy={ev['accuracy']:.3f}, "
        f"accuracy of always saying yes={ev['base_rate']:.3f}",
        "Coefficients (standardized features): "
        + ", ".join(f"{k}={v:+.2f}" for k, v in ml.coefficients(model).items()),
    ]
    text = "\n".join(lines)
    print(text)
    (out / "report.txt").write_text(text + "\n")

    plt.figure(figsize=(9, 5))
    for r, lab in [(r_base, "Pairs (z-score rules)"), (r_ml, "Pairs + logistic regression"),
                   (r_spy, "SPY buy and hold")]:
        plt.plot((1 + r.fillna(0)).cumprod(), label=lab)
    plt.title("Growth of $1 on the test period (after costs)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=130); plt.close()

    name = pairs[0].name
    zt = z[name].loc[test_start:]
    pt = pos_ml[name].loc[test_start:]
    plt.figure(figsize=(9, 4))
    plt.plot(zt, lw=0.8, label="z-score")
    plt.axhline(a.entry, color="r", ls="--", lw=0.7); plt.axhline(-a.entry, color="r", ls="--", lw=0.7)
    plt.fill_between(zt.index, -6, 6, where=pt != 0, color="green", alpha=0.15, label="in a trade (with filter)")
    plt.ylim(-6, 6)
    plt.title(f"{name}: z-score and trades on the test period")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "example_pair.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
