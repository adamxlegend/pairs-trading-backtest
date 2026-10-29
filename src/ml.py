"""Logistic regression filter: should we take this mean-reversion trade?

Label (for a day where |z| is large): 1 if a mean-reversion trade opened at
the close would have made money over the next H days, else 0.
Trained on the TRAIN period only, which comes after pair selection and
before the test period."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["z", "abs_z", "dz1", "dz5", "vol_ratio"]


def build_dataset(spreads, z, horizon=5):
    """One row per (date, pair). Features use data up to day t only.
    The label looks `horizon` days ahead, so it is for training/scoring only."""
    frames = []
    for col in spreads.columns:
        s, zc = spreads[col], z[col]
        d = pd.DataFrame(index=s.index)
        d["pair"] = col
        d["z"] = zc
        d["abs_z"] = zc.abs()
        d["dz1"] = zc.diff(1)
        d["dz5"] = zc.diff(5)
        d["vol_ratio"] = s.diff().rolling(10).std() / s.diff().rolling(60).std()
        direction = -np.sign(zc)  # revert: short if z>0, long if z<0
        future_move = s.shift(-horizon) - s
        d["label"] = (direction * future_move > 0).astype(float)
        d.loc[future_move.isna() | zc.isna(), "label"] = np.nan
        frames.append(d)
    return pd.concat(frames).rename_axis("date").reset_index()


def train_model(data, train_start, train_end, horizon=5, min_abs_z=1.5):
    """Fit on train-period rows only. The last `horizon` days are dropped
    because their labels would peek into the test period."""
    dates = data["date"].drop_duplicates().sort_values()
    in_train = dates[(dates >= train_start) & (dates <= train_end)]
    last_ok = in_train.iloc[-1 - horizon]  # purge
    rows = data[(data["date"] >= train_start) & (data["date"] <= last_ok)]
    rows = rows[(rows["abs_z"] >= min_abs_z)].dropna(subset=FEATURES + ["label"])
    rows = rows.sort_values("date")
    if len(rows) < 60 or rows["label"].nunique() < 2:
        raise ValueError(f"Not enough training rows ({len(rows)}) to fit the model.")

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    grid = GridSearchCV(
        pipe,
        {"logisticregression__C": [0.01, 0.1, 1, 10]},
        cv=TimeSeriesSplit(n_splits=4, gap=horizon * data["pair"].nunique()),
        scoring="neg_log_loss",
    )
    grid.fit(rows[FEATURES], rows["label"])
    base_rate = float(rows["label"].mean())
    return grid.best_estimator_, len(rows), grid.best_params_["logisticregression__C"], base_rate


def approval_table(model, data, index, columns, threshold=0.5):
    """True where the model says the trade is worth taking."""
    ok = pd.DataFrame(False, index=index, columns=columns)
    d = data.dropna(subset=FEATURES)
    prob = model.predict_proba(d[FEATURES])[:, 1]
    for (dt, pair), p in zip(zip(d["date"], d["pair"]), prob):
        ok.at[dt, pair] = p > threshold
    return ok


def evaluate(model, data, test_start, test_end, threshold, horizon=5, min_abs_z=2.0):
    """Out-of-sample classification quality on trade-candidate rows."""
    dates = data["date"].drop_duplicates().sort_values()
    in_test = dates[(dates >= test_start) & (dates <= test_end)]
    last_ok = in_test.iloc[-1 - horizon]
    rows = data[(data["date"] >= test_start) & (data["date"] <= last_ok)]
    rows = rows[rows["abs_z"] >= min_abs_z].dropna(subset=FEATURES + ["label"])
    if len(rows) < 20 or rows["label"].nunique() < 2:
        return {"n": len(rows), "auc": np.nan, "accuracy": np.nan, "base_rate": np.nan}
    prob = model.predict_proba(rows[FEATURES])[:, 1]
    return {
        "n": len(rows),
        "auc": roc_auc_score(rows["label"], prob),
        "accuracy": float(((prob > threshold) == rows["label"]).mean()),
        "base_rate": float(rows["label"].mean()),  # accuracy of always saying "yes"
    }


def coefficients(model):
    lr = model.named_steps["logisticregression"]
    return dict(zip(FEATURES, lr.coef_[0]))
