"""Ranking model: LightGBM regression tuned with Optuna.

Label: the fund's next-month return expressed as a cross-sectional percentile (0..1). We only care
about the ordering of funds, and percentiles strip out the market-wide move and outliers.

Metric: Rank IC = Spearman correlation between model scores and realised next-month returns on one
rebalance date, averaged over dates. The standard ranking metric in fund/stock selection.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rank_ic_by_date(dates, y_true, y_pred) -> pd.Series:
    df = pd.DataFrame({"date": np.asarray(dates), "y": np.asarray(y_true), "p": np.asarray(y_pred)})
    df = df.dropna()
    if df.empty:
        return pd.Series(index=pd.DatetimeIndex([], name="date"), dtype=float)
    return df.groupby("date")[["y", "p"]].apply(lambda g: g["y"].corr(g["p"], method="spearman"))


def suggest_params(trial: optuna.Trial) -> dict:
    """Optuna search space (deliberately conservative: financial labels are very noisy)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 800, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def make_lgbm(params: dict, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**params, subsample_freq=1, random_state=seed, n_jobs=2, verbose=-1)


def tune_lgbm(Z_tr, y_tr, Z_va, y_va, d_va, n_trials: int, seed: int) -> tuple[dict, float, pd.DataFrame]:
    """Train on the train fold, score mean Rank IC on the validation fold, let Optuna (TPE) maximise it."""

    def objective(trial):
        model = make_lgbm(suggest_params(trial), seed).fit(Z_tr, y_tr)
        return float(rank_ic_by_date(d_va, y_va, model.predict(Z_va)).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params"))
    return study.best_params, float(study.best_value), trials
