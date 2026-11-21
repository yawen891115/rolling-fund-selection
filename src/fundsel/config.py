"""Single source of truth for every tunable number in the pipeline.

Two kinds of parameters live here:
  * Portfolio / evaluation parameters (K, buffer, rebalance timing, fees, risk-free rate) were fixed
    *before* looking at any backtest result and are never tuned, to keep the out-of-sample test honest.
  * Model hyper-parameters are tuned by Optuna, but only on validation folds inside each training window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ---------------- paths ----------------
    raw_dir: str = "data/raw"
    out_dir: str = "outputs"

    # ---------------- universe ----------------
    universe_types: tuple = ("股票型", "混合型-偏股")   # Eastmoney types: active stock funds + equity-heavy hybrids
    # Funds with lock-up / periodic-open structures cannot be redeemed monthly -> not investable here
    exclude_name_pattern: str = r"持有|定开|定期开放|封闭|滚动"
    min_history_days: int = 252       # need ~1 year of NAV history (>=90% of trading days) to be eligible
    max_daily_abs_ret: float = 0.25   # |daily return| > 25% is treated as a data error

    # ---------------- time ----------------
    first_signal: str = "2015-12-31"  # first month-end with factors (2 years of look-back before it)
    test_start: str = "2019-01-01"    # out-of-sample start
    test_end: str = "2022-12-31"      # out-of-sample end (the 2019-2022 window)
    holdout_end: str = "2024-12-31"   # end of the second evaluation window

    # ---------------- representation ----------------
    n_components: int = 6             # both PCA and the autoencoder compress 30 factors to 6 dims
    winsor_q: float = 0.01            # cross-sectional winsorization at 1% / 99%

    # ---------------- autoencoder ----------------
    ae_hidden_choices: tuple = (12, 16, 24)
    ae_epochs: int = 200
    ae_patience: int = 15
    ae_trials: int = 8                # Optuna trials for the AE (objective: validation reconstruction MSE)

    # ---------------- ranking model (LightGBM) ----------------
    lgbm_trials: int = 25             # Optuna trials for LightGBM (objective: mean validation Rank IC)
    valid_months: int = 12            # last 12 months of each training window = validation fold
    embargo_months: int = 1           # 1-month gap between the train fold and the validation fold

    # ---------------- portfolio ----------------
    top_k: int = 10                   # hold 10 funds
    buffer_k: int = 20                # keep a holding while it still ranks inside the top 20 (cuts turnover)
    cov_lookback: int = 126           # ~6 months of daily returns for the risk-parity covariance
    exec_lag: int = 1                 # signal at month-end close, trade at the next trading day's NAV
    buy_fee: float = 0.0015           # subscription fee (typical 90% platform discount on 1.5%)
    sell_fee: float = 0.005           # redemption fee for holdings between 7 days and 1 year

    # ---------------- evaluation ----------------
    rf: float = 0.015                 # risk-free rate: 1-year PBoC deposit benchmark rate
    benchmark: str = "930950"         # CSI Equity Fund Index (中证偏股型基金指数)
    seed: int = 42

    methods: tuple = field(default=("pca", "ae", "raw"))

    def to_dict(self) -> dict:
        return asdict(self)
