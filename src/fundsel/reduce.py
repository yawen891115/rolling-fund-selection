"""Representation: 30 factors -> 6 dimensions.

    PCAReducer       linear: the first 6 principal components
    AEReducer        non-linear: autoencoder 30 -> h -> 6 -> h -> 30, keep the 6-unit bottleneck
    IdentityReducer  no reduction, all 30 factors (ablation: does compression help at all?)

Same interface as sklearn: fit on the training window only, then transform train and test rows.
"""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.decomposition import PCA

def recon_r2(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Reconstruction R^2 = 1 - SSE / total variance (for PCA this equals the explained variance)."""
    return float(1.0 - ((X - X_hat) ** 2).sum() / ((X - X.mean(axis=0)) ** 2).sum())


class IdentityReducer:
    name = "raw"

    def fit(self, X, X_valid=None):
        return self

    def transform(self, X):
        return X

    def info(self, X=None):
        return {"dim": None}


class PCAReducer:
    name = "pca"

    def __init__(self, n_components: int = 6, seed: int = 42):
        self.pca = PCA(n_components=n_components, random_state=seed)

    def fit(self, X, X_valid=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def info(self, X=None):
        evr = self.pca.explained_variance_ratio_
        out = {"explained_variance_ratio": [round(float(v), 4) for v in evr],
               "cum_explained": round(float(evr.sum()), 4)}
        if X is not None:
            out["recon_r2"] = round(recon_r2(X, self.pca.inverse_transform(self.pca.transform(X))), 4)
        return out


# ---------------------------------------------------------------------------
# autoencoder
# ---------------------------------------------------------------------------
class AEReducer:
    """Autoencoder compression; hidden width, learning rate and weight decay are chosen by Optuna
    to minimise validation reconstruction error."""

    name = "ae"

    def __init__(self, n_components=6, hidden_choices=(12, 16, 24), epochs=200, patience=15,
                 n_trials=8, seed=42, params: dict | None = None):
        self.k, self.hidden_choices = n_components, hidden_choices
        self.epochs, self.patience, self.n_trials, self.seed = epochs, patience, n_trials, seed
        self.params = params            # if given, skip tuning
        self.model = None
        self.history: dict = {}
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            torch = None
        self.torch = torch

    # ---- train one autoencoder with fixed hyper-parameters; returns (model, validation MSE, epochs)
    def _train(self, X, X_val, hidden, lr, wd):
        torch = self.torch
        if torch is None:
            return self._train_sklearn(X, X_val, hidden, lr, wd)
        from collections import OrderedDict
        from torch import nn

        torch.manual_seed(self.seed)
        Xt = torch.tensor(X, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        model = nn.Sequential(OrderedDict([
            ("encoder", nn.Sequential(nn.Linear(X.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, self.k))),
            ("decoder", nn.Sequential(nn.Linear(self.k, hidden), nn.Tanh(), nn.Linear(hidden, X.shape[1]))),
        ]))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            model.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(512):   # mini-batches
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), Xt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(Xv), Xv).item()
            if v < best - 1e-5:                        # early stopping on validation reconstruction error
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        model.load_state_dict(best_state)
        return model, best, epoch + 1

    def _train_sklearn(self, X, X_val, hidden, lr, wd):
        from sklearn.neural_network import MLPRegressor

        m = MLPRegressor(hidden_layer_sizes=(hidden, self.k, hidden), activation="tanh",
                         learning_rate_init=lr, alpha=wd, max_iter=self.epochs, batch_size=512,
                         early_stopping=True, n_iter_no_change=self.patience, random_state=self.seed)
        m.fit(X, X)
        return m, float(((m.predict(X_val) - X_val) ** 2).mean()), m.n_iter_

    # ---- public API
    def fit(self, X, X_valid=None):
        if X_valid is None:                            # no validation set given: hold out the last 10% (in time)
            cut = int(len(X) * 0.9)
            X, X_valid = X[:cut], X[cut:]
        if self.params is None:                        # Optuna search
            def objective(trial):
                hidden = trial.suggest_categorical("hidden", list(self.hidden_choices))
                lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
                wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                return self._train(X, X_valid, hidden, lr, wd)[1]

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(objective, n_trials=self.n_trials)
            self.params = study.best_params
        p = self.params
        self.model, val_mse, epochs = self._train(X, X_valid, p["hidden"], p["lr"], p["weight_decay"])
        self.history = {"params": p, "valid_mse": round(val_mse, 5), "epochs": epochs,
                        "backend": "torch" if self.torch is not None else "sklearn"}
        return self

    def transform(self, X):
        torch = self.torch
        if torch is not None:
            with torch.no_grad():
                return self.model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        h = X                                          # sklearn: manual forward pass up to the bottleneck
        for W, b in zip(self.model.coefs_[:2], self.model.intercepts_[:2]):
            h = np.tanh(h @ W + b)
        return h

    def reconstruct(self, X):
        torch = self.torch
        if torch is not None:
            with torch.no_grad():
                return self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return self.model.predict(X)

    def info(self, X=None):
        out = dict(self.history)
        if X is not None:
            out["recon_r2"] = round(recon_r2(X, self.reconstruct(X)), 4)
        return out


def make_reducer(method: str, cfg, ae_params: dict | None = None):
    if method == "pca":
        return PCAReducer(cfg.n_components, cfg.seed)
    if method == "ae":
        return AEReducer(cfg.n_components, cfg.ae_hidden_choices, cfg.ae_epochs, cfg.ae_patience,
                         cfg.ae_trials, cfg.seed, params=ae_params)
    if method == "raw":
        return IdentityReducer()
    raise ValueError(method)
