"""
Baseline prediction models for daily WTI oil return forecasting.
All models follow the sklearn interface: .fit(X, y) and .predict(X).
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class ARModel:
    """
    Autoregressive model using lagged target values.
    Wraps LinearRegression but constructs AR features internally.
    For use in the evaluation framework, the target lags should be
    pre-computed and included in the feature set.
    """
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class HistoricalMean:
    """Predicts the expanding mean of training data."""
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class XGBoostModel:
    """XGBoost wrapper with sensible defaults for financial time series."""
    def __init__(self, **kwargs):
        defaults = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'min_samples_leaf': 20,
            'random_state': 42,
        }
        defaults.update(kwargs)
        self.model = GradientBoostingRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RidgeModel:
    """Ridge regression with regularization."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel:
    """Random Forest wrapper."""
    def __init__(self, **kwargs):
        defaults = {
            'n_estimators': 200,
            'max_depth': 6,
            'min_samples_leaf': 20,
            'random_state': 42,
            'n_jobs': -1,
        }
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def get_baseline_models():
    """Return dict of all baseline models."""
    return {
        'HistMean': HistoricalMean(),
        'AR_Linear': ARModel(),
        'Ridge': RidgeModel(alpha=1.0),
        'GBR': XGBoostModel(),
        'RF': RandomForestModel(),
    }
