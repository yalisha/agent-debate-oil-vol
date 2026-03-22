"""
Rolling-origin evaluation framework for daily WTI oil price forecasting.
Supports multiple models, multiple horizons, and regime-conditional metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class ForecastResult:
    """Result for a single forecast origin."""
    origin_date: pd.Timestamp
    horizon: int
    y_true: float
    y_pred: float
    model_name: str
    regime: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class RollingOriginEvaluator:
    """
    Walk-forward evaluation engine.

    Parameters
    ----------
    train_window : int
        Number of days in the training window.
    horizons : list of int
        Forecast horizons (e.g., [1, 5, 20] for 1-day, 1-week, 1-month).
    step : int
        Number of days to advance between origins.
    min_train : int
        Minimum training samples required before starting evaluation.
    """

    def __init__(self, train_window=504, horizons=None, step=1, min_train=252):
        self.train_window = train_window
        self.horizons = horizons or [1, 5, 20]
        self.step = step
        self.min_train = min_train

    def run(self, df, features, target, models, regime_col=None):
        """
        Run rolling-origin evaluation.

        Parameters
        ----------
        df : DataFrame
            Must be sorted by date, contain `features`, `target`, and optionally `regime_col`.
        features : list of str
            Feature column names.
        target : str
            Target column name.
        models : dict
            {model_name: model_object} where each model has .fit(X, y) and .predict(X).
        regime_col : str, optional
            Column name for regime labels.

        Returns
        -------
        list of ForecastResult
        """
        df = df.dropna(subset=features + [target]).reset_index(drop=True)
        n = len(df)
        results = []

        max_horizon = max(self.horizons)
        start_idx = max(self.train_window, self.min_train)

        for origin in range(start_idx, n - max_horizon, self.step):
            train_start = max(0, origin - self.train_window)
            train = df.iloc[train_start:origin]

            X_train = train[features].values
            y_train = train[target].values

            for h in self.horizons:
                test_idx = origin + h - 1
                if test_idx >= n:
                    continue

                X_test = df.iloc[origin:origin + 1][features].values
                y_true = df.iloc[test_idx][target]
                regime = df.iloc[origin].get(regime_col) if regime_col else None

                for model_name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)[0]

                        results.append(ForecastResult(
                            origin_date=df.iloc[origin]['date'],
                            horizon=h,
                            y_true=y_true,
                            y_pred=y_pred,
                            model_name=model_name,
                            regime=int(regime) if regime is not None and not np.isnan(regime) else None,
                        ))
                    except Exception as e:
                        pass  # skip failed fits

            if (origin - start_idx) % 200 == 0 and origin > start_idx:
                print(f"  Progress: {origin - start_idx}/{n - max_horizon - start_idx}")

        return results


def compute_metrics(results, group_by=None):
    """
    Compute forecast evaluation metrics.

    Parameters
    ----------
    results : list of ForecastResult
    group_by : list of str, optional
        Fields to group by (e.g., ['model_name', 'horizon', 'regime']).

    Returns
    -------
    DataFrame with metrics.
    """
    df = pd.DataFrame([vars(r) for r in results])

    if group_by is None:
        group_by = ['model_name', 'horizon']

    group_by = [g for g in group_by if g in df.columns and df[g].notna().any()]

    metrics = []
    for keys, group in df.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = group['y_true'].values
        y_pred = group['y_pred'].values

        n = len(y_true)
        errors = y_true - y_pred

        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / (np.abs(y_true) + 1e-8))) * 100

        # Directional accuracy
        if 'y_true' in group.columns:
            dir_true = np.sign(y_true)
            dir_pred = np.sign(y_pred)
            da = np.mean(dir_true == dir_pred) * 100
        else:
            da = np.nan

        # R-squared (OOS)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        row = dict(zip(group_by, keys))
        row.update({
            'n_obs': n,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'DA': da,
            'R2_OOS': r2_oos,
        })
        metrics.append(row)

    return pd.DataFrame(metrics)
