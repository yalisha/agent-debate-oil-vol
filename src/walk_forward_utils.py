"""
Shared helpers for walk-forward evaluation with forward-looking targets.

For an h-step-ahead target y_t that depends on outcomes from t+1 to t+h,
the label for sample t is only fully observed at time t+h. When forecasting
sample `split_idx`, supervised training must therefore exclude labels later
than `split_idx - h`.
"""

import numpy as np


LABEL_EMBARGO = 20


def embargoed_train_end(split_idx, horizon=LABEL_EMBARGO):
    """Exclusive end index for labels observable when forecasting split_idx."""
    return max(0, split_idx - horizon + 1)


def embargoed_train_idx(split_idx, horizon=LABEL_EMBARGO):
    """Training indices available when forecasting split_idx."""
    return np.arange(0, embargoed_train_end(split_idx, horizon))


def rolling_window_bounds(split_idx, train_window, horizon=LABEL_EMBARGO):
    """Rolling-window start/end bounds with forward-label embargo applied."""
    train_end = embargoed_train_end(split_idx, horizon)
    train_start = max(0, train_end - train_window)
    return train_start, train_end
