"""
XGBoost / LSTM / Transformer baselines for volatility prediction.
Train once on pre-2020 data, predict 2020-2025. Simple and fast.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent


def load_data():
    macro = pd.read_csv(BASE / "data/oil_macro_daily.csv", parse_dates=["date"])
    gdelt = pd.read_csv(BASE / "data/gdelt_daily_features.csv", parse_dates=["date"])
    df = macro.merge(gdelt, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["fwd_vol_20d"] = df["wti_vol_20d"].shift(-1)
    df = df.dropna(subset=["fwd_vol_20d"]).reset_index(drop=True)
    return df


def build_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["vol_20d"] = df["wti_vol_20d"]
    feat["vol_60d"] = df["wti_vol_60d"]
    feat["vol_ratio"] = df["wti_vol_20d"] / (df["wti_vol_60d"] + 1e-8)
    for lag in [1, 2, 3, 5, 10]:
        feat[f"vol_lag_{lag}"] = df["wti_vol_20d"].shift(lag)
    feat["vol_week"] = df["wti_vol_20d"].rolling(5).mean()
    feat["vol_month"] = df["wti_vol_20d"].rolling(22).mean()
    feat["vol_diff_1"] = df["wti_vol_20d"].diff(1)
    feat["vol_diff_5"] = df["wti_vol_20d"].diff(5)
    feat["abs_return"] = df["wti_return"].abs()
    feat["abs_ret_5d"] = df["wti_return"].abs().rolling(5).mean()
    feat["ret_sq_5d"] = (df["wti_return"] ** 2).rolling(5).mean()
    feat["mom_5d"] = df["wti_mom_5d"]
    feat["mom_20d"] = df["wti_mom_20d"]
    feat["vix"] = df["VIX"]
    feat["vix_change"] = df["VIX"].diff()
    feat["dxy_return"] = df["dxy_return"]
    feat["spread_10y2y"] = df["YIELD_SPREAD_10Y_2Y"]
    for col in ["oil_goldstein_mean", "oil_conflict_share",
                "oil_material_conflict_share", "oil_tone_mean",
                "oil_net_coop", "n_events_oil"]:
        if col in df.columns:
            feat[col] = df[col]
            feat[f"{col}_ma5"] = df[col].rolling(5).mean()
    return feat


def make_sequences(X, y, seq_len=20):
    """Build (seq_len, n_features) sequences for LSTM/Transformer."""
    seqs_X, seqs_y = [], []
    for i in range(seq_len, len(X)):
        seqs_X.append(X[i - seq_len:i])
        seqs_y.append(y[i])
    return np.array(seqs_X), np.array(seqs_y)


def run_xgboost(train_X, train_y, test_X):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        random_state=42, verbosity=0,
    )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return np.clip(pred, 0.05, None)


def run_lstm(train_X_seq, train_y_seq, test_X_seq, n_features, seq_len=20):
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                                batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMModel(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_X_t = torch.FloatTensor(train_X_seq).to(device)
    train_y_t = torch.FloatTensor(train_y_seq).to(device)
    test_X_t = torch.FloatTensor(test_X_seq).to(device)

    # Train
    model.train()
    batch_size = 64
    for epoch in range(50):
        perm = torch.randperm(len(train_X_t))
        for b_start in range(0, len(train_X_t), batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(train_X_t[b_idx]), train_y_t[b_idx])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                train_loss = loss_fn(model(train_X_t), train_y_t).item()
                print(f"    LSTM epoch {epoch+1}/50, train MSE={train_loss:.6f}")

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(test_X_t).cpu().numpy()
    return np.clip(pred, 0.05, None)


def run_transformer(train_X_seq, train_y_seq, test_X_seq, n_features, seq_len=20):
    import torch
    import torch.nn as nn

    class VolTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, n_layers=2, max_len=20):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
            x = self.encoder(x)
            return self.fc(x[:, -1, :]).squeeze(-1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VolTransformer(n_features, max_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_X_t = torch.FloatTensor(train_X_seq).to(device)
    train_y_t = torch.FloatTensor(train_y_seq).to(device)
    test_X_t = torch.FloatTensor(test_X_seq).to(device)

    model.train()
    batch_size = 64
    for epoch in range(50):
        perm = torch.randperm(len(train_X_t))
        for b_start in range(0, len(train_X_t), batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(train_X_t[b_idx]), train_y_t[b_idx])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                train_loss = loss_fn(model(train_X_t), train_y_t).item()
                print(f"    Transformer epoch {epoch+1}/50, train MSE={train_loss:.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(test_X_t).cpu().numpy()
    return np.clip(pred, 0.05, None)


def main():
    print("Loading data...")
    df = load_data()
    feat = build_features(df)
    target = df["fwd_vol_20d"].values

    feature_cols = [c for c in feat.columns if feat[c].dtype in [np.float64, np.float32, int]]
    n_features = len(feature_cols)
    print(f"Features: {n_features}")

    # Train/test split: pre-2020 = train, 2020+ = test
    split_mask = df["date"] >= "2020-01-01"
    train_idx = df[~split_mask].index
    test_idx = df[split_mask].index
    print(f"Train: {len(train_idx)} days, Test: {len(test_idx)} days")

    # Prepare flat features (XGBoost)
    all_feat = feat[feature_cols].fillna(0).values
    train_X = all_feat[train_idx]
    train_y = target[train_idx]
    test_X = all_feat[test_idx]
    test_y = target[test_idx]
    test_dates = df.iloc[test_idx]["date"].values
    test_persist = df.iloc[test_idx]["wti_vol_20d"].values

    # Remove NaN rows from train
    valid = ~np.isnan(train_y)
    train_X, train_y = train_X[valid], train_y[valid]
    print(f"Train (clean): {len(train_X)}")

    results = {}

    # ── XGBoost ──────────────────────────────────────────
    print("\nRunning XGBoost...")
    xgb_pred = run_xgboost(train_X, train_y, test_X)
    results["XGBoost"] = xgb_pred
    print(f"  Done. RMSE={np.sqrt(np.mean((xgb_pred - test_y)**2)):.4f}")

    # ── Prepare sequences for LSTM/Transformer ───────────
    seq_len = 20
    # Normalize features
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-8
    all_feat_norm = (all_feat - mean) / std

    # Build sequences from ALL data, then split
    all_seqs_X, all_seqs_y = make_sequences(all_feat_norm, target, seq_len)
    # The sequence at index i uses data from [i, i+seq_len) to predict target at i+seq_len
    # So sequence dates correspond to df index (seq_len) onward
    seq_date_idx = np.arange(seq_len, len(df))

    train_seq_mask = seq_date_idx < test_idx[0]
    test_seq_mask = seq_date_idx >= test_idx[0]

    train_X_seq = all_seqs_X[train_seq_mask]
    train_y_seq = all_seqs_y[train_seq_mask]
    test_X_seq = all_seqs_X[test_seq_mask]
    test_y_seq = all_seqs_y[test_seq_mask]

    # Remove NaN from train sequences
    valid_seq = ~np.isnan(train_y_seq)
    train_X_seq = train_X_seq[valid_seq]
    train_y_seq = train_y_seq[valid_seq]

    print(f"\nSequence data: train={len(train_X_seq)}, test={len(test_X_seq)}")

    # ── LSTM ─────────────────────────────────────────────
    print("\nRunning LSTM...")
    lstm_pred = run_lstm(train_X_seq, train_y_seq, test_X_seq, n_features, seq_len)
    results["LSTM"] = lstm_pred
    print(f"  Done. RMSE={np.sqrt(np.mean((lstm_pred - test_y_seq)**2)):.4f}")

    # ── Transformer ──────────────────────────────────────
    print("\nRunning Transformer...")
    tf_pred = run_transformer(train_X_seq, train_y_seq, test_X_seq, n_features, seq_len)
    results["Transformer"] = tf_pred
    print(f"  Done. RMSE={np.sqrt(np.mean((tf_pred - test_y_seq)**2)):.4f}")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DEEP LEARNING VOLATILITY BASELINES (train once, predict all)")
    print(f"{'=' * 70}")

    bins = [0, 0.20, 0.35, 0.55, 10]
    regime_labels = ["low", "normal", "elevated", "crisis"]

    all_rows = []

    # XGBoost uses flat features (same length as test_idx)
    for name in ["XGBoost"]:
        pred = results[name]
        err_sq = (pred - test_y) ** 2
        rmse = np.sqrt(err_sq.mean())
        mae = np.abs(pred - test_y).mean()
        print(f"  {name:15s}: RMSE={rmse:.4f}  MAE={mae:.4f}  (n={len(pred)})")

        rdf = pd.DataFrame({
            "date": [str(d)[:10] for d in test_dates],
            "actual_vol": test_y, "pred_vol": pred,
            "wti_vol_20d": test_persist, "model": name,
        })
        rdf["vol_regime"] = pd.cut(rdf["wti_vol_20d"], bins=bins, labels=regime_labels)
        all_rows.append(rdf)

    # LSTM/Transformer use sequences (may be slightly shorter due to seq_len)
    test_seq_dates = test_dates[:len(test_y_seq)]  # align
    test_seq_persist = test_persist[:len(test_y_seq)]

    for name in ["LSTM", "Transformer"]:
        pred = results[name]
        err_sq = (pred - test_y_seq) ** 2
        rmse = np.sqrt(err_sq.mean())
        mae = np.abs(pred - test_y_seq).mean()
        print(f"  {name:15s}: RMSE={rmse:.4f}  MAE={mae:.4f}  (n={len(pred)})")

        rdf = pd.DataFrame({
            "date": [str(d)[:10] for d in test_seq_dates],
            "actual_vol": test_y_seq, "pred_vol": pred,
            "wti_vol_20d": test_seq_persist, "model": name,
        })
        rdf["vol_regime"] = pd.cut(rdf["wti_vol_20d"], bins=bins, labels=regime_labels)
        all_rows.append(rdf)

    # Regime breakdown
    print(f"\n--- By Vol Regime ---")
    combined = pd.concat(all_rows, ignore_index=True)
    combined["error_sq"] = (combined["pred_vol"] - combined["actual_vol"]) ** 2

    for name in ["XGBoost", "LSTM", "Transformer"]:
        sub = combined[combined["model"] == name]
        parts = [f"{name:15s}:"]
        for regime in regime_labels:
            rsub = sub[sub["vol_regime"] == regime]
            if len(rsub) > 0:
                r = np.sqrt(rsub["error_sq"].mean())
                parts.append(f"{regime}={r:.4f}(n={len(rsub)})")
        print(f"  {' '.join(parts)}")

    # Save
    out_path = BASE / "results" / "vol_baselines_dl.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
