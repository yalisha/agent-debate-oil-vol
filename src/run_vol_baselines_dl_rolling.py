"""
XGBoost / LSTM / Transformer baselines with quarterly rolling retrain.
Retrain every 63 trading days (~1 quarter), predict next quarter.
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


SEQ_LEN = 20
RETRAIN_EVERY = 63  # quarterly


def train_xgb(train_X, train_y):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        random_state=42, verbosity=0,
    )
    model.fit(train_X, train_y)
    return model


def train_lstm(train_X_seq, train_y_seq, n_features):
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

    X_t = torch.FloatTensor(train_X_seq).to(device)
    y_t = torch.FloatTensor(train_y_seq).to(device)

    model.train()
    for epoch in range(30):
        perm = torch.randperm(len(X_t))
        for b in range(0, len(X_t), 64):
            idx = perm[b:b+64]
            optimizer.zero_grad()
            loss_fn(model(X_t[idx]), y_t[idx]).backward()
            optimizer.step()

    return model, device


def train_transformer(train_X_seq, train_y_seq, n_features):
    import torch
    import torch.nn as nn

    class VolTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, n_layers=2):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.fc = nn.Linear(d_model, 1)
        def forward(self, x):
            x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
            return self.fc(self.encoder(x)[:, -1, :]).squeeze(-1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VolTransformer(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X_t = torch.FloatTensor(train_X_seq).to(device)
    y_t = torch.FloatTensor(train_y_seq).to(device)

    model.train()
    for epoch in range(30):
        perm = torch.randperm(len(X_t))
        for b in range(0, len(X_t), 64):
            idx = perm[b:b+64]
            optimizer.zero_grad()
            loss_fn(model(X_t[idx]), y_t[idx]).backward()
            optimizer.step()

    return model, device


def predict_torch(model, device, test_X_seq):
    import torch
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(test_X_seq).to(device)).cpu().numpy()
    return np.clip(pred, 0.05, None)


def main():
    import torch

    print("Loading data...")
    df = load_data()
    feat = build_features(df)
    target = df["fwd_vol_20d"].values

    feature_cols = [c for c in feat.columns if feat[c].dtype in [np.float64, np.float32, int]]
    n_features = len(feature_cols)
    all_feat = feat[feature_cols].fillna(0).values

    # Test period: 2020+
    test_start = df[df["date"] >= "2020-01-01"].index[0]
    print(f"Test start index: {test_start}, retrain every {RETRAIN_EVERY} days")

    # Determine retrain points
    test_indices = list(range(test_start, len(df)))
    retrain_points = list(range(0, len(test_indices), RETRAIN_EVERY))
    n_retrains = len(retrain_points)
    print(f"Total retrains: {n_retrains} (for {len(test_indices)} test days)")

    # Storage
    preds = {name: np.full(len(test_indices), np.nan) for name in ["XGBoost", "LSTM", "Transformer"]}

    for ri, rp in enumerate(retrain_points):
        chunk_start = rp
        chunk_end = min(rp + RETRAIN_EVERY, len(test_indices))
        abs_test_start = test_indices[chunk_start]
        abs_test_end = test_indices[chunk_end - 1] + 1

        # Train on all data up to this chunk
        train_end = test_indices[chunk_start]
        train_X = all_feat[:train_end]
        train_y = target[:train_end]
        valid = ~np.isnan(train_y)
        train_X_clean, train_y_clean = train_X[valid], train_y[valid]

        if len(train_X_clean) < 100:
            continue

        # Flat test data for XGBoost
        test_X_flat = all_feat[test_indices[chunk_start]:test_indices[chunk_end - 1] + 1]

        # Normalize for DL
        mean = train_X_clean.mean(axis=0)
        std = train_X_clean.std(axis=0) + 1e-8
        all_norm = (all_feat - mean) / std

        # Build sequences for train
        train_seqs_X, train_seqs_y = [], []
        for j in range(SEQ_LEN, train_end):
            if not np.isnan(target[j]):
                train_seqs_X.append(all_norm[j - SEQ_LEN:j])
                train_seqs_y.append(target[j])
        train_seqs_X = np.array(train_seqs_X)
        train_seqs_y = np.array(train_seqs_y)

        # Build sequences for test chunk
        test_seqs_X = []
        test_seq_indices = []
        for ti in range(chunk_start, chunk_end):
            abs_idx = test_indices[ti]
            if abs_idx >= SEQ_LEN:
                test_seqs_X.append(all_norm[abs_idx - SEQ_LEN:abs_idx])
                test_seq_indices.append(ti)
        test_seqs_X = np.array(test_seqs_X) if test_seqs_X else np.empty((0, SEQ_LEN, n_features))

        print(f"  Retrain {ri+1}/{n_retrains}: train={len(train_X_clean)}, "
              f"test_chunk={chunk_end - chunk_start}, date={df.iloc[test_indices[chunk_start]]['date'].date()}")

        # XGBoost
        xgb_model = train_xgb(train_X_clean, train_y_clean)
        xgb_pred = np.clip(xgb_model.predict(test_X_flat), 0.05, None)
        preds["XGBoost"][chunk_start:chunk_end] = xgb_pred

        # LSTM
        if len(train_seqs_X) > 50 and len(test_seqs_X) > 0:
            lstm_model, device = train_lstm(train_seqs_X, train_seqs_y, n_features)
            lstm_pred = predict_torch(lstm_model, device, test_seqs_X)
            for pi, ti in enumerate(test_seq_indices):
                preds["LSTM"][ti] = lstm_pred[pi]

        # Transformer
        if len(train_seqs_X) > 50 and len(test_seqs_X) > 0:
            tf_model, device = train_transformer(train_seqs_X, train_seqs_y, n_features)
            tf_pred = predict_torch(tf_model, device, test_seqs_X)
            for pi, ti in enumerate(test_seq_indices):
                preds["Transformer"][ti] = tf_pred[pi]

    # ── Results ──────────────────────────────────────────
    test_y = target[test_start:test_start + len(test_indices)]
    test_dates = df.iloc[test_indices]["date"].values
    test_persist = df.iloc[test_indices]["wti_vol_20d"].values

    bins = [0, 0.20, 0.35, 0.55, 10]
    regime_labels = ["low", "normal", "elevated", "crisis"]

    print(f"\n{'=' * 70}")
    print(f"DL BASELINES (quarterly rolling retrain, {n_retrains} retrains)")
    print(f"{'=' * 70}")

    all_rows = []
    for name in ["XGBoost", "LSTM", "Transformer"]:
        pred = preds[name]
        valid_mask = ~np.isnan(pred)
        p, a = pred[valid_mask], test_y[valid_mask]
        rmse = np.sqrt(np.mean((p - a) ** 2))
        mae = np.abs(p - a).mean()
        print(f"  {name:15s}: RMSE={rmse:.4f}  MAE={mae:.4f}  (n={valid_mask.sum()})")

        rdf = pd.DataFrame({
            "date": [str(d)[:10] for d in test_dates[valid_mask]],
            "actual_vol": a, "pred_vol": p,
            "wti_vol_20d": test_persist[valid_mask], "model": name,
        })
        rdf["vol_regime"] = pd.cut(rdf["wti_vol_20d"], bins=bins, labels=regime_labels)
        all_rows.append(rdf)

    combined = pd.concat(all_rows, ignore_index=True)
    combined["error_sq"] = (combined["pred_vol"] - combined["actual_vol"]) ** 2

    print(f"\n--- By Vol Regime ---")
    for name in ["XGBoost", "LSTM", "Transformer"]:
        sub = combined[combined["model"] == name]
        parts = [f"{name:15s}:"]
        for regime in regime_labels:
            rsub = sub[sub["vol_regime"] == regime]
            if len(rsub) > 0:
                r = np.sqrt(rsub["error_sq"].mean())
                parts.append(f"{regime}={r:.4f}(n={len(rsub)})")
        print(f"  {' '.join(parts)}")

    out_path = BASE / "results" / "vol_baselines_dl_rolling.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
