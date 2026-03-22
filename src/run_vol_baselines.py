"""
ML baselines for volatility prediction (same target as debate system).
Predict next-day 20d realized volatility using walk-forward evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

BASE = Path(__file__).parent.parent


def load_data():
    macro = pd.read_csv(BASE / "data/oil_macro_daily.csv", parse_dates=["date"])
    gdelt = pd.read_csv(BASE / "data/gdelt_daily_features.csv", parse_dates=["date"])
    df = macro.merge(gdelt, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["fwd_vol_20d"] = df["wti_vol_20d"].shift(-1)
    df = df.dropna(subset=["fwd_vol_20d"]).reset_index(drop=True)
    return df


def build_features(df):
    """Build feature columns for volatility prediction."""
    feat = pd.DataFrame(index=df.index)

    # Lagged vol
    feat["vol_20d"] = df["wti_vol_20d"]
    feat["vol_60d"] = df["wti_vol_60d"]
    feat["vol_ratio"] = df["wti_vol_20d"] / (df["wti_vol_60d"] + 1e-8)

    # Vol lags
    for lag in [1, 2, 3, 5, 10]:
        feat[f"vol_lag_{lag}"] = df["wti_vol_20d"].shift(lag)

    # HAR components
    feat["vol_week"] = df["wti_vol_20d"].rolling(5).mean()
    feat["vol_month"] = df["wti_vol_20d"].rolling(22).mean()

    # Vol momentum
    feat["vol_diff_1"] = df["wti_vol_20d"].diff(1)
    feat["vol_diff_5"] = df["wti_vol_20d"].diff(5)

    # Return features
    feat["abs_return"] = df["wti_return"].abs()
    feat["abs_ret_5d"] = df["wti_return"].abs().rolling(5).mean()
    feat["ret_sq_5d"] = (df["wti_return"] ** 2).rolling(5).mean()

    # Momentum
    feat["mom_5d"] = df["wti_mom_5d"]
    feat["mom_20d"] = df["wti_mom_20d"]

    # Macro
    feat["vix"] = df["VIX"]
    feat["vix_change"] = df["VIX"].diff()
    feat["dxy_return"] = df["dxy_return"]
    feat["spread_10y2y"] = df["YIELD_SPREAD_10Y_2Y"]

    # GDELT
    for col in ["oil_goldstein_mean", "oil_conflict_share",
                "oil_material_conflict_share", "oil_tone_mean",
                "oil_net_coop", "n_events_oil"]:
        if col in df.columns:
            feat[col] = df[col]
            feat[f"{col}_ma5"] = df[col].rolling(5).mean()

    return feat


def get_models():
    return {
        "HistMean": None,
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.001),
        "GBR": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42),
        "RF": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
    }


def main():
    print("Loading data...")
    df = load_data()
    feat = build_features(df)
    target = df["fwd_vol_20d"]

    # Eval dates: 2020 onward (same as debate)
    eval_mask = df["date"] >= "2020-01-01"
    eval_indices = df[eval_mask].index.tolist()
    print(f"Eval dates: {len(eval_indices)} (2020-01-01 to {df.date.max().date()})")

    feature_cols = [c for c in feat.columns if feat[c].dtype in [np.float64, np.float32, int]]
    models = get_models()
    train_window = 252

    all_results = {name: [] for name in models}

    for i, idx in enumerate(eval_indices):
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(eval_indices)}")

        train_start = max(0, idx - train_window)
        train_df = feat.iloc[train_start:idx].copy()
        train_y = target.iloc[train_start:idx].copy()

        # Drop NaN rows
        valid = train_df.notna().all(axis=1) & train_y.notna()
        train_X = train_df.loc[valid, feature_cols]
        train_y = train_y.loc[valid]

        if len(train_X) < 50:
            continue

        test_X = feat.iloc[[idx]][feature_cols]
        if test_X.isna().any(axis=1).iloc[0]:
            # Fill NaN with 0 for test
            test_X = test_X.fillna(0)

        actual = target.iloc[idx]
        date_str = str(df.iloc[idx]["date"].date())
        persist_vol = df.iloc[idx]["wti_vol_20d"]

        for name, model in models.items():
            if name == "HistMean":
                pred = train_y.mean()
            else:
                try:
                    model.fit(train_X, train_y)
                    pred = model.predict(test_X)[0]
                    pred = max(pred, 0.05)  # floor
                except Exception:
                    pred = persist_vol

            all_results[name].append({
                "date": date_str,
                "actual_vol": actual,
                "pred_vol": pred,
                "persist_vol": persist_vol,
                "error_sq": (pred - actual) ** 2,
                "persist_error_sq": (persist_vol - actual) ** 2,
                "regime": df.iloc[idx].get("regime", ""),
                "wti_vol_20d": persist_vol,
            })

    # Summary
    print(f"\n{'=' * 70}")
    print("VOLATILITY PREDICTION BASELINES")
    print(f"{'=' * 70}")

    summary_rows = []
    for name in models:
        rdf = pd.DataFrame(all_results[name])
        rmse = np.sqrt(rdf["error_sq"].mean())
        mae = np.sqrt(rdf["error_sq"]).mean()
        n = len(rdf)
        summary_rows.append({"model": name, "n": n, "RMSE": rmse, "MAE": mae})
        print(f"  {name:12s}: RMSE={rmse:.4f}  MAE={mae:.4f}  (n={n})")

    # Regime breakdown
    print(f"\n--- By Vol Regime ---")
    bins = [0, 0.20, 0.35, 0.55, 10]
    labels = ["low", "normal", "elevated", "crisis"]

    for name in models:
        rdf = pd.DataFrame(all_results[name])
        rdf["vol_regime"] = pd.cut(rdf["wti_vol_20d"], bins=bins, labels=labels)
        parts = [f"{name:12s}:"]
        for regime in labels:
            sub = rdf[rdf["vol_regime"] == regime]
            if len(sub) > 0:
                r = np.sqrt(sub["error_sq"].mean())
                parts.append(f"{regime}={r:.4f}(n={len(sub)})")
        print(f"  {' '.join(parts)}")

    # Save
    out_path = BASE / "results" / "vol_baselines_full.csv"
    rows = []
    for name in models:
        for r in all_results[name]:
            r["model"] = name
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(BASE / "results" / "vol_baselines_summary.csv", index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
