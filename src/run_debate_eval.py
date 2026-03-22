"""
Walk-forward evaluation of the multi-agent debate volatility forecasting system.

Usage:
  python src/run_debate_eval.py --mode test     # 10 days, verify pipeline
  python src/run_debate_eval.py --mode crisis   # ~200 extreme-vol days only
  python src/run_debate_eval.py --mode full     # all ~1300 days from 2020
  python src/run_debate_eval.py --mode full --resume  # resume interrupted run
"""

import argparse
import json
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from debate_system import (
    DebateEngine, InfluenceGraph, AttributionEngine,
    SingleAgentBaseline, InterventionExperiment,
    prepare_market_data, prepare_agent_data, AGENT_IDS,
)

# ── Statistical Baselines ─────────────────────────────────

def garch_forecast(returns: pd.Series) -> float:
    """GARCH(1,1) volatility forecast. Returns annualized vol."""
    from arch import arch_model
    try:
        scaled = returns.dropna() * 100
        if len(scaled) < 50:
            return scaled.std() / 100 * np.sqrt(252)
        model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = model.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1)
        daily_var = forecast.variance.values[-1, 0] / 10000
        return np.sqrt(daily_var * 252)
    except Exception:
        return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.25


def har_forecast(vol_series: pd.Series) -> float:
    """HAR (Heterogeneous AutoRegressive) volatility forecast.
    RV_t+1 = c + b1*RV_t + b5*RV_t^(w) + b22*RV_t^(m) + e
    where RV^(w) = avg of last 5 days, RV^(m) = avg of last 22 days.
    """
    from sklearn.linear_model import LinearRegression
    vol = vol_series.dropna()
    if len(vol) < 30:
        return vol.iloc[-1] if len(vol) > 0 else 0.25

    # Build HAR features
    rv_d = vol.values  # daily RV
    rv_w = pd.Series(rv_d).rolling(5).mean().values   # weekly avg
    rv_m = pd.Series(rv_d).rolling(22).mean().values   # monthly avg

    # Align: need at least 22 days of history
    valid = ~(np.isnan(rv_w) | np.isnan(rv_m))
    if valid.sum() < 10:
        return vol.iloc[-1]

    X = np.column_stack([rv_d[:-1], rv_w[:-1], rv_m[:-1]])
    y = rv_d[1:]
    # Trim to valid rows
    valid_rows = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid_rows], y[valid_rows]

    if len(X) < 10:
        return vol.iloc[-1]

    model = LinearRegression().fit(X, y)
    # Predict next
    last_features = np.array([[rv_d[-1], rv_w[-1], rv_m[-1]]])
    if np.isnan(last_features).any():
        return vol.iloc[-1]
    pred = model.predict(last_features)[0]
    return max(pred, 0.05)  # floor at 5%


# ── JSON Serialization ────────────────────────────────────

def clean_for_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    return obj


# ── Data Loading ──────────────────────────────────────────

def load_data(horizon: int = 20) -> tuple:
    """Load and merge macro + GDELT data."""
    base = Path(__file__).parent.parent

    macro = pd.read_csv(base / "data/oil_macro_daily.csv", parse_dates=["date"])
    gdelt = pd.read_csv(base / "data/gdelt_daily_features.csv", parse_dates=["date"])

    # Merge on date (inner join: only dates with both macro and GDELT)
    df = macro.merge(gdelt, on="date", how="inner").sort_values("date").reset_index(drop=True)

    vol_col = f"wti_vol_{horizon}d"
    fwd_col = f"fwd_rv_{horizon}d"

    # Target: true forward-looking realized vol
    # fwd_rv_20d = std(returns over next 20 trading days), already in data

    # Drop rows without target
    df = df.dropna(subset=[fwd_col]).reset_index(drop=True)

    # Store horizon info for downstream use
    df.attrs["horizon"] = horizon
    df.attrs["vol_col"] = vol_col
    df.attrs["fwd_col"] = fwd_col

    return df


def select_eval_dates(df: pd.DataFrame, mode: str) -> list:
    """Select evaluation date indices based on mode."""
    vol_col = df.attrs.get("vol_col", "wti_vol_20d")
    horizon = df.attrs.get("horizon", 20)

    # Adjust thresholds for shorter horizons (5d vol is noisier/higher)
    crisis_thresh = 0.45 if horizon >= 20 else 0.55
    low_thresh = 0.25 if horizon >= 20 else 0.20

    # Start from 2020
    start_mask = df["date"] >= "2020-01-01"
    eval_pool = df[start_mask].index.tolist()

    if mode == "test":
        high_vol = df[(start_mask) & (df[vol_col] > crisis_thresh)].index.tolist()
        low_vol = df[(start_mask) & (df[vol_col] < low_thresh)].index.tolist()
        selected = []
        if high_vol:
            step = max(1, len(high_vol) // 5)
            selected += high_vol[::step][:5]
        if low_vol:
            step = max(1, len(low_vol) // 5)
            selected += low_vol[::step][:5]
        return sorted(selected)[:10]

    elif mode == "crisis":
        return df[(start_mask) & (df[vol_col] > crisis_thresh)].index.tolist()

    elif mode == "weekly":
        return eval_pool[::5]

    elif mode == "monthly":
        return eval_pool[::21]

    elif mode == "full":
        return eval_pool

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Checkpoint Utilities ─────────────────────────────────

def get_checkpoint_paths(mode: str) -> tuple:
    """Return deterministic checkpoint file paths for a given mode."""
    base = Path(__file__).parent.parent / "results"
    base.mkdir(exist_ok=True)
    csv_path = base / f"debate_eval_{mode}_checkpoint.csv"
    jsonl_path = base / f"debate_attribution_{mode}_checkpoint.jsonl"
    return csv_path, jsonl_path


def load_completed_dates(csv_path: Path) -> set:
    """Load set of already-evaluated dates from checkpoint CSV."""
    if not csv_path.exists():
        return set()
    try:
        existing = pd.read_csv(csv_path)
        dates = set(existing["date"].astype(str).tolist())
        return dates
    except Exception:
        return set()


def append_result_csv(csv_path: Path, result: dict):
    """Append a single result row to checkpoint CSV."""
    write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0
    pd.DataFrame([result]).to_csv(csv_path, mode='a', header=write_header, index=False)


def append_attribution_jsonl(jsonl_path: Path, report: dict):
    """Append a single attribution report as JSONL."""
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(clean_for_json(report)) + '\n')


def finalize_checkpoint(mode: str, csv_path: Path, jsonl_path: Path):
    """Convert checkpoint files to final timestamped output."""
    base = csv_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Copy CSV
    final_csv = base / f"debate_eval_{mode}_{timestamp}.csv"
    if csv_path.exists():
        import shutil
        shutil.copy2(csv_path, final_csv)
        print(f"  Final CSV: {final_csv}")

    # Convert JSONL to JSON array
    final_json = base / f"debate_attribution_{mode}_{timestamp}.json"
    if jsonl_path.exists():
        reports = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    reports.append(json.loads(line))
        with open(final_json, 'w') as f:
            json.dump(reports, f, indent=2)
        print(f"  Final JSON: {final_json}")

    return final_csv, final_json


# ── Main Pipeline ─────────────────────────────────────────

def run_evaluation(mode: str = "test", n_rounds: int = 2,
                   n_attrib_samples: int = 500, resume: bool = False,
                   horizon: int = 20):
    """Run walk-forward debate evaluation with checkpoint support."""
    print(f"Loading data (horizon={horizon}d)...")
    df = load_data(horizon=horizon)
    vol_col = df.attrs["vol_col"]
    fwd_col = df.attrs["fwd_col"]
    print(f"  Total rows: {len(df)}, date range: {df.date.min().date()} to {df.date.max().date()}")

    eval_indices = select_eval_dates(df, mode)
    print(f"  Mode: {mode}, eval days: {len(eval_indices)}")

    # Checkpoint setup (include horizon in path for h!=20)
    ckpt_mode = f"{mode}_h{horizon}" if horizon != 20 else mode
    csv_path, jsonl_path = get_checkpoint_paths(ckpt_mode)
    completed_dates = set()
    if resume:
        completed_dates = load_completed_dates(csv_path)
        if completed_dates:
            print(f"  Resuming: {len(completed_dates)} dates already completed, "
                  f"{len(eval_indices) - len(completed_dates)} remaining")
        else:
            print(f"  No checkpoint found, starting fresh")
    else:
        # Fresh run: remove old checkpoints
        if csv_path.exists():
            csv_path.unlink()
        if jsonl_path.exists():
            jsonl_path.unlink()

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    N_PARALLEL_DAYS = 3  # number of days to evaluate in parallel
    write_lock = threading.Lock()

    n_total = len(eval_indices)
    n_done_counter = [len(completed_dates)]  # mutable counter for threads
    n_errors_counter = [0]
    t_start = time.time()

    def process_one_day(idx):
        """Process a single evaluation day. Thread-safe."""
        # Each thread gets its own engine instances to avoid shared state
        eng = DebateEngine(n_rounds=n_rounds, temperature=0.3, call_delay=0.0)
        sa = SingleAgentBaseline(temperature=0.3, call_delay=0.0)
        ae = AttributionEngine(n_samples=n_attrib_samples)
        iv = InterventionExperiment(ae)

        row = df.iloc[idx]
        date_str = str(row["date"].date())
        actual_vol = row[fwd_col]
        persist_vol = row[vol_col]

        if date_str in completed_dates:
            return None

        with write_lock:
            n_done_counter[0] += 1
            n_done = n_done_counter[0]
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1) * 3600
            remaining = n_total - n_done
            eta_hours = remaining / max(rate, 0.1)
            print(f"\n[{n_done}/{n_total}] {date_str}  "
                  f"vol_{horizon}d={persist_vol:.3f}  target={actual_vol:.3f}  "
                  f"ETA={eta_hours:.1f}h")

        try:
            market_data = prepare_market_data(row, horizon=horizon)
            agent_data = prepare_agent_data(row, gdelt_row=row, horizon=horizon)

            record = eng.run_debate(
                date=date_str, market_data=market_data,
                agent_data=agent_data, actual_vol=actual_vol,
                persistence_vol=persist_vol,
            )
            record._persistence_vol = persist_vol

            single_vol = sa.forecast(
                date=date_str, market_data=market_data,
                agent_data=agent_data, persistence_vol=persist_vol,
            )

            train_start = max(0, idx - 252)
            returns_history = df.iloc[train_start:idx]["wti_return"].dropna()
            garch_vol = garch_forecast(returns_history)

            vol_history = df.iloc[train_start:idx][vol_col].dropna()
            har_vol = har_forecast(vol_history)

            graph = InfluenceGraph(record)
            report = ae.attribution_report(record, graph)
            intv = iv.run_intervention(record, graph, k=2)

            result = {
                "date": date_str,
                "actual_vol": actual_vol,
                "debate_vol": record.final_vol_forecast,
                "single_vol": single_vol,
                "garch_vol": garch_vol,
                "har_vol": har_vol,
                "persist_vol": persist_vol,
                "debate_error": (record.final_vol_forecast - actual_vol) ** 2,
                "single_error": (single_vol - actual_vol) ** 2,
                "garch_error": (garch_vol - actual_vol) ** 2,
                "har_error": (har_vol - actual_vol) ** 2,
                "persist_error": (persist_vol - actual_vol) ** 2,
                "n_influence_edges": len(record.influence_edges),
                "who": report["who"],
                "when": report["when"],
                "what": report["what"],
                "how": " -> ".join(report["how"]),
                vol_col: persist_vol,
            }

            if intv:
                result["intv_original"] = intv["original_error"]
                result["intv_random"] = intv["random_intervention"]
                result["intv_shapley"] = intv["shapley_intervention"]
                result["intv_myerson"] = intv["myerson_intervention"]
                result["intv_shapley_reduction"] = intv["shapley_reduction"]
                result["intv_myerson_reduction"] = intv["myerson_reduction"]
                result["intv_random_reduction"] = intv["random_reduction"]

            for agent_id in AGENT_IDS:
                result[f"shapley_{agent_id}"] = report["shapley"][agent_id]
                result[f"myerson_{agent_id}"] = report["myerson"][agent_id]
                result[f"behavior_{agent_id}"] = report["behaviors"][agent_id]

            with write_lock:
                append_result_csv(csv_path, result)
                append_attribution_jsonl(jsonl_path, report)

            print(f"  [{date_str}] Debate:{record.final_vol_forecast:.3f} Single:{single_vol:.3f} "
                  f"HAR:{har_vol:.3f} GARCH:{garch_vol:.3f} Persist:{persist_vol:.3f}  "
                  f"WHO={report['who']} WHAT={report['what']}")

            # Return token stats for aggregation
            return eng.total_tokens + sa.total_tokens, eng.fail_count + sa.fail_count, eng.call_count + sa.call_count

        except Exception as e:
            with write_lock:
                n_errors_counter[0] += 1
            print(f"  ERROR on {date_str}: {e}")
            return None

    # Filter out already completed indices
    pending_indices = [idx for idx in eval_indices
                       if str(df.iloc[idx]["date"].date()) not in completed_dates]

    total_tokens = 0
    total_fails = 0
    total_calls = 0

    with ThreadPoolExecutor(max_workers=N_PARALLEL_DAYS) as pool:
        futures = {pool.submit(process_one_day, idx): idx for idx in pending_indices}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                tokens, fails, calls = result
                total_tokens += tokens
                total_fails += fails
                total_calls += calls
            if n_errors_counter[0] > 50:
                print(f"\n  Too many errors ({n_errors_counter[0]}), stopping.")
                pool.shutdown(wait=False, cancel_futures=True)
                break

    n_errors = n_errors_counter[0]

    # ── Summary Statistics ────────────────────────────────
    if not csv_path.exists():
        print("No results generated.")
        return None, None

    results_df = pd.read_csv(csv_path)
    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY ({mode}, horizon={horizon}d, n={len(results_df)})")
    print(f"{'=' * 70}")

    baselines = [
        ("Debate(7-agent)", "debate_error"),
        ("Single-agent", "single_error"),
        ("HAR", "har_error"),
        ("GARCH", "garch_error"),
        ("Persistence", "persist_error"),
    ]

    print(f"\n--- Forecast Accuracy ---")
    for name, col in baselines:
        if col in results_df.columns:
            rmse = np.sqrt(results_df[col].mean())
            mae = np.sqrt(results_df[col]).mean()
            print(f"  {name:18s}: RMSE={rmse:.4f}  MAE={mae:.4f}")

    # By vol regime
    if len(results_df) > 20 and vol_col in results_df.columns:
        print(f"\n--- By Vol Regime ---")
        if horizon <= 5:
            bins = [0, 0.20, 0.40, 0.65, 10]
        else:
            bins = [0, 0.20, 0.35, 0.55, 10]
        results_df["vol_regime"] = pd.cut(
            results_df[vol_col],
            bins=bins,
            labels=["low", "normal", "elevated", "crisis"]
        )
        for regime in ["low", "normal", "elevated", "crisis"]:
            subset = results_df[results_df["vol_regime"] == regime]
            if len(subset) > 0:
                parts = [f"{regime:10s} (n={len(subset):4d}):"]
                for name, col in baselines:
                    if col in results_df.columns:
                        r = np.sqrt(subset[col].mean())
                        parts.append(f"{name.split('(')[0].strip()[:6]}={r:.4f}")
                print(f"  {' '.join(parts)}")

    # Intervention results
    if "intv_myerson_reduction" in results_df.columns:
        print(f"\n--- Intervention Experiment (top-2 agents replaced) ---")
        for name, col in [("Myerson-targeted", "intv_myerson_reduction"),
                          ("Shapley-targeted", "intv_shapley_reduction"),
                          ("Random", "intv_random_reduction")]:
            vals = results_df[col].dropna()
            if len(vals) > 0:
                print(f"  {name:20s}: avg error reduction = {vals.mean()*100:.1f}%  "
                      f"(median={vals.median()*100:.1f}%)")

    # Attribution summary
    if "who" in results_df.columns:
        print(f"\n--- Attribution Patterns ---")
        print(f"  Most frequent WHO:")
        who_counts = results_df["who"].value_counts()
        for agent, count in who_counts.head(3).items():
            print(f"    {agent}: {count} times ({count/len(results_df)*100:.1f}%)")

        print(f"  Most frequent WHAT:")
        what_counts = results_df["what"].value_counts()
        for behav, count in what_counts.head(3).items():
            print(f"    {behav}: {count} times ({count/len(results_df)*100:.1f}%)")

    # Shapley vs Myerson divergence
    shapley_cols = [c for c in results_df.columns if c.startswith("shapley_")]
    if shapley_cols:
        print(f"\n--- Shapley vs Myerson Divergence ---")
        for agent_id in AGENT_IDS:
            s_col = f"shapley_{agent_id}"
            m_col = f"myerson_{agent_id}"
            if s_col in results_df.columns and m_col in results_df.columns:
                mean_div = (results_df[s_col] - results_df[m_col]).abs().mean()
                print(f"  {agent_id:20s}: avg |S-M| = {mean_div:.6f}")

    print(f"\n--- LLM Usage ---")
    print(f"  Total calls: {total_calls}")
    print(f"  Failed calls: {total_fails}")
    success_rate = (total_calls - total_fails) / max(total_calls, 1)
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Total tokens: {total_tokens:,}")
    if n_errors > 0:
        print(f"  Date-level errors: {n_errors}")

    # Finalize: create timestamped copies
    final_csv, final_json = finalize_checkpoint(ckpt_mode, csv_path, jsonl_path)
    print(f"\nCheckpoint: {csv_path}")
    print(f"Results saved to: {final_csv}")

    return results_df, None


# ── Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test",
                        choices=["test", "crisis", "weekly", "monthly", "full"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--attrib-samples", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 20],
                        help="Volatility horizon in days (5 or 20)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip already-evaluated dates)")
    args = parser.parse_args()

    run_evaluation(mode=args.mode, n_rounds=args.rounds,
                   n_attrib_samples=args.attrib_samples, resume=args.resume,
                   horizon=args.horizon)
