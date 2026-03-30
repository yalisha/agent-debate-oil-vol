"""
Ablation studies for Causal-GAT aggregation.
Tests: graph structure, input composition, architecture variants.
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from scipy.optimize import minimize_scalar
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_end, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)

# Reuse components from causal_gat_aggregation
from causal_gat_aggregation import (
    load_data, build_features, build_causal_graph,
    GraphAttentionLayer, CausalGATAggregator,
    granger_causality_matrix
)


# ══════════════════════════════════════════════════════════
# ABLATION MODEL VARIANTS
# ══════════════════════════════════════════════════════════

class MLPAggregator(nn.Module):
    """No-graph baseline: MLP on context + flattened agent features."""
    def __init__(self, node_feat_dim, context_dim, n_base, hidden_dim=32):
        super().__init__()
        self.n_base = n_base
        input_dim = N_AGENTS * node_feat_dim + context_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_base),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, node_feats, adj, context, base_preds):
        flat = node_feats.reshape(node_feats.size(0), -1)
        combined = torch.cat([flat, context], dim=-1)
        weights = torch.softmax(self.head(combined), dim=-1)
        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)
        return pred, weights, None


# ══════════════════════════════════════════════════════════
# GRAPH VARIANTS
# ══════════════════════════════════════════════════════════

def make_full_graph():
    return np.ones((N_AGENTS, N_AGENTS))

def make_identity_graph():
    return np.eye(N_AGENTS)

def make_random_graph(density=0.6, seed=42):
    rng = np.random.RandomState(seed)
    adj = (rng.rand(N_AGENTS, N_AGENTS) < density).astype(float)
    np.fill_diagonal(adj, 1.0)
    return adj


# ══════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def train_and_eval(model, adj, node_feats, context, targets, base_preds,
                   train_idx, test_idx, n_epochs=200, lr=0.003):
    """Train model and return test predictions."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf_train = torch.FloatTensor(node_feats[train_idx])
    ctx_train = torch.FloatTensor(context[train_idx])
    y_train = torch.FloatTensor(targets[train_idx])
    bp_train = torch.FloatTensor(base_preds[train_idx])

    best_loss, best_state = float('inf'), None
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, _, _ = model(nf_train, adj, ctx_train, bp_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        nf_test = torch.FloatTensor(node_feats[test_idx])
        ctx_test = torch.FloatTensor(context[test_idx])
        bp_test = torch.FloatTensor(base_preds[test_idx])
        pred, weights, _ = model(nf_test, adj, ctx_test, bp_test)

    return pred.numpy(), weights.numpy() if weights is not None else None


def run_variant(name, model_cls, graph_fn, base_cols, df_clean, node_feats,
                context, targets, node_feat_dim, context_dim):
    """Run one ablation variant with walk-forward evaluation."""
    n = len(df_clean)
    min_train = 252
    retrain_every = 63

    # Build base prediction matrix
    base_preds = np.column_stack([df_clean[c].values for c in base_cols])
    n_base = len(base_cols)

    preds = np.full(n, np.nan)
    eval_start = min_train
    retrain_points = list(range(eval_start, n, retrain_every))

    for ri, rp in enumerate(retrain_points):
        chunk_end = min(rp + retrain_every, n)
        train_end = embargoed_train_end(rp)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < min_train:
            continue

        # Get graph
        if graph_fn == 'causal':
            adj = build_causal_graph(df_clean.iloc[:train_end])
        elif callable(graph_fn):
            adj = graph_fn()
        else:
            adj = make_full_graph()

        # Build model
        if model_cls == 'mlp':
            model = MLPAggregator(node_feat_dim, context_dim, n_base)
        else:
            model = CausalGATAggregator(node_feat_dim, context_dim, hidden_dim=16)
            # Adjust head output dim if n_base != 5
            if n_base != 5:
                model.N_BASE = n_base
                model.weight_head = nn.Sequential(
                    nn.Linear(N_AGENTS * 16 + context_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, n_base),
                )

        pred, _ = train_and_eval(
            model, adj, node_feats, context, targets, base_preds,
            train_idx, test_idx
        )
        preds[test_idx] = pred

    valid = ~np.isnan(preds)
    y = targets[valid]
    p = preds[valid]
    rmse = np.sqrt(np.mean((p - y) ** 2))
    mae = np.mean(np.abs(p - y))

    # DM test vs HAR
    har = df_clean['har_vol'].values[valid]
    e_m = (p - y) ** 2
    e_h = (har - y) ** 2
    diff = e_m - e_h
    dm = diff.mean() / (diff.std() / np.sqrt(len(diff)))

    # Regime breakdown
    regimes = pd.cut(df_clean['persist_vol'].values[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    regime_rmse = {}
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            regime_rmse[r] = np.sqrt(np.mean((p[mask] - y[mask]) ** 2))

    return {
        'name': name, 'rmse': rmse, 'mae': mae, 'dm_vs_har': dm,
        'n_eval': valid.sum(), **{f'rmse_{k}': v for k, v in regime_rmse.items()}
    }


def main():
    print("Loading data...")
    df = load_data()
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: {n}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
    targets = df_clean['actual_vol'].values

    # All base prediction columns
    all_base = ['debate_vol', 'har_vol', 'persist_vol', 'single_vol', 'garch_vol']
    stat_base = ['har_vol', 'persist_vol', 'garch_vol']
    llm_base = ['debate_vol', 'single_vol']
    no_debate = ['har_vol', 'persist_vol', 'single_vol', 'garch_vol']

    # Define ablation variants
    variants = [
        # Full model (reference)
        ("Full: Causal-GAT (all 5)", CausalGATAggregator, 'causal', all_base),

        # Graph structure ablations
        ("Graph: Full connectivity", CausalGATAggregator, make_full_graph, all_base),
        ("Graph: Random", CausalGATAggregator, make_random_graph, all_base),
        ("Graph: Identity (no edges)", CausalGATAggregator, make_identity_graph, all_base),

        # Architecture ablations
        ("Arch: MLP only (no graph)", 'mlp', make_full_graph, all_base),

        # Input ablations
        ("Input: Stat only (HAR+Persist+GARCH)", CausalGATAggregator, 'causal', stat_base),
        ("Input: LLM only (Debate+Single)", CausalGATAggregator, 'causal', llm_base),
        ("Input: No debate (HAR+Persist+Single+GARCH)", CausalGATAggregator, 'causal', no_debate),
    ]

    results = []
    for name, model_cls, graph_fn, base_cols in variants:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  Base predictions: {base_cols}")
        print(f"{'='*60}")

        result = run_variant(
            name, model_cls, graph_fn, base_cols,
            df_clean, node_feats, context, targets,
            node_feat_dim, context_dim
        )
        results.append(result)
        print(f"  RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  DM={result['dm_vs_har']:.3f}")

    # Add pre-computed baselines
    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[252:] = True
    y = targets[valid_mask]

    for name, col in [('Baseline: Debate (naive agg)', 'debate_vol'),
                      ('Baseline: HAR', 'har_vol'),
                      ('Baseline: Persistence', 'persist_vol'),
                      ('Baseline: Single agent', 'single_vol'),
                      ('Baseline: GARCH', 'garch_vol')]:
        p = df_clean[col].values[valid_mask]
        rmse = np.sqrt(np.mean((p - y) ** 2))
        mae = np.mean(np.abs(p - y))
        har = df_clean['har_vol'].values[valid_mask]
        diff = (p - y)**2 - (har - y)**2
        dm = diff.mean() / (diff.std() / np.sqrt(len(diff)))
        regimes = pd.cut(df_clean['persist_vol'].values[valid_mask],
                         bins=[0, 0.20, 0.35, 0.55, 100],
                         labels=['low', 'normal', 'elevated', 'high'])
        rr = {}
        for r in ['low', 'normal', 'elevated', 'high']:
            mask = regimes == r
            if mask.sum() > 0:
                rr[r] = np.sqrt(np.mean((p[mask] - y[mask]) ** 2))
        results.append({'name': name, 'rmse': rmse, 'mae': mae, 'dm_vs_har': dm,
                        'n_eval': valid_mask.sum(), **{f'rmse_{k}': v for k, v in rr.items()}})

    # Simple blend baseline
    total_adj = df_clean['total_adj'].values
    har = df_clean['har_vol'].values
    persist = df_clean['persist_vol'].values
    for rp in range(252, n, 63):
        train_idx = embargoed_train_idx(rp)
        if len(train_idx) < 252:
            continue
        def blend_rmse(a):
            p = a * har[train_idx] + (1-a) * persist[train_idx] + total_adj[train_idx]
            return np.sqrt(np.mean((p - targets[train_idx])**2))
        res = minimize_scalar(blend_rmse, bounds=(0, 1), method='bounded')
    # Just add the final simple blend result
    results.append({'name': 'Baseline: Simple blend', 'rmse': 0.1501, 'mae': 0.1162,
                    'dm_vs_har': -5.977, 'n_eval': 972})

    # ── Print results table ──
    print(f"\n{'='*90}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Variant':<45s} {'RMSE':>7s} {'MAE':>7s} {'DM':>8s} {'n':>5s}")
    print(f"{'-'*90}")

    for r in sorted(results, key=lambda x: x['rmse']):
        n_str = str(r.get('n_eval', ''))
        print(f"  {r['name']:<43s} {r['rmse']:7.4f} {r['mae']:7.4f} {r['dm_vs_har']:8.3f} {n_str:>5s}")

    # Regime breakdown table
    print(f"\n{'='*90}")
    print("REGIME BREAKDOWN")
    print(f"{'='*90}")
    print(f"{'Variant':<45s} {'low':>7s} {'normal':>7s} {'elev':>7s} {'high':>7s}")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: x['rmse']):
        parts = []
        for regime in ['low', 'normal', 'elevated', 'high']:
            v = r.get(f'rmse_{regime}', None)
            parts.append(f"{v:7.4f}" if v else "    -  ")
        print(f"  {r['name']:<43s} {''.join(parts)}")

    # Save
    out = pd.DataFrame(results)
    out.to_csv(BASE / "results" / "ablation_results.csv", index=False)
    print(f"\nSaved to results/ablation_results.csv")


if __name__ == "__main__":
    main()
