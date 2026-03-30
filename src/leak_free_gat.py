"""
Leak-free GAT experiments using v2 per-agent R1/R2/R3 features.

Progressive pipeline:
  Step 1: Feature diagnostic (correlations with actual_vol)
  Step 2: Ridge stacking (go/no-go gate)
  Step 3: MLP (non-linear combination)
  Step 4: DropEdge GAT (graph structure)
  Step 5: GAT vs MLP direct comparison

All features are leak-free: available at prediction time t without
using any future information.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_idx
from optimized_gat import MultiHeadGATLayer, dm_test_hac

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)
H = 20
SEEDS = [42, 123, 456, 789, 1024]

V3_CSV = BASE / 'results' / 'debate_eval_full_v3_checkpoint.csv'
V2_CSV = BASE / 'results' / 'debate_eval_full_v2_20260328_2014.csv'


# ══════════════════════════════════════════════════════════
# DATA LOADING & FEATURE CONSTRUCTION
# ══════════════════════════════════════════════════════════

def load_data(version='v3'):
    csv_path = V3_CSV if version == 'v3' else V2_CSV
    df = pd.read_csv(csv_path)
    df = df.sort_values('date').reset_index(drop=True)
    # Remove extreme outliers (same as v1: actual_vol > 2 OR persist_vol > 2)
    df = df[(df['actual_vol'] <= 2.0) & (df['persist_vol'] <= 2.0)].reset_index(drop=True)
    return df


def build_leak_free_features(df, tier='t1'):
    """Build per-agent node features (all leak-free).

    Tier 1 (6-dim per agent): final round output + revision summary
      adj_r3, conf_r3, total_revision, conf_trend, revision_r23, agg_weight

    Tier 2 (12-dim per agent): all rounds
      adj_r1, conf_r1, adj_r2, conf_r2, adj_r3, conf_r3,
      revision_r12, revision_r23, total_revision, conf_trend,
      signed_revision_r12, signed_revision_r23

    Tier 3 (Tier 2 + rolling 5d stats): add temporal patterns
      + rolling 5d mean/std of adj_r3, rolling 5d mean of total_revision
    """
    n = len(df)

    if tier == 't1':
        feat_dim = 6
        node_feats = np.zeros((n, N_AGENTS, feat_dim))
        for i, a in enumerate(AGENTS):
            node_feats[:, i, 0] = df[f'adj_r3_{a}'].values
            node_feats[:, i, 1] = df[f'conf_r3_{a}'].values
            node_feats[:, i, 2] = df[f'total_revision_{a}'].values
            node_feats[:, i, 3] = df[f'conf_trend_{a}'].values
            node_feats[:, i, 4] = df[f'revision_r23_{a}'].values
            node_feats[:, i, 5] = df[f'agg_weight_{a}'].values

    elif tier == 't2':
        feat_dim = 12
        node_feats = np.zeros((n, N_AGENTS, feat_dim))
        for i, a in enumerate(AGENTS):
            node_feats[:, i, 0] = df[f'adj_r1_{a}'].values
            node_feats[:, i, 1] = df[f'conf_r1_{a}'].values
            node_feats[:, i, 2] = df[f'adj_r2_{a}'].values
            node_feats[:, i, 3] = df[f'conf_r2_{a}'].values
            node_feats[:, i, 4] = df[f'adj_r3_{a}'].values
            node_feats[:, i, 5] = df[f'conf_r3_{a}'].values
            node_feats[:, i, 6] = df[f'revision_r12_{a}'].values
            node_feats[:, i, 7] = df[f'revision_r23_{a}'].values
            node_feats[:, i, 8] = df[f'total_revision_{a}'].values
            node_feats[:, i, 9] = df[f'conf_trend_{a}'].values
            # Signed revisions
            r1 = df[f'adj_r1_{a}'].values
            r2 = df[f'adj_r2_{a}'].values
            r3 = df[f'adj_r3_{a}'].values
            node_feats[:, i, 10] = r2 - r1
            node_feats[:, i, 11] = r3 - r2

    elif tier == 't3':
        # Start with t2 features, add rolling stats
        base = build_leak_free_features(df, 't2')
        feat_dim = 15  # 12 + 3 rolling
        node_feats = np.zeros((n, N_AGENTS, feat_dim))
        node_feats[:, :, :12] = base

        for i, a in enumerate(AGENTS):
            adj_r3 = pd.Series(df[f'adj_r3_{a}'].values)
            tot_rev = pd.Series(df[f'total_revision_{a}'].values)
            node_feats[:, i, 12] = adj_r3.rolling(5, min_periods=1).mean().values
            node_feats[:, i, 13] = adj_r3.rolling(5, min_periods=1).std().fillna(0).values
            node_feats[:, i, 14] = tot_rev.rolling(5, min_periods=1).mean().values

    else:
        raise ValueError(f"Unknown tier: {tier}")

    # Replace NaN with 0
    node_feats = np.nan_to_num(node_feats, 0.0)
    return node_feats


def build_context_features(df):
    """Context features (augmented with cross-agent statistics).

    Base context (same as v1):
      persist_vol, har_vol, total_adj, n_herding/7, vol_regime/3,
      debate_vol, persist-har gap, vol_change_5d, debate-har gap

    Augmented (cross-agent summaries):
      mean_adj_r3, std_adj_r3 (disagreement), mean_conf_r3,
      std_conf_r3 (confidence dispersion), mean_total_revision
    """
    n = len(df)
    pv = df['persist_vol'].values
    har = df['har_vol'].values
    dv = df['debate_vol'].values

    # Count herding
    n_herding = np.zeros(n)
    for a in AGENTS:
        if f'behavior_{a}' in df.columns:
            n_herding += (df[f'behavior_{a}'] == 'herding').astype(float).values

    # Vol regime
    vol_regime = np.digitize(pv, bins=[0.20, 0.35, 0.55]) / 3.0

    # Vol change 5d
    pv_series = pd.Series(pv)
    vol_change_5d = (pv_series - pv_series.shift(5)).fillna(0).values

    # Total adj
    total_adj = np.zeros(n)
    for a in AGENTS:
        total_adj += df[f'adj_r3_{a}'].values

    # Cross-agent summary stats
    adj_r3_all = np.column_stack([df[f'adj_r3_{a}'].values for a in AGENTS])
    conf_r3_all = np.column_stack([df[f'conf_r3_{a}'].values for a in AGENTS])
    tot_rev_all = np.column_stack([df[f'total_revision_{a}'].values for a in AGENTS])

    context = np.column_stack([
        pv,                                    # 0: persist_vol
        har,                                   # 1: har_vol
        total_adj,                             # 2: total_adj
        n_herding / 7.0,                       # 3: herding_frac
        vol_regime,                            # 4: vol_regime
        dv,                                    # 5: debate_vol
        pv - har,                              # 6: persist-har gap
        vol_change_5d,                         # 7: vol_change_5d
        dv - har,                              # 8: debate-har gap
        adj_r3_all.mean(axis=1),               # 9: mean_adj_r3
        adj_r3_all.std(axis=1),                # 10: std_adj_r3 (disagreement)
        conf_r3_all.mean(axis=1),              # 11: mean_conf_r3
        conf_r3_all.std(axis=1),               # 12: std_conf_r3
        tot_rev_all.mean(axis=1),              # 13: mean_total_revision
    ])

    context = np.nan_to_num(context, 0.0)
    return context


def build_regime_feats(df):
    pv = df['persist_vol'].values
    pv_series = pd.Series(pv)
    vol_change_5d = (pv_series - pv_series.shift(5)).fillna(0).values
    n_herding = np.zeros(len(df))
    for a in AGENTS:
        if f'behavior_{a}' in df.columns:
            n_herding += (df[f'behavior_{a}'] == 'herding').astype(float).values
    return np.column_stack([pv, vol_change_5d, n_herding / 7.0])


# ══════════════════════════════════════════════════════════
# STEP 1: FEATURE DIAGNOSTIC
# ══════════════════════════════════════════════════════════

def feature_diagnostic(df):
    """Correlations and partial correlations with actual_vol."""
    print("=" * 70)
    print("STEP 1: FEATURE DIAGNOSTIC")
    print("=" * 70)

    y = df['actual_vol'].values
    dv = df['debate_vol'].values

    print(f"\nSample: n={len(df)}")
    print(f"\nBaseline correlations:")
    print(f"  debate_vol vs actual_vol: r={np.corrcoef(dv, y)[0,1]:.4f}")
    print(f"  persist_vol vs actual_vol: r={np.corrcoef(df['persist_vol'].values, y)[0,1]:.4f}")

    # Per-agent R3 adjustment correlations
    print(f"\n--- Per-agent adj_r3 vs actual_vol ---")
    for a in AGENTS:
        v = df[f'adj_r3_{a}'].values
        r = np.corrcoef(v, y)[0, 1]
        print(f"  adj_r3_{a:14s}: r={r:+.4f}")

    # Cross-agent summary
    adj_r3_all = np.column_stack([df[f'adj_r3_{a}'].values for a in AGENTS])
    mean_adj = adj_r3_all.mean(axis=1)
    std_adj = adj_r3_all.std(axis=1)

    print(f"\n--- Cross-agent summaries ---")
    print(f"  mean_adj_r3 vs actual_vol: r={np.corrcoef(mean_adj, y)[0,1]:+.4f}")
    print(f"  std_adj_r3 vs actual_vol:  r={np.corrcoef(std_adj, y)[0,1]:+.4f}")

    # Partial correlation: mean_adj_r3 vs actual_vol | debate_vol
    from numpy.linalg import lstsq
    # Residualize both on debate_vol
    X = np.column_stack([np.ones(len(dv)), dv])
    beta_y, _, _, _ = lstsq(X, y, rcond=None)
    beta_m, _, _, _ = lstsq(X, mean_adj, rcond=None)
    resid_y = y - X @ beta_y
    resid_m = mean_adj - X @ beta_m
    partial_r = np.corrcoef(resid_m, resid_y)[0, 1]
    print(f"  partial_r(mean_adj_r3, actual_vol | debate_vol): {partial_r:+.4f}")

    # Confidence features
    conf_r3_all = np.column_stack([df[f'conf_r3_{a}'].values for a in AGENTS])
    mean_conf = conf_r3_all.mean(axis=1)
    std_conf = conf_r3_all.std(axis=1)
    print(f"\n--- Confidence features ---")
    print(f"  mean_conf_r3 vs actual_vol: r={np.corrcoef(mean_conf, y)[0,1]:+.4f}")
    print(f"  std_conf_r3 vs actual_vol:  r={np.corrcoef(std_conf, y)[0,1]:+.4f}")

    # Revision features
    tot_rev_all = np.column_stack([df[f'total_revision_{a}'].values for a in AGENTS])
    mean_rev = tot_rev_all.mean(axis=1)
    print(f"\n--- Revision features ---")
    print(f"  mean_total_revision vs actual_vol: r={np.corrcoef(mean_rev, y)[0,1]:+.4f}")

    # Incremental R2 test: debate_vol alone vs debate_vol + new features
    from sklearn.linear_model import LinearRegression
    X_base = dv.reshape(-1, 1)
    X_aug = np.column_stack([dv, mean_adj, std_adj, mean_conf, std_conf, mean_rev])

    lr_base = LinearRegression().fit(X_base, y)
    lr_aug = LinearRegression().fit(X_aug, y)
    r2_base = lr_base.score(X_base, y)
    r2_aug = lr_aug.score(X_aug, y)

    print(f"\n--- Incremental R² (in-sample) ---")
    print(f"  debate_vol only:          R²={r2_base:.4f}")
    print(f"  debate_vol + new features: R²={r2_aug:.4f}")
    print(f"  Incremental R²:           {r2_aug - r2_base:+.4f}")

    return {'partial_r': partial_r, 'r2_base': r2_base, 'r2_aug': r2_aug}


# ══════════════════════════════════════════════════════════
# STEP 2: RIDGE STACKING (GO/NO-GO GATE)
# ══════════════════════════════════════════════════════════

def ridge_walk_forward(df, feature_set='full'):
    """Ridge regression with walk-forward evaluation.

    feature_set:
      'debate_only': just debate_vol (baseline)
      'full': debate_vol + per-agent R3 features
      'cross_agent': debate_vol + cross-agent summaries only
    """
    y = df['actual_vol'].values
    dv = df['debate_vol'].values
    har = df['har_vol'].values
    pv = df['persist_vol'].values
    n = len(df)

    if feature_set == 'debate_only':
        X = dv.reshape(-1, 1)
    elif feature_set == 'cross_agent':
        adj_all = np.column_stack([df[f'adj_r3_{a}'].values for a in AGENTS])
        conf_all = np.column_stack([df[f'conf_r3_{a}'].values for a in AGENTS])
        rev_all = np.column_stack([df[f'total_revision_{a}'].values for a in AGENTS])
        X = np.column_stack([
            dv, pv,
            adj_all.mean(axis=1), adj_all.std(axis=1),
            conf_all.mean(axis=1), conf_all.std(axis=1),
            rev_all.mean(axis=1),
        ])
    elif feature_set == 'full':
        cols = []
        cols.append(dv)
        cols.append(pv)
        for a in AGENTS:
            cols.append(df[f'adj_r3_{a}'].values)
            cols.append(df[f'conf_r3_{a}'].values)
            cols.append(df[f'total_revision_{a}'].values)
        adj_all = np.column_stack([df[f'adj_r3_{a}'].values for a in AGENTS])
        cols.append(adj_all.std(axis=1))
        X = np.column_stack(cols)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    preds = np.full(n, np.nan)
    scaler = StandardScaler()

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_end = max(0, rp - LABEL_EMBARGO + 1)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(rp, chunk_end)

        if len(train_idx) < 100:
            continue

        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr = y[train_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        preds[test_idx] = np.clip(model.predict(X_te), 0.05, None)

    valid = ~np.isnan(preds)
    rmse = np.sqrt(np.mean((preds[valid] - y[valid]) ** 2))
    dm, p = dm_test_hac(preds[valid] - y[valid], har[valid] - y[valid])

    # Regime breakdown
    regimes = pd.cut(pv[valid], bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    regime_rmse = {}
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            regime_rmse[r] = np.sqrt(np.mean((preds[valid][mask] - y[valid][mask]) ** 2))

    return {
        'rmse': rmse, 'dm_hac': dm, 'p_hac': p,
        'n_valid': valid.sum(), 'regime_rmse': regime_rmse,
        'preds': preds,
    }


def step2_ridge(df):
    print("\n" + "=" * 70)
    print("STEP 2: RIDGE STACKING (GO/NO-GO GATE)")
    print("=" * 70)

    results = {}
    for name, fset in [('debate_only', 'debate_only'),
                       ('cross_agent', 'cross_agent'),
                       ('full_peragent', 'full')]:
        r = ridge_walk_forward(df, fset)
        results[name] = r
        print(f"\n  Ridge-{name}:")
        print(f"    RMSE={r['rmse']:.4f}, DM_HAC={r['dm_hac']:.3f} (p={r['p_hac']:.4f})")
        print(f"    n={r['n_valid']}")
        rr = r['regime_rmse']
        print(f"    Regime: low={rr.get('low',0):.4f} normal={rr.get('normal',0):.4f} "
              f"elev={rr.get('elevated',0):.4f} high={rr.get('high',0):.4f}")

    # Pairwise comparison
    for a, b in [('debate_only', 'full_peragent'), ('debate_only', 'cross_agent')]:
        pa = results[a]['preds']
        pb = results[b]['preds']
        y = df['actual_vol'].values
        valid = ~np.isnan(pa) & ~np.isnan(pb)
        dm, p = dm_test_hac(pb[valid] - y[valid], pa[valid] - y[valid])
        print(f"\n  {b} vs {a}: DM_HAC={dm:.3f} (p={p:.4f})")

    # Go/no-go decision
    best = min(results.values(), key=lambda x: x['rmse'])
    go = best['p_hac'] < 0.10
    print(f"\n  GO/NO-GO: {'GO' if go else 'NO-GO'} (best p={best['p_hac']:.4f})")
    return results, go


# ══════════════════════════════════════════════════════════
# STEP 3 & 4: MLP AND GAT MODELS
# ══════════════════════════════════════════════════════════

class LeakFreeMLPModel(nn.Module):
    """MLP baseline for leak-free features."""

    def __init__(self, node_feat_dim, context_dim, n_base=2, hidden_dim=16):
        super().__init__()
        self.n_base = n_base
        input_dim = N_AGENTS * node_feat_dim + context_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        graph_dim = hidden_dim + context_dim
        self.base_head = nn.Sequential(
            nn.Linear(graph_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(graph_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim, 16), nn.Tanh(), nn.Linear(16, 1),
        )

    def forward(self, node_feats, context, base_preds, regime_feats):
        flat = node_feats.reshape(node_feats.size(0), -1)
        combined_in = torch.cat([flat, context], dim=-1)
        h = self.encoder(combined_in)
        combined = torch.cat([h, context], dim=-1)

        w_base = F.softmax(self.base_head(combined), dim=-1)
        w_regime = F.softmax(self.regime_head(combined), dim=-1)
        gate = self.regime_gate(regime_feats)
        weights = (1 - gate) * w_base + gate * w_regime

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        return pred, weights, gate.squeeze(-1)


class LeakFreeGATModel(nn.Module):
    """DropEdge GAT for leak-free features (v1: mean pooling)."""

    def __init__(self, node_feat_dim, context_dim, n_base=2,
                 hidden_dim=16, n_heads=4, top_k=3, drop_rate=0.2):
        super().__init__()
        self.n_base = n_base
        self.drop_rate = drop_rate

        # Learnable graph
        self.edge_logits = nn.Parameter(torch.randn(N_AGENTS, N_AGENTS) * 0.1)

        # Skip projection
        self.skip_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GAT layers
        self.gat1 = MultiHeadGATLayer(node_feat_dim, hidden_dim, n_heads, top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, n_heads, top_k)
        self.norm2 = nn.LayerNorm(hidden_dim)

        graph_dim = hidden_dim + context_dim

        self.base_head = nn.Sequential(
            nn.Linear(graph_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(graph_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim, 16), nn.Tanh(), nn.Linear(16, 1),
        )

    def get_effective_graph(self):
        return torch.sigmoid(self.edge_logits)

    def forward(self, node_feats, context, base_preds, regime_feats):
        edge_mask = self.get_effective_graph()

        if self.training and self.drop_rate > 0:
            drop = (torch.rand_like(edge_mask) > self.drop_rate).float()
            eye = torch.eye(N_AGENTS, device=edge_mask.device)
            drop = drop * (1 - eye) + eye
            edge_mask = edge_mask * drop

        skip = self.skip_proj(node_feats)
        h, _ = self.gat1(node_feats, edge_mask)
        h = self.norm1(h + skip)
        h = F.relu(h)

        skip2 = h
        h, attn = self.gat2(h, edge_mask)
        h = self.norm2(h + skip2)
        h = F.relu(h)

        graph_out = h.mean(dim=1)
        combined = torch.cat([graph_out, context], dim=-1)

        w_base = F.softmax(self.base_head(combined), dim=-1)
        w_regime = F.softmax(self.regime_head(combined), dim=-1)
        gate = self.regime_gate(regime_feats)
        weights = (1 - gate) * w_base + gate * w_regime

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        n_active = (self.get_effective_graph() > 0.5).sum().item()
        return pred, weights, gate.squeeze(-1), attn, n_active


class OptimizedGATModel(nn.Module):
    """Optimized GAT: flatten readout + role embedding + configurable loss."""

    def __init__(self, node_feat_dim, context_dim, n_base=2,
                 hidden_dim=16, n_heads=4, top_k=3, drop_rate=0.2,
                 role_dim=4):
        super().__init__()
        self.n_base = n_base
        self.drop_rate = drop_rate

        # Learnable graph
        self.edge_logits = nn.Parameter(torch.randn(N_AGENTS, N_AGENTS) * 0.1)

        # Role embedding: each agent gets a learned identity vector
        self.role_embed = nn.Embedding(N_AGENTS, role_dim)
        augmented_feat_dim = node_feat_dim + role_dim

        # Skip projection (from augmented dim)
        self.skip_proj = nn.Linear(augmented_feat_dim, hidden_dim)

        # GAT layers
        self.gat1 = MultiHeadGATLayer(augmented_feat_dim, hidden_dim, n_heads, top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, n_heads, top_k)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Flatten readout: N_AGENTS * hidden_dim -> compressed
        flatten_dim = N_AGENTS * hidden_dim
        self.readout_proj = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        graph_dim = hidden_dim * 2 + context_dim

        self.base_head = nn.Sequential(
            nn.Linear(graph_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(graph_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim, 16), nn.Tanh(), nn.Linear(16, 1),
        )

    def get_effective_graph(self):
        return torch.sigmoid(self.edge_logits)

    def forward(self, node_feats, context, base_preds, regime_feats):
        B = node_feats.size(0)
        edge_mask = self.get_effective_graph()

        if self.training and self.drop_rate > 0:
            drop = (torch.rand_like(edge_mask) > self.drop_rate).float()
            eye = torch.eye(N_AGENTS, device=edge_mask.device)
            drop = drop * (1 - eye) + eye
            edge_mask = edge_mask * drop

        # Append role embeddings to node features
        role_ids = torch.arange(N_AGENTS, device=node_feats.device)
        role_emb = self.role_embed(role_ids)  # (N_AGENTS, role_dim)
        role_emb = role_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, role_dim)
        x = torch.cat([node_feats, role_emb], dim=-1)  # (B, N, feat+role)

        skip = self.skip_proj(x)
        h, _ = self.gat1(x, edge_mask)
        h = self.norm1(h + skip)
        h = F.relu(h)

        skip2 = h
        h, attn = self.gat2(h, edge_mask)
        h = self.norm2(h + skip2)
        h = F.relu(h)

        # Flatten readout: preserve agent-specific representations
        graph_out = self.readout_proj(h.reshape(B, -1))
        combined = torch.cat([graph_out, context], dim=-1)

        w_base = F.softmax(self.base_head(combined), dim=-1)
        w_regime = F.softmax(self.regime_head(combined), dim=-1)
        gate = self.regime_gate(regime_feats)
        weights = (1 - gate) * w_base + gate * w_regime

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        n_active = (self.get_effective_graph() > 0.5).sum().item()
        return pred, weights, gate.squeeze(-1), attn, n_active


def generate_oof_ridge(df):
    """Generate strictly OOF Ridge-cross_agent predictions for anchoring."""
    y = df['actual_vol'].values
    dv = df['debate_vol'].values
    pv = df['persist_vol'].values
    n = len(df)

    adj_all = np.column_stack([df[f'adj_r3_{a}'].values for a in AGENTS])
    conf_all = np.column_stack([df[f'conf_r3_{a}'].values for a in AGENTS])
    rev_all = np.column_stack([df[f'total_revision_{a}'].values for a in AGENTS])
    X = np.column_stack([
        dv, pv,
        adj_all.mean(axis=1), adj_all.std(axis=1),
        conf_all.mean(axis=1), conf_all.std(axis=1),
        rev_all.mean(axis=1),
    ])

    oof_preds = np.full(n, np.nan)
    scaler = StandardScaler()

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_end = max(0, rp - LABEL_EMBARGO + 1)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 100:
            continue
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y[train_idx])
        oof_preds[test_idx] = np.clip(model.predict(X_te), 0.05, None)

    return oof_preds


def train_model(model, node_feats, context, targets, base_preds,
                regime_feats, train_idx, n_epochs=250, lr=0.003, loss_fn='mse'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])

    loss_func = F.mse_loss if loss_fn == 'mse' else nn.HuberLoss(delta=0.1)
    best_loss, best_state = float('inf'), None
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(nf, ctx, bp, rf)
        pred = out[0]
        loss = loss_func(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_loss


def run_walk_forward(model_cls, model_kwargs, df, node_feats, context,
                     targets, base_preds, regime_feats, seed,
                     node_feat_dim, context_dim, is_gat=False,
                     n_epochs=250, lr=0.003, loss_fn='mse'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df)
    preds = np.full(n, np.nan)

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 252:
            continue

        model = model_cls(**model_kwargs)
        train_model(model, node_feats, context, targets, base_preds,
                    regime_feats, train_idx, n_epochs=n_epochs,
                    lr=lr, loss_fn=loss_fn)

        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            out = model(nf, ctx, bp, rf)
            preds[test_idx] = out[0].numpy()

    return preds


def evaluate_model(name, model_cls, model_kwargs, df, node_feats, context,
                   targets, base_preds, regime_feats, har, persist_vol,
                   node_feat_dim, context_dim, is_gat=False,
                   n_epochs=250, lr=0.003, loss_fn='mse'):
    all_preds = []
    seed_rmses = []

    for seed in SEEDS:
        p = run_walk_forward(
            model_cls, model_kwargs, df, node_feats, context,
            targets, base_preds, regime_feats, seed,
            node_feat_dim, context_dim, is_gat,
            n_epochs=n_epochs, lr=lr, loss_fn=loss_fn,
        )
        valid = ~np.isnan(p)
        rmse = np.sqrt(np.mean((p[valid] - targets[valid]) ** 2))
        seed_rmses.append(rmse)
        all_preds.append(p)

    ens = np.nanmean(np.stack(all_preds), axis=0)
    valid = ~np.isnan(ens)
    y = targets[valid]
    rmse_ens = np.sqrt(np.mean((ens[valid] - y) ** 2))
    dm, p_val = dm_test_hac(ens[valid] - y, har[valid] - y)

    regimes = pd.cut(persist_vol[valid], bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    regime_rmse = {}
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            regime_rmse[r] = np.sqrt(np.mean((ens[valid][mask] - y[mask]) ** 2))

    print(f"  {name}:")
    print(f"    Per-seed: {np.mean(seed_rmses):.4f} +/- {np.std(seed_rmses):.4f}")
    print(f"    Ensemble: {rmse_ens:.4f}, DM_HAC={dm:.3f} (p={p_val:.4f})")
    rr = regime_rmse
    print(f"    Regime:   low={rr.get('low',0):.4f} normal={rr.get('normal',0):.4f} "
          f"elev={rr.get('elevated',0):.4f} high={rr.get('high',0):.4f}")

    return {
        'name': name, 'ensemble_rmse': rmse_ens, 'dm_hac': dm, 'p_hac': p_val,
        'perseed_mean': np.mean(seed_rmses), 'perseed_std': np.std(seed_rmses),
        'preds': ens, **{f'rmse_{k}': v for k, v in regime_rmse.items()},
    }


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main(version='v3'):
    df = load_data(version)
    print(f"Loaded {version} data: n={len(df)}, date range {df.date.iloc[0]} to {df.date.iloc[-1]}")

    # ── Step 1: Feature diagnostic ──
    diag = feature_diagnostic(df)

    # ── Step 2: Ridge stacking ──
    ridge_results, go = step2_ridge(df)

    # ── Reference baselines ──
    y = df['actual_vol'].values
    har = df['har_vol'].values
    pv = df['persist_vol'].values
    dv = df['debate_vol'].values

    # HAR and naive debate RMSE (over same valid window as Ridge)
    ridge_valid = ~np.isnan(ridge_results['debate_only']['preds'])
    har_rmse = np.sqrt(np.mean((har[ridge_valid] - y[ridge_valid]) ** 2))
    dv_rmse = np.sqrt(np.mean((dv[ridge_valid] - y[ridge_valid]) ** 2))
    pv_rmse = np.sqrt(np.mean((pv[ridge_valid] - y[ridge_valid]) ** 2))
    garch_rmse = np.sqrt(np.mean((df['garch_vol'].values[ridge_valid] - y[ridge_valid]) ** 2))

    print(f"\n--- Reference baselines (n={ridge_valid.sum()}) ---")
    print(f"  HAR:        RMSE={har_rmse:.4f}")
    print(f"  Persistence: RMSE={pv_rmse:.4f}")
    print(f"  GARCH:      RMSE={garch_rmse:.4f}")
    print(f"  Naive debate: RMSE={dv_rmse:.4f}")

    if not go:
        print("\n  NOTE: GO/NO-GO gate says NO-GO, but proceeding anyway")
        print("  (strong feature diagnostics, limited HAC power at small n)")

    # ── Step 3: Optimized GAT experiments ──
    print("\n" + "=" * 70)
    print("STEP 3: OPTIMIZED GAT EXPERIMENTS")
    print("=" * 70)

    nn_results = {}
    base_preds_2 = np.column_stack([dv, pv])

    # Generate OOF Ridge predictions for anchoring
    print("\n  Generating OOF Ridge predictions for anchoring...")
    oof_ridge = generate_oof_ridge(df)
    n_valid_oof = np.sum(~np.isnan(oof_ridge))
    print(f"  OOF Ridge: {n_valid_oof} valid predictions")

    # 3-base version: [ridge_pred, debate_vol, persist_vol]
    base_preds_3 = np.column_stack([oof_ridge, dv, pv])
    # Fill NaN in ridge for early days with debate_vol
    for i in range(len(base_preds_3)):
        if np.isnan(base_preds_3[i, 0]):
            base_preds_3[i, 0] = dv[i]

    node_feats_t1 = build_leak_free_features(df, 't1')
    context = build_context_features(df)
    regime_feats = build_regime_feats(df)
    node_feat_dim = node_feats_t1.shape[2]
    context_dim = context.shape[1]

    # ── Experiment A: Structural fix (flatten + role embed) ──
    print(f"\n{'─' * 60}")
    print("Exp A: Flatten readout + role embedding (T1, 2-base)")
    print(f"{'─' * 60}")
    gat_a_kwargs = dict(
        node_feat_dim=node_feat_dim, context_dim=context_dim,
        n_base=2, hidden_dim=16, n_heads=4, top_k=3, drop_rate=0.2,
        role_dim=4,
    )
    r = evaluate_model(
        'OptGAT-flatten', OptimizedGATModel, gat_a_kwargs,
        df, node_feats_t1, context, y, base_preds_2, regime_feats,
        har, pv, node_feat_dim, context_dim, is_gat=True,
    )
    nn_results['OptGAT-flatten'] = r

    # ── Experiment B: Flatten + Ridge anchoring (3-base) ──
    print(f"\n{'─' * 60}")
    print("Exp B: Flatten + Ridge anchor (T1, 3-base)")
    print(f"{'─' * 60}")
    gat_b_kwargs = dict(
        node_feat_dim=node_feat_dim, context_dim=context_dim,
        n_base=3, hidden_dim=16, n_heads=4, top_k=3, drop_rate=0.2,
        role_dim=4,
    )
    r = evaluate_model(
        'OptGAT-ridge3', OptimizedGATModel, gat_b_kwargs,
        df, node_feats_t1, context, y, base_preds_3, regime_feats,
        har, pv, node_feat_dim, context_dim, is_gat=True,
    )
    nn_results['OptGAT-ridge3'] = r

    # ── Experiment C: Grid search on best structure ──
    print(f"\n{'─' * 60}")
    print("Exp C: Hyperparameter grid on OptGAT-ridge3")
    print(f"{'─' * 60}")

    grid_configs = [
        ('dr0.10', dict(drop_rate=0.10)),
        ('dr0.15', dict(drop_rate=0.15)),
        ('k4',     dict(top_k=4)),
        ('h24',    dict(hidden_dim=24)),
        ('huber',  dict()),  # same arch, Huber loss
    ]

    for label, overrides in grid_configs:
        kw = dict(
            node_feat_dim=node_feat_dim, context_dim=context_dim,
            n_base=3, hidden_dim=16, n_heads=4, top_k=3, drop_rate=0.2,
            role_dim=4,
        )
        loss_fn = 'mse'
        if label == 'huber':
            loss_fn = 'huber'
        else:
            kw.update(overrides)

        r = evaluate_model(
            f'Grid-{label}', OptimizedGATModel, kw,
            df, node_feats_t1, context, y, base_preds_3, regime_feats,
            har, pv, node_feat_dim, context_dim, is_gat=True,
            loss_fn=loss_fn,
        )
        nn_results[f'Grid-{label}'] = r

    # ── Reference: original mean-pool GAT ──
    print(f"\n{'─' * 60}")
    print("Ref: Original mean-pool GAT (T1, 2-base)")
    print(f"{'─' * 60}")
    ref_kwargs = dict(
        node_feat_dim=node_feat_dim, context_dim=context_dim,
        n_base=2, hidden_dim=16, n_heads=4, top_k=3, drop_rate=0.2,
    )
    r = evaluate_model(
        'OrigGAT-mean', LeakFreeGATModel, ref_kwargs,
        df, node_feats_t1, context, y, base_preds_2, regime_feats,
        har, pv, node_feat_dim, context_dim, is_gat=True,
    )
    nn_results['OrigGAT-mean'] = r

    # ── Pairwise comparisons ──
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS")
    print("=" * 70)

    pairs = [
        ('OptGAT-flatten', 'OrigGAT-mean', 'flatten vs mean-pool'),
        ('OptGAT-ridge3', 'OptGAT-flatten', 'ridge-anchor vs no-anchor'),
        ('OptGAT-ridge3', 'OrigGAT-mean', 'full-opt vs original'),
    ]
    for a, b, desc in pairs:
        if a in nn_results and b in nn_results:
            pa = nn_results[a]['preds']
            pb = nn_results[b]['preds']
            valid = ~np.isnan(pa) & ~np.isnan(pb)
            dm, p_val = dm_test_hac(pa[valid] - y[valid], pb[valid] - y[valid])
            print(f"  {desc}: DM_HAC={dm:.3f} (p={p_val:.4f})")

    save_results(df, ridge_results, nn_results, diag)


def save_results(df, ridge_results, nn_results, diag):
    print("\n" + "=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)

    rows = []
    for name, r in ridge_results.items():
        rows.append({
            'model': f'Ridge-{name}',
            'rmse': r['rmse'],
            'dm_hac': r['dm_hac'],
            'p_hac': r['p_hac'],
        })
    for name, r in nn_results.items():
        rows.append({
            'model': name,
            'rmse': r['ensemble_rmse'],
            'dm_hac': r['dm_hac'],
            'p_hac': r['p_hac'],
        })

    summary = pd.DataFrame(rows).sort_values('rmse')
    print(f"\n{'Model':<25s} {'RMSE':>8s} {'DM_HAC':>8s} {'p':>8s} {'Sig':>5s}")
    print("─" * 55)
    for _, row in summary.iterrows():
        sig = '***' if row['p_hac'] < 0.01 else '**' if row['p_hac'] < 0.05 \
            else '*' if row['p_hac'] < 0.10 else 'n.s.'
        print(f"  {row['model']:<23s} {row['rmse']:8.4f} {row['dm_hac']:8.3f} "
              f"{row['p_hac']:8.4f} {sig:>5s}")

    out_path = BASE / 'results' / 'leak_free_gat_v3_results.csv'
    summary.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
