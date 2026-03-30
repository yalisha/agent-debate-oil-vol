"""
Causal Graph Attention Network for Multi-Agent Debate Aggregation.

Upgrades the naive confidence-weighted mean aggregation with:
1. Causal Discovery: PC algorithm on agent Shapley time series → causal adjacency
2. Graph Attention Network: learns context-dependent aggregation on the causal graph
3. Dynamic anchor blending: learns to mix persistence and HAR anchors

Walk-forward evaluation to avoid look-ahead bias.
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_end, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)


# ══════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════

def load_data():
    """Load debate results and attribution data."""
    df = pd.read_csv(BASE / "results/debate_eval_full_20260320_2343.csv")
    with open(BASE / "results/debate_attribution_full_20260320_2343.json") as f:
        attrib = json.load(f)

    # Add graph degree from attribution JSON
    for i, rec in enumerate(attrib):
        if i < len(df):
            for agent in AGENTS:
                df.loc[i, f'degree_{agent}'] = rec.get('graph_degree', {}).get(agent, 0)

    # Encode behaviors
    behavior_map = {'herding': 0, 'anchored': 1, 'independent': 2, 'overconfident': 3}
    for agent in AGENTS:
        col = f'behavior_{agent}'
        df[f'beh_enc_{agent}'] = df[col].map(behavior_map).fillna(1).astype(int)

    # Derived features
    df['total_adj'] = df['debate_vol'] - df['persist_vol']
    df['n_herding'] = sum((df[f'behavior_{agent}'] == 'herding').astype(int) for agent in AGENTS)
    df['vol_regime'] = pd.cut(df['persist_vol'],
                              bins=[0, 0.20, 0.35, 0.55, 100],
                              labels=[0, 1, 2, 3]).astype(int)
    return df


# ══════════════════════════════════════════════════════════
# 2. CAUSAL DISCOVERY
# ══════════════════════════════════════════════════════════

def granger_causality_matrix(df, max_lag=5, alpha=0.005):
    """Compute pairwise Granger causality between agent Shapley time series.
    Returns adjacency matrix where A[i,j]=1 means agent i Granger-causes agent j.
    """
    from scipy import stats

    shapley_cols = [f'shapley_{a}' for a in AGENTS]
    data = df[shapley_cols].values
    n = len(data)
    adj = np.zeros((N_AGENTS, N_AGENTS))

    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if i == j:
                continue
            # Restricted model: y_j ~ lags of y_j
            # Unrestricted model: y_j ~ lags of y_j + lags of y_i
            y = data[max_lag:, j]
            X_r = np.column_stack([data[max_lag - k - 1:n - k - 1, j] for k in range(max_lag)])
            X_u = np.column_stack([X_r] +
                                  [data[max_lag - k - 1:n - k - 1, i] for k in range(max_lag)])

            # OLS
            n_obs = len(y)
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
            rss_r = np.sum((y - X_r @ beta_r) ** 2)
            rss_u = np.sum((y - X_u @ beta_u) ** 2)

            # F-test
            df1 = max_lag
            df2 = n_obs - 2 * max_lag
            if df2 <= 0 or rss_u <= 0:
                continue
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_val = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_val < alpha:
                adj[i, j] = 1.0

    return adj


def build_causal_graph(df_train):
    """Build causal adjacency matrix from training data."""
    adj = granger_causality_matrix(df_train)
    # Add self-loops
    adj = adj + np.eye(N_AGENTS)
    # Print causal edges
    edges = []
    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if adj[i, j] > 0 and i != j:
                edges.append(f"{AGENTS[i]} -> {AGENTS[j]}")
    print(f"  Causal edges ({len(edges)}): {', '.join(edges[:10])}")
    return adj


# ══════════════════════════════════════════════════════════
# 3. GRAPH ATTENTION NETWORK
# ══════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        """x: (batch, n_nodes, in_dim), adj: (n_nodes, n_nodes)"""
        h = self.W(x)  # (batch, n_nodes, out_dim)
        B, N, D = h.shape

        # Compute attention coefficients
        h_i = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        e = self.leaky(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (B, N, N)

        # Mask by adjacency
        mask = torch.tensor(adj, dtype=torch.float32, device=x.device).unsqueeze(0)
        e = e.masked_fill(mask == 0, float('-inf'))
        alpha = torch.softmax(e, dim=-1)  # (B, N, N)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        out = torch.bmm(alpha, h)  # (B, N, out_dim)
        return out, alpha


class CausalGATAggregator(nn.Module):
    """
    Causal GAT-based meta-learner.

    GNN processes agent interaction graph → graph-level embedding →
    combined with base predictions → MLP outputs combination weights
    for 5 base forecasts (debate, HAR, persistence, single, GARCH).

    Final prediction = softmax_weights . [debate, HAR, persist, single, GARCH]
    """
    N_BASE = 5  # number of base forecasts

    def __init__(self, node_feat_dim, context_dim, hidden_dim=16):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_feat_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)

        # Graph readout → weight prediction head
        self.weight_head = nn.Sequential(
            nn.Linear(N_AGENTS * hidden_dim + context_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.N_BASE),  # one weight per base forecast
        )

        # Residual correction head (small learned adjustment)
        self.residual_head = nn.Sequential(
            nn.Linear(N_AGENTS * hidden_dim + context_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, node_feats, adj, context, base_preds):
        """
        node_feats: (batch, 7, node_feat_dim)
        adj: (7, 7) adjacency matrix
        context: (batch, context_dim)
        base_preds: (batch, 5) [debate, har, persist, single, garch]

        Returns: final prediction, weights, attention
        """
        h, attn1 = self.gat1(node_feats, adj)
        h = torch.relu(h)
        h, attn2 = self.gat2(h, adj)
        h = torch.relu(h)

        graph_embed = h.reshape(h.size(0), -1)
        combined = torch.cat([graph_embed, context], dim=-1)

        # Softmax weights over base forecasts
        raw_weights = self.weight_head(combined)
        weights = torch.softmax(raw_weights, dim=-1)  # (batch, 5)

        # Weighted combination
        weighted_pred = (weights * base_preds).sum(dim=-1)  # (batch,)

        # Small residual correction
        residual = self.residual_head(combined).squeeze(-1) * 0.05  # scaled small

        pred = weighted_pred + residual
        pred = torch.clamp(pred, min=0.05)

        return pred, weights, attn2


# ══════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def build_features(df):
    """Build node features and context features for each day."""
    # Compute rolling agent features
    for agent in AGENTS:
        scol = f'shapley_{agent}'
        df[f'shapley_ma5_{agent}'] = df[scol].rolling(5, min_periods=1).mean()
        df[f'shapley_std5_{agent}'] = df[scol].rolling(5, min_periods=1).std().fillna(0)
        # Herding streak: rolling sum of herding indicator over 5 days
        herd_ind = (df[f'behavior_{agent}'] == 'herding').astype(float)
        df[f'herd_streak_{agent}'] = herd_ind.rolling(5, min_periods=1).sum()

    # Node features per agent (7 nodes):
    #   [shapley, myerson, behavior_encoded, degree,
    #    shapley_ma5, shapley_std5, herd_streak]
    node_feat_dim = 7

    node_feats = np.zeros((len(df), N_AGENTS, node_feat_dim))
    for i, agent in enumerate(AGENTS):
        node_feats[:, i, 0] = df[f'shapley_{agent}'].values
        node_feats[:, i, 1] = df[f'myerson_{agent}'].values
        node_feats[:, i, 2] = df[f'beh_enc_{agent}'].values / 3.0
        node_feats[:, i, 3] = df.get(f'degree_{agent}', pd.Series(0, index=df.index)).values / 6.0
        node_feats[:, i, 4] = df[f'shapley_ma5_{agent}'].values
        node_feats[:, i, 5] = df[f'shapley_std5_{agent}'].values
        node_feats[:, i, 6] = df[f'herd_streak_{agent}'].values / 5.0

    # Context features
    df['vol_change_5d'] = df['persist_vol'].diff(5).fillna(0)
    df['debate_har_gap'] = df['debate_vol'] - df['har_vol']
    context = np.column_stack([
        df['persist_vol'].values,
        df['har_vol'].values,
        df['total_adj'].values,
        df['n_herding'].values / 7.0,
        df['vol_regime'].values / 3.0,
        df['debate_vol'].values,
        (df['persist_vol'] - df['har_vol']).values,
        df['vol_change_5d'].values,
        df['debate_har_gap'].values,
    ])
    context_dim = context.shape[1]

    return node_feats, context, node_feat_dim, context_dim


# ══════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════

def train_model(model, adj, node_feats, context, targets,
                base_preds, train_idx, n_epochs=200, lr=0.003):
    """Train the GAT model on training data with cosine LR schedule."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf_train = torch.FloatTensor(node_feats[train_idx])
    ctx_train = torch.FloatTensor(context[train_idx])
    y_train = torch.FloatTensor(targets[train_idx])
    bp_train = torch.FloatTensor(base_preds[train_idx])

    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, weights, _ = model(nf_train, adj, ctx_train, bp_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def predict(model, adj, node_feats, context, base_preds, idx):
    """Predict on given indices."""
    model.eval()
    with torch.no_grad():
        nf = torch.FloatTensor(node_feats[idx])
        ctx = torch.FloatTensor(context[idx])
        bp = torch.FloatTensor(base_preds[idx])
        pred, weights, attn = model(nf, adj, ctx, bp)
    return pred.numpy(), weights.numpy()


# ══════════════════════════════════════════════════════════
# 6. WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    df = load_data()

    # Clean sample
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: {n} days")

    # Build features
    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
    targets = df_clean['actual_vol'].values
    persist = df_clean['persist_vol'].values
    har = df_clean['har_vol'].values
    debate_vol = df_clean['debate_vol'].values
    single_vol = df_clean['single_vol'].values
    garch_vol = df_clean['garch_vol'].values
    total_adj_vals = df_clean['total_adj'].values

    # Base predictions matrix: [debate, HAR, persist, single, GARCH]
    base_preds = np.column_stack([debate_vol, har, persist, single_vol, garch_vol])
    base_names = ['Debate', 'HAR', 'Persist', 'Single', 'GARCH']

    # Walk-forward parameters
    min_train = 252
    retrain_every = 63
    n_epochs = 200

    # Storage
    preds_gat = np.full(n, np.nan)
    weight_history = np.full((n, 5), np.nan)

    # Simple baselines
    preds_simple_blend = np.full(n, np.nan)

    eval_start = min_train
    retrain_points = list(range(eval_start, n, retrain_every))

    print(f"\nWalk-forward evaluation: {len(retrain_points)} retrain windows")
    print(f"Label embargo: {LABEL_EMBARGO} trading days")
    print(f"Eval period: day {eval_start} to {n}")

    for ri, rp in enumerate(retrain_points):
        chunk_end = min(rp + retrain_every, n)
        train_end = embargoed_train_end(rp)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)

        if len(train_idx) < min_train:
            continue

        print(f"\n  Retrain {ri+1}/{len(retrain_points)}: "
              f"train=0:{train_end}, test={rp}:{chunk_end}, "
              f"date={df_clean.iloc[rp].get('date', '?')}")

        causal_adj = build_causal_graph(df_clean.iloc[:train_end])

        model = CausalGATAggregator(node_feat_dim, context_dim, hidden_dim=16)
        final_loss = train_model(
            model, causal_adj, node_feats, context, targets,
            base_preds, train_idx, n_epochs=n_epochs
        )
        print(f"  Train loss: {final_loss:.6f}")

        pred, weights = predict(
            model, causal_adj, node_feats, context, base_preds, test_idx
        )
        preds_gat[test_idx] = pred
        weight_history[test_idx] = weights

        # Simple blend baseline
        from scipy.optimize import minimize_scalar
        def blend_rmse(a):
            p = a * har[train_idx] + (1-a) * persist[train_idx] + total_adj_vals[train_idx]
            return np.sqrt(np.mean((p - targets[train_idx])**2))
        res = minimize_scalar(blend_rmse, bounds=(0, 1), method='bounded')
        opt_a = res.x
        preds_simple_blend[test_idx] = (opt_a * har[test_idx] +
                                         (1 - opt_a) * persist[test_idx] +
                                         total_adj_vals[test_idx])

    # ══════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════
    valid = ~np.isnan(preds_gat)
    n_eval = valid.sum()

    print(f"\n{'=' * 70}")
    print(f"RESULTS (n={n_eval} eval days)")
    print(f"{'=' * 70}")

    methods = {
        'Causal-GAT': preds_gat[valid],
        'Simple Blend': preds_simple_blend[valid],
        'Debate (original)': debate_vol[valid],
        'HAR': har[valid],
        'Persistence': persist[valid],
    }

    y = targets[valid]
    for name, pred in methods.items():
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        mae = np.mean(np.abs(pred - y))
        print(f"  {name:20s}: RMSE={rmse:.4f}  MAE={mae:.4f}")

    # DM tests vs HAR
    print(f"\n--- Diebold-Mariano vs HAR ---")
    e_har = (har[valid] - y) ** 2
    for name, pred in methods.items():
        if name == 'HAR':
            continue
        e_m = (pred - y) ** 2
        diff = e_m - e_har
        dm = diff.mean() / (diff.std() / np.sqrt(len(diff)))
        sig = '*' if abs(dm) > 1.96 else ''
        print(f"  {name:20s}: DM={dm:.3f} {sig}")

    # Regime breakdown
    print(f"\n--- By Regime ---")
    regimes = df_clean['vol_regime'].values[valid]
    regime_names = {0: 'low', 1: 'normal', 2: 'elevated', 3: 'high'}
    for rv, rname in regime_names.items():
        mask = regimes == rv
        if mask.sum() > 0:
            rmse_gat = np.sqrt(np.mean((preds_gat[valid][mask] - y[mask]) ** 2))
            rmse_har = np.sqrt(np.mean((har[valid][mask] - y[mask]) ** 2))
            rmse_deb = np.sqrt(np.mean((debate_vol[valid][mask] - y[mask]) ** 2))
            print(f"  {rname:10s}: GAT={rmse_gat:.4f}  HAR={rmse_har:.4f}  "
                  f"Debate={rmse_deb:.4f}  n={mask.sum()}")

    # Weight analysis
    valid_weights = weight_history[valid]
    print(f"\n--- Base Forecast Weights (mean) ---")
    for i, name in enumerate(base_names):
        print(f"  {name:10s}: {valid_weights[:, i].mean():.3f} ± {valid_weights[:, i].std():.3f}")

    print(f"\n--- Weights by Regime ---")
    for rv, rname in regime_names.items():
        mask = regimes == rv
        if mask.sum() > 0:
            w = valid_weights[mask].mean(axis=0)
            parts = ' '.join(f'{base_names[i]}={w[i]:.3f}' for i in range(5))
            print(f"  {rname:10s}: {parts}")

    # Save results
    out = df_clean[valid].copy()
    out['gat_pred'] = preds_gat[valid]
    for i, name in enumerate(base_names):
        out[f'w_{name.lower()}'] = valid_weights[:, i]
    out_path = BASE / "results" / "causal_gat_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
