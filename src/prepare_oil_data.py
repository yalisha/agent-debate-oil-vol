"""
Extract and prepare WTI oil price series with macro features from the gold prediction CSV.
Output: clean daily oil price dataset with returns and volatility measures.
"""

import pandas as pd
import numpy as np

INPUT_PATH = "/Users/mac/computerscience/17Agent可解释预测/data/新gold_prediction_features.csv"
OUTPUT_PATH = "/Users/mac/computerscience/17Agent可解释预测/data/oil_macro_daily.csv"


def main():
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Select oil-relevant columns
    cols = ['date', 'WTI_OIL', 'VIX', 'DXY', 'YIELD_2Y', 'YIELD_5Y',
            'YIELD_10Y', 'FED_FUNDS', 'REAL_YIELD_10Y',
            'YIELD_SPREAD_10Y_2Y', 'SHANGHAI_COMPOSITE']

    oil = df[cols].copy()
    oil = oil.rename(columns={'WTI_OIL': 'wti_price'})

    # Drop rows where WTI is missing
    oil = oil.dropna(subset=['wti_price']).reset_index(drop=True)

    # Compute returns
    oil['wti_return'] = oil['wti_price'].pct_change()
    # Log return: use log(price/prev) where both are positive;
    # for negative oil price days (Apr 2020), fall back to simple return
    ratio = oil['wti_price'] / oil['wti_price'].shift(1)
    oil['wti_log_return'] = np.where(
        (ratio > 0) & ratio.notna(),
        np.log(ratio),
        oil['wti_return']  # fallback: simple return for negative price transitions
    )

    # Rolling volatility (backward-looking, for agent context)
    oil['wti_vol_5d'] = oil['wti_log_return'].rolling(5).std() * np.sqrt(252)
    oil['wti_vol_20d'] = oil['wti_log_return'].rolling(20).std() * np.sqrt(252)
    oil['wti_vol_60d'] = oil['wti_log_return'].rolling(60).std() * np.sqrt(252)

    # Forward-looking realized volatility (prediction targets)
    # fwd_rv_Xd_t = std(log_returns[t+1 : t+X+1]) * sqrt(252)
    for w in [5, 20]:
        oil[f'fwd_rv_{w}d'] = (oil['wti_log_return'].shift(-1)
                                .rolling(w).std()
                                .shift(-(w - 1)) * np.sqrt(252))

    # Price momentum
    oil['wti_mom_5d'] = oil['wti_price'].pct_change(5)
    oil['wti_mom_20d'] = oil['wti_price'].pct_change(20)

    # DXY return
    oil['dxy_return'] = oil['DXY'].pct_change()

    # VIX change
    oil['vix_change'] = oil['VIX'].diff()

    # Term spread change
    oil['spread_change'] = oil['YIELD_SPREAD_10Y_2Y'].diff()

    oil.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(oil)} rows to {OUTPUT_PATH}")
    print(f"Date range: {oil['date'].min()} ~ {oil['date'].max()}")
    print(f"WTI price range: {oil['wti_price'].min():.2f} ~ {oil['wti_price'].max():.2f}")
    print(f"\nMissing values:")
    print(oil.isnull().sum()[oil.isnull().sum() > 0])


if __name__ == '__main__':
    main()
