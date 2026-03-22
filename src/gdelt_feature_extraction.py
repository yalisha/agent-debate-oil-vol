"""
Extract daily geopolitical risk features from GDELT event data.
Focus on oil-producing countries and conflict/cooperation dynamics.
"""

import zipfile
import os
import pandas as pd
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

GDELT_DIR = "/Users/mac/computerscience/第三方资料/gdelt/gdelt_events"
OUTPUT_PATH = "/Users/mac/computerscience/17Agent可解释预测/data/gdelt_daily_features.csv"

# GDELT 1.0 column names (58 columns)
GDELT_COLS = [
    'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate',
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode',
    'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode', 'Actor2EthnicCode',
    'Actor2Religion1Code', 'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass',
    'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone',
    'Actor1Geo_Type', 'Actor1Geo_FullName', 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
    'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
    'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code',
    'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
    'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code', 'ActionGeo_ADM2Code',
    'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID',
]

# Major oil-producing / geopolitically relevant countries
OIL_COUNTRIES = {
    'SAU', 'IRN', 'IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'BHR',  # Gulf
    'LBY', 'NGA', 'AGO', 'DZA',  # Africa
    'VEN', 'ECU', 'COL', 'BRA', 'MEX',  # Americas
    'RUS', 'KAZ', 'NOR', 'USA', 'CAN',  # Others
}

# CAMEO QuadClass: 1=Verbal Coop, 2=Material Coop, 3=Verbal Conflict, 4=Material Conflict
# EventRootCode 14=Protest, 17=Coerce, 18=Assault, 19=Fight, 20=Mass violence
CONFLICT_CODES = {'14', '17', '18', '19', '20'}


def process_single_file(filepath):
    """Extract features from a single GDELT daily zip file."""
    try:
        date_str = os.path.basename(filepath).split('.')[0]  # e.g. '20130401'

        with zipfile.ZipFile(filepath) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, sep='\t', header=None, dtype=str,
                                 on_bad_lines='skip')

        # Select and rename columns by position index
        # Columns: 7=Actor1CountryCode, 17=Actor2CountryCode,
        # 28=EventRootCode, 29=QuadClass, 30=GoldsteinScale,
        # 31=NumMentions, 32=NumSources, 33=NumArticles, 34=AvgTone
        col_map = {7: 'Actor1CountryCode', 17: 'Actor2CountryCode',
                   28: 'EventRootCode', 29: 'QuadClass',
                   30: 'GoldsteinScale', 31: 'NumMentions',
                   32: 'NumSources', 33: 'NumArticles', 34: 'AvgTone'}
        available = {k: v for k, v in col_map.items() if k < df.shape[1]}
        df = df[list(available.keys())].rename(columns=available)

        # Convert numeric columns
        for col in ['GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['QuadClass'] = pd.to_numeric(df['QuadClass'], errors='coerce')
        df['EventRootCode'] = df['EventRootCode'].astype(str).str.strip()

        # Filter: events involving oil-producing countries
        oil_mask = (
            df['Actor1CountryCode'].isin(OIL_COUNTRIES) |
            df['Actor2CountryCode'].isin(OIL_COUNTRIES)
        )
        oil_df = df[oil_mask]

        n_total = len(df)
        n_oil = len(oil_df)

        if n_oil == 0:
            return {
                'date': date_str,
                'n_events_total': n_total,
                'n_events_oil': 0,
            }

        # Global features
        global_goldstein_mean = df['GoldsteinScale'].mean()
        global_tone_mean = df['AvgTone'].mean()

        # Oil-country features
        oil_goldstein_mean = oil_df['GoldsteinScale'].mean()
        oil_goldstein_std = oil_df['GoldsteinScale'].std()
        oil_tone_mean = oil_df['AvgTone'].mean()
        oil_tone_std = oil_df['AvgTone'].std()
        oil_mentions_sum = oil_df['NumMentions'].sum()
        oil_articles_sum = oil_df['NumArticles'].sum()

        # QuadClass distribution for oil events
        qc = oil_df['QuadClass'].value_counts()
        n_verbal_coop = qc.get(1, 0)
        n_material_coop = qc.get(2, 0)
        n_verbal_conflict = qc.get(3, 0)
        n_material_conflict = qc.get(4, 0)

        # Conflict intensity
        conflict_events = oil_df[oil_df['EventRootCode'].isin(CONFLICT_CODES)]
        n_conflict = len(conflict_events)
        conflict_goldstein = conflict_events['GoldsteinScale'].mean() if n_conflict > 0 else 0

        # Geopolitical risk index: conflict share weighted by negative Goldstein
        conflict_share = (n_verbal_conflict + n_material_conflict) / n_oil if n_oil > 0 else 0
        material_conflict_share = n_material_conflict / n_oil if n_oil > 0 else 0

        # Net cooperation score
        net_coop = ((n_verbal_coop + n_material_coop) - (n_verbal_conflict + n_material_conflict)) / n_oil

        return {
            'date': date_str,
            'n_events_total': n_total,
            'n_events_oil': n_oil,
            'oil_goldstein_mean': oil_goldstein_mean,
            'oil_goldstein_std': oil_goldstein_std,
            'oil_tone_mean': oil_tone_mean,
            'oil_tone_std': oil_tone_std,
            'oil_mentions_sum': oil_mentions_sum,
            'oil_articles_sum': oil_articles_sum,
            'oil_verbal_coop': n_verbal_coop,
            'oil_material_coop': n_material_coop,
            'oil_verbal_conflict': n_verbal_conflict,
            'oil_material_conflict': n_material_conflict,
            'oil_n_conflict_events': n_conflict,
            'oil_conflict_goldstein': conflict_goldstein,
            'oil_conflict_share': conflict_share,
            'oil_material_conflict_share': material_conflict_share,
            'oil_net_coop': net_coop,
            'global_goldstein_mean': global_goldstein_mean,
            'global_tone_mean': global_tone_mean,
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def main():
    files = sorted(glob(os.path.join(GDELT_DIR, "*.zip")))
    print(f"Found {len(files)} GDELT files to process")

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_file, f): f for f in files}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  Processed {done}/{len(files)}")
            result = future.result()
            if result is not None:
                results.append(result)

    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")


if __name__ == '__main__':
    main()
