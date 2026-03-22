"""
Construct natural language event descriptions from GDELT structured event data.
These descriptions serve as LLM input for information extraction at each prediction point.

GDELT CAMEO event codes are translated into human-readable descriptions
for oil-producing country events, enabling the LLM to synthesize
geopolitical context for forecasting.
"""

import zipfile
import os
import pandas as pd
import numpy as np
from glob import glob

GDELT_DIR = "/Users/mac/computerscience/第三方资料/gdelt/gdelt_events"
OUTPUT_DIR = "/Users/mac/computerscience/17Agent可解释预测/data/gdelt_summaries"

# CAMEO QuadClass descriptions
QUAD_CLASS = {
    1: 'verbal cooperation',
    2: 'material cooperation',
    3: 'verbal conflict',
    4: 'material conflict',
}

# CAMEO root event code descriptions (top-level)
EVENT_ROOT_DESC = {
    '1': 'made a public statement',
    '2': 'expressed intent or appeal',
    '3': 'expressed intent to cooperate',
    '4': 'engaged in consultation or negotiation',
    '5': 'engaged in diplomatic cooperation',
    '6': 'provided material cooperation or aid',
    '7': 'provided economic or military assistance',
    '8': 'made a political concession or agreement',
    '9': 'investigated or inspected',
    '10': 'demanded or coerced',
    '11': 'rejected, disapproved, or protested',
    '12': 'issued threats',
    '13': 'engaged in protest or demonstration',
    '14': 'escalated with political actions',
    '15': 'imposed sanctions or restrictions',
    '16': 'used coercive force',
    '17': 'reduced or severed relations',
    '18': 'used military force',
    '19': 'engaged in armed conflict',
    '20': 'committed acts of mass violence',
}

# Oil-producing countries full names
COUNTRY_NAMES = {
    'SAU': 'Saudi Arabia', 'IRN': 'Iran', 'IRQ': 'Iraq',
    'KWT': 'Kuwait', 'ARE': 'UAE', 'QAT': 'Qatar',
    'OMN': 'Oman', 'BHR': 'Bahrain', 'LBY': 'Libya',
    'NGA': 'Nigeria', 'AGO': 'Angola', 'DZA': 'Algeria',
    'VEN': 'Venezuela', 'ECU': 'Ecuador', 'COL': 'Colombia',
    'BRA': 'Brazil', 'MEX': 'Mexico', 'RUS': 'Russia',
    'KAZ': 'Kazakhstan', 'NOR': 'Norway', 'USA': 'United States',
    'CAN': 'Canada', 'CHN': 'China', 'GBR': 'United Kingdom',
    'ISR': 'Israel', 'TUR': 'Turkey', 'SYR': 'Syria',
    'YEM': 'Yemen', 'EGY': 'Egypt', 'JPN': 'Japan',
    'DEU': 'Germany', 'FRA': 'France', 'IND': 'India',
}

OIL_COUNTRIES = {
    'SAU', 'IRN', 'IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'BHR',
    'LBY', 'NGA', 'AGO', 'DZA', 'VEN', 'ECU', 'COL', 'BRA',
    'MEX', 'RUS', 'KAZ', 'NOR', 'USA', 'CAN',
}


def describe_event(row):
    """Convert a single GDELT event row into a natural language description."""
    actor1 = COUNTRY_NAMES.get(row.get('Actor1CountryCode', ''), row.get('Actor1CountryCode', 'Unknown'))
    actor2 = COUNTRY_NAMES.get(row.get('Actor2CountryCode', ''), row.get('Actor2CountryCode', 'Unknown'))
    root_code = str(row.get('EventRootCode', '')).strip()
    action = EVENT_ROOT_DESC.get(root_code, f'took action (code {root_code})')

    goldstein = row.get('GoldsteinScale', 0)
    tone = row.get('AvgTone', 0)
    mentions = row.get('NumMentions', 0)

    severity = ''
    if isinstance(goldstein, (int, float)) and not np.isnan(goldstein):
        if goldstein <= -7:
            severity = ' (extreme hostility)'
        elif goldstein <= -3:
            severity = ' (significant tension)'
        elif goldstein >= 7:
            severity = ' (strong cooperation)'
        elif goldstein >= 3:
            severity = ' (positive engagement)'

    desc = f"{actor1} {action} toward {actor2}{severity}"

    if isinstance(mentions, (int, float)) and not np.isnan(mentions) and mentions > 50:
        desc += f" [high media attention: {int(mentions)} mentions]"

    return desc


def summarize_day(date_str, filepath):
    """
    Generate a structured summary of oil-relevant GDELT events for one day.
    Returns a dict with the date and a structured text summary.
    """
    try:
        with zipfile.ZipFile(filepath) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, sep='\t', header=None, dtype=str,
                                 on_bad_lines='skip')
    except Exception:
        return None

    col_map = {
        5: 'Actor1Code', 7: 'Actor1CountryCode',
        15: 'Actor2Code', 17: 'Actor2CountryCode',
        28: 'EventRootCode', 29: 'QuadClass',
        30: 'GoldsteinScale', 31: 'NumMentions',
        32: 'NumSources', 33: 'NumArticles', 34: 'AvgTone',
    }
    available = {k: v for k, v in col_map.items() if k < df.shape[1]}
    df = df[list(available.keys())].rename(columns=available)

    for col in ['GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['QuadClass'] = pd.to_numeric(df.get('QuadClass'), errors='coerce')

    # Filter oil-country events
    oil_mask = (
        df['Actor1CountryCode'].isin(OIL_COUNTRIES) |
        df['Actor2CountryCode'].isin(OIL_COUNTRIES)
    )
    oil_df = df[oil_mask].copy()

    if len(oil_df) == 0:
        return {'date': date_str, 'n_events': 0, 'summary': 'No oil-relevant events.'}

    # Sort by abs(GoldsteinScale) * NumMentions to get most impactful events
    oil_df['impact'] = oil_df['GoldsteinScale'].abs() * np.log1p(oil_df['NumMentions'].fillna(0))
    oil_df = oil_df.sort_values('impact', ascending=False)

    # Take top 15 most impactful events
    top_events = oil_df.head(15)
    descriptions = [describe_event(row) for _, row in top_events.iterrows()]

    # Aggregate stats
    n_conflict = len(oil_df[oil_df['QuadClass'].isin([3, 4])])
    n_coop = len(oil_df[oil_df['QuadClass'].isin([1, 2])])
    avg_goldstein = oil_df['GoldsteinScale'].mean()

    header = (
        f"Date: {date_str} | "
        f"Oil-country events: {len(oil_df)} | "
        f"Conflict: {n_conflict} | Cooperation: {n_coop} | "
        f"Avg Goldstein: {avg_goldstein:.2f}"
    )

    summary = header + "\nKey events:\n" + "\n".join(f"- {d}" for d in descriptions)

    return {
        'date': date_str,
        'n_events': len(oil_df),
        'n_conflict': n_conflict,
        'n_coop': n_coop,
        'avg_goldstein': avg_goldstein,
        'summary': summary,
    }


def build_daily_summaries(start_date='2013-04-01', end_date='2026-03-12'):
    """Build daily event summaries for the full sample period."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob(os.path.join(GDELT_DIR, "*.zip")))
    print(f"Processing {len(files)} GDELT files for event descriptions...")

    results = []
    for i, filepath in enumerate(files):
        date_str = os.path.basename(filepath).split('.')[0]
        result = summarize_day(date_str, filepath)
        if result:
            results.append(result)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}")

    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)

    # Save summaries
    df.to_parquet(f"{OUTPUT_DIR}/gdelt_daily_summaries.parquet", index=False)

    # Also save just the text summaries as a lookup dict
    summaries = dict(zip(df['date'].dt.strftime('%Y-%m-%d'), df['summary']))

    print(f"Saved {len(df)} daily summaries")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Show example
    print("\n=== Example summary (first date) ===")
    print(df.iloc[0]['summary'])

    return df


if __name__ == '__main__':
    build_daily_summaries()
