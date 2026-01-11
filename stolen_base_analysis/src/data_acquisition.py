"""
Data Acquisition for Stolen Base Decision Analysis

Pulls statcast data from 2015-2024 and processes into steal opportunities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pybaseball import statcast, cache
import re
import warnings

warnings.filterwarnings('ignore')

# Enable caching to avoid repeated downloads
cache.enable()


def pull_season_data(year: int) -> pd.DataFrame:
    """Pull full season statcast data for a given year."""
    print(f"Pulling {year} data...")

    # Regular season roughly March 28 - October 5
    # Postseason through early November
    try:
        data = statcast(start_dt=f'{year}-03-20', end_dt=f'{year}-11-10')
        data['season'] = year
        return data
    except Exception as e:
        print(f"Error pulling {year}: {e}")
        return pd.DataFrame()


def pull_all_seasons(start_year: int = 2015, end_year: int = 2024) -> pd.DataFrame:
    """Pull statcast data for all seasons."""
    all_data = []

    for year in range(start_year, end_year + 1):
        season_data = pull_season_data(year)
        if len(season_data) > 0:
            all_data.append(season_data)
            print(f"  {year}: {len(season_data):,} pitches")

    if not all_data:
        raise ValueError("No data retrieved")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined):,} pitches across {end_year - start_year + 1} seasons")

    return combined


def identify_steal_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns identifying steal attempts and outcomes.

    Steals are identified from the 'des' (description) column which contains
    text like "Christian Yelich steals (1) 2nd base" or "Joey Wendle caught stealing 2nd".
    """
    df = df.copy()

    # Initialize columns
    df['steal_attempt'] = False
    df['steal_success'] = False
    df['caught_stealing'] = False
    df['steal_base'] = None

    # Pattern for successful steals: "Player steals (N) 2nd/3rd base"
    # or "Player steals 2nd/3rd"
    steal_success_pattern = r'steals\s+(?:\(\d+\)\s+)?(2nd|3rd|home)'

    # Pattern for caught stealing: "caught stealing 2nd/3rd/home"
    caught_stealing_pattern = r'caught stealing (2nd|3rd|home)'

    # Apply patterns
    des_col = df['des'].fillna('')

    # Successful steals
    success_mask = des_col.str.contains(steal_success_pattern, case=False, regex=True)
    df.loc[success_mask, 'steal_success'] = True
    df.loc[success_mask, 'steal_attempt'] = True

    # Extract base stolen for successes
    def extract_steal_base(des):
        match = re.search(steal_success_pattern, des, re.IGNORECASE)
        if match:
            base = match.group(1).upper()
            if base == '2ND':
                return '2B'
            elif base == '3RD':
                return '3B'
            elif base == 'HOME':
                return 'HOME'
        return None

    df.loc[success_mask, 'steal_base'] = des_col[success_mask].apply(extract_steal_base)

    # Caught stealing
    cs_mask = des_col.str.contains(caught_stealing_pattern, case=False, regex=True)
    df.loc[cs_mask, 'caught_stealing'] = True
    df.loc[cs_mask, 'steal_attempt'] = True

    # Extract base for caught stealing
    def extract_cs_base(des):
        match = re.search(caught_stealing_pattern, des, re.IGNORECASE)
        if match:
            base = match.group(1).upper()
            if base == '2ND':
                return '2B'
            elif base == '3RD':
                return '3B'
            elif base == 'HOME':
                return 'HOME'
        return None

    df.loc[cs_mask, 'steal_base'] = des_col[cs_mask].apply(extract_cs_base)

    return df


def create_steal_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset of steal opportunities.

    A steal opportunity is defined as:
    - Runner on 1st or 2nd base (potential to steal 2nd or 3rd)
    - Less than 2 outs
    - We aggregate to plate appearance level for tractability

    For each PA meeting these criteria, we record whether a steal was attempted.
    """
    df = identify_steal_events(df)

    # Filter to situations with runner in stealing position
    # Runner on 1st (can steal 2nd) or runner on 2nd (can steal 3rd)
    steal_position = (df['on_1b'].notna()) | (df['on_2b'].notna())
    less_than_2_outs = df['outs_when_up'] < 2

    opportunities = df[steal_position & less_than_2_outs].copy()

    n_attempts = opportunities['steal_attempt'].sum()
    n_success = opportunities['steal_success'].sum()

    print(f"\nSteal opportunities: {len(opportunities):,} pitches")
    print(f"  Steal attempts: {n_attempts:,}")
    if n_attempts > 0:
        print(f"  Success rate: {n_success / n_attempts:.1%}")

    return opportunities


def aggregate_to_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level data to plate appearance level.

    For each PA, we record:
    - Game state at start of PA
    - Whether any steal attempt occurred during PA
    - Outcome of steal attempt if any
    """
    # Create PA identifier
    df['pa_id'] = df['game_pk'].astype(str) + '_' + df['at_bat_number'].astype(str)

    # Aggregate by PA
    pa_agg = df.groupby('pa_id').agg({
        # Game identifiers
        'game_pk': 'first',
        'game_date': 'first',
        'season': 'first',
        'inning': 'first',
        'inning_topbot': 'first',

        # Teams
        'home_team': 'first',
        'away_team': 'first',

        # Game state at PA start
        'outs_when_up': 'first',
        'on_1b': 'first',
        'on_2b': 'first',
        'on_3b': 'first',
        'home_score': 'first',
        'away_score': 'first',

        # Players
        'batter': 'first',
        'pitcher': 'first',
        'p_throws': 'first',

        # Steal outcomes
        'steal_attempt': 'any',
        'steal_success': 'any',
        'caught_stealing': 'any',
        'steal_base': lambda x: x.dropna().iloc[0] if x.notna().any() else None,

        # Count at first pitch (proxy for count during steal)
        'balls': 'first',
        'strikes': 'first',

        # Pitch count in PA
        'pitch_number': 'count'
    }).reset_index()

    pa_agg.rename(columns={'pitch_number': 'n_pitches'}, inplace=True)

    # Calculate score differential from batting team perspective
    pa_agg['is_home'] = pa_agg['inning_topbot'] == 'Bot'
    pa_agg['score_diff'] = np.where(
        pa_agg['is_home'],
        pa_agg['home_score'] - pa_agg['away_score'],
        pa_agg['away_score'] - pa_agg['home_score']
    )

    # Runner info
    pa_agg['runner_on_1b'] = pa_agg['on_1b'].notna()
    pa_agg['runner_on_2b'] = pa_agg['on_2b'].notna()
    pa_agg['runner_on_3b'] = pa_agg['on_3b'].notna()

    # Create base-out state string
    def get_base_state(row):
        bases = []
        if row['runner_on_1b']:
            bases.append('1')
        if row['runner_on_2b']:
            bases.append('2')
        if row['runner_on_3b']:
            bases.append('3')
        return '-'.join(bases) if bases else '_'

    pa_agg['base_state'] = pa_agg.apply(get_base_state, axis=1)

    print(f"\nAggregated to {len(pa_agg):,} plate appearances")
    print(f"  With steal attempt: {pa_agg['steal_attempt'].sum():,} ({pa_agg['steal_attempt'].mean():.1%})")

    return pa_agg


def compute_win_probability(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add win probability estimates for each game state.

    We need WP for:
    - Current state
    - State after successful steal
    - State after caught stealing
    """
    pa_df = pa_df.copy()

    # Simple logistic WP model based on inning and score diff
    def simple_wp(inning, score_diff, outs, is_home):
        """Simplified win probability estimate."""
        # Base WP from score diff (each run ≈ 6-8% WP in mid-game)
        wp = 0.5 + 0.07 * np.clip(score_diff, -7, 7)

        # Adjust for inning (later innings = more certain)
        # In late innings, leads are more valuable
        if inning >= 7:
            leverage = 1.5
        elif inning >= 4:
            leverage = 1.2
        else:
            leverage = 1.0
        wp = 0.5 + (wp - 0.5) * leverage

        # Home team slight advantage (especially in close games)
        if is_home:
            wp += 0.03

        return np.clip(wp, 0.02, 0.98)

    pa_df['wp_current'] = pa_df.apply(
        lambda r: simple_wp(r['inning'], r['score_diff'], r['outs_when_up'], r['is_home']),
        axis=1
    )

    # WP impact of successful steal
    # Run expectancy changes from stealing (approximate)
    # Stealing 2nd: runner on 1st → runner on 2nd (RE increases ~0.18 runs)
    # Stealing 3rd: runner on 2nd → runner on 3rd (RE increases ~0.16 runs)
    # Each run ≈ 4-5% WP in typical game state

    def wp_after_steal_success(row):
        """WP after successful steal (runner advances)."""
        base = row.get('steal_base', '2B')
        # Approximate WP gain from advancing runner
        if base == '2B':
            wp_gain = 0.015  # Runner 1st → 2nd
        elif base == '3B':
            wp_gain = 0.020  # Runner 2nd → 3rd (more valuable)
        else:  # HOME
            wp_gain = 0.08  # Stealing home rare but big if successful

        return np.clip(row['wp_current'] + wp_gain, 0.02, 0.98)

    def wp_after_caught_stealing(row):
        """WP after caught stealing (out added, runner removed)."""
        # CS is costly: lose runner + add out
        # Losing an out costs ~0.03-0.05 WP depending on situation
        # Losing runner costs additional WP
        outs = row['outs_when_up']

        if outs == 0:
            wp_loss = 0.04  # First out relatively less costly
        elif outs == 1:
            wp_loss = 0.05  # Second out more costly
        else:
            wp_loss = 0.06  # Would be third out (inning over)

        return np.clip(row['wp_current'] - wp_loss, 0.02, 0.98)

    pa_df['wp_success'] = pa_df.apply(wp_after_steal_success, axis=1)
    pa_df['wp_fail'] = pa_df.apply(wp_after_caught_stealing, axis=1)

    return pa_df


def main():
    """Main data acquisition pipeline."""
    data_dir = Path('stolen_base_analysis/data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check if raw data already exists
    raw_files = [
        raw_dir / 'statcast_2015_2024.parquet',
        raw_dir / 'statcast_2023_2024.parquet'
    ]

    raw_file = None
    for f in raw_files:
        if f.exists():
            raw_file = f
            break

    if raw_file is not None:
        print(f"Loading cached raw data from {raw_file}...")
        raw_data = pd.read_parquet(raw_file)
    else:
        print("Pulling statcast data (this may take a while)...")
        raw_data = pull_all_seasons(2023, 2024)
        raw_data.to_parquet(raw_dir / 'statcast_2023_2024.parquet')
        print(f"Saved raw data to {raw_dir / 'statcast_2023_2024.parquet'}")

    # Create steal opportunities
    print("\nProcessing steal opportunities...")
    opportunities = create_steal_opportunities(raw_data)

    # Aggregate to PA level
    print("\nAggregating to plate appearances...")
    pa_data = aggregate_to_plate_appearances(opportunities)

    # Add win probability
    print("\nComputing win probabilities...")
    pa_data = compute_win_probability(pa_data)

    # Save processed data
    output_file = processed_dir / 'steal_opportunities.parquet'
    pa_data.to_parquet(output_file)
    print(f"\nSaved processed data to {output_file}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total plate appearances with steal opportunity: {len(pa_data):,}")
    print(f"Steal attempts: {pa_data['steal_attempt'].sum():,} ({pa_data['steal_attempt'].mean():.2%})")

    if pa_data['steal_attempt'].sum() > 0:
        success_rate = pa_data['steal_success'].sum() / pa_data['steal_attempt'].sum()
        print(f"Success rate (when attempted): {success_rate:.1%}")

    print("\nBy season:")
    by_season = pa_data.groupby('season').agg({
        'steal_attempt': ['sum', 'mean'],
        'steal_success': 'sum',
        'pa_id': 'count'
    }).round(4)
    by_season.columns = ['attempts', 'attempt_rate', 'successes', 'n_opps']
    by_season['success_rate'] = (by_season['successes'] / by_season['attempts']).round(3)
    print(by_season.to_string())

    return pa_data


if __name__ == "__main__":
    pa_data = main()
