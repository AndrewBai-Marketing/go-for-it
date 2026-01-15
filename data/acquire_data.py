"""
Data acquisition script for NFL 4th down analysis.
Uses nfl_data_py to get play-by-play data.
"""

import ssl
# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path

def acquire_pbp_data(seasons: list[int], cache_path: Path = None) -> pd.DataFrame:
    """
    Acquire play-by-play data for specified seasons.

    Args:
        seasons: List of NFL seasons to fetch
        cache_path: Optional path to cache the data

    Returns:
        DataFrame with play-by-play data
    """
    print(f"Fetching play-by-play data for seasons: {seasons}")

    if cache_path and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    pbp = nfl.import_pbp_data(seasons)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pbp.to_parquet(cache_path)
        print(f"Cached data to {cache_path}")

    return pbp


def clean_pbp_data(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Clean play-by-play data for analysis.

    - Filter to regular season
    - Remove plays with missing critical data
    - Create team_won outcome variable
    """
    print(f"Starting with {len(pbp):,} plays")

    # Filter to regular season
    df = pbp[pbp['season_type'] == 'REG'].copy()
    print(f"After filtering to regular season: {len(df):,} plays")

    # Keep only actual plays (not timeouts, penalties, etc.)
    df = df[df['play_type'].isin(['pass', 'run', 'punt', 'field_goal', 'qb_kneel', 'qb_spike'])].copy()
    print(f"After filtering to actual plays: {len(df):,} plays")

    # Remove plays with missing critical data
    critical_cols = [
        'down', 'ydstogo', 'yardline_100', 'score_differential',
        'game_seconds_remaining', 'posteam', 'defteam',
        'home_team', 'away_team', 'home_score', 'away_score'
    ]
    df = df.dropna(subset=critical_cols)
    print(f"After dropping missing critical data: {len(df):,} plays")

    # Create team_won outcome: did the possession team win the game?
    # Need to get final scores
    game_results = df.groupby('game_id').agg({
        'home_team': 'first',
        'away_team': 'first',
        'home_score': 'max',
        'away_score': 'max'
    }).reset_index()

    # Determine winner
    game_results['winner'] = np.where(
        game_results['home_score'] > game_results['away_score'],
        game_results['home_team'],
        np.where(
            game_results['away_score'] > game_results['home_score'],
            game_results['away_team'],
            'TIE'
        )
    )

    # Merge back to plays
    df = df.merge(game_results[['game_id', 'winner']], on='game_id', how='left')

    # team_won: 1 if possession team won, 0 if lost, 0.5 if tie
    df['team_won'] = np.where(
        df['posteam'] == df['winner'],
        1,
        np.where(df['winner'] == 'TIE', 0.5, 0)
    )

    # Remove ties for cleaner binary outcome
    df = df[df['winner'] != 'TIE']
    print(f"After removing ties: {len(df):,} plays")

    # Convert to appropriate types
    df['down'] = df['down'].astype(int)
    df['ydstogo'] = df['ydstogo'].astype(int)
    df['yardline_100'] = df['yardline_100'].astype(int)

    # Create useful derived features
    df['time_remaining_minutes'] = df['game_seconds_remaining'] / 60
    df['score_diff'] = df['score_differential'].astype(int)

    # Timeouts - use forward fill within game, then fill remaining with 3
    # Missing timeouts are rare (~5%) and mostly on no_play types already filtered out
    df = df.sort_values(['game_id', 'play_id'])
    df['posteam_timeouts'] = df.groupby('game_id')['posteam_timeouts_remaining'].ffill().bfill().fillna(3).astype(int)
    df['defteam_timeouts'] = df.groupby('game_id')['defteam_timeouts_remaining'].ffill().bfill().fillna(3).astype(int)
    df['timeout_diff'] = df['posteam_timeouts'] - df['defteam_timeouts']

    return df


def extract_fourth_downs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 4th down plays for decision analysis.
    """
    fourth_downs = df[df['down'] == 4].copy()
    print(f"Found {len(fourth_downs):,} fourth down plays")

    # Categorize what the team actually did
    # Note: qb_kneel and qb_spike on 4th down are excluded - these are clock management
    # plays in garbage time, not actual fourth down decisions
    fourth_downs = fourth_downs[~fourth_downs['play_type'].isin(['qb_kneel', 'qb_spike'])].copy()

    fourth_downs['actual_decision'] = np.where(
        fourth_downs['play_type'] == 'punt', 'punt',
        np.where(
            fourth_downs['play_type'] == 'field_goal', 'field_goal',
            'go_for_it'  # run or pass
        )
    )

    # Cap yards to go at 15 for modeling purposes, with indicator
    fourth_downs['ydstogo_capped'] = fourth_downs['ydstogo'].clip(upper=15)
    fourth_downs['ydstogo_was_capped'] = (fourth_downs['ydstogo'] > 15).astype(int)

    # Add field goal distance (yardline_100 + 17 yards for snap/hold)
    fourth_downs['fg_distance'] = fourth_downs['yardline_100'] + 17

    return fourth_downs


def extract_punt_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract punt plays for punt distance modeling.
    """
    punts = df[df['play_type'] == 'punt'].copy()

    # Calculate net punt yards
    # Net yards = punt distance - return yards (if any)
    punts['punt_net_yards'] = punts['kick_distance'].fillna(0) - punts['return_yards'].fillna(0)

    # Filter out blocked punts and extreme outliers
    punts = punts[punts['punt_blocked'] != 1]
    punts = punts[punts['punt_net_yards'].between(-10, 80)]

    print(f"Found {len(punts):,} valid punt plays")
    return punts


def extract_field_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract field goal attempts for FG probability modeling.
    """
    fgs = df[df['play_type'] == 'field_goal'].copy()

    # Create binary outcome
    fgs['fg_made'] = (fgs['field_goal_result'] == 'made').astype(int)

    # FG distance = yardline + 17
    fgs['fg_distance'] = fgs['yardline_100'] + 17

    # Keep all FG attempts - let the Bayesian model handle distance uncertainty
    # Only filter out clearly invalid data (negative distances, etc.)
    fgs = fgs[fgs['fg_distance'] >= 17]  # Minimum possible: at goal line + snap/hold

    print(f"Found {len(fgs):,} field goal attempts")
    return fgs


def extract_fourth_down_attempts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 4th down go-for-it attempts for conversion modeling.
    """
    attempts = df[
        (df['down'] == 4) &
        (df['play_type'].isin(['run', 'pass']))
    ].copy()

    # Use nflfastR's fourth_down_converted field (handles penalties, turnovers correctly)
    # Fall back to yards_gained comparison only if the field is missing
    if 'fourth_down_converted' in attempts.columns:
        attempts['converted'] = attempts['fourth_down_converted'].fillna(0).astype(int)
    else:
        # Fallback for older data or missing field
        attempts['converted'] = (attempts['yards_gained'] >= attempts['ydstogo']).astype(int)

    # Cap distance at 15 and add indicator for capped values
    attempts['ydstogo_capped'] = attempts['ydstogo'].clip(upper=15)
    attempts['ydstogo_was_capped'] = (attempts['ydstogo'] > 15).astype(int)

    print(f"Found {len(attempts):,} 4th down go-for-it attempts")
    return attempts


if __name__ == "__main__":
    # Set up paths
    data_dir = Path(__file__).parent

    # Acquire data
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    pbp = acquire_pbp_data(seasons, cache_path=data_dir / 'raw_pbp.parquet')

    print(f"\nRaw data shape: {pbp.shape}")
    print(f"Columns: {len(pbp.columns)}")

    # Clean data
    df = clean_pbp_data(pbp)

    # Save cleaned data
    df.to_parquet(data_dir / 'cleaned_pbp.parquet')
    print(f"\nSaved cleaned data: {len(df):,} plays")

    # Extract subsets for modeling
    fourth_downs = extract_fourth_downs(df)
    fourth_downs.to_parquet(data_dir / 'fourth_downs.parquet')

    punts = extract_punt_plays(df)
    punts.to_parquet(data_dir / 'punts.parquet')

    fgs = extract_field_goals(df)
    fgs.to_parquet(data_dir / 'field_goals.parquet')

    attempts = extract_fourth_down_attempts(df)
    attempts.to_parquet(data_dir / 'fourth_down_attempts.parquet')

    # Summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total regular season plays: {len(df):,}")
    print(f"Fourth down situations: {len(fourth_downs):,}")
    print(f"  - Punts: {(fourth_downs['actual_decision'] == 'punt').sum():,}")
    print(f"  - Field goals: {(fourth_downs['actual_decision'] == 'field_goal').sum():,}")
    print(f"  - Go for it: {(fourth_downs['actual_decision'] == 'go_for_it').sum():,}")
    print(f"Punt plays (for distance model): {len(punts):,}")
    print(f"Field goal attempts: {len(fgs):,}")
    print(f"  - Make rate: {fgs['fg_made'].mean():.1%}")
    print(f"4th down go-for-it attempts: {len(attempts):,}")
    print(f"  - Conversion rate: {attempts['converted'].mean():.1%}")
