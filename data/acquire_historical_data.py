"""
Acquire historical NFL data for era comparison analysis.

Era 1: 2006-2012 (post-Romer 2006 paper, pre-analytics departments)
Era 2: 2019-2024 (already have this data)
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from main acquisition script
from acquire_data import (
    clean_pbp_data,
    extract_fourth_downs,
    extract_punt_plays,
    extract_field_goals,
    extract_fourth_down_attempts
)


def acquire_historical_data(seasons: list[int], data_dir: Path):
    """
    Download and process historical NFL data.
    """
    print(f"Acquiring data for seasons: {seasons}")

    # Download raw data
    cache_path = data_dir / 'historical_raw_pbp.parquet'

    if cache_path.exists():
        print(f"Loading cached historical data from {cache_path}")
        pbp = pd.read_parquet(cache_path)
    else:
        print("Downloading historical play-by-play data...")
        pbp = nfl.import_pbp_data(seasons)
        pbp.to_parquet(cache_path)
        print(f"Cached raw data to {cache_path}")

    print(f"Raw historical data: {len(pbp):,} plays")

    # Clean data using same pipeline
    print("\nCleaning historical data...")
    df = clean_pbp_data(pbp)

    # Save cleaned data
    df.to_parquet(data_dir / 'historical_cleaned_pbp.parquet')
    print(f"Saved cleaned historical data: {len(df):,} plays")

    # Extract subsets for modeling
    print("\nExtracting subsets...")

    fourth_downs = extract_fourth_downs(df)
    fourth_downs.to_parquet(data_dir / 'historical_fourth_downs.parquet')

    punts = extract_punt_plays(df)
    punts.to_parquet(data_dir / 'historical_punts.parquet')

    fgs = extract_field_goals(df)
    fgs.to_parquet(data_dir / 'historical_field_goals.parquet')

    attempts = extract_fourth_down_attempts(df)
    attempts.to_parquet(data_dir / 'historical_fourth_down_attempts.parquet')

    # Summary
    print("\n" + "="*60)
    print("HISTORICAL DATA SUMMARY (2006-2012)")
    print("="*60)
    print(f"Total regular season plays: {len(df):,}")
    print(f"Fourth down situations: {len(fourth_downs):,}")
    print(f"  - Punts: {(fourth_downs['actual_decision'] == 'punt').sum():,}")
    print(f"  - Field goals: {(fourth_downs['actual_decision'] == 'field_goal').sum():,}")
    print(f"  - Go for it: {(fourth_downs['actual_decision'] == 'go_for_it').sum():,}")
    print(f"Punt plays: {len(punts):,}")
    print(f"Field goal attempts: {len(fgs):,}")
    print(f"  - Make rate: {fgs['fg_made'].mean():.1%}")
    print(f"4th down go-for-it attempts: {len(attempts):,}")
    print(f"  - Conversion rate: {attempts['converted'].mean():.1%}")

    return df, fourth_downs, punts, fgs, attempts


if __name__ == "__main__":
    data_dir = Path(__file__).parent

    # Era 1: Post-Romer (2006), pre-analytics boom
    historical_seasons = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

    df, fourth_downs, punts, fgs, attempts = acquire_historical_data(
        historical_seasons, data_dir
    )
