"""
Data Acquisition for MLB Salary Efficiency Analysis

Pulls batting/pitching stats with WAR from pybaseball and merges with salary data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pybaseball import batting_stats, pitching_stats, cache
import warnings
import time

warnings.filterwarnings('ignore')

# Enable pybaseball caching
cache.enable()


def pull_batting_stats(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """
    Pull batting statistics with WAR for all seasons.

    Uses Fangraphs data via pybaseball.
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        print(f"  Pulling batting stats for {year}...")
        try:
            # qual=50 means minimum 50 PA
            df = batting_stats(year, qual=50)
            df['Season'] = year
            all_data.append(df)
            time.sleep(0.5)  # Be nice to the API
        except Exception as e:
            print(f"    Error for {year}: {e}")
            continue

    if not all_data:
        raise ValueError("No batting data retrieved")

    batting = pd.concat(all_data, ignore_index=True)
    print(f"  Total batting records: {len(batting):,}")

    return batting


def pull_pitching_stats(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """
    Pull pitching statistics with WAR for all seasons.
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        print(f"  Pulling pitching stats for {year}...")
        try:
            # qual=20 means minimum 20 IP
            df = pitching_stats(year, qual=20)
            df['Season'] = year
            all_data.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error for {year}: {e}")
            continue

    if not all_data:
        raise ValueError("No pitching data retrieved")

    pitching = pd.concat(all_data, ignore_index=True)
    print(f"  Total pitching records: {len(pitching):,}")

    return pitching


def standardize_batting_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for batting data.
    """
    # Map Fangraphs column names to our standard names
    col_map = {
        'Name': 'name',
        'Team': 'team',
        'Season': 'season',
        'Age': 'age',
        'G': 'games',
        'PA': 'pa',
        'AB': 'ab',
        'H': 'hits',
        'HR': 'hr',
        'R': 'runs',
        'RBI': 'rbi',
        'SB': 'sb',
        'BB': 'bb',
        'SO': 'so',
        'AVG': 'avg',
        'OBP': 'obp',
        'SLG': 'slg',
        'OPS': 'ops',
        'wOBA': 'woba',
        'WAR': 'war',
        'wRC+': 'wrc_plus',
        'Off': 'off_runs',
        'Def': 'def_runs',
        'BsR': 'bsr',  # Baserunning runs
        'Pos': 'position',
        # Fangraphs uses different WAR component names
        'Batting': 'batting_runs',
        'Baserunning': 'baserunning_runs',
        'Fielding': 'fielding_runs',
    }

    df = df.copy()

    # Rename columns that exist
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Handle player ID
    if 'IDfg' in df.columns:
        df['player_id'] = df['IDfg']
    elif 'playerid' in df.columns:
        df['player_id'] = df['playerid']
    else:
        df['player_id'] = df.index

    return df


def standardize_pitching_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for pitching data.
    """
    col_map = {
        'Name': 'name',
        'Team': 'team',
        'Season': 'season',
        'Age': 'age',
        'W': 'wins',
        'L': 'losses',
        'G': 'games',
        'GS': 'games_started',
        'IP': 'ip',
        'SO': 'so',
        'BB': 'bb',
        'HR': 'hr_allowed',
        'ERA': 'era',
        'FIP': 'fip',
        'xFIP': 'xfip',
        'WHIP': 'whip',
        'WAR': 'war',
        'K/9': 'k_per_9',
        'BB/9': 'bb_per_9',
    }

    df = df.copy()
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    if 'IDfg' in df.columns:
        df['player_id'] = df['IDfg']
    elif 'playerid' in df.columns:
        df['player_id'] = df['playerid']
    else:
        df['player_id'] = df.index

    df['is_pitcher'] = True

    return df


def estimate_service_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate service time class based on age and career games.

    Rough heuristics:
    - Players < 25 years old: likely pre-arb
    - Players 25-28: likely arb
    - Players > 28: likely free agent

    This is a simplification - real service time depends on days on roster.
    """
    df = df.copy()

    # Simple age-based heuristic
    df['service_time_class'] = 'free_agent'
    df.loc[df['age'] < 25, 'service_time_class'] = 'pre_arb'
    df.loc[(df['age'] >= 25) & (df['age'] < 28), 'service_time_class'] = 'arb'

    return df


def create_salary_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create estimated salary data based on WAR and service time.

    Since we don't have actual salary data readily available via pybaseball,
    we'll use known market rates to create realistic estimates for modeling.

    This is a PLACEHOLDER - ideally we'd scrape actual salaries from
    Baseball-Reference or Spotrac.

    Known market rates (approximate $/WAR for free agents):
    - 2000-2005: ~$3-4M per WAR
    - 2006-2010: ~$4-5M per WAR
    - 2011-2015: ~$6-7M per WAR
    - 2016-2020: ~$8-9M per WAR
    - 2021-2024: ~$9-10M per WAR
    """
    df = df.copy()

    # Market rate by year ($/WAR for free agents)
    market_rates = {
        2000: 3.0, 2001: 3.2, 2002: 3.4, 2003: 3.6, 2004: 3.8,
        2005: 4.0, 2006: 4.3, 2007: 4.6, 2008: 4.9, 2009: 5.0,
        2010: 5.2, 2011: 5.5, 2012: 6.0, 2013: 6.5, 2014: 7.0,
        2015: 7.5, 2016: 8.0, 2017: 8.3, 2018: 8.5, 2019: 8.8,
        2020: 9.0, 2021: 9.2, 2022: 9.5, 2023: 9.8, 2024: 10.0
    }

    # Add some noise to make it realistic
    np.random.seed(42)

    def estimate_salary(row):
        base_rate = market_rates.get(row['season'], 8.0) * 1e6

        # Adjust for service time
        if row['service_time_class'] == 'pre_arb':
            # Pre-arb players paid league minimum (~$700K)
            return 0.7e6 + np.random.normal(0, 0.1e6)
        elif row['service_time_class'] == 'arb':
            # Arb players get ~40-60% of market rate
            rate_mult = np.random.uniform(0.4, 0.6)
            return max(1.0e6, row['war'] * base_rate * rate_mult + np.random.normal(0, 1e6))
        else:
            # Free agents get market rate with noise
            # Star premium: higher WAR players get disproportionately more
            war = max(0.1, row['war'])  # Avoid issues with zero/negative WAR
            star_mult = 1.0 + 0.1 * (war - 2)  # 10% premium per WAR above 2
            star_mult = max(0.8, min(1.5, star_mult))

            base_salary = war * base_rate * star_mult
            noise = np.random.normal(0, 0.15 * base_salary)
            return max(1.0e6, base_salary + noise)

    df['salary'] = df.apply(estimate_salary, axis=1)

    return df


def process_and_merge_data(batting: pd.DataFrame, pitching: pd.DataFrame) -> pd.DataFrame:
    """
    Process and combine batting and pitching data into unified dataset.
    """
    print("Processing data...")

    # Standardize columns
    batting = standardize_batting_columns(batting)
    pitching = standardize_pitching_columns(pitching)

    # Add service time estimates
    batting = estimate_service_time(batting)
    pitching = estimate_service_time(pitching)

    # For position players, use batting data
    batting['is_pitcher'] = False

    # Select key columns
    batting_cols = ['player_id', 'name', 'team', 'season', 'age', 'games', 'pa',
                    'hr', 'sb', 'avg', 'obp', 'slg', 'ops', 'woba', 'war',
                    'off_runs', 'def_runs', 'bsr', 'position', 'service_time_class', 'is_pitcher']
    batting_cols = [c for c in batting_cols if c in batting.columns]

    pitching_cols = ['player_id', 'name', 'team', 'season', 'age', 'games', 'ip',
                     'wins', 'losses', 'so', 'era', 'fip', 'whip', 'war',
                     'service_time_class', 'is_pitcher']
    pitching_cols = [c for c in pitching_cols if c in pitching.columns]

    # Combine
    batting_subset = batting[batting_cols].copy()
    pitching_subset = pitching[pitching_cols].copy()
    pitching_subset['position'] = 'P'

    # Merge - keep only one record per player-season
    # For two-way players, batting takes precedence unless WAR is higher for pitching
    combined = pd.concat([batting_subset, pitching_subset], ignore_index=True)

    # For duplicate player-seasons, keep the one with higher WAR
    combined = combined.sort_values('war', ascending=False)
    combined = combined.drop_duplicates(subset=['player_id', 'season'], keep='first')

    # Add salary estimates
    combined = create_salary_estimates(combined)

    # Filter to players with positive WAR for main analysis
    combined = combined[combined['war'] > 0].copy()

    print(f"Final dataset: {len(combined):,} player-seasons")
    print(f"Seasons: {combined['season'].min()} to {combined['season'].max()}")
    print(f"Unique players: {combined['player_id'].nunique():,}")

    return combined


def main():
    """Pull all data and save to parquet."""
    data_dir = Path('salary_efficiency/data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check if raw data already exists
    batting_path = raw_dir / 'batting_2000_2024.parquet'
    pitching_path = raw_dir / 'pitching_2000_2024.parquet'

    if batting_path.exists() and pitching_path.exists():
        print("Loading cached raw data...")
        batting = pd.read_parquet(batting_path)
        pitching = pd.read_parquet(pitching_path)
    else:
        print("Pulling batting statistics (2000-2024)...")
        batting = pull_batting_stats(2000, 2024)
        batting.to_parquet(batting_path)
        print(f"Saved batting data to {batting_path}")

        print("\nPulling pitching statistics (2000-2024)...")
        pitching = pull_pitching_stats(2000, 2024)
        pitching.to_parquet(pitching_path)
        print(f"Saved pitching data to {pitching_path}")

    # Process and merge
    print("\nProcessing and merging data...")
    df = process_and_merge_data(batting, pitching)

    # Save processed data - convert object columns to string to avoid parquet issues
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    processed_path = processed_dir / 'player_seasons.parquet'
    df.to_parquet(processed_path)
    print(f"\nSaved processed data to {processed_path}")

    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total player-seasons: {len(df):,}")
    print(f"\nBy service time class:")
    print(df['service_time_class'].value_counts())
    print(f"\nBy position (top 10):")
    if 'position' in df.columns:
        print(df['position'].value_counts().head(10))
    print(f"\nWAR distribution:")
    print(df['war'].describe())
    print(f"\nSalary distribution (estimated):")
    print(df['salary'].describe())

    return df


if __name__ == "__main__":
    df = main()
