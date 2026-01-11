"""
Data Acquisition v2: Use REAL salary data from Lahman database

Key changes from v1:
- Use actual observed salaries from Lahman's Salaries table
- Merge with batting/pitching WAR from FanGraphs
- No more simulated salaries
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import requests
from io import BytesIO
from zipfile import ZipFile

warnings.filterwarnings('ignore')

# pybaseball imports
from pybaseball import batting_stats, pitching_stats


def pull_lahman_salaries() -> pd.DataFrame:
    """Pull real salary data from Lahman database."""
    print("Pulling Lahman salary data...")

    # Try multiple potential URLs for Lahman data
    # Note: seanlahman/baseballdatabank is the working source
    salary_urls = [
        "https://raw.githubusercontent.com/seanlahman/baseballdatabank/master/core/Salaries.csv",
        "https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Salaries.csv",
        "https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/main/core/Salaries.csv",
    ]

    # Note: older versions use Master.csv, newer use People.csv
    people_urls = [
        "https://raw.githubusercontent.com/seanlahman/baseballdatabank/master/core/Master.csv",
        "https://raw.githubusercontent.com/seanlahman/baseballdatabank/master/core/People.csv",
        "https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/People.csv",
        "https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/main/core/People.csv",
    ]

    sal = None
    ppl = None

    # Try salary URLs
    for url in salary_urls:
        try:
            print(f"  Trying: {url}")
            sal = pd.read_csv(url)
            print(f"  Success! Loaded {len(sal):,} salary records")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if sal is None:
        raise ValueError("Could not download Lahman salary data from any source")

    # Try people URLs
    for url in people_urls:
        try:
            print(f"  Trying: {url}")
            ppl = pd.read_csv(url)
            print(f"  Success! Loaded {len(ppl):,} people records")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if ppl is None:
        # Create name key directly from salary data if People.csv not available
        print("  WARNING: Could not load People.csv, using playerID only")
        sal['name_key'] = sal['playerID'].str.lower()
        return sal[sal['yearID'] >= 2000]

    print(f"  Raw salaries: {len(sal):,} records")
    print(f"  Years: {sal['yearID'].min()}-{sal['yearID'].max()}")

    # Create name key for matching
    ppl['name_key'] = (ppl['nameFirst'].fillna('').str.lower() + ' ' +
                       ppl['nameLast'].fillna('').str.lower())

    # Merge salary with player info
    sal = sal.merge(
        ppl[['playerID', 'name_key', 'nameFirst', 'nameLast', 'birthYear']],
        on='playerID',
        how='left'
    )

    # Filter to 2000+ (Lahman salary data ends at 2016)
    sal = sal[sal['yearID'] >= 2000]
    print(f"  After year filter (2000+): {len(sal):,}")
    print(f"  Max year in Lahman salaries: {sal['yearID'].max()}")

    return sal


def pull_fangraphs_war(start_year: int = 2000, end_year: int = 2023) -> pd.DataFrame:
    """Pull WAR data from FanGraphs via pybaseball or use cached data."""

    # Check for cached data first
    cache_bat = Path('salary_efficiency/data/raw/batting_2000_2024.parquet')
    cache_pit = Path('salary_efficiency/data/raw/pitching_2000_2024.parquet')

    if cache_bat.exists() and cache_pit.exists():
        print(f"\nUsing cached FanGraphs data...")
        batting = pd.read_parquet(cache_bat)
        pitching = pd.read_parquet(cache_pit)
        # Filter to year range
        batting = batting[(batting['Season'] >= start_year) & (batting['Season'] <= end_year)]
        pitching = pitching[(pitching['Season'] >= start_year) & (pitching['Season'] <= end_year)]
        print(f"  Batting records (cached): {len(batting):,}")
        print(f"  Pitching records (cached): {len(pitching):,}")
    else:
        print(f"\nPulling FanGraphs batting stats {start_year}-{end_year}...")
        batting = batting_stats(start_year, end_year, qual=0)
        print(f"  Batting records: {len(batting):,}")

        print(f"\nPulling FanGraphs pitching stats {start_year}-{end_year}...")
        pitching = pitching_stats(start_year, end_year, qual=0)
        print(f"  Pitching records: {len(pitching):,}")

    # Filter to players with meaningful WAR
    batting = batting[batting['WAR'] >= 0.1]
    pitching = pitching[pitching['WAR'] >= 0.1]
    print(f"  After WAR filter - Batting: {len(batting):,}, Pitching: {len(pitching):,}")

    # Standardize column names
    batting_clean = batting[['Name', 'Team', 'Season', 'Age', 'WAR']].copy()
    batting_clean['position_type'] = 'batter'

    pitching_clean = pitching[['Name', 'Team', 'Season', 'Age', 'WAR']].copy()
    pitching_clean['position_type'] = 'pitcher'

    # Combine
    war_df = pd.concat([batting_clean, pitching_clean], ignore_index=True)

    # Create name key for matching
    war_df['name_key'] = war_df['Name'].str.lower().str.strip()

    # Aggregate WAR by player-season (some players bat and pitch)
    war_agg = war_df.groupby(['name_key', 'Season', 'Team']).agg({
        'Name': 'first',
        'Age': 'first',
        'WAR': 'sum',
        'position_type': 'first'
    }).reset_index()

    print(f"\nTotal player-seasons with WAR >= 0.1: {len(war_agg):,}")

    return war_agg


def merge_salary_war(sal_df: pd.DataFrame, war_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge salary and WAR data.

    This is tricky because:
    - Lahman uses playerID
    - FanGraphs uses player names
    - Team abbreviations may differ
    """
    print("\nMerging salary and WAR data...")

    # Standardize team names (Lahman uses different abbreviations)
    team_map = {
        'ANA': 'LAA', 'CAL': 'LAA', 'MON': 'WSN', 'FLO': 'MIA', 'FLA': 'MIA',
        'TBD': 'TBR', 'TBA': 'TBR', 'CHN': 'CHC', 'CHA': 'CHW', 'KCA': 'KCR',
        'LAN': 'LAD', 'NYA': 'NYY', 'NYN': 'NYM', 'SDN': 'SDP', 'SFN': 'SFG',
        'SLN': 'STL', 'WAS': 'WSN', 'WSH': 'WSN'
    }

    sal_df = sal_df.copy()
    sal_df['team_std'] = sal_df['teamID'].replace(team_map)

    war_df = war_df.copy()
    war_df['team_std'] = war_df['Team'].replace(team_map)

    # Merge on name + year + team
    merged = war_df.merge(
        sal_df[['name_key', 'yearID', 'team_std', 'salary', 'playerID']],
        left_on=['name_key', 'Season', 'team_std'],
        right_on=['name_key', 'yearID', 'team_std'],
        how='inner'
    )

    print(f"  Matched records: {len(merged):,}")

    # Clean up columns
    merged = merged.rename(columns={
        'Season': 'season',
        'team_std': 'team',
        'Name': 'name',
        'Age': 'age',
        'WAR': 'war',
        'position_type': 'position'
    })

    merged = merged[['name', 'playerID', 'season', 'team', 'age', 'war',
                     'salary', 'position']]

    # Remove duplicates (keep highest salary if multiple)
    merged = merged.sort_values('salary', ascending=False)
    merged = merged.drop_duplicates(subset=['playerID', 'season'], keep='first')

    print(f"  After dedup: {len(merged):,}")

    return merged


def estimate_service_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate service time class based on years in dataset.

    More accurate than age-based estimation.
    """
    df = df.copy()

    # Count years per player
    player_first_year = df.groupby('playerID')['season'].min().reset_index()
    player_first_year.columns = ['playerID', 'first_season']

    df = df.merge(player_first_year, on='playerID')
    df['years_in_mlb'] = df['season'] - df['first_season']

    # Service time classification
    # 0-2 years: pre-arb
    # 3-5 years: arb
    # 6+ years: free agent eligible
    def classify_service(years):
        if years <= 2:
            return 'pre_arb'
        elif years <= 5:
            return 'arb'
        else:
            return 'free_agent'

    df['service_time_class'] = df['years_in_mlb'].apply(classify_service)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate the merged dataset."""
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)

    # Check team count
    n_teams = df['team'].nunique()
    print(f"\nUnique teams: {n_teams}")
    if n_teams > 30:
        print("  WARNING: More than 30 teams detected!")
        print(f"  Teams: {sorted(df['team'].unique())}")

    # Check year range
    print(f"Year range: {df['season'].min()}-{df['season'].max()}")

    # Check salary distribution
    print(f"\nSalary distribution:")
    print(f"  Min: ${df['salary'].min():,.0f}")
    print(f"  Median: ${df['salary'].median():,.0f}")
    print(f"  Mean: ${df['salary'].mean():,.0f}")
    print(f"  Max: ${df['salary'].max():,.0f}")

    # Check WAR distribution
    print(f"\nWAR distribution:")
    print(f"  Min: {df['war'].min():.2f}")
    print(f"  Median: {df['war'].median():.2f}")
    print(f"  Mean: {df['war'].mean():.2f}")
    print(f"  Max: {df['war'].max():.2f}")

    # Service time breakdown
    print(f"\nService time breakdown:")
    print(df['service_time_class'].value_counts())

    # Check for obvious issues
    print(f"\nData quality checks:")
    print(f"  Missing salaries: {df['salary'].isna().sum()}")
    print(f"  Missing WAR: {df['war'].isna().sum()}")
    print(f"  Zero salaries: {(df['salary'] == 0).sum()}")
    print(f"  Negative WAR: {(df['war'] < 0).sum()}")


def main():
    """Main data acquisition pipeline using real salary data."""

    output_dir = Path('salary_efficiency/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pull Lahman salaries
    sal_df = pull_lahman_salaries()

    # Pull FanGraphs WAR
    war_df = pull_fangraphs_war(start_year=2000, end_year=2023)

    # Merge
    merged = merge_salary_war(sal_df, war_df)

    # Estimate service time
    merged = estimate_service_time(merged)

    # Validate
    validate_data(merged)

    # Save
    output_path = output_dir / 'player_seasons_real_salary.parquet'

    # Convert object columns to string for parquet
    for col in merged.select_dtypes(include=['object']).columns:
        merged[col] = merged[col].astype(str)

    merged.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total records: {len(merged):,}")

    return merged


if __name__ == "__main__":
    df = main()
