"""
Team-Level Win Probability Loss Analysis

For each team and season, computes expected wins lost due to suboptimal
fourth down decisions using an expanding window approach.

Key features:
- Uses only information available at the time (expanding window)
- Starts analyzing in 2006 (after 7 years of training data: 1999-2005)
- Aggregates WP loss by team and season
- Converts WP loss to expected wins lost

Output: Large table with teams as rows, seasons as columns
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.bayesian_models import (
    ConversionModel, PuntModel, FieldGoalModel, WinProbabilityModel, ClockConsumptionModel
)
from analysis.decision_framework import BayesianDecisionAnalyzer, GameState
from data.acquire_data import (
    clean_pbp_data, extract_fourth_downs, extract_punt_plays,
    extract_field_goals, extract_fourth_down_attempts
)


def load_or_download_data(data_dir: Path, start_year: int = 1999, end_year: int = 2024):
    """Load all available play-by-play data."""
    cache_path = data_dir / f'all_pbp_{start_year}_{end_year}.parquet'

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # Try alternative file paths
    alt_path = data_dir / 'all_cleaned_1999_2024.parquet'
    if alt_path.exists():
        print(f"Loading from {alt_path}")
        return pd.read_parquet(alt_path)

    # Download if necessary
    print(f"Downloading play-by-play data for {start_year}-{end_year}...")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import nfl_data_py as nfl

    seasons = list(range(start_year, end_year + 1))
    pbp = nfl.import_pbp_data(seasons)
    pbp.to_parquet(cache_path)
    print(f"Saved to {cache_path}")

    return pbp


def prepare_cumulative_training_data(pbp: pd.DataFrame, end_year: int):
    """Prepare cumulative training data through end_year (inclusive)."""
    cumulative_pbp = pbp[pbp['season'] <= end_year].copy()
    cleaned = clean_pbp_data(cumulative_pbp)

    fourth_downs = extract_fourth_downs(cleaned)
    punts = extract_punt_plays(cleaned)
    fgs = extract_field_goals(cleaned)
    attempts = extract_fourth_down_attempts(cleaned)

    return {
        'cleaned': cleaned,
        'fourth_downs': fourth_downs,
        'punts': punts,
        'fgs': fgs,
        'attempts': attempts,
        'raw_pbp': cumulative_pbp
    }


def fit_models_on_data(data: dict, n_samples: int = 500):
    """Fit all Bayesian models on given data."""
    models = {}

    # Conversion model
    conversion = ConversionModel()
    if len(data['attempts']) > 100:
        conversion.fit(data['attempts'], n_samples=n_samples)
        models['conversion'] = conversion
    else:
        print(f"  Warning: Only {len(data['attempts'])} attempts, skipping")
        return None

    # Punt model
    punt = PuntModel()
    if len(data['punts']) > 100:
        punt.fit(data['punts'], n_samples=n_samples)
        models['punt'] = punt
    else:
        return None

    # FG model
    fg = FieldGoalModel()
    if len(data['fgs']) > 100:
        fg.fit(data['fgs'], n_samples=n_samples)
        models['fg'] = fg
    else:
        return None

    # WP model
    wp = WinProbabilityModel()
    if len(data['cleaned']) > 10000:
        wp.fit(data['cleaned'], n_samples=n_samples)
        models['wp'] = wp
    else:
        return None

    # Bayesian clock model
    clock = ClockConsumptionModel()
    clock.fit(data['raw_pbp'], n_samples=n_samples)
    models['clock'] = clock

    return models


def analyze_team_season_wp_loss(
    fourth_downs_df: pd.DataFrame,
    analyzer: BayesianDecisionAnalyzer
) -> pd.DataFrame:
    """
    Analyze WP loss for each play in the given fourth down data.

    Returns DataFrame with play-level WP costs.
    """
    results = []

    for idx, row in tqdm(fourth_downs_df.iterrows(), total=len(fourth_downs_df),
                         desc="Analyzing plays", leave=False):
        try:
            state = GameState(
                field_pos=int(row['yardline_100']),
                yards_to_go=int(row['ydstogo']),
                score_diff=int(row['score_diff']),
                time_remaining=int(row['game_seconds_remaining']),
                timeout_diff=int(row.get('timeout_diff', 0))
            )

            result = analyzer.analyze(state)
            actual = row['actual_decision']

            # Get WP for actual decision
            if actual == 'go_for_it':
                wp_actual = result.wp_go
            elif actual == 'punt':
                wp_actual = result.wp_punt
            elif actual == 'field_goal':
                wp_actual = result.wp_fg
            else:
                continue

            # Get WP for optimal decision
            wp_optimal = max(result.wp_go, result.wp_punt, result.wp_fg)

            # WP cost (always >= 0)
            wp_cost = wp_optimal - wp_actual

            # Get team - use posteam (possession team)
            team = row.get('posteam', row.get('possession_team', None))

            results.append({
                'game_id': row.get('game_id', idx),
                'play_id': row.get('play_id', idx),
                'season': row.get('season'),
                'team': team,
                'actual_decision': actual,
                'optimal_action': result.optimal_action,
                'wp_actual': wp_actual,
                'wp_optimal': wp_optimal,
                'wp_cost': wp_cost,
                'decision_margin': result.decision_margin,
                'is_mistake': actual != result.optimal_action,
                'is_clear_mistake': (actual != result.optimal_action) and (result.decision_margin >= 0.02),
            })

        except Exception as e:
            continue

    return pd.DataFrame(results)


def run_team_wp_loss_analysis(
    start_train_year: int = 1999,
    first_test_year: int = 2006,
    last_test_year: int = 2024,
    n_samples: int = 500
) -> pd.DataFrame:
    """
    Run the full expanding window team WP loss analysis.

    For each test year Y:
    - Train on start_train_year through Y-1
    - Analyze all 4th downs in year Y
    - Compute WP loss per team
    """
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    pbp = load_or_download_data(data_dir, start_train_year, last_test_year)
    print(f"Total plays: {len(pbp):,}")
    print(f"Seasons: {sorted(pbp['season'].unique())}")

    # Prepare fourth downs for each test year
    print("\n" + "=" * 80)
    print("PREPARING TEST YEAR DATA")
    print("=" * 80)

    test_year_fourth_downs = {}
    for year in range(first_test_year, last_test_year + 1):
        year_pbp = pbp[pbp['season'] == year].copy()
        year_cleaned = clean_pbp_data(year_pbp)
        year_4th = extract_fourth_downs(year_cleaned)
        # Add season column if not present
        if 'season' not in year_4th.columns:
            year_4th['season'] = year
        test_year_fourth_downs[year] = year_4th
        print(f"  {year}: {len(year_4th):,} fourth down plays")

    # Run expanding window analysis
    print("\n" + "=" * 80)
    print("EXPANDING WINDOW ANALYSIS")
    print("=" * 80)

    all_play_results = []

    for test_year in range(first_test_year, last_test_year + 1):
        train_end_year = test_year - 1

        print(f"\n--- Test Year: {test_year} (Training: {start_train_year}-{train_end_year}) ---")

        # Train models on data available at start of test year
        train_data = prepare_cumulative_training_data(pbp, train_end_year)
        print(f"  Training plays: {len(train_data['cleaned']):,}")
        print(f"  Training 4th down attempts: {len(train_data['attempts']):,}")

        models = fit_models_on_data(train_data, n_samples=n_samples)

        if models is None:
            print(f"  Skipping {test_year} due to insufficient training data")
            continue

        analyzer = BayesianDecisionAnalyzer(models)

        # Analyze each play in test year
        test_plays = test_year_fourth_downs[test_year]
        print(f"  Analyzing {len(test_plays):,} plays...")

        year_results = analyze_team_season_wp_loss(test_plays, analyzer)

        if len(year_results) > 0:
            all_play_results.append(year_results)

            # Print summary for this year
            by_team = year_results.groupby('team').agg({
                'wp_cost': 'sum',
                'is_mistake': 'sum',
                'game_id': 'count'
            }).rename(columns={'game_id': 'n_plays'})

            total_wp_loss = year_results['wp_cost'].sum()
            total_mistakes = year_results['is_mistake'].sum()
            print(f"  Total WP loss: {total_wp_loss:.3f} ({total_wp_loss * 17:.2f} expected wins)")
            print(f"  Total mistakes: {total_mistakes:,} / {len(year_results):,}")

    # Combine all results
    results_df = pd.concat(all_play_results, ignore_index=True)
    print(f"\n\nTotal plays analyzed: {len(results_df):,}")

    # Save play-level results
    results_df.to_parquet(output_dir / 'team_wp_loss_play_level.parquet')

    return results_df


def generate_team_season_table(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Generate the main table: Expected wins lost by team and season.

    Rows: Teams
    Columns: Seasons
    Values: Expected wins lost (WP_loss * games_per_season / 17 games assumption)
    """
    # Aggregate by team and season
    team_season = results_df.groupby(['team', 'season']).agg({
        'wp_cost': 'sum',
        'is_mistake': 'sum',
        'is_clear_mistake': 'sum',
        'game_id': 'count'
    }).rename(columns={'game_id': 'n_plays'})

    team_season = team_season.reset_index()

    # Calculate expected wins lost
    # WP cost is already per-play, sum gives total WP lost per season
    # We interpret this as: if you played the season 1000 times,
    # you'd win 1000 * wp_cost fewer games on average
    team_season['expected_wins_lost'] = team_season['wp_cost']

    # Create pivot table
    pivot = team_season.pivot(index='team', columns='season', values='expected_wins_lost')

    # Fill NaN with 0 (team didn't exist or no data)
    pivot = pivot.fillna(0)

    # Add row totals (average per season)
    pivot['Average'] = pivot.mean(axis=1)

    # Add column totals (league average per season)
    col_means = pivot.drop(columns='Average').mean(axis=0)
    col_means['Average'] = pivot['Average'].mean()
    pivot.loc['LEAGUE AVG'] = col_means

    # Sort by average wins lost (descending = worst offenders first)
    # Exclude LEAGUE AVG from sorting
    teams_sorted = pivot.drop(index='LEAGUE AVG').sort_values('Average', ascending=False)
    pivot = pd.concat([teams_sorted, pivot.loc[['LEAGUE AVG']]])

    # Round for display
    pivot_display = pivot.round(3)

    # Save as CSV
    pivot_display.to_csv(output_dir / 'team_season_wins_lost.csv')

    # Also save the detailed aggregation
    team_season.to_csv(output_dir / 'team_season_wp_loss_detail.csv', index=False)

    return pivot


def generate_latex_table(pivot_df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for the slides."""

    # Get season columns (excluding Average)
    seasons = [c for c in pivot_df.columns if c != 'Average']

    # For slides, we might want to show fewer columns
    # Show every other year to fit on slide
    display_seasons = seasons[::2]  # Every other year
    if seasons[-1] not in display_seasons:
        display_seasons.append(seasons[-1])

    display_cols = display_seasons + ['Average']

    # Get top 10 worst teams + league average
    teams_to_show = list(pivot_df.drop(index='LEAGUE AVG').head(10).index) + ['LEAGUE AVG']

    pivot_subset = pivot_df.loc[teams_to_show, display_cols]

    # Generate LaTeX
    latex = """\\begin{table}[H]
\\centering
\\caption{Expected Wins Lost from Suboptimal Fourth Down Decisions}
\\label{tab:team_wins_lost}
\\tiny
\\begin{tabular}{l""" + "r" * len(display_cols) + """}
\\toprule
\\textbf{Team} & """ + " & ".join([f"\\textbf{{{s}}}" for s in display_cols]) + """ \\\\
\\midrule
"""

    for team in teams_to_show:
        if team == 'LEAGUE AVG':
            latex += "\\midrule\n"
        row_vals = [f"{pivot_subset.loc[team, c]:.2f}" for c in display_cols]
        team_display = team if team != 'LEAGUE AVG' else '\\textit{League Avg}'
        latex += team_display + " & " + " & ".join(row_vals) + " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_dir / 'team_wins_lost_table.tex', 'w') as f:
        f.write(latex)

    # Also generate a full table (all teams, all seasons)
    full_latex = """\\begin{table}[H]
\\centering
\\caption{Expected Wins Lost by Team and Season (2006--2024)}
\\label{tab:team_wins_lost_full}
\\tiny
\\setlength{\\tabcolsep}{2pt}
\\begin{tabular}{l""" + "r" * len(seasons) + """r}
\\toprule
\\textbf{Team} & """ + " & ".join([f"\\textbf{{'{str(s)[2:]}}}" for s in seasons]) + """ & \\textbf{Avg} \\\\
\\midrule
"""

    for team in pivot_df.index:
        if team == 'LEAGUE AVG':
            full_latex += "\\midrule\n"
        row_vals = [f"{pivot_df.loc[team, c]:.2f}" for c in seasons + ['Average']]
        team_display = team if team != 'LEAGUE AVG' else '\\textit{League Avg}'
        full_latex += team_display + " & " + " & ".join(row_vals) + " \\\\\n"

    full_latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_dir / 'team_wins_lost_full_table.tex', 'w') as f:
        f.write(full_latex)

    print(f"LaTeX tables saved to {output_dir}")


def print_summary_stats(pivot_df: pd.DataFrame, results_df: pd.DataFrame):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Drop league average for team-level stats
    teams_only = pivot_df.drop(index='LEAGUE AVG', errors='ignore')

    print(f"\nTeams analyzed: {len(teams_only)}")
    print(f"Seasons analyzed: {len([c for c in pivot_df.columns if c != 'Average'])}")
    print(f"Total plays: {len(results_df):,}")
    print(f"Total mistakes: {results_df['is_mistake'].sum():,}")
    print(f"Total clear mistakes (>=2% margin): {results_df['is_clear_mistake'].sum():,}")

    print(f"\n--- League-Wide Statistics ---")
    avg_per_team = pivot_df.loc['LEAGUE AVG', 'Average']
    print(f"Average expected wins lost per team per season: {avg_per_team:.3f}")

    total_league_wp_loss = results_df['wp_cost'].sum()
    seasons = results_df['season'].nunique()
    print(f"Total league WP loss: {total_league_wp_loss:.2f}")
    print(f"Average per season (league-wide): {total_league_wp_loss/seasons:.2f}")

    print(f"\n--- Worst Offenders (by average wins lost/season) ---")
    worst = teams_only.sort_values('Average', ascending=False).head(10)
    for team in worst.index:
        print(f"  {team}: {worst.loc[team, 'Average']:.3f} expected wins lost/season")

    print(f"\n--- Best Decision-Makers (by average wins lost/season) ---")
    best = teams_only.sort_values('Average', ascending=True).head(10)
    for team in best.index:
        print(f"  {team}: {best.loc[team, 'Average']:.3f} expected wins lost/season")

    print(f"\n--- Year-by-Year League Trends ---")
    seasons_cols = [c for c in pivot_df.columns if c != 'Average']
    for season in seasons_cols:
        league_avg = pivot_df.loc['LEAGUE AVG', season]
        print(f"  {season}: {league_avg:.3f} avg wins lost per team")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the analysis
    # 7-year minimum training window: 1999-2005 -> test 2006
    results_df = run_team_wp_loss_analysis(
        start_train_year=1999,
        first_test_year=2006,
        last_test_year=2024,
        n_samples=500  # Reduced for speed, increase for publication
    )

    # Generate team x season table
    pivot_df = generate_team_season_table(results_df, output_dir)

    # Generate LaTeX tables
    generate_latex_table(pivot_df, output_dir)

    # Print summary statistics
    print_summary_stats(pivot_df, results_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_dir}")
