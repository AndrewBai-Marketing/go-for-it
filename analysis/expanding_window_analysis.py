"""
Expanding Window Analysis: Were optimal 4th down decisions knowable in real-time?

For each test year Y (2017-2024):
1. Train models on 2006 through Y-1 (information available at start of season Y)
2. Apply to year Y's 4th downs to get "ex ante" optimal action
3. Compare to "ex post" optimal (full 2006-2024 model)
4. Compare to actual coach decisions

Key question: What % of mistakes were "inexcusable" (both ex ante and ex post agreed)?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.bayesian_models import (
    ConversionModel, PuntModel, FieldGoalModel, WinProbabilityModel,
    HierarchicalFieldGoalModel, HierarchicalPuntModel
)
try:
    from models.hierarchical_off_def_model import HierarchicalOffDefConversionModel
    HAS_OFF_DEF_MODEL = True
except ImportError:
    HAS_OFF_DEF_MODEL = False
from analysis.decision_framework import BayesianDecisionAnalyzer, GameState
from data.acquire_data import (
    clean_pbp_data, extract_fourth_downs, extract_punt_plays,
    extract_field_goals, extract_fourth_down_attempts
)


def download_all_data(data_dir: Path, start_year: int = 1999, end_year: int = 2024):
    """Download all available nflfastR data."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import nfl_data_py as nfl

    cache_path = data_dir / f'all_pbp_{start_year}_{end_year}.parquet'

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Downloading play-by-play data for {start_year}-{end_year}...")
    seasons = list(range(start_year, end_year + 1))
    pbp = nfl.import_pbp_data(seasons)
    pbp.to_parquet(cache_path)
    print(f"Saved to {cache_path}")

    return pbp


def prepare_year_data(pbp: pd.DataFrame, year: int):
    """Extract training data for a specific year."""
    year_pbp = pbp[pbp['season'] == year].copy()
    cleaned = clean_pbp_data(year_pbp)

    fourth_downs = extract_fourth_downs(cleaned)
    punts = extract_punt_plays(cleaned)
    fgs = extract_field_goals(cleaned)
    attempts = extract_fourth_down_attempts(cleaned)

    return {
        'cleaned': cleaned,
        'fourth_downs': fourth_downs,
        'punts': punts,
        'fgs': fgs,
        'attempts': attempts
    }


def prepare_cumulative_data(pbp: pd.DataFrame, end_year: int):
    """Prepare cumulative training data from start through end_year (inclusive)."""
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
        'attempts': attempts
    }


def fit_models_on_data(data: dict, n_samples: int = 1000, use_hierarchical: bool = True):
    """
    Fit all Bayesian models on given data.

    Args:
        data: Dict with 'attempts', 'punts', 'fgs', 'cleaned' DataFrames
        n_samples: Number of posterior samples
        use_hierarchical: If True, use hierarchical models with team/kicker effects
    """
    # Conversion model - use hierarchical off/def model if available
    if use_hierarchical and HAS_OFF_DEF_MODEL:
        conversion = HierarchicalOffDefConversionModel()
        if len(data['attempts']) > 500:  # Need more data for hierarchical
            try:
                conversion.fit(data['attempts'], n_samples=n_samples)
                print(f"    Fitted hierarchical off/def conversion model")
            except Exception as e:
                print(f"    Warning: Hierarchical conversion failed ({e}), using basic model")
                conversion = ConversionModel()
                conversion.fit(data['attempts'], n_samples=n_samples)
        else:
            print(f"    Using basic conversion model (only {len(data['attempts'])} attempts)")
            conversion = ConversionModel()
            conversion.fit(data['attempts'], n_samples=n_samples)
    else:
        conversion = ConversionModel()
        if len(data['attempts']) > 100:
            conversion.fit(data['attempts'], n_samples=n_samples)
        else:
            print(f"  Warning: Only {len(data['attempts'])} attempts, skipping conversion model")
            return None

    # Punt model - use hierarchical punter model
    if use_hierarchical:
        punt = HierarchicalPuntModel()
        if len(data['punts']) > 500:  # Need more data for hierarchical
            try:
                punt.fit(data['punts'], n_samples=n_samples)
                n_punters = len(punt.punter_effects)
                print(f"    Fitted hierarchical punt model with {n_punters} punters")
            except Exception as e:
                print(f"    Warning: Hierarchical punt failed ({e}), using basic model")
                punt = PuntModel()
                punt.fit(data['punts'], n_samples=n_samples)
        else:
            print(f"    Using basic punt model (only {len(data['punts'])} punts)")
            punt = PuntModel()
            punt.fit(data['punts'], n_samples=n_samples)
    else:
        punt = PuntModel()
        if len(data['punts']) > 100:
            punt.fit(data['punts'], n_samples=n_samples)
        else:
            print(f"  Warning: Only {len(data['punts'])} punts, skipping punt model")
            return None

    # FG model - use hierarchical kicker model
    if use_hierarchical:
        fg = HierarchicalFieldGoalModel()
        if len(data['fgs']) > 500:  # Need more data for hierarchical
            try:
                fg.fit(data['fgs'], n_samples=n_samples)
                n_kickers = len(fg.kicker_effects)
                print(f"    Fitted hierarchical FG model with {n_kickers} kickers")
            except Exception as e:
                print(f"    Warning: Hierarchical FG failed ({e}), using basic model")
                fg = FieldGoalModel()
                fg.fit(data['fgs'], n_samples=n_samples)
        else:
            print(f"    Using basic FG model (only {len(data['fgs'])} FGs)")
            fg = FieldGoalModel()
            fg.fit(data['fgs'], n_samples=n_samples)
    else:
        fg = FieldGoalModel()
        if len(data['fgs']) > 100:
            fg.fit(data['fgs'], n_samples=n_samples)
        else:
            print(f"  Warning: Only {len(data['fgs'])} FGs, skipping FG model")
            return None

    # WP model
    wp = WinProbabilityModel()
    if len(data['cleaned']) > 10000:
        wp.fit(data['cleaned'], n_samples=n_samples)
    else:
        print(f"  Warning: Only {len(data['cleaned'])} plays, skipping WP model")
        return None

    return {
        'conversion': conversion,
        'punt': punt,
        'fg': fg,
        'wp': wp
    }


def analyze_play(state: GameState, analyzer: BayesianDecisionAnalyzer):
    """Analyze a single 4th down play."""
    try:
        result = analyzer.analyze(state)
        return {
            'optimal_action': result.optimal_action,
            'wp_go': result.wp_go,
            'wp_punt': result.wp_punt,
            'wp_fg': result.wp_fg,
            'prob_go_best': result.prob_go_best,
            'prob_punt_best': result.prob_punt_best,
            'prob_fg_best': result.prob_fg_best,
            'decision_margin': result.decision_margin,
        }
    except Exception as e:
        return None


def build_team_kicker_mapping(fgs_df: pd.DataFrame, season: int) -> dict:
    """
    Build a mapping from team -> primary kicker for a given season.

    Uses data up through the given season to find each team's most recent
    primary kicker (the one with most FG attempts in their most recent season).
    """
    # Filter to data up through this season
    df = fgs_df[fgs_df['season'] <= season].copy()

    if len(df) == 0:
        return {}

    # For each team, find the most recent season they have data for
    team_kickers = {}

    for team in df['posteam'].dropna().unique():
        team_df = df[df['posteam'] == team]
        if len(team_df) == 0:
            continue

        # Get the most recent season for this team
        most_recent = team_df['season'].max()
        recent_df = team_df[team_df['season'] == most_recent]

        # Find the kicker with most attempts in that season
        kicker_counts = recent_df.groupby('kicker_player_id').size()
        if len(kicker_counts) > 0:
            primary_kicker = kicker_counts.idxmax()
            team_kickers[team] = primary_kicker

    return team_kickers


def build_team_punter_mapping(punts_df: pd.DataFrame, season: int) -> dict:
    """
    Build a mapping from team -> primary punter for a given season.

    Uses data up through the given season to find each team's most recent
    primary punter (the one with most punt attempts in their most recent season).
    """
    # Filter to data up through this season
    df = punts_df[punts_df['season'] <= season].copy()

    if len(df) == 0:
        return {}

    # For each team, find the most recent season they have data for
    team_punters = {}

    for team in df['posteam'].dropna().unique():
        team_df = df[df['posteam'] == team]
        if len(team_df) == 0:
            continue

        # Get the most recent season for this team
        most_recent = team_df['season'].max()
        recent_df = team_df[team_df['season'] == most_recent]

        # Find the punter with most attempts in that season
        punter_counts = recent_df.groupby('punter_player_id').size()
        if len(punter_counts) > 0:
            primary_punter = punter_counts.idxmax()
            team_punters[team] = primary_punter

    return team_punters


def run_expanding_window_analysis(
    start_train_year: int = 1999,
    first_test_year: int = 2006,
    last_test_year: int = 2024,
    n_samples: int = 1000
):
    """
    Run the full expanding window analysis.

    For each test year Y:
    - Train on start_train_year through Y-1
    - Test on year Y
    """
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download all data
    print("="*80)
    print("STEP 1: DOWNLOADING ALL DATA")
    print("="*80)

    pbp = download_all_data(data_dir, start_train_year, last_test_year)
    print(f"Total plays: {len(pbp):,}")
    print(f"Seasons: {sorted(pbp['season'].unique())}")

    # Step 2: Prepare test year fourth downs
    print("\n" + "="*80)
    print("STEP 2: PREPARING TEST DATA")
    print("="*80)

    test_fourth_downs = {}
    for year in range(first_test_year, last_test_year + 1):
        year_data = prepare_year_data(pbp, year)
        test_fourth_downs[year] = year_data['fourth_downs']
        print(f"  {year}: {len(year_data['fourth_downs']):,} 4th down plays")

    # Step 3: Train ex post model (full sample)
    print("\n" + "="*80)
    print("STEP 3: TRAINING EX POST MODEL (FULL SAMPLE)")
    print("="*80)

    full_data = prepare_cumulative_data(pbp, last_test_year)
    print(f"Full sample: {len(full_data['cleaned']):,} plays")
    print(f"  4th down attempts: {len(full_data['attempts']):,}")
    print(f"  Punts: {len(full_data['punts']):,}")
    print(f"  FGs: {len(full_data['fgs']):,}")

    ex_post_models = fit_models_on_data(full_data, n_samples=n_samples, use_hierarchical=True)
    ex_post_analyzer = BayesianDecisionAnalyzer(ex_post_models)

    # Build team -> kicker/punter mappings for full sample (for ex post analysis)
    full_team_kicker_map = build_team_kicker_mapping(full_data['fgs'], last_test_year)
    full_team_punter_map = build_team_punter_mapping(full_data['punts'], last_test_year)
    print(f"  Full sample team-kicker mappings: {len(full_team_kicker_map)} teams")
    print(f"  Full sample team-punter mappings: {len(full_team_punter_map)} teams")

    # Step 4: Train expanding window models and analyze
    print("\n" + "="*80)
    print("STEP 4: EXPANDING WINDOW ANALYSIS")
    print("="*80)

    all_results = []

    for test_year in range(first_test_year, last_test_year + 1):
        train_end_year = test_year - 1

        print(f"\n--- Test Year: {test_year} (Training on {start_train_year}-{train_end_year}) ---")

        # Train ex ante model (only data available before test year)
        train_data = prepare_cumulative_data(pbp, train_end_year)
        print(f"  Training data: {len(train_data['cleaned']):,} plays")
        print(f"    4th down attempts: {len(train_data['attempts']):,}")

        ex_ante_models = fit_models_on_data(train_data, n_samples=n_samples, use_hierarchical=True)

        if ex_ante_models is None:
            print(f"  Skipping {test_year} due to insufficient training data")
            continue

        ex_ante_analyzer = BayesianDecisionAnalyzer(ex_ante_models)

        # Build team -> kicker/punter mappings for this training window
        # This tells us each team's primary kicker/punter as of the training data cutoff
        team_kicker_map = build_team_kicker_mapping(train_data['fgs'], train_end_year)
        team_punter_map = build_team_punter_mapping(train_data['punts'], train_end_year)
        print(f"    Team-kicker mappings: {len(team_kicker_map)} teams")
        print(f"    Team-punter mappings: {len(team_punter_map)} teams")

        # Analyze each 4th down in test year
        test_plays = test_fourth_downs[test_year]

        # Filter out plays with <60 seconds remaining (per methodology)
        # WP models are unreliable in end-of-game situations
        test_plays = test_plays[test_plays['game_seconds_remaining'] >= 60].copy()
        print(f"  Analyzing {len(test_plays):,} 4th down plays (after 60s filter)...")

        for idx, row in tqdm(test_plays.iterrows(), total=len(test_plays),
                            desc=f"  Year {test_year}"):
            try:
                # Get team info
                off_team = row.get('posteam', None)
                def_team = row.get('defteam', None)

                # Look up the team's kicker and punter from training data
                kicker_id = team_kicker_map.get(off_team, None)
                punter_id = team_punter_map.get(off_team, None)

                state = GameState(
                    field_pos=int(row['yardline_100']),
                    yards_to_go=int(row['ydstogo']),
                    score_diff=int(row['score_diff']),
                    time_remaining=int(row['game_seconds_remaining']),
                    timeout_diff=int(row.get('timeout_diff', 0)),
                    off_team=off_team,
                    def_team=def_team,
                    kicker_id=kicker_id,
                    punter_id=punter_id
                )

                # Ex ante analysis (what was knowable at the time)
                ex_ante_result = analyze_play(state, ex_ante_analyzer)

                # Ex post analysis (with full hindsight)
                # Use full sample kicker/punter for ex post (may differ from ex ante)
                ex_post_kicker = full_team_kicker_map.get(off_team, None)
                ex_post_punter = full_team_punter_map.get(off_team, None)
                ex_post_state = GameState(
                    field_pos=int(row['yardline_100']),
                    yards_to_go=int(row['ydstogo']),
                    score_diff=int(row['score_diff']),
                    time_remaining=int(row['game_seconds_remaining']),
                    timeout_diff=int(row.get('timeout_diff', 0)),
                    off_team=off_team,
                    def_team=def_team,
                    kicker_id=ex_post_kicker,
                    punter_id=ex_post_punter
                )
                ex_post_result = analyze_play(ex_post_state, ex_post_analyzer)

                if ex_ante_result is None or ex_post_result is None:
                    continue

                actual = row['actual_decision']

                # Compute confidence (probability that optimal action is best)
                if ex_ante_result['optimal_action'] == 'go_for_it':
                    ex_ante_confidence = ex_ante_result['prob_go_best']
                elif ex_ante_result['optimal_action'] == 'punt':
                    ex_ante_confidence = ex_ante_result['prob_punt_best']
                else:
                    ex_ante_confidence = ex_ante_result['prob_fg_best']

                all_results.append({
                    'season': test_year,
                    'game_id': row.get('game_id', idx),
                    'play_id': row.get('play_id', idx),

                    # Game state
                    'field_pos': state.field_pos,
                    'yards_to_go': state.yards_to_go,
                    'score_diff': state.score_diff,
                    'time_remaining': state.time_remaining,

                    # Decisions
                    'actual_decision': actual,
                    'ex_ante_optimal': ex_ante_result['optimal_action'],
                    'ex_post_optimal': ex_post_result['optimal_action'],

                    # Match indicators
                    'ex_ante_match': actual == ex_ante_result['optimal_action'],
                    'ex_post_match': actual == ex_post_result['optimal_action'],
                    'models_agree': ex_ante_result['optimal_action'] == ex_post_result['optimal_action'],

                    # Confidence
                    'ex_ante_confidence': ex_ante_confidence,
                    'ex_ante_margin': ex_ante_result['decision_margin'],
                    'ex_post_margin': ex_post_result['decision_margin'],

                    # Win probabilities (ex ante)
                    'ex_ante_wp_go': ex_ante_result['wp_go'],
                    'ex_ante_wp_punt': ex_ante_result['wp_punt'],
                    'ex_ante_wp_fg': ex_ante_result['wp_fg'],

                    # Training window
                    'train_start': start_train_year,
                    'train_end': train_end_year,
                })

            except Exception as e:
                continue

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    print(f"\n\nTotal plays analyzed: {len(results_df):,}")

    # Save play-level results
    results_df.to_csv(output_dir / 'expanding_window_results.csv', index=False)
    results_df.to_parquet(output_dir / 'expanding_window_results.parquet')

    return results_df


def generate_summary_tables(results_df: pd.DataFrame, output_dir: Path):
    """Generate summary tables from the analysis."""

    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)

    # Table 1: Year-by-year comparison
    print("\n--- Table 1: Year-by-Year Comparison ---")

    yearly = results_df.groupby('season').agg({
        'ex_ante_match': ['sum', 'mean'],
        'ex_post_match': ['sum', 'mean'],
        'models_agree': 'mean',
        'game_id': 'count'
    }).round(3)

    yearly.columns = ['Ex Ante Correct', 'Ex Ante Rate', 'Ex Post Correct',
                      'Ex Post Rate', 'Agreement Rate', 'N Plays']

    print(yearly.to_string())
    yearly.to_csv(output_dir / 'yearly_comparison.csv')

    # Table 2: Overall summary
    print("\n--- Table 2: Overall Summary ---")

    total_plays = len(results_df)
    ex_ante_correct = results_df['ex_ante_match'].sum()
    ex_post_correct = results_df['ex_post_match'].sum()
    agree = results_df['models_agree'].sum()

    print(f"Total 4th down plays: {total_plays:,}")
    print(f"Ex ante optimal rate: {ex_ante_correct:,} ({ex_ante_correct/total_plays:.1%})")
    print(f"Ex post optimal rate: {ex_post_correct:,} ({ex_post_correct/total_plays:.1%})")
    print(f"Models agree rate: {agree:,} ({agree/total_plays:.1%})")

    # Table 3: Mistake decomposition
    print("\n--- Table 3: Mistake Decomposition ---")

    # When models agree on optimal action
    agree_df = results_df[results_df['models_agree']]
    agree_correct = agree_df['ex_ante_match'].sum()  # Same as ex_post_match when they agree

    # When models disagree
    disagree_df = results_df[~results_df['models_agree']]
    disagree_ex_ante_correct = disagree_df['ex_ante_match'].sum()
    disagree_ex_post_correct = disagree_df['ex_post_match'].sum()

    print(f"\nWhen models AGREE on optimal action ({len(agree_df):,} plays, {len(agree_df)/total_plays:.1%}):")
    print(f"  Coach was correct: {agree_correct:,} ({agree_correct/len(agree_df):.1%})")
    print(f"  Coach was WRONG (inexcusable): {len(agree_df) - agree_correct:,} ({(len(agree_df) - agree_correct)/len(agree_df):.1%})")

    print(f"\nWhen models DISAGREE on optimal action ({len(disagree_df):,} plays, {len(disagree_df)/total_plays:.1%}):")
    print(f"  Coach matched ex ante: {disagree_ex_ante_correct:,} ({disagree_ex_ante_correct/len(disagree_df):.1%})")
    print(f"  Coach matched ex post: {disagree_ex_post_correct:,} ({disagree_ex_post_correct/len(disagree_df):.1%})")

    # Detailed mistake categories
    print("\n--- Table 4: Detailed Mistake Categories ---")

    # Category 1: Both agree, coach was wrong ("Inexcusable")
    inexcusable = len(agree_df) - agree_correct
    inexcusable_pct = inexcusable / total_plays

    # Category 2: Models disagree, coach matched ex ante but not ex post ("Unlucky")
    unlucky = ((disagree_df['ex_ante_match']) & (~disagree_df['ex_post_match'])).sum()
    unlucky_pct = unlucky / total_plays

    # Category 3: Models disagree, coach matched ex post but not ex ante ("Lucky")
    lucky = ((~disagree_df['ex_ante_match']) & (disagree_df['ex_post_match'])).sum()
    lucky_pct = lucky / total_plays

    # Category 4: Models disagree, coach wrong on both ("Double wrong")
    double_wrong = ((~disagree_df['ex_ante_match']) & (~disagree_df['ex_post_match'])).sum()
    double_wrong_pct = double_wrong / total_plays

    print(f"Inexcusable mistakes (both models agreed, coach wrong): {inexcusable:,} ({inexcusable_pct:.1%})")
    print(f"Unlucky (ex ante right, ex post wrong, coach matched ex ante): {unlucky:,} ({unlucky_pct:.1%})")
    print(f"Lucky (ex ante wrong, ex post right, coach matched ex post): {lucky:,} ({lucky_pct:.1%})")
    print(f"Double wrong (models disagreed, coach wrong on both): {double_wrong:,} ({double_wrong_pct:.1%})")

    # The key punchline
    print("\n" + "="*80)
    print("THE PUNCHLINE")
    print("="*80)

    stable_pct = len(agree_df) / total_plays
    stable_correct_pct = agree_correct / len(agree_df) if len(agree_df) > 0 else 0

    print(f"""
In {stable_pct:.1%} of 4th down plays, the ex ante and ex post optimal decisions AGREE.
These are "stable" situations where the correct answer was knowable in real-time.

Among these stable situations, coaches only got {stable_correct_pct:.1%} right.

The information existed. Coaches didn't use it.

Inexcusable mistakes (both models said X, coach did Y): {inexcusable:,} plays ({inexcusable_pct:.1%} of all 4th downs)
""")

    # By decision type
    print("\n--- Inexcusable Mistakes by Decision Type ---")
    inexcusable_df = agree_df[~agree_df['ex_ante_match']]

    for actual in ['go_for_it', 'punt', 'field_goal']:
        for optimal in ['go_for_it', 'punt', 'field_goal']:
            if actual != optimal:
                count = len(inexcusable_df[
                    (inexcusable_df['actual_decision'] == actual) &
                    (inexcusable_df['ex_ante_optimal'] == optimal)
                ])
                if count > 0:
                    print(f"  Did {actual}, should have {optimal}: {count:,}")

    # By confidence level
    print("\n--- Inexcusable Mistakes by Ex Ante Confidence ---")
    for lo, hi, label in [(0.5, 0.7, '50-70%'), (0.7, 0.9, '70-90%'), (0.9, 1.0, '90%+')]:
        subset = inexcusable_df[
            (inexcusable_df['ex_ante_confidence'] >= lo) &
            (inexcusable_df['ex_ante_confidence'] < hi)
        ]
        print(f"  {label} confidence: {len(subset):,} mistakes")

    return yearly


def create_visualization(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization of results."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Year-by-year optimal rates
        ax1 = axes[0, 0]
        yearly = results_df.groupby('season').agg({
            'ex_ante_match': 'mean',
            'ex_post_match': 'mean',
            'models_agree': 'mean'
        })

        ax1.plot(yearly.index, yearly['ex_ante_match'], 'o-', label='Ex Ante Optimal Rate', linewidth=2)
        ax1.plot(yearly.index, yearly['ex_post_match'], 's-', label='Ex Post Optimal Rate', linewidth=2)
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Coach Optimal Rate')
        ax1.set_title('Coach Decision Quality Over Time')
        ax1.legend()
        ax1.set_ylim(0.5, 1.0)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Agreement rate over time
        ax2 = axes[0, 1]
        ax2.bar(yearly.index, yearly['models_agree'], color='steelblue', alpha=0.7)
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Agreement Rate')
        ax2.set_title('Ex Ante vs Ex Post Model Agreement')
        ax2.set_ylim(0, 1)
        ax2.axhline(yearly['models_agree'].mean(), color='red', linestyle='--',
                   label=f'Mean: {yearly["models_agree"].mean():.1%}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Mistake decomposition pie chart
        ax3 = axes[1, 0]
        agree_df = results_df[results_df['models_agree']]
        disagree_df = results_df[~results_df['models_agree']]

        correct_stable = agree_df['ex_ante_match'].sum()
        inexcusable = len(agree_df) - correct_stable
        correct_unstable = disagree_df['ex_ante_match'].sum() + disagree_df['ex_post_match'].sum()
        # Avoid double counting when both match (shouldn't happen when models disagree)
        both_match = ((disagree_df['ex_ante_match']) & (disagree_df['ex_post_match'])).sum()
        correct_unstable -= both_match
        wrong_unstable = len(disagree_df) - correct_unstable

        sizes = [correct_stable, inexcusable, correct_unstable, wrong_unstable]
        labels = [f'Correct (stable)\n{correct_stable:,}',
                  f'INEXCUSABLE\n{inexcusable:,}',
                  f'Correct (unstable)\n{correct_unstable:,}',
                  f'Wrong (unstable)\n{wrong_unstable:,}']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
        explode = (0, 0.1, 0, 0)  # Explode the inexcusable slice

        ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90)
        ax3.set_title('Mistake Decomposition')

        # Plot 4: Confidence distribution for inexcusable mistakes
        ax4 = axes[1, 1]
        inexcusable_df = agree_df[~agree_df['ex_ante_match']]
        ax4.hist(inexcusable_df['ex_ante_confidence'], bins=20, color='indianred',
                alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Ex Ante Confidence (P(optimal action is best))')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Confidence Distribution for Inexcusable Mistakes (n={len(inexcusable_df):,})')
        ax4.axvline(0.95, color='red', linestyle='--', label='>95% confidence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'optimal_rate_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to {output_dir / 'optimal_rate_comparison.png'}")

    except ImportError:
        print("matplotlib not available, skipping visualization")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the analysis with extended data
    # 7-year minimum training window: 1999-2005 -> test 2006
    results_df = run_expanding_window_analysis(
        start_train_year=1999,
        first_test_year=2006,
        last_test_year=2024,
        n_samples=1000  # Reduced for speed
    )

    # Generate summary tables
    generate_summary_tables(results_df, output_dir)

    # Create visualization
    figures_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    create_visualization(results_df, figures_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to {output_dir}")
