"""
Analyze all 4th down decisions in the dataset.

Compares:
1. Bayesian optimal decisions
2. CHMM robust decisions (at various λ)
3. Actual coach decisions

Generates tables and summary statistics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.bayesian_models import load_all_models
from analysis.decision_framework import (
    BayesianDecisionAnalyzer, CHMMDecisionAnalyzer,
    GameState, DecisionResult,
    create_structured_models, load_structured_models
)


def analyze_fourth_down_dataset(
    fourth_downs: pd.DataFrame,
    analyzer: BayesianDecisionAnalyzer,
    max_plays: int = None
) -> pd.DataFrame:
    """
    Analyze all 4th down plays using Bayesian decision framework.

    Args:
        fourth_downs: DataFrame with 4th down plays
        analyzer: BayesianDecisionAnalyzer instance
        max_plays: Maximum number of plays to analyze (for testing)

    Returns:
        DataFrame with analysis results for each play
    """
    if max_plays:
        fourth_downs = fourth_downs.head(max_plays)

    results = []

    for idx, row in tqdm(fourth_downs.iterrows(), total=len(fourth_downs), desc="Analyzing"):
        try:
            state = GameState(
                field_pos=int(row['yardline_100']),
                yards_to_go=int(row['ydstogo']),
                score_diff=int(row['score_diff']),
                time_remaining=int(row['game_seconds_remaining']),
                timeout_diff=int(row.get('timeout_diff', 0))
            )

            result = analyzer.analyze(state)

            results.append({
                'game_id': row.get('game_id', idx),
                'play_id': row.get('play_id', idx),
                'season': row.get('season'),
                'posteam': row.get('posteam'),

                # Game state
                'field_pos': state.field_pos,
                'yards_to_go': state.yards_to_go,
                'score_diff': state.score_diff,
                'time_remaining': state.time_remaining,
                'quarter': 4 - (state.time_remaining // 900),  # Approximate quarter

                # Actual decision
                'actual_decision': row['actual_decision'],

                # Bayesian analysis
                'wp_go': result.wp_go,
                'wp_punt': result.wp_punt,
                'wp_fg': result.wp_fg,
                'optimal_action': result.optimal_action,
                'decision_margin': result.decision_margin,
                'prob_go_best': result.prob_go_best,
                'prob_punt_best': result.prob_punt_best,
                'prob_fg_best': result.prob_fg_best,

                # Derived
                'is_optimal': row['actual_decision'] == result.optimal_action,
                'wp_actual': result.wp_go if row['actual_decision'] == 'go_for_it'
                            else result.wp_punt if row['actual_decision'] == 'punt'
                            else result.wp_fg,
                'wp_best': max(result.wp_go, result.wp_punt, result.wp_fg),
                'wp_lost': max(result.wp_go, result.wp_punt, result.wp_fg) -
                          (result.wp_go if row['actual_decision'] == 'go_for_it'
                           else result.wp_punt if row['actual_decision'] == 'punt'
                           else result.wp_fg),
            })
        except Exception as e:
            print(f"Error analyzing play {idx}: {e}")
            continue

    return pd.DataFrame(results)


def generate_summary_tables(analysis_df: pd.DataFrame, output_dir: Path):
    """
    Generate summary tables from the analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)

    # Table 1: Overall decision accuracy
    print("\n--- Table 1: Overall Decision Quality ---")
    total_plays = len(analysis_df)
    optimal_plays = analysis_df['is_optimal'].sum()
    pct_optimal = optimal_plays / total_plays * 100

    print(f"Total 4th down plays analyzed: {total_plays:,}")
    print(f"Plays where actual = optimal: {optimal_plays:,} ({pct_optimal:.1f}%)")
    print(f"Plays with suboptimal decisions: {total_plays - optimal_plays:,} ({100-pct_optimal:.1f}%)")

    # Breakdown by actual decision
    print("\nBy actual decision:")
    for decision in ['go_for_it', 'punt', 'field_goal']:
        subset = analysis_df[analysis_df['actual_decision'] == decision]
        if len(subset) > 0:
            pct = subset['is_optimal'].mean() * 100
            print(f"  {decision}: {len(subset):,} plays, {pct:.1f}% optimal")

    # Table 2: Mistake patterns
    print("\n--- Table 2: Mistake Patterns ---")
    mistakes = analysis_df[~analysis_df['is_optimal']]

    print(f"\nTotal mistakes: {len(mistakes):,}")
    print("\nMistake type breakdown:")
    for actual in ['go_for_it', 'punt', 'field_goal']:
        for optimal in ['go_for_it', 'punt', 'field_goal']:
            if actual != optimal:
                count = len(mistakes[(mistakes['actual_decision'] == actual) &
                                    (mistakes['optimal_action'] == optimal)])
                if count > 0:
                    print(f"  Did {actual}, should have {optimal}: {count:,}")

    # Table 3: WP cost of suboptimal decisions
    print("\n--- Table 3: Win Probability Cost ---")
    total_wp_lost = analysis_df['wp_lost'].sum()
    avg_wp_lost = analysis_df['wp_lost'].mean()
    avg_wp_lost_mistakes = mistakes['wp_lost'].mean() if len(mistakes) > 0 else 0

    print(f"Total WP lost across all decisions: {total_wp_lost:.1f} percentage points")
    print(f"Average WP lost per decision: {avg_wp_lost:.3f} pp ({avg_wp_lost*100:.2f}%)")
    print(f"Average WP lost per MISTAKE: {avg_wp_lost_mistakes:.3f} pp ({avg_wp_lost_mistakes*100:.2f}%)")

    # Per game and per season estimates
    games = analysis_df['game_id'].nunique() if 'game_id' in analysis_df else total_plays / 15
    seasons = analysis_df['season'].nunique() if 'season' in analysis_df else 6

    wp_per_game = total_wp_lost / games
    print(f"\nEstimated WP lost per game: {wp_per_game:.3f} pp")
    print(f"This translates to approximately {wp_per_game / 100 * 17:.2f} wins lost per 17-game season")

    # Table 4: Decision certainty analysis
    print("\n--- Table 4: Decision Certainty Analysis ---")

    # Obvious decisions (high certainty)
    obvious_go = analysis_df[analysis_df['prob_go_best'] > 0.95]
    obvious_punt = analysis_df[analysis_df['prob_punt_best'] > 0.95]
    obvious_fg = analysis_df[analysis_df['prob_fg_best'] > 0.95]

    print(f"\nObvious 'go for it' situations (P > 95%): {len(obvious_go):,}")
    if len(obvious_go) > 0:
        went_for_it = (obvious_go['actual_decision'] == 'go_for_it').mean() * 100
        print(f"  Teams actually went for it: {went_for_it:.1f}%")

    print(f"\nObvious 'punt' situations (P > 95%): {len(obvious_punt):,}")
    if len(obvious_punt) > 0:
        punted = (obvious_punt['actual_decision'] == 'punt').mean() * 100
        print(f"  Teams actually punted: {punted:.1f}%")

    print(f"\nObvious 'FG' situations (P > 95%): {len(obvious_fg):,}")
    if len(obvious_fg) > 0:
        kicked_fg = (obvious_fg['actual_decision'] == 'field_goal').mean() * 100
        print(f"  Teams actually kicked FG: {kicked_fg:.1f}%")

    # Close calls
    close_calls = analysis_df[
        (analysis_df['prob_go_best'] < 0.6) &
        (analysis_df['prob_punt_best'] < 0.6) &
        (analysis_df['prob_fg_best'] < 0.6)
    ]
    print(f"\nClose calls (no action > 60% probable): {len(close_calls):,}")

    # Table 5: By field position and distance
    print("\n--- Table 5: Optimal Decision by Field Position and Distance ---")

    # Create buckets
    analysis_df['field_pos_bucket'] = pd.cut(
        analysis_df['field_pos'],
        bins=[0, 10, 20, 35, 50, 100],
        labels=['Red zone (1-10)', 'Scoring range (11-20)', 'FG range (21-35)',
                'Midfield (36-50)', 'Own territory (51+)']
    )

    analysis_df['ytg_bucket'] = pd.cut(
        analysis_df['yards_to_go'],
        bins=[0, 1, 3, 5, 10, 100],
        labels=['4th & inches', '4th & short (1-3)', '4th & medium (4-5)',
                '4th & long (6-10)', '4th & forever (10+)']
    )

    print("\nOptimal decision rates by field position:")
    for fp_bucket in analysis_df['field_pos_bucket'].cat.categories:
        subset = analysis_df[analysis_df['field_pos_bucket'] == fp_bucket]
        if len(subset) > 0:
            go_rate = (subset['optimal_action'] == 'go_for_it').mean() * 100
            punt_rate = (subset['optimal_action'] == 'punt').mean() * 100
            fg_rate = (subset['optimal_action'] == 'field_goal').mean() * 100
            print(f"  {fp_bucket}: Go {go_rate:.0f}% | Punt {punt_rate:.0f}% | FG {fg_rate:.0f}%")

    print("\nOptimal decision rates by yards to go:")
    for ytg_bucket in analysis_df['ytg_bucket'].cat.categories:
        subset = analysis_df[analysis_df['ytg_bucket'] == ytg_bucket]
        if len(subset) > 0:
            go_rate = (subset['optimal_action'] == 'go_for_it').mean() * 100
            punt_rate = (subset['optimal_action'] == 'punt').mean() * 100
            fg_rate = (subset['optimal_action'] == 'field_goal').mean() * 100
            print(f"  {ytg_bucket}: Go {go_rate:.0f}% | Punt {punt_rate:.0f}% | FG {fg_rate:.0f}%")

    # Table 6: Time trends
    if 'season' in analysis_df.columns:
        print("\n--- Table 6: Time Trends (Has the NFL gotten less conservative?) ---")

        seasonal = analysis_df.groupby('season').agg({
            'actual_decision': lambda x: (x == 'go_for_it').mean(),
            'optimal_action': lambda x: (x == 'go_for_it').mean(),
            'is_optimal': 'mean',
            'wp_lost': 'mean'
        }).round(3)

        seasonal.columns = ['Actual Go Rate', 'Optimal Go Rate', 'Pct Optimal', 'Avg WP Lost']
        print(seasonal.to_string())

    # Save detailed results
    analysis_df.to_parquet(output_dir / 'decision_analysis_full.parquet')
    print(f"\nSaved full analysis to {output_dir / 'decision_analysis_full.parquet'}")

    return analysis_df


def analyze_with_chmm(
    sample_plays: pd.DataFrame,
    structured_models: dict,
    lambda_values: list = None
) -> pd.DataFrame:
    """
    Analyze sample plays using CHMM framework at various λ values.
    """
    if lambda_values is None:
        lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 100.0]

    chmm = CHMMDecisionAnalyzer(structured_models)

    results = []

    for idx, row in tqdm(sample_plays.iterrows(), total=len(sample_plays),
                         desc="CHMM Analysis"):
        try:
            state = GameState(
                field_pos=int(row['yardline_100']),
                yards_to_go=int(row['ydstogo']),
                score_diff=int(row['score_diff']),
                time_remaining=int(row['game_seconds_remaining']),
                timeout_diff=int(row.get('timeout_diff', 0))
            )

            result_row = {
                'play_idx': idx,
                'field_pos': state.field_pos,
                'yards_to_go': state.yards_to_go,
                'score_diff': state.score_diff,
                'actual_decision': row['actual_decision'],
            }

            for lam in lambda_values:
                decision = chmm.optimal_decision(state, lam)
                result_row[f'optimal_lambda_{lam}'] = decision['optimal_action']
                result_row[f'wp_go_lambda_{lam}'] = decision['action_values']['go_for_it']
                result_row[f'worst_model_go_lambda_{lam}'] = decision['worst_models']['go_for_it']

            results.append(result_row)

        except Exception as e:
            print(f"Error: {e}")
            continue

    return pd.DataFrame(results)


def estimate_implied_lambda(analysis_df: pd.DataFrame,
                            structured_models: dict) -> float:
    """
    Estimate the λ value that best explains actual coach decisions.
    """
    from scipy.optimize import minimize_scalar

    chmm = CHMMDecisionAnalyzer(structured_models)

    # Sample for speed
    sample = analysis_df.sample(min(500, len(analysis_df)), random_state=42)

    def loss(lam):
        correct = 0
        for idx, row in sample.iterrows():
            state = GameState(
                field_pos=int(row['field_pos']),
                yards_to_go=int(row['yards_to_go']),
                score_diff=int(row['score_diff']),
                time_remaining=int(row['time_remaining']),
            )
            decision = chmm.optimal_decision(state, lam)
            if decision['optimal_action'] == row['actual_decision']:
                correct += 1
        return -correct / len(sample)

    result = minimize_scalar(loss, bounds=(0.1, 10), method='bounded')
    return result.x


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'

    # Load data
    print("Loading 4th down data...")
    fourth_downs = pd.read_parquet(data_dir / 'fourth_downs.parquet')
    print(f"Loaded {len(fourth_downs):,} 4th down plays")

    # Load models
    print("Loading models...")
    models = load_all_models(models_dir)

    # Create analyzer
    analyzer = BayesianDecisionAnalyzer(models)

    # Analyze all plays
    print("\nAnalyzing all 4th down decisions...")
    analysis_df = analyze_fourth_down_dataset(fourth_downs, analyzer)

    # Generate summary tables
    analysis_df = generate_summary_tables(analysis_df, output_dir)

    # CHMM analysis on sample
    print("\n" + "="*80)
    print("CHMM MISSPECIFICATION-ROBUST ANALYSIS")
    print("="*80)

    # Check if structured models exist, otherwise create them
    structured_models_path = models_dir / 'structured_models.pkl'
    if not structured_models_path.exists():
        print("\nCreating structured models for CHMM analysis...")
        structured_models = create_structured_models(data_dir, models_dir, n_bootstrap=500)
    else:
        print("\nLoading existing structured models...")
        structured_models = load_structured_models(models_dir)

    # Sample plays for CHMM (it's slower)
    print("\nAnalyzing sample with CHMM framework...")
    sample = analysis_df.sample(min(1000, len(analysis_df)), random_state=42)
    chmm_results = analyze_with_chmm(sample, structured_models)

    print("\n--- CHMM Results: How optimal decision varies with lambda ---")
    lambda_cols = [c for c in chmm_results.columns if c.startswith('optimal_lambda')]
    for col in lambda_cols:
        lam = col.split('_')[-1]
        go_rate = (chmm_results[col] == 'go_for_it').mean() * 100
        print(f"  lambda = {lam}: Go for it rate = {go_rate:.1f}%")

    # Compare to actual
    actual_go_rate = (chmm_results['actual_decision'] == 'go_for_it').mean() * 100
    print(f"\n  Actual coaches: Go for it rate = {actual_go_rate:.1f}%")

    # Estimate implied lambda
    print("\nEstimating implied ambiguity aversion (lambda)...")
    try:
        implied_lambda = estimate_implied_lambda(analysis_df, structured_models)
        print(f"Implied lambda that best matches coach behavior: {implied_lambda:.2f}")
    except Exception as e:
        print(f"Could not estimate implied λ: {e}")

    # Save CHMM results
    chmm_results.to_parquet(output_dir / 'chmm_analysis.parquet')
    print(f"\nSaved CHMM analysis to {output_dir / 'chmm_analysis.parquet'}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
