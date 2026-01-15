"""
Era Comparison Analysis: Did coaches learn from analytics?

Compares 4th down decision-making across two eras:
- Era 1: 2006-2012 (post-Romer 2006, pre-analytics departments)
- Era 2: 2019-2024 (full analytics era)

Hypotheses:
H1: Go-for-it rate increased (coaches listened to analytics)
H2: Overall optimal rate stayed flat (they're not smarter)
H3: Conditional on going for it, optimal rate DECREASED (they over-corrected)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.bayesian_models import (
    ConversionModel, PuntModel, FieldGoalModel, WinProbabilityModel,
    fit_all_models, load_all_models
)
from analysis.decision_framework import BayesianDecisionAnalyzer, GameState


def fit_era_models(data_dir: Path, models_dir: Path, prefix: str = ""):
    """
    Fit Bayesian models for a specific era.

    Args:
        data_dir: Path to data files
        models_dir: Path to save models
        prefix: Prefix for model files (e.g., "historical_")
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load era-specific data
    attempts = pd.read_parquet(data_dir / f'{prefix}fourth_down_attempts.parquet')
    punts = pd.read_parquet(data_dir / f'{prefix}punts.parquet')
    fgs = pd.read_parquet(data_dir / f'{prefix}field_goals.parquet')
    all_plays = pd.read_parquet(data_dir / f'{prefix}cleaned_pbp.parquet')

    print(f"\n{'='*60}")
    print(f"FITTING MODELS FOR {prefix.upper() or 'MODERN'} ERA")
    print(f"{'='*60}")
    print(f"4th down attempts: {len(attempts):,}")
    print(f"Punts: {len(punts):,}")
    print(f"FG attempts: {len(fgs):,}")
    print(f"Total plays (for WP): {len(all_plays):,}")

    # Fit conversion model (non-hierarchical for fair comparison)
    print("\nFitting conversion model...")
    conversion_model = ConversionModel()
    conversion_model.fit(attempts, n_samples=2000)
    conversion_model.save(models_dir / f'{prefix}conversion_model.pkl')

    # Fit punt model
    print("\nFitting punt model...")
    punt_model = PuntModel()
    punt_model.fit(punts, n_samples=2000)
    punt_model.save(models_dir / f'{prefix}punt_model.pkl')

    # Fit FG model (non-hierarchical)
    print("\nFitting FG model...")
    fg_model = FieldGoalModel()
    fg_model.fit(fgs, n_samples=2000)
    fg_model.save(models_dir / f'{prefix}fg_model.pkl')

    # Fit WP model
    print("\nFitting WP model...")
    wp_model = WinProbabilityModel()
    wp_model.fit(all_plays, n_samples=2000)
    wp_model.save(models_dir / f'{prefix}wp_model.pkl')

    return {
        'conversion': conversion_model,
        'punt': punt_model,
        'fg': fg_model,
        'wp': wp_model
    }


def load_era_models(models_dir: Path, prefix: str = "") -> dict:
    """Load models for a specific era."""
    return {
        'conversion': ConversionModel().load(models_dir / f'{prefix}conversion_model.pkl'),
        'punt': PuntModel().load(models_dir / f'{prefix}punt_model.pkl'),
        'fg': FieldGoalModel().load(models_dir / f'{prefix}fg_model.pkl'),
        'wp': WinProbabilityModel().load(models_dir / f'{prefix}wp_model.pkl')
    }


def analyze_era_decisions(
    fourth_downs: pd.DataFrame,
    analyzer: BayesianDecisionAnalyzer,
    era_name: str
) -> pd.DataFrame:
    """
    Analyze all 4th down decisions for an era.
    """
    results = []

    for idx, row in tqdm(fourth_downs.iterrows(), total=len(fourth_downs),
                         desc=f"Analyzing {era_name}"):
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
                'season': row.get('season'),
                'field_pos': state.field_pos,
                'yards_to_go': state.yards_to_go,
                'score_diff': state.score_diff,
                'time_remaining': state.time_remaining,

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
            })
        except Exception as e:
            continue

    df = pd.DataFrame(results)
    df['wp_lost'] = df['wp_best'] - df['wp_actual']

    return df


def compute_era_metrics(analysis_df: pd.DataFrame) -> dict:
    """
    Compute key metrics for an era.
    """
    total = len(analysis_df)

    # Decision rates
    go_rate = (analysis_df['actual_decision'] == 'go_for_it').mean()
    punt_rate = (analysis_df['actual_decision'] == 'punt').mean()
    fg_rate = (analysis_df['actual_decision'] == 'field_goal').mean()

    # Overall optimal rate
    optimal_rate = analysis_df['is_optimal'].mean()

    # Optimal rate by decision type
    go_plays = analysis_df[analysis_df['actual_decision'] == 'go_for_it']
    punt_plays = analysis_df[analysis_df['actual_decision'] == 'punt']
    fg_plays = analysis_df[analysis_df['actual_decision'] == 'field_goal']

    optimal_given_go = go_plays['is_optimal'].mean() if len(go_plays) > 0 else np.nan
    optimal_given_punt = punt_plays['is_optimal'].mean() if len(punt_plays) > 0 else np.nan
    optimal_given_fg = fg_plays['is_optimal'].mean() if len(fg_plays) > 0 else np.nan

    # WP cost
    avg_wp_lost = analysis_df['wp_lost'].mean()
    total_wp_lost = analysis_df['wp_lost'].sum()

    # Obvious situations
    obvious_go = analysis_df[analysis_df['prob_go_best'] > 0.95]
    obvious_go_rate = (obvious_go['actual_decision'] == 'go_for_it').mean() if len(obvious_go) > 0 else np.nan

    obvious_punt = analysis_df[analysis_df['prob_punt_best'] > 0.95]
    obvious_punt_rate = (obvious_punt['actual_decision'] == 'punt').mean() if len(obvious_punt) > 0 else np.nan

    # Over-aggressive: went for it when shouldn't have
    go_when_suboptimal = go_plays[~go_plays['is_optimal']]
    over_aggressive_rate = len(go_when_suboptimal) / total if len(go_plays) > 0 else np.nan

    # Under-aggressive: punted/kicked when should have gone for it
    should_go = analysis_df[analysis_df['optimal_action'] == 'go_for_it']
    under_aggressive_rate = (should_go['actual_decision'] != 'go_for_it').mean() if len(should_go) > 0 else np.nan

    return {
        'n_plays': total,
        'go_rate': go_rate,
        'punt_rate': punt_rate,
        'fg_rate': fg_rate,
        'optimal_rate': optimal_rate,
        'optimal_given_go': optimal_given_go,
        'optimal_given_punt': optimal_given_punt,
        'optimal_given_fg': optimal_given_fg,
        'avg_wp_lost': avg_wp_lost,
        'total_wp_lost': total_wp_lost,
        'obvious_go_compliance': obvious_go_rate,
        'obvious_punt_compliance': obvious_punt_rate,
        'over_aggressive_rate': over_aggressive_rate,
        'under_aggressive_rate': under_aggressive_rate,
        'n_obvious_go': len(obvious_go),
        'n_obvious_punt': len(obvious_punt),
    }


def compute_metrics_by_distance(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics broken down by yards to go.
    """
    # Create distance buckets
    analysis_df = analysis_df.copy()
    analysis_df['distance_bucket'] = pd.cut(
        analysis_df['yards_to_go'],
        bins=[0, 1, 3, 6, 15],
        labels=['4th & 1', '4th & 2-3', '4th & 4-6', '4th & 7+']
    )

    results = []
    for bucket in analysis_df['distance_bucket'].dropna().unique():
        subset = analysis_df[analysis_df['distance_bucket'] == bucket]
        metrics = compute_era_metrics(subset)
        metrics['distance'] = bucket
        results.append(metrics)

    return pd.DataFrame(results)


def run_era_comparison():
    """
    Main function to run the full era comparison analysis.
    """
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # STEP 1: Fit models for historical era
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: FITTING HISTORICAL ERA MODELS (2006-2012)")
    print("="*80)

    historical_models_exist = (models_dir / 'historical_conversion_model.pkl').exists()

    if not historical_models_exist:
        historical_models = fit_era_models(data_dir, models_dir, prefix="historical_")
    else:
        print("Loading existing historical models...")
        historical_models = load_era_models(models_dir, prefix="historical_")

    # ========================================
    # STEP 2: Load/use modern era models
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: LOADING MODERN ERA MODELS (2019-2024)")
    print("="*80)

    # Use non-hierarchical for fair comparison
    modern_models_exist = (models_dir / 'conversion_model.pkl').exists()

    if not modern_models_exist:
        modern_models = fit_era_models(data_dir, models_dir, prefix="")
    else:
        print("Loading existing modern models...")
        modern_models = load_era_models(models_dir, prefix="")

    # ========================================
    # STEP 3: Analyze historical 4th downs
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: ANALYZING HISTORICAL 4TH DOWN DECISIONS")
    print("="*80)

    historical_fourth_downs = pd.read_parquet(data_dir / 'historical_fourth_downs.parquet')
    historical_analyzer = BayesianDecisionAnalyzer(historical_models)
    historical_analysis = analyze_era_decisions(
        historical_fourth_downs, historical_analyzer, "Era 1 (2006-2012)"
    )
    historical_analysis['era'] = 'Era 1 (2006-2012)'

    # ========================================
    # STEP 4: Analyze modern 4th downs
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: ANALYZING MODERN 4TH DOWN DECISIONS")
    print("="*80)

    modern_fourth_downs = pd.read_parquet(data_dir / 'fourth_downs.parquet')
    modern_analyzer = BayesianDecisionAnalyzer(modern_models)
    modern_analysis = analyze_era_decisions(
        modern_fourth_downs, modern_analyzer, "Era 2 (2019-2024)"
    )
    modern_analysis['era'] = 'Era 2 (2019-2024)'

    # ========================================
    # STEP 5: Compute and compare metrics
    # ========================================
    print("\n" + "="*80)
    print("STEP 5: ERA COMPARISON RESULTS")
    print("="*80)

    historical_metrics = compute_era_metrics(historical_analysis)
    modern_metrics = compute_era_metrics(modern_analysis)

    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': [
            'Total 4th downs',
            'Go-for-it rate',
            'Punt rate',
            'FG rate',
            'Overall optimal rate',
            'Optimal rate | went for it',
            'Optimal rate | punted',
            'Optimal rate | kicked FG',
            'Avg WP lost per decision',
            'Compliance in obvious go situations',
            'Compliance in obvious punt situations',
            'Over-aggressive rate (went for it when suboptimal)',
            'Under-aggressive rate (didnt go when should have)',
        ],
        'Era 1 (2006-2012)': [
            f"{historical_metrics['n_plays']:,}",
            f"{historical_metrics['go_rate']:.1%}",
            f"{historical_metrics['punt_rate']:.1%}",
            f"{historical_metrics['fg_rate']:.1%}",
            f"{historical_metrics['optimal_rate']:.1%}",
            f"{historical_metrics['optimal_given_go']:.1%}",
            f"{historical_metrics['optimal_given_punt']:.1%}",
            f"{historical_metrics['optimal_given_fg']:.1%}",
            f"{historical_metrics['avg_wp_lost']*100:.3f} pp",
            f"{historical_metrics['obvious_go_compliance']:.1%}",
            f"{historical_metrics['obvious_punt_compliance']:.1%}",
            f"{historical_metrics['over_aggressive_rate']:.1%}",
            f"{historical_metrics['under_aggressive_rate']:.1%}",
        ],
        'Era 2 (2019-2024)': [
            f"{modern_metrics['n_plays']:,}",
            f"{modern_metrics['go_rate']:.1%}",
            f"{modern_metrics['punt_rate']:.1%}",
            f"{modern_metrics['fg_rate']:.1%}",
            f"{modern_metrics['optimal_rate']:.1%}",
            f"{modern_metrics['optimal_given_go']:.1%}",
            f"{modern_metrics['optimal_given_punt']:.1%}",
            f"{modern_metrics['optimal_given_fg']:.1%}",
            f"{modern_metrics['avg_wp_lost']*100:.3f} pp",
            f"{modern_metrics['obvious_go_compliance']:.1%}",
            f"{modern_metrics['obvious_punt_compliance']:.1%}",
            f"{modern_metrics['over_aggressive_rate']:.1%}",
            f"{modern_metrics['under_aggressive_rate']:.1%}",
        ],
    })

    # Print results
    print("\n" + "="*80)
    print("ERA COMPARISON: KEY METRICS")
    print("="*80)
    print(comparison.to_string(index=False))

    # Test hypotheses
    print("\n" + "="*80)
    print("HYPOTHESIS TESTS")
    print("="*80)

    h1_result = modern_metrics['go_rate'] > historical_metrics['go_rate']
    h1_diff = modern_metrics['go_rate'] - historical_metrics['go_rate']
    print(f"\nH1: Go-for-it rate increased")
    print(f"    Era 1: {historical_metrics['go_rate']:.1%} -> Era 2: {modern_metrics['go_rate']:.1%}")
    print(f"    Change: {h1_diff:+.1%}")
    print(f"    Result: {'SUPPORTED' if h1_result else 'NOT SUPPORTED'}")

    h2_diff = modern_metrics['optimal_rate'] - historical_metrics['optimal_rate']
    h2_result = abs(h2_diff) < 0.05  # within 5pp is "flat"
    print(f"\nH2: Overall optimal rate stayed flat")
    print(f"    Era 1: {historical_metrics['optimal_rate']:.1%} -> Era 2: {modern_metrics['optimal_rate']:.1%}")
    print(f"    Change: {h2_diff:+.1%}")
    print(f"    Result: {'SUPPORTED' if h2_result else 'NOT SUPPORTED'}")

    h3_result = modern_metrics['optimal_given_go'] < historical_metrics['optimal_given_go']
    h3_diff = modern_metrics['optimal_given_go'] - historical_metrics['optimal_given_go']
    print(f"\nH3: Conditional on going for it, optimal rate DECREASED")
    print(f"    Era 1: {historical_metrics['optimal_given_go']:.1%} -> Era 2: {modern_metrics['optimal_given_go']:.1%}")
    print(f"    Change: {h3_diff:+.1%}")
    print(f"    Result: {'SUPPORTED' if h3_result else 'NOT SUPPORTED'}")

    # Additional insight: are they going for it in the RIGHT situations now?
    print("\n" + "-"*80)
    print("ADDITIONAL ANALYSIS")
    print("-"*80)

    print(f"\nObvious go situations (>95% prob best):")
    print(f"    Era 1: {historical_metrics['n_obvious_go']:,} situations, {historical_metrics['obvious_go_compliance']:.1%} compliance")
    print(f"    Era 2: {modern_metrics['n_obvious_go']:,} situations, {modern_metrics['obvious_go_compliance']:.1%} compliance")

    print(f"\nOver-aggressive decisions (went for it when suboptimal):")
    print(f"    Era 1: {historical_metrics['over_aggressive_rate']:.1%} of all 4th downs")
    print(f"    Era 2: {modern_metrics['over_aggressive_rate']:.1%} of all 4th downs")

    print(f"\nUnder-aggressive decisions (didnt go when should have):")
    print(f"    Era 1: {historical_metrics['under_aggressive_rate']:.1%} of situations where go was optimal")
    print(f"    Era 2: {modern_metrics['under_aggressive_rate']:.1%} of situations where go was optimal")

    # By distance breakdown
    print("\n" + "="*80)
    print("BREAKDOWN BY YARDS TO GO")
    print("="*80)

    hist_by_dist = compute_metrics_by_distance(historical_analysis)
    mod_by_dist = compute_metrics_by_distance(modern_analysis)

    print("\nGo-for-it rate by distance:")
    print(f"{'Distance':<15} {'Era 1':>12} {'Era 2':>12} {'Change':>12}")
    print("-"*55)
    for dist in ['4th & 1', '4th & 2-3', '4th & 4-6', '4th & 7+']:
        h_row = hist_by_dist[hist_by_dist['distance'] == dist]
        m_row = mod_by_dist[mod_by_dist['distance'] == dist]
        if len(h_row) > 0 and len(m_row) > 0:
            h_rate = h_row['go_rate'].iloc[0]
            m_rate = m_row['go_rate'].iloc[0]
            print(f"{dist:<15} {h_rate:>11.1%} {m_rate:>11.1%} {m_rate - h_rate:>+11.1%}")

    print("\nOptimal rate given went for it, by distance:")
    print(f"{'Distance':<15} {'Era 1':>12} {'Era 2':>12} {'Change':>12}")
    print("-"*55)
    for dist in ['4th & 1', '4th & 2-3', '4th & 4-6', '4th & 7+']:
        h_row = hist_by_dist[hist_by_dist['distance'] == dist]
        m_row = mod_by_dist[mod_by_dist['distance'] == dist]
        if len(h_row) > 0 and len(m_row) > 0:
            h_rate = h_row['optimal_given_go'].iloc[0]
            m_rate = m_row['optimal_given_go'].iloc[0]
            if pd.notna(h_rate) and pd.notna(m_rate):
                print(f"{dist:<15} {h_rate:>11.1%} {m_rate:>11.1%} {m_rate - h_rate:>+11.1%}")

    # Save results
    comparison.to_csv(output_dir / 'era_comparison.csv', index=False)

    # Save detailed metrics
    all_metrics = pd.DataFrame([
        {**historical_metrics, 'era': 'Era 1 (2006-2012)'},
        {**modern_metrics, 'era': 'Era 2 (2019-2024)'}
    ])
    all_metrics.to_csv(output_dir / 'era_metrics_detailed.csv', index=False)

    # Combine analyses and save
    combined = pd.concat([historical_analysis, modern_analysis], ignore_index=True)
    combined.to_parquet(output_dir / 'era_comparison_full.parquet')

    print(f"\n\nResults saved to {output_dir}")
    print("  - era_comparison.csv")
    print("  - era_metrics_detailed.csv")
    print("  - era_comparison_full.parquet")

    return comparison, historical_metrics, modern_metrics


if __name__ == "__main__":
    comparison, hist, mod = run_era_comparison()
