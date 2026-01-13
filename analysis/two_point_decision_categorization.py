"""
Two-Point Conversion Decision Categorization Analysis

Analogous to fourth down analysis, this evaluates PAT vs 2-point decisions:
1. How many decisions are optimal according to the model?
2. Are coaches learning over time?
3. What are the costliest mistakes?

Categories (same as fourth down):
- "Close call": Decision margin < 2% WP (reasonable disagreement)
- "Moderate": Decision margin 2-5% WP
- "Clear": Decision margin > 5% WP (should be obvious)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Tuple
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from analysis.two_point_analysis import (
    TwoPointState,
    HierarchicalOffDefTwoPointModel,
    HierarchicalPATModel,
    TwoPointDecisionAnalyzer,
    prepare_two_point_data
)


def load_all_pat_2pt_decisions(data_path: Path, start_year: int = 2015) -> pd.DataFrame:
    """
    Load ALL post-touchdown extra point decisions (both PAT attempts and 2pt attempts).

    The key insight: Every touchdown creates a decision point. We need to:
    1. Find all touchdowns
    2. Identify what decision the team made (PAT or 2pt)
    3. Compute what the optimal decision would have been

    Args:
        data_path: Path to all_pbp_1999_2024.parquet
        start_year: First year to include (2015 = post-PAT rule change)

    Returns:
        DataFrame with one row per PAT/2pt decision
    """
    print(f"Loading play-by-play data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Filter to post-rule-change era
    df = df[df['season'] >= start_year].copy()
    print(f"  Seasons {start_year}-{df['season'].max()}: {len(df):,} plays")

    # Find all extra point decisions (both PAT and 2pt attempts)
    # PAT attempts
    pat_plays = df[df['extra_point_attempt'] == 1].copy()
    pat_plays = pat_plays[pat_plays['extra_point_result'].notna()].copy()
    pat_plays['actual_decision'] = 'pat'
    pat_plays['decision_success'] = (pat_plays['extra_point_result'] == 'good').astype(int)

    # 2pt attempts
    two_pt_plays = df[df['two_point_attempt'] == 1].copy()
    two_pt_plays = two_pt_plays[two_pt_plays['two_point_conv_result'].notna()].copy()
    two_pt_plays['actual_decision'] = 'two_point'
    two_pt_plays['decision_success'] = (two_pt_plays['two_point_conv_result'] == 'success').astype(int)

    # Combine
    all_decisions = pd.concat([pat_plays, two_pt_plays], ignore_index=True)

    # Compute score differential BEFORE the touchdown
    # The score_differential column in nflfastR is the score difference at the start of the play
    # For a PAT/2pt play, this is AFTER the TD but BEFORE the extra point
    # So score_diff_pre_td = score_differential - 6
    all_decisions['score_diff_pre_td'] = all_decisions['score_differential'] - 6

    # Time remaining in seconds
    # game_seconds_remaining is already in the data
    all_decisions['time_remaining'] = all_decisions['game_seconds_remaining']

    # Sort by game and time
    all_decisions = all_decisions.sort_values(['game_id', 'play_id']).reset_index(drop=True)

    print(f"\nTotal extra point decisions found:")
    print(f"  PAT attempts: {len(pat_plays):,}")
    print(f"  2pt attempts: {len(two_pt_plays):,}")
    print(f"  Total: {len(all_decisions):,}")

    # Distribution by season
    print("\nBy season:")
    season_counts = all_decisions.groupby(['season', 'actual_decision']).size().unstack(fill_value=0)
    season_counts['two_pt_rate'] = season_counts['two_point'] / (season_counts['pat'] + season_counts['two_point'])
    print(season_counts.to_string())

    return all_decisions


def analyze_all_decisions(decisions: pd.DataFrame,
                         two_pt_model,
                         pat_model,
                         wp_model) -> pd.DataFrame:
    """
    For each decision, compute optimal action and categorize.

    Args:
        decisions: DataFrame from load_all_pat_2pt_decisions
        two_pt_model: Trained 2pt conversion model
        pat_model: Trained PAT model
        wp_model: Win probability model

    Returns:
        DataFrame with optimal action and categorization for each decision
    """
    analyzer = TwoPointDecisionAnalyzer(two_pt_model, pat_model, wp_model)

    results = []
    total = len(decisions)

    print(f"\nAnalyzing {total:,} decisions...")

    for idx, row in decisions.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx:,}/{total:,} ({idx/total:.1%})")

        # Create state
        state = TwoPointState(
            score_diff_pre_td=int(row['score_diff_pre_td']),
            time_remaining=int(row['time_remaining']) if pd.notna(row['time_remaining']) else 1800,
            posteam=row.get('posteam'),
            defteam=row.get('defteam'),
            kicker_id=row.get('kicker_player_id')
        )

        # Get model recommendation
        try:
            analysis = analyzer.analyze(state)

            results.append({
                'game_id': row['game_id'],
                'play_id': row['play_id'],
                'season': row['season'],
                'week': row.get('week'),
                'posteam': row.get('posteam'),
                'defteam': row.get('defteam'),
                'score_diff_pre_td': state.score_diff_pre_td,
                'time_remaining': state.time_remaining,
                'actual_decision': row['actual_decision'],
                'decision_success': row['decision_success'],
                'optimal_decision': analysis['optimal_action'],
                'wp_pat': analysis['wp_pat'],
                'wp_2pt': analysis['wp_2pt'],
                'wp_margin': abs(analysis['wp_margin']),  # Always positive
                'wp_margin_signed': analysis['wp_margin'],  # Positive = 2pt better
                'prob_2pt_better': analysis['prob_2pt_better'],
                'p_pat': analysis['p_pat'],
                'p_2pt': analysis['p_2pt'],
            })
        except Exception as e:
            print(f"  Error on row {idx}: {e}")
            continue

    result_df = pd.DataFrame(results)

    # Categorize decisions
    result_df = categorize_decisions(result_df)

    return result_df


def categorize_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize decisions based on decision margin.

    Categories:
    - Close call: margin < 2% WP
    - Moderate: margin 2-5% WP
    - Clear: margin > 5% WP
    """
    df = df.copy()

    # Was the coach's decision correct?
    df['coach_correct'] = df['actual_decision'] == df['optimal_decision']

    # Decision clarity based on margin
    df['decision_clarity'] = pd.cut(
        df['wp_margin'],
        bins=[0, 0.02, 0.05, 1.0],
        labels=['close_call', 'moderate', 'clear'],
        include_lowest=True
    )

    # Confidence category based on P(2pt better)
    # Transform to "confidence in optimal" = max(P(2pt better), 1 - P(2pt better))
    df['confidence'] = df['prob_2pt_better'].apply(lambda p: max(p, 1-p))
    df['confidence_category'] = pd.cut(
        df['confidence'],
        bins=[0.5, 0.6, 0.8, 0.95, 1.0],
        labels=['toss_up', 'lean', 'clear', 'obvious'],
        include_lowest=True
    )

    # Mistake type (only if coach was wrong)
    df['mistake_type'] = 'correct'

    df.loc[
        ~df['coach_correct'] & (df['decision_clarity'] == 'close_call'),
        'mistake_type'
    ] = 'difference_of_opinion'

    df.loc[
        ~df['coach_correct'] & (df['decision_clarity'] == 'moderate'),
        'mistake_type'
    ] = 'questionable'

    df.loc[
        ~df['coach_correct'] & (df['decision_clarity'] == 'clear'),
        'mistake_type'
    ] = 'clear_mistake'

    # Egregious: margin > 10%
    df.loc[
        ~df['coach_correct'] & (df['wp_margin'] > 0.10),
        'mistake_type'
    ] = 'egregious'

    # WP cost of mistake
    df['wp_cost'] = df.apply(
        lambda row: row['wp_margin'] if not row['coach_correct'] else 0,
        axis=1
    )

    return df


def generate_summary_statistics(df: pd.DataFrame):
    """Generate summary statistics for the categorization."""
    print("\n" + "="*80)
    print("TWO-POINT CONVERSION DECISION CATEGORIZATION SUMMARY")
    print("="*80)

    total = len(df)
    total_pat = (df['actual_decision'] == 'pat').sum()
    total_2pt = (df['actual_decision'] == 'two_point').sum()

    print(f"\nTotal decisions analyzed: {total:,}")
    print(f"  PAT attempts: {total_pat:,} ({total_pat/total:.1%})")
    print(f"  2pt attempts: {total_2pt:,} ({total_2pt/total:.1%})")

    # Success rates
    pat_success = df[df['actual_decision'] == 'pat']['decision_success'].mean()
    two_pt_success = df[df['actual_decision'] == 'two_point']['decision_success'].mean()
    print(f"\nSuccess rates:")
    print(f"  PAT: {pat_success:.1%}")
    print(f"  2pt: {two_pt_success:.1%}")

    # By decision clarity
    print("\n--- Distribution by Decision Clarity ---")
    clarity_dist = df['decision_clarity'].value_counts()
    for cat in ['close_call', 'moderate', 'clear']:
        count = clarity_dist.get(cat, 0)
        print(f"  {cat:12s}: {count:6,} ({count/total:5.1%})")

    # By mistake type
    print("\n--- Distribution by Mistake Type ---")
    mistake_dist = df['mistake_type'].value_counts()
    for cat in ['correct', 'difference_of_opinion', 'questionable', 'clear_mistake', 'egregious']:
        count = mistake_dist.get(cat, 0)
        print(f"  {cat:22s}: {count:6,} ({count/total:5.1%})")

    # THE KEY BREAKDOWN
    print("\n" + "="*80)
    print("THE KEY BREAKDOWN: COACH OPTIMALITY")
    print("="*80)

    total_correct = df['coach_correct'].sum()
    total_wrong = (~df['coach_correct']).sum()

    debatable = len(df[df['mistake_type'] == 'difference_of_opinion'])
    questionable = len(df[df['mistake_type'] == 'questionable'])
    clear_mistakes = len(df[df['mistake_type'] == 'clear_mistake'])
    egregious = len(df[df['mistake_type'] == 'egregious'])

    print(f"""
Total PAT/2pt decisions analyzed: {total:,}
Coaches made optimal decision: {total_correct:,} ({total_correct/total:.1%})
Coaches deviated from optimal: {total_wrong:,} ({total_wrong/total:.1%})

Of the deviations:

  DEBATABLE (margin < 2% WP):
    {debatable:,} plays ({debatable/total:.1%} of all, {debatable/total_wrong:.1%} of deviations)
    These are "differences of opinion" - essentially a coin flip.

  QUESTIONABLE (margin 2-5% WP):
    {questionable:,} plays ({questionable/total:.1%} of all, {questionable/total_wrong:.1%} of deviations)
    The model has a preference, but it's not overwhelming.

  CLEAR MISTAKES (margin 5-10% WP):
    {clear_mistakes:,} plays ({clear_mistakes/total:.1%} of all, {clear_mistakes/total_wrong:.1%} of deviations)
    The model strongly recommends a different action.

  EGREGIOUS (margin > 10% WP):
    {egregious:,} plays ({egregious/total:.1%} of all, {egregious/total_wrong:.1%} of deviations)
    Truly costly - the optimal choice was obvious.
""")

    # Total WP cost
    if total_wrong > 0:
        total_wp_cost = df['wp_cost'].sum()
        print(f"Total WP cost from suboptimal decisions: {total_wp_cost:.1f} percentage points")
        print(f"Average WP cost per suboptimal decision: {total_wp_cost/total_wrong:.2f} pp")


def analyze_trends_over_time(df: pd.DataFrame):
    """Analyze how decision quality has changed over time."""
    print("\n" + "="*80)
    print("TRENDS OVER TIME: ARE COACHES LEARNING?")
    print("="*80)

    by_season = df.groupby('season').agg({
        'coach_correct': 'mean',
        'actual_decision': lambda x: (x == 'two_point').mean(),
        'optimal_decision': lambda x: (x == 'two_point').mean(),
        'wp_cost': 'sum',
        'game_id': 'count'
    }).rename(columns={
        'coach_correct': 'optimal_rate',
        'actual_decision': 'actual_2pt_rate',
        'optimal_decision': 'optimal_2pt_rate',
        'game_id': 'n_decisions'
    })

    print("\n--- By Season ---")
    print(f"{'Season':<8} {'N':<7} {'Optimal%':<10} {'Actual 2pt%':<12} {'Optimal 2pt%':<12} {'WP Cost':<10}")
    print("-" * 60)

    for season, row in by_season.iterrows():
        print(f"{season:<8} {row['n_decisions']:<7.0f} {row['optimal_rate']*100:<10.1f} "
              f"{row['actual_2pt_rate']*100:<12.1f} {row['optimal_2pt_rate']*100:<12.1f} "
              f"{row['wp_cost']:<10.2f}")

    # Calculate trend
    from scipy import stats
    seasons = by_season.index.values
    optimal_rates = by_season['optimal_rate'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(seasons, optimal_rates)

    print(f"\n--- Trend Analysis ---")
    print(f"Linear trend in optimal decision rate:")
    print(f"  Slope: {slope*100:+.2f} percentage points per year")
    print(f"  R-squared: {r_value**2:.3f}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        if slope > 0:
            print("  => Coaches ARE improving at PAT/2pt decisions (statistically significant)")
        else:
            print("  => Coaches are getting WORSE at PAT/2pt decisions (statistically significant)")
    else:
        print("  => No statistically significant trend in decision quality")

    # 2pt rate trend
    actual_2pt_rates = by_season['actual_2pt_rate'].values
    slope2, _, r2, p2, _ = stats.linregress(seasons, actual_2pt_rates)

    print(f"\nLinear trend in 2pt attempt rate:")
    print(f"  Slope: {slope2*100:+.2f} percentage points per year")
    print(f"  R-squared: {r2**2:.3f}")
    print(f"  P-value: {p2:.4f}")

    # Compare actual vs optimal 2pt rates
    print("\n--- Actual vs Optimal 2pt Rates ---")
    overall_actual = df['actual_decision'].apply(lambda x: x == 'two_point').mean()
    overall_optimal = df['optimal_decision'].apply(lambda x: x == 'two_point').mean()

    print(f"Overall actual 2pt rate: {overall_actual:.1%}")
    print(f"Overall optimal 2pt rate: {overall_optimal:.1%}")

    if overall_actual > overall_optimal:
        print(f"=> Coaches go for 2 {overall_actual - overall_optimal:.1%} MORE often than optimal")
    else:
        print(f"=> Coaches go for 2 {overall_optimal - overall_actual:.1%} LESS often than optimal")

    return by_season


def find_worst_decisions(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Find the costliest PAT/2pt decisions."""
    print("\n" + "="*80)
    print(f"THE {n} WORST PAT/2PT DECISIONS ({df['season'].min()}-{df['season'].max()})")
    print("="*80)

    worst = df[~df['coach_correct']].nlargest(n, 'wp_cost').copy()

    for i, (_, row) in enumerate(worst.iterrows(), 1):
        print(f"\n{i}. {row['season']} Week {row.get('week', '?')}: {row['posteam']} vs {row['defteam']}")
        print(f"   Score diff pre-TD: {row['score_diff_pre_td']:+d}, Time remaining: {row['time_remaining']//60:.0f}:{row['time_remaining']%60:02.0f}")
        print(f"   Actual: {row['actual_decision'].upper()}, Optimal: {row['optimal_decision'].upper()}")
        print(f"   WP(PAT): {row['wp_pat']:.1%}, WP(2pt): {row['wp_2pt']:.1%}")
        print(f"   WP Cost: {row['wp_cost']:.2%} ({row['mistake_type']})")

    return worst


def save_results(df: pd.DataFrame, by_season: pd.DataFrame, output_dir: Path):
    """Save analysis results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    df.to_parquet(output_dir / 'two_point_decision_analysis.parquet')
    df.to_csv(output_dir / 'two_point_decision_analysis.csv', index=False)

    # Save by-season summary
    by_season.to_csv(output_dir / 'two_point_by_season.csv')

    # Save worst decisions
    worst = df[~df['coach_correct']].nlargest(20, 'wp_cost')
    worst.to_csv(output_dir / 'two_point_worst_decisions.csv', index=False)

    print(f"\nResults saved to {output_dir}")


def main():
    """Run the full two-point decision analysis."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    models_dir = base_dir / 'models'
    output_dir = base_dir / 'outputs' / 'tables'

    # Load decisions
    decisions = load_all_pat_2pt_decisions(
        data_dir / 'all_pbp_1999_2024.parquet',
        start_year=2015
    )

    # Load or train models
    print("\n--- Loading Models ---")

    # Try to load existing models
    off_def_model_path = models_dir / 'hierarchical_off_def_two_point_model.pkl'
    pat_model_path = models_dir / 'hierarchical_pat_model.pkl'
    wp_model_path = models_dir / 'wp_model.pkl'

    if off_def_model_path.exists():
        print(f"Loading 2pt model from {off_def_model_path}")
        two_pt_model = HierarchicalOffDefTwoPointModel.load(off_def_model_path)
    else:
        print("Training new 2pt off/def model...")
        two_pt_data, _ = prepare_two_point_data(data_dir / 'all_pbp_1999_2024.parquet', start_year=2015)
        two_pt_model = HierarchicalOffDefTwoPointModel()
        two_pt_model.fit(two_pt_data)
        two_pt_model.save(off_def_model_path)

    if pat_model_path.exists():
        print(f"Loading PAT model from {pat_model_path}")
        pat_model = HierarchicalPATModel.load(pat_model_path)
    else:
        print("Training new PAT model...")
        _, pat_data = prepare_two_point_data(data_dir / 'all_pbp_1999_2024.parquet', start_year=2015)
        pat_model = HierarchicalPATModel()
        pat_model.fit(pat_data)
        pat_model.save(pat_model_path)

    # Load win probability model
    print(f"Loading WP model from {wp_model_path}")
    from models.bayesian_models import WinProbabilityModel
    wp_model = WinProbabilityModel()
    wp_model = wp_model.load(wp_model_path)

    # Analyze all decisions
    results = analyze_all_decisions(decisions, two_pt_model, pat_model, wp_model)

    # Generate reports
    generate_summary_statistics(results)
    by_season = analyze_trends_over_time(results)
    find_worst_decisions(results)

    # Save results
    save_results(results, by_season, output_dir)

    return results


if __name__ == "__main__":
    results = main()
