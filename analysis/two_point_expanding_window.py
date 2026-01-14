"""
Two-Point Conversion Expanding Window Analysis

For each test year Y (2015-2024):
1. Train models on 2015 through Y-1 (information available at start of season Y)
2. Apply to year Y's PAT/2pt decisions to get "ex ante" optimal action
3. Compare to "ex post" optimal (full 2015-2024 model)
4. Compare to actual coach decisions

Key question: What % of decisions were knowable in real-time?

Note: Analysis starts from 2015 due to PAT rule change (kicked from 15-yard line).
First test year is 2016 (trained on 2015 only).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import sys
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))

from analysis.two_point_analysis import (
    TwoPointState,
    HierarchicalOffDefTwoPointModel,
    HierarchicalPATModel,
    TwoPointDecisionAnalyzer,
    prepare_two_point_data
)


def prepare_cumulative_2pt_data(pbp: pd.DataFrame, end_year: int, start_year: int = 2015):
    """Prepare cumulative training data from start_year through end_year (inclusive)."""
    mask = (pbp['season'] >= start_year) & (pbp['season'] <= end_year)
    cumulative_pbp = pbp[mask].copy()

    # Two-point conversions
    two_pt = cumulative_pbp[cumulative_pbp['two_point_attempt'] == 1].copy()
    two_pt = two_pt[two_pt['two_point_conv_result'].notna()].copy()

    # PAT attempts
    pat = cumulative_pbp[cumulative_pbp['extra_point_attempt'] == 1].copy()
    pat = pat[pat['extra_point_result'].notna()].copy()

    return two_pt, pat


def fit_2pt_models(two_pt_data: pd.DataFrame, pat_data: pd.DataFrame,
                   wp_model, n_samples: int = 2000):
    """Fit 2pt and PAT models on given data."""
    # 2pt model (off/def effects)
    two_pt_model = HierarchicalOffDefTwoPointModel(n_samples=n_samples)
    if len(two_pt_data) > 50:
        two_pt_model.fit(two_pt_data, min_attempts=3)
    else:
        print(f"  Warning: Only {len(two_pt_data)} 2pt attempts, using basic model")
        # Fall back to population-level model
        from analysis.two_point_analysis import TwoPointConversionModel
        two_pt_model = TwoPointConversionModel(n_samples=n_samples)
        two_pt_model.fit(two_pt_data)

    # PAT model
    pat_model = HierarchicalPATModel(n_samples=n_samples)
    if len(pat_data) > 100:
        pat_model.fit(pat_data)
    else:
        print(f"  Warning: Only {len(pat_data)} PAT attempts, using basic model")
        from analysis.two_point_analysis import PATModel
        pat_model = PATModel(n_samples=n_samples)
        pat_model.fit(pat_data)

    return two_pt_model, pat_model


def load_all_decisions_year(pbp: pd.DataFrame, year: int) -> pd.DataFrame:
    """Load all PAT/2pt decisions for a specific year."""
    df = pbp[pbp['season'] == year].copy()

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
    all_decisions['score_diff_pre_td'] = all_decisions['score_differential'] - 6
    all_decisions['time_remaining'] = all_decisions['game_seconds_remaining']

    return all_decisions


def analyze_decision(row: pd.Series, analyzer: TwoPointDecisionAnalyzer) -> dict:
    """Analyze a single PAT/2pt decision."""
    try:
        state = TwoPointState(
            score_diff_pre_td=int(row['score_diff_pre_td']),
            time_remaining=int(row['time_remaining']) if pd.notna(row['time_remaining']) else 1800,
            posteam=row.get('posteam'),
            defteam=row.get('defteam'),
            kicker_id=row.get('kicker_player_id')
        )

        analysis = analyzer.analyze(state)

        return {
            'optimal_action': analysis['optimal_action'],
            'wp_pat': analysis['wp_pat'],
            'wp_2pt': analysis['wp_2pt'],
            'wp_margin': abs(analysis['wp_margin']),
            'wp_margin_signed': analysis['wp_margin'],
            'prob_2pt_better': analysis['prob_2pt_better'],
        }
    except Exception as e:
        return None


def run_expanding_window_analysis(
    first_test_year: int = 2016,  # First year we can test (trained on 2015)
    last_test_year: int = 2024,
    n_samples: int = 2000
):
    """
    Run the full expanding window analysis for 2pt conversions.

    For each test year Y:
    - Train on 2015 through Y-1
    - Test on year Y
    """
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    pbp_path = data_dir / 'all_pbp_1999_2024.parquet'
    print(f"Loading from {pbp_path}...")
    pbp = pd.read_parquet(pbp_path)
    print(f"Total plays: {len(pbp):,}")

    # Filter to post-rule-change era (2015+)
    pbp = pbp[pbp['season'] >= 2015].copy()
    print(f"Post-2015 plays: {len(pbp):,}")

    # Load WP model (we use the same one for all years for consistency)
    print("\nLoading win probability model...")
    from models.bayesian_models import WinProbabilityModel
    wp_model = WinProbabilityModel()
    wp_model_path = models_dir / 'wp_model.pkl'
    wp_model = wp_model.load(wp_model_path)

    # Train ex post model (full sample)
    print("\n" + "=" * 80)
    print("TRAINING EX POST MODEL (FULL SAMPLE)")
    print("=" * 80)

    full_2pt, full_pat = prepare_cumulative_2pt_data(pbp, last_test_year, start_year=2015)
    print(f"Full sample 2pt attempts: {len(full_2pt):,}")
    print(f"Full sample PAT attempts: {len(full_pat):,}")

    ex_post_2pt_model, ex_post_pat_model = fit_2pt_models(full_2pt, full_pat, wp_model, n_samples)
    ex_post_analyzer = TwoPointDecisionAnalyzer(ex_post_2pt_model, ex_post_pat_model, wp_model)

    # Results storage
    all_results = []
    year_summaries = []

    # Run expanding window for each test year
    print("\n" + "=" * 80)
    print("RUNNING EXPANDING WINDOW ANALYSIS")
    print("=" * 80)

    for test_year in range(first_test_year, last_test_year + 1):
        print(f"\n--- Test Year: {test_year} (Training: 2015-{test_year-1}) ---")

        # Get test year decisions
        test_decisions = load_all_decisions_year(pbp, test_year)
        print(f"  Test decisions: {len(test_decisions):,}")

        if len(test_decisions) == 0:
            continue

        # Train ex ante model (through test_year - 1)
        train_2pt, train_pat = prepare_cumulative_2pt_data(pbp, test_year - 1, start_year=2015)
        print(f"  Training data: {len(train_2pt):,} 2pt, {len(train_pat):,} PAT")

        ex_ante_2pt_model, ex_ante_pat_model = fit_2pt_models(train_2pt, train_pat, wp_model, n_samples)
        ex_ante_analyzer = TwoPointDecisionAnalyzer(ex_ante_2pt_model, ex_ante_pat_model, wp_model)

        # Analyze each decision
        year_results = []

        for idx, row in tqdm(test_decisions.iterrows(), total=len(test_decisions), desc="  Analyzing"):
            # Ex ante analysis
            ex_ante = analyze_decision(row, ex_ante_analyzer)
            if ex_ante is None:
                continue

            # Ex post analysis
            ex_post = analyze_decision(row, ex_post_analyzer)
            if ex_post is None:
                continue

            result = {
                'game_id': row['game_id'],
                'play_id': row['play_id'],
                'season': test_year,
                'week': row.get('week'),
                'posteam': row.get('posteam'),
                'defteam': row.get('defteam'),
                'score_diff_pre_td': row['score_diff_pre_td'],
                'time_remaining': row['time_remaining'],
                'actual_decision': row['actual_decision'],
                'decision_success': row['decision_success'],

                # Ex ante (what was knowable at time of decision)
                'ex_ante_optimal': ex_ante['optimal_action'],
                'ex_ante_wp_margin': ex_ante['wp_margin'],
                'ex_ante_prob_2pt_better': ex_ante['prob_2pt_better'],

                # Ex post (with full hindsight)
                'ex_post_optimal': ex_post['optimal_action'],
                'ex_post_wp_margin': ex_post['wp_margin'],
                'ex_post_prob_2pt_better': ex_post['prob_2pt_better'],
            }

            # Derived metrics
            result['coach_agrees_ex_ante'] = result['actual_decision'] == result['ex_ante_optimal']
            result['coach_agrees_ex_post'] = result['actual_decision'] == result['ex_post_optimal']
            result['ex_ante_agrees_ex_post'] = result['ex_ante_optimal'] == result['ex_post_optimal']

            # Categorize
            if result['coach_agrees_ex_ante']:
                result['category'] = 'correct'
            elif result['ex_ante_optimal'] == result['ex_post_optimal']:
                # Both ex ante and ex post agree coach was wrong - clear mistake
                result['category'] = 'knowable_mistake'
            else:
                # Ex ante and ex post disagree - hindsight issue
                result['category'] = 'hindsight_only'

            year_results.append(result)

        # Year summary
        year_df = pd.DataFrame(year_results)

        n_total = len(year_df)
        n_correct = (year_df['coach_agrees_ex_ante']).sum()
        n_ex_ante_wrong = (~year_df['coach_agrees_ex_ante']).sum()
        n_knowable = (year_df['category'] == 'knowable_mistake').sum()
        n_hindsight = (year_df['category'] == 'hindsight_only').sum()
        agreement_rate = (year_df['ex_ante_agrees_ex_post']).mean()

        summary = {
            'year': test_year,
            'n_decisions': n_total,
            'n_correct_ex_ante': n_correct,
            'pct_correct_ex_ante': n_correct / n_total if n_total > 0 else 0,
            'n_ex_ante_wrong': n_ex_ante_wrong,
            'pct_ex_ante_wrong': n_ex_ante_wrong / n_total if n_total > 0 else 0,
            'n_knowable_mistake': n_knowable,
            'pct_knowable': n_knowable / n_total if n_total > 0 else 0,
            'n_hindsight_only': n_hindsight,
            'pct_hindsight': n_hindsight / n_total if n_total > 0 else 0,
            'ex_ante_ex_post_agreement': agreement_rate,
            'actual_2pt_rate': (year_df['actual_decision'] == 'two_point').mean(),
            'ex_ante_optimal_2pt_rate': (year_df['ex_ante_optimal'] == 'two_point').mean(),
            'ex_post_optimal_2pt_rate': (year_df['ex_post_optimal'] == 'two_point').mean(),
        }

        year_summaries.append(summary)
        all_results.extend(year_results)

        print(f"  Results: {n_correct/n_total:.1%} optimal (ex ante), "
              f"{agreement_rate:.1%} ex ante/ex post agreement")

    # Create DataFrames
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(year_summaries)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: EX ANTE VS EX POST")
    print("=" * 80)

    print("\n--- By Year ---")
    print(summary_df[['year', 'n_decisions', 'pct_correct_ex_ante',
                      'ex_ante_ex_post_agreement', 'actual_2pt_rate',
                      'ex_ante_optimal_2pt_rate']].to_string(index=False))

    # Overall statistics
    total_decisions = len(results_df)
    total_correct = results_df['coach_agrees_ex_ante'].sum()
    total_knowable = (results_df['category'] == 'knowable_mistake').sum()
    total_hindsight = (results_df['category'] == 'hindsight_only').sum()
    overall_agreement = results_df['ex_ante_agrees_ex_post'].mean()

    print(f"\n--- Overall Statistics ---")
    print(f"Total decisions analyzed: {total_decisions:,}")
    print(f"Coach optimal (ex ante): {total_correct:,} ({total_correct/total_decisions:.1%})")
    print(f"Knowable mistakes: {total_knowable:,} ({total_knowable/total_decisions:.1%})")
    print(f"Hindsight-only issues: {total_hindsight:,} ({total_hindsight/total_decisions:.1%})")
    print(f"Ex ante / ex post agreement: {overall_agreement:.1%}")

    # Trend analysis
    print("\n--- Trend Analysis: Is Decision Quality Improving? ---")
    years = summary_df['year'].values
    optimal_rates = summary_df['pct_correct_ex_ante'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, optimal_rates)

    print(f"Linear trend in optimal decision rate:")
    print(f"  Slope: {slope*100:+.2f} percentage points per year")
    print(f"  R-squared: {r_value**2:.3f}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        direction = "improving" if slope > 0 else "worsening"
        print(f"  => Coaches ARE {direction} at 2pt decisions (p < 0.05)")
    else:
        print(f"  => No statistically significant trend")

    # 2pt rate trend
    actual_2pt_rates = summary_df['actual_2pt_rate'].values
    slope2, _, r2, p2, _ = stats.linregress(years, actual_2pt_rates)

    print(f"\nLinear trend in actual 2pt attempt rate:")
    print(f"  Slope: {slope2*100:+.2f} percentage points per year")
    print(f"  R-squared: {r2**2:.3f}")
    print(f"  P-value: {p2:.4f}")

    # Save results
    results_df.to_parquet(output_dir / 'two_point_expanding_window.parquet')
    results_df.to_csv(output_dir / 'two_point_expanding_window.csv', index=False)
    summary_df.to_csv(output_dir / 'two_point_expanding_window_summary.csv', index=False)

    print(f"\nResults saved to {output_dir}")

    # Generate LaTeX table
    generate_latex_table(summary_df, output_dir)

    return results_df, summary_df


def generate_latex_table(summary_df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for the paper."""

    latex = r"""\begin{table}[H]
\centering
\caption{Two-Point Conversion Decision Quality Over Time (Expanding Window)}
\label{tab:2pt_expanding_window}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Year} & \textbf{N} & \textbf{Optimal\%} & \textbf{Ex Ante/Ex Post} & \textbf{Actual 2pt\%} & \textbf{Optimal 2pt\%} \\
\midrule
"""

    for _, row in summary_df.iterrows():
        latex += f"{int(row['year'])} & {int(row['n_decisions']):,} & "
        latex += f"{row['pct_correct_ex_ante']*100:.1f}\\% & "
        latex += f"{row['ex_ante_ex_post_agreement']*100:.1f}\\% & "
        latex += f"{row['actual_2pt_rate']*100:.1f}\\% & "
        latex += f"{row['ex_ante_optimal_2pt_rate']*100:.1f}\\% \\\\\n"

    # Add average row
    latex += r"\midrule" + "\n"
    latex += f"\\textit{{Average}} & "
    latex += f"{int(summary_df['n_decisions'].mean()):,} & "
    latex += f"{summary_df['pct_correct_ex_ante'].mean()*100:.1f}\\% & "
    latex += f"{summary_df['ex_ante_ex_post_agreement'].mean()*100:.1f}\\% & "
    latex += f"{summary_df['actual_2pt_rate'].mean()*100:.1f}\\% & "
    latex += f"{summary_df['ex_ante_optimal_2pt_rate'].mean()*100:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'two_point_expanding_window_table.tex', 'w') as f:
        f.write(latex)

    print(f"\nLaTeX table saved to {output_dir / 'two_point_expanding_window_table.tex'}")


if __name__ == "__main__":
    results_df, summary_df = run_expanding_window_analysis()
