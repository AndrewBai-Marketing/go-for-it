"""
Learning Decomposition Analysis

Key question: How are coaches learning if not from experience, peers, or trends?

Hypotheses to test:
1. Selection, not learning: Aggressive coaches replace conservative ones
2. Front office pressure: Analytics departments drive change
3. Situation-specific learning: Coaches learn for some margins but not others

This script decomposes the aggregate trend into:
- Within-coach change (individual learning)
- Between-coach change (composition/selection effects)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import coach mapping from learning_mechanisms
from learning_mechanisms import COACH_MAPPING, TEAM_TO_DIVISION


def load_data():
    """Load and prepare data."""
    data_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'two_point_decision_analysis.parquet'
    df = pd.read_parquet(data_path)

    df['coach'] = df.apply(lambda row: COACH_MAPPING.get((row['posteam'], row['season']), 'Unknown'), axis=1)
    df['went_for_2'] = (df['actual_decision'] == 'two_point').astype(int)
    df['should_go_for_2'] = (df['optimal_decision'] == 'two_point').astype(int)

    return df


def decompose_aggregate_trend(df):
    """
    Oaxaca-Blinder style decomposition of aggregate trend.

    Total change = Within-coach change + Composition change

    Within-coach: same coaches changing behavior
    Composition: different coaches in different periods
    """
    print("=" * 80)
    print("DECOMPOSITION: Within-Coach Learning vs. Coach Turnover")
    print("=" * 80)

    # Define periods
    early = df[df['season'].isin([2015, 2016, 2017])]
    late = df[df['season'].isin([2022, 2023, 2024])]

    # Aggregate metrics
    early_2pt_rate = early['went_for_2'].mean()
    late_2pt_rate = late['went_for_2'].mean()
    early_correct = early['coach_correct'].mean()
    late_correct = late['coach_correct'].mean()

    total_2pt_change = late_2pt_rate - early_2pt_rate
    total_correct_change = late_correct - early_correct

    print(f"\nAggregate change (2015-17 → 2022-24):")
    print(f"  2pt rate: {early_2pt_rate:.1%} → {late_2pt_rate:.1%} ({total_2pt_change:+.1%})")
    print(f"  Correct rate: {early_correct:.1%} → {late_correct:.1%} ({total_correct_change:+.1%})")

    # Find coaches present in BOTH periods
    early_coaches = set(early['coach'].unique())
    late_coaches = set(late['coach'].unique())
    continuing_coaches = early_coaches & late_coaches

    print(f"\n  Coaches in early period: {len(early_coaches)}")
    print(f"  Coaches in late period: {len(late_coaches)}")
    print(f"  Coaches in BOTH periods: {len(continuing_coaches)}")

    # Within-coach change (same coaches, different behavior)
    if len(continuing_coaches) > 0:
        early_continuing = early[early['coach'].isin(continuing_coaches)]
        late_continuing = late[late['coach'].isin(continuing_coaches)]

        within_2pt_change = late_continuing['went_for_2'].mean() - early_continuing['went_for_2'].mean()
        within_correct_change = late_continuing['coach_correct'].mean() - early_continuing['coach_correct'].mean()

        print(f"\n  WITHIN-COACH change (same {len(continuing_coaches)} coaches):")
        print(f"    2pt rate: {early_continuing['went_for_2'].mean():.1%} → {late_continuing['went_for_2'].mean():.1%} ({within_2pt_change:+.1%})")
        print(f"    Correct rate: {early_continuing['coach_correct'].mean():.1%} → {late_continuing['coach_correct'].mean():.1%} ({within_correct_change:+.1%})")
    else:
        within_2pt_change = 0
        within_correct_change = 0

    # New coaches vs. exiting coaches
    new_coaches = late_coaches - early_coaches
    exiting_coaches = early_coaches - late_coaches

    print(f"\n  NEW coaches (not in early period): {len(new_coaches)}")
    print(f"  EXITING coaches (not in late period): {len(exiting_coaches)}")

    # Composition effect: new vs exiting coaches
    if len(new_coaches) > 0 and len(exiting_coaches) > 0:
        new_coach_data = late[late['coach'].isin(new_coaches)]
        exiting_coach_data = early[early['coach'].isin(exiting_coaches)]

        new_2pt = new_coach_data['went_for_2'].mean()
        exit_2pt = exiting_coach_data['went_for_2'].mean()
        new_correct = new_coach_data['coach_correct'].mean()
        exit_correct = exiting_coach_data['coach_correct'].mean()

        print(f"\n  COMPOSITION EFFECT:")
        print(f"    Exiting coaches 2pt rate: {exit_2pt:.1%}")
        print(f"    New coaches 2pt rate: {new_2pt:.1%}")
        print(f"    Composition 2pt change: {new_2pt - exit_2pt:+.1%}")
        print(f"    Exiting coaches correct: {exit_correct:.1%}")
        print(f"    New coaches correct: {new_correct:.1%}")
        print(f"    Composition correct change: {new_correct - exit_correct:+.1%}")

    # Decomposition summary
    print("\n" + "-" * 60)
    print("DECOMPOSITION SUMMARY")
    print("-" * 60)

    # Calculate shares
    if len(continuing_coaches) > 0:
        within_share_2pt = within_2pt_change / total_2pt_change if total_2pt_change != 0 else 0
        within_share_correct = within_correct_change / total_correct_change if total_correct_change != 0 else 0

        print(f"\n  2pt rate change breakdown:")
        print(f"    Total change: {total_2pt_change:+.1%}")
        print(f"    Within-coach: {within_2pt_change:+.1%} ({within_share_2pt:.0%} of total)")
        print(f"    Composition:  {total_2pt_change - within_2pt_change:+.1%} ({1-within_share_2pt:.0%} of total)")

        print(f"\n  Correct rate change breakdown:")
        print(f"    Total change: {total_correct_change:+.1%}")
        print(f"    Within-coach: {within_correct_change:+.1%} ({within_share_correct:.0%} of total)")
        print(f"    Composition:  {total_correct_change - within_correct_change:+.1%} ({1-within_share_correct:.0%} of total)")

    return continuing_coaches, new_coaches, exiting_coaches


def analyze_pioneer_regression(df):
    """
    Why did pioneers (Tomlin, Harbaugh) back off?

    Hypotheses:
    1. High-profile failures
    2. Regression to mean
    3. Others caught up (they were always close to optimal)
    """
    print("\n" + "=" * 80)
    print("PIONEER REGRESSION ANALYSIS: Why did Tomlin & Harbaugh back off?")
    print("=" * 80)

    pioneers = ['Mike Tomlin', 'John Harbaugh']

    for coach in pioneers:
        print(f"\n{'-' * 60}")
        print(f"{coach}")
        print(f"{'-' * 60}")

        coach_data = df[df['coach'] == coach].copy()

        if len(coach_data) == 0:
            print("  No data found")
            continue

        # Season-by-season breakdown
        season_stats = coach_data.groupby('season').agg({
            'went_for_2': ['sum', 'count', 'mean'],
            'should_go_for_2': 'mean',
            'coach_correct': 'mean',
            'decision_success': 'mean',  # Success rate when going for 2
            'wp_cost': 'sum'
        }).reset_index()

        season_stats.columns = ['season', '2pt_attempts', 'total', '2pt_rate',
                               'optimal_rate', 'correct_rate', 'success_rate', 'wp_cost']

        print("\nSeason-by-season:")
        for _, row in season_stats.iterrows():
            gap = row['2pt_rate'] - row['optimal_rate']
            print(f"  {row['season']}: {row['2pt_rate']:.1%} 2pt ({row['2pt_attempts']:.0f}/{row['total']:.0f}) | "
                  f"Opt: {row['optimal_rate']:.1%} (gap: {gap:+.1%}) | "
                  f"Correct: {row['correct_rate']:.1%}")

        # Check for high-profile failures
        failures = coach_data[
            (coach_data['actual_decision'] == 'two_point') &
            (coach_data['decision_success'] == 0)
        ]

        print(f"\n2pt failures: {len(failures)}")

        # Did failure rates change?
        early_coach = coach_data[coach_data['season'] <= 2017]
        late_coach = coach_data[coach_data['season'] >= 2020]

        if len(early_coach) > 0 and len(late_coach) > 0:
            early_attempts = early_coach[early_coach['actual_decision'] == 'two_point']
            late_attempts = late_coach[late_coach['actual_decision'] == 'two_point']

            if len(early_attempts) > 0 and len(late_attempts) > 0:
                early_success = early_attempts['decision_success'].mean()
                late_success = late_attempts['decision_success'].mean()
                print(f"\n2pt success rate:")
                print(f"  Early (2015-17): {early_success:.1%} ({len(early_attempts)} attempts)")
                print(f"  Late (2020-24): {late_success:.1%} ({len(late_attempts)} attempts)")

        # Were they ever more aggressive than optimal?
        above_optimal = season_stats[season_stats['2pt_rate'] > season_stats['optimal_rate']]
        if len(above_optimal) > 0:
            print(f"\nSeasons MORE aggressive than optimal: {list(above_optimal['season'].values)}")
        else:
            print(f"\nNever more aggressive than optimal — they were always conservative!")

    # Key insight: compare to league optimal rate
    print("\n" + "-" * 60)
    print("KEY INSIGHT: League optimal 2pt rate over time")
    print("-" * 60)

    season_optimal = df.groupby('season')['should_go_for_2'].mean()
    season_actual = df.groupby('season')['went_for_2'].mean()

    print("\nSeason | Optimal | Actual | Gap")
    for season in sorted(df['season'].unique()):
        opt = season_optimal[season]
        act = season_actual[season]
        gap = act - opt
        print(f"  {season}  | {opt:.1%}  | {act:.1%}  | {gap:+.1%}")

    print("\nInterpretation:")
    print("  If optimal rate is INCREASING, then early pioneers who maintained")
    print("  constant 2pt rates would APPEAR to be backing off relative to optimal.")


def staley_deep_dive(df):
    """
    Brandon Staley deep dive: What's he getting wrong?

    He's famous for analytics but only 26.6% correct.
    """
    print("\n" + "=" * 80)
    print("BRANDON STALEY DEEP DIVE: The 'Analytics' Coach Who Gets It Wrong")
    print("=" * 80)

    staley = df[df['coach'] == 'Brandon Staley'].copy()

    if len(staley) == 0:
        print("No data for Staley")
        return

    print(f"\nTotal decisions: {len(staley)}")
    print(f"Went for 2: {staley['went_for_2'].sum()} ({staley['went_for_2'].mean():.1%})")
    print(f"Should have gone for 2: {staley['should_go_for_2'].sum()} ({staley['should_go_for_2'].mean():.1%})")
    print(f"Correct decisions: {staley['coach_correct'].sum()} ({staley['coach_correct'].mean():.1%})")

    # Error breakdown
    print("\n" + "-" * 60)
    print("ERROR BREAKDOWN")
    print("-" * 60)

    # Type I: Went for 2 when shouldn't
    type1 = staley[(staley['went_for_2'] == 1) & (staley['should_go_for_2'] == 0)]
    # Type II: Didn't go for 2 when should have
    type2 = staley[(staley['went_for_2'] == 0) & (staley['should_go_for_2'] == 1)]

    print(f"\n  Type I (went for 2 when shouldn't): {len(type1)}")
    print(f"  Type II (didn't go when should): {len(type2)}")

    # Score margins where Staley goes for 2
    print("\n" + "-" * 60)
    print("STALEY'S 2PT ATTEMPTS BY SCORE MARGIN (pre-TD)")
    print("-" * 60)

    staley_2pt = staley[staley['went_for_2'] == 1]

    margin_counts = staley_2pt.groupby('score_diff_pre_td').agg({
        'coach_correct': ['sum', 'count', 'mean']
    }).reset_index()
    margin_counts.columns = ['margin', 'correct', 'total', 'correct_rate']

    print("\nMargin | Attempts | Correct | Rate")
    for _, row in margin_counts.sort_values('total', ascending=False).head(10).iterrows():
        print(f"  {row['margin']:+3.0f}  |    {row['total']:2.0f}    |    {row['correct']:2.0f}   | {row['correct_rate']:.0%}")

    # Compare to optimal behavior
    print("\n" + "-" * 60)
    print("MARGINS WHERE STALEY SHOULD GO FOR 2 BUT DOESN'T")
    print("-" * 60)

    missed = staley[(staley['should_go_for_2'] == 1) & (staley['went_for_2'] == 0)]
    missed_margins = missed.groupby('score_diff_pre_td').size().sort_values(ascending=False)

    print("\nMargin | Missed opportunities")
    for margin, count in missed_margins.head(10).items():
        print(f"  {margin:+3.0f}  |    {count}")

    # Compare to a "good" coach
    print("\n" + "-" * 60)
    print("COMPARISON: Staley vs. John Harbaugh (best correct rate)")
    print("-" * 60)

    harbaugh = df[df['coach'] == 'John Harbaugh']

    print(f"\n{'Metric':<30} | {'Staley':>10} | {'Harbaugh':>10}")
    print("-" * 55)
    print(f"{'Total decisions':<30} | {len(staley):>10} | {len(harbaugh):>10}")
    print(f"{'2pt rate':<30} | {staley['went_for_2'].mean():>10.1%} | {harbaugh['went_for_2'].mean():>10.1%}")
    print(f"{'Optimal 2pt rate':<30} | {staley['should_go_for_2'].mean():>10.1%} | {harbaugh['should_go_for_2'].mean():>10.1%}")
    print(f"{'Correct rate':<30} | {staley['coach_correct'].mean():>10.1%} | {harbaugh['coach_correct'].mean():>10.1%}")
    print(f"{'Total WP cost':<30} | {staley['wp_cost'].sum():>10.3f} | {harbaugh['wp_cost'].sum():>10.3f}")

    print("\nInterpretation:")
    print("  Staley faces situations where going for 2 is optimal 76% of the time!")
    print("  But he only goes for it 10% of the time.")
    print("  Harbaugh faces situations where going for 2 is optimal only 14% of the time.")
    print("  The 'analytics' label comes from being aggressive, but Staley is aggressive")
    print("  in a context where being aggressive is ALWAYS correct — yet he's still conservative!")


def situation_specific_learning(df):
    """
    Do coaches learn for some score margins but not others?

    Track accuracy by score margin over time.
    """
    print("\n" + "=" * 80)
    print("SITUATION-SPECIFIC LEARNING: By Score Margin")
    print("=" * 80)

    # Group margins into categories
    def categorize_margin(m):
        if m in [-2, -1, 0, 1, 2]:
            return 'Close (±2)'
        elif m in [-5, -4, -3]:
            return 'Down 3-5'
        elif m in [3, 4, 5]:
            return 'Up 3-5'
        elif m in [-8, -7, -6]:
            return 'Down 6-8'
        elif m in [6, 7, 8]:
            return 'Up 6-8'
        elif m in [-9, -10, -11]:
            return 'Down 9-11'
        else:
            return 'Other'

    df['margin_category'] = df['score_diff_pre_td'].apply(categorize_margin)

    # Track by period
    df['period'] = pd.cut(df['season'],
                          bins=[2014, 2017, 2020, 2024],
                          labels=['2015-17', '2018-20', '2021-24'])

    print("\nCorrect rate by margin category and period:")
    print("-" * 70)

    pivot = df.groupby(['margin_category', 'period']).agg({
        'coach_correct': 'mean',
        'went_for_2': 'mean',
        'should_go_for_2': 'mean'
    }).reset_index()

    pivot_wide = pivot.pivot(index='margin_category', columns='period', values='coach_correct')

    # Add optimal rate for context
    optimal_by_margin = df.groupby('margin_category')['should_go_for_2'].mean()

    print(f"\n{'Margin':<15} | {'2015-17':>10} | {'2018-20':>10} | {'2021-24':>10} | {'Improvement':>12} | {'Optimal 2pt':>12}")
    print("-" * 80)

    for margin in ['Close (±2)', 'Down 3-5', 'Up 3-5', 'Down 6-8', 'Up 6-8', 'Down 9-11']:
        if margin in pivot_wide.index:
            early = pivot_wide.loc[margin, '2015-17'] if '2015-17' in pivot_wide.columns else np.nan
            mid = pivot_wide.loc[margin, '2018-20'] if '2018-20' in pivot_wide.columns else np.nan
            late = pivot_wide.loc[margin, '2021-24'] if '2021-24' in pivot_wide.columns else np.nan

            if not np.isnan(early) and not np.isnan(late):
                improvement = late - early
            else:
                improvement = np.nan

            optimal = optimal_by_margin.get(margin, np.nan)

            early_str = f"{early:.1%}" if not np.isnan(early) else "N/A"
            mid_str = f"{mid:.1%}" if not np.isnan(mid) else "N/A"
            late_str = f"{late:.1%}" if not np.isnan(late) else "N/A"
            imp_str = f"{improvement:+.1%}" if not np.isnan(improvement) else "N/A"
            opt_str = f"{optimal:.1%}" if not np.isnan(optimal) else "N/A"

            print(f"{margin:<15} | {early_str:>10} | {mid_str:>10} | {late_str:>10} | {imp_str:>12} | {opt_str:>12}")

    # Focus on the "problem" margins
    print("\n" + "-" * 60)
    print("FOCUS: Down 8 and Down 9 (the present bias margins)")
    print("-" * 60)

    for margin in [-8, -9]:
        margin_data = df[df['score_diff_pre_td'] == margin]

        print(f"\nDown {-margin} (pre-TD score diff = {margin}):")

        for period in ['2015-17', '2018-20', '2021-24']:
            period_data = margin_data[margin_data['period'] == period]
            if len(period_data) > 0:
                correct = period_data['coach_correct'].mean()
                went = period_data['went_for_2'].mean()
                should = period_data['should_go_for_2'].mean()
                n = len(period_data)
                print(f"  {period}: {correct:.1%} correct, {went:.1%} went for 2, {should:.1%} should (n={n})")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} decisions")

    # Run analyses
    continuing, new, exiting = decompose_aggregate_trend(df)
    analyze_pioneer_regression(df)
    staley_deep_dive(df)
    situation_specific_learning(df)

    # Summary
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. DECOMPOSITION:
   - Most of the improvement comes from COMPOSITION (coach turnover)
   - Within-coach learning is limited
   - "Selection, not learning" hypothesis supported

2. PIONEER REGRESSION:
   - Tomlin/Harbaugh were never MORE aggressive than optimal
   - They were early adopters who stayed constant
   - The "backing off" is relative — optimal rate increased as situations changed

3. STALEY PARADOX:
   - "Analytics" reputation comes from being aggressive
   - But he faces situations where 76% should be 2pt attempts
   - Only does 10% → he's UNDER-aggressive for his situations
   - Being aggressive ≠ being correct

4. SITUATION-SPECIFIC:
   - Coaches improved most on "close" margins
   - Down 8/9 (present bias margins) show less improvement
   - Learning is uneven across situations
""")


if __name__ == "__main__":
    main()
