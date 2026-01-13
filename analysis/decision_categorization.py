"""
Decision Categorization Analysis

Categorizes fourth down decisions into:
1. Close calls (reasonable disagreement) vs Clear mistakes (inexcusable)
2. Finds most controversial decisions of all time
3. Finds worst decisions according to the model

Categories:
- "Close call": Decision margin < 2% WP (reasonable people could disagree)
- "Moderate": Decision margin 2-5% WP
- "Clear": Decision margin > 5% WP (should be obvious)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def load_and_merge_data():
    """Load expanding window results and merge with game context."""
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'

    # Load expanding window analysis results
    results = pd.read_parquet(output_dir / 'expanding_window_results.parquet')

    # Load fourth downs data for game context
    fourth_downs = pd.read_parquet(data_dir / 'fourth_downs.parquet')

    # Select relevant context columns
    context_cols = [
        'game_id', 'play_id', 'posteam', 'defteam', 'week', 'season_type',
        'home_team', 'away_team', 'game_date', 'qtr', 'desc'
    ]
    context = fourth_downs[context_cols].copy()

    # Merge
    merged = results.merge(
        context,
        on=['game_id', 'play_id'],
        how='left'
    )

    return merged


def categorize_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize decisions based on decision margin (how clear the optimal choice was).

    Categories:
    - Close call: margin < 2% WP
    - Moderate: margin 2-5% WP
    - Clear: margin > 5% WP
    """
    df = df.copy()

    # Decision clarity based on ex ante margin
    df['decision_clarity'] = pd.cut(
        df['ex_ante_margin'],
        bins=[0, 0.02, 0.05, 1.0],
        labels=['close_call', 'moderate', 'clear'],
        include_lowest=True
    )

    # Alternative: based on confidence (probability optimal action is best)
    df['confidence_category'] = pd.cut(
        df['ex_ante_confidence'],
        bins=[0, 0.6, 0.8, 0.95, 1.0],
        labels=['toss_up', 'lean', 'clear', 'obvious'],
        include_lowest=True
    )

    # Was the coach wrong?
    df['coach_wrong'] = ~df['ex_ante_match']

    # Mistake severity: only matters if coach was wrong
    df['mistake_type'] = 'correct'

    # Close calls where coach disagreed (reasonable)
    df.loc[
        df['coach_wrong'] & (df['decision_clarity'] == 'close_call'),
        'mistake_type'
    ] = 'difference_of_opinion'

    # Moderate situations where coach was wrong
    df.loc[
        df['coach_wrong'] & (df['decision_clarity'] == 'moderate'),
        'mistake_type'
    ] = 'questionable'

    # Clear situations where coach was wrong (inexcusable)
    df.loc[
        df['coach_wrong'] & (df['decision_clarity'] == 'clear'),
        'mistake_type'
    ] = 'clear_mistake'

    # Among clear mistakes, distinguish truly egregious ones (>10% WP margin)
    df.loc[
        df['coach_wrong'] & (df['ex_ante_margin'] > 0.10),
        'mistake_type'
    ] = 'egregious'

    # WP cost of mistake
    df['wp_cost'] = df.apply(
        lambda row: calculate_wp_cost(row) if row['coach_wrong'] else 0,
        axis=1
    )

    return df


def calculate_wp_cost(row):
    """
    Calculate the WP cost of the coach's decision vs optimal.

    The cost is simply the ex_ante_margin - the difference between the best
    and second-best options. This is correct even when one option (like a
    very long FG) has 0 WP because it's infeasible.

    Note: We use ex_ante_margin rather than computing wp_optimal - wp_actual
    because infeasible options (wp=0) would give misleading costs.
    """
    # The margin already captures the cost correctly
    return row['ex_ante_margin']


def generate_summary_statistics(df: pd.DataFrame):
    """Generate summary statistics for the categorization."""
    print("="*80)
    print("DECISION CATEGORIZATION SUMMARY")
    print("="*80)

    total = len(df)

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
    print("THE KEY BREAKDOWN: INEXCUSABLE vs DEBATABLE")
    print("="*80)

    total_wrong = df['coach_wrong'].sum()

    debatable = len(df[df['mistake_type'] == 'difference_of_opinion'])
    questionable = len(df[df['mistake_type'] == 'questionable'])
    clear_mistakes = len(df[df['mistake_type'] == 'clear_mistake'])
    egregious = len(df[df['mistake_type'] == 'egregious'])

    print(f"""
Total 4th down plays analyzed: {total:,}
Total where coach deviated from model: {total_wrong:,} ({total_wrong/total:.1%})

Of these deviations:

  DEBATABLE (margin < 2% WP):
    {debatable:,} plays ({debatable/total:.1%} of all, {debatable/total_wrong:.1%} of deviations)
    These are "differences of opinion" - the model says one thing, but it's a close call.
    Reasonable coaches could disagree.

  QUESTIONABLE (margin 2-5% WP):
    {questionable:,} plays ({questionable/total:.1%} of all, {questionable/total_wrong:.1%} of deviations)
    The model has a clear preference, but it's not overwhelming.

  CLEAR MISTAKES (margin 5-10% WP):
    {clear_mistakes:,} plays ({clear_mistakes/total:.1%} of all, {clear_mistakes/total_wrong:.1%} of deviations)
    The model strongly recommends a different action.

  EGREGIOUS (margin > 10% WP):
    {egregious:,} plays ({egregious/total:.1%} of all, {egregious/total_wrong:.1%} of deviations)
    Truly inexcusable - the optimal choice was obvious.
""")

    # Total WP cost
    total_wp_cost = df['wp_cost'].sum()
    print(f"Total WP cost from suboptimal decisions: {total_wp_cost:.1f} percentage points")
    print(f"Average WP cost per suboptimal decision: {total_wp_cost/total_wrong:.2f} pp")

    # By category
    for cat in ['difference_of_opinion', 'questionable', 'clear_mistake', 'egregious']:
        cat_df = df[df['mistake_type'] == cat]
        if len(cat_df) > 0:
            cat_cost = cat_df['wp_cost'].sum()
            print(f"  {cat}: {cat_cost:.1f} pp total ({cat_cost/len(cat_df):.2f} pp avg)")


def find_most_controversial(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Find the most controversial decisions - close calls where coach and model disagreed.

    These are high-stakes situations where the decision was genuinely difficult.
    """
    # Close calls where coach disagreed with model
    controversial = df[
        (df['decision_clarity'] == 'close_call') &
        df['coach_wrong']
    ].copy()

    # Sort by how close it was (smallest margin = most controversial)
    # But also consider stakes (time remaining, score)
    controversial['controversy_score'] = (
        1 / (controversial['ex_ante_margin'] + 0.001) *  # Closer = more controversial
        np.abs(controversial['score_diff']).clip(upper=14) / 14  # Close game bonus
    )

    return controversial.nlargest(n, 'controversy_score')


def find_worst_decisions(df: pd.DataFrame, n: int = 20, exclude_end_of_game: bool = False) -> pd.DataFrame:
    """
    Find the worst decisions of all time according to the model.

    Ranked by WP cost (how much the decision hurt the team).

    Parameters:
        df: DataFrame with categorized decisions
        n: Number of worst decisions to return
        exclude_end_of_game: If True, exclude late-game situations where opponent
                             can run out the clock. Default is now False because
                             the clock-adjusted model properly accounts for
                             asymmetric time consumption in end-of-game scenarios.

    Note on Clock Model:
    --------------------
    With the clock-adjusted decision model (use_clock_model=True in
    BayesianDecisionAnalyzer), the model now properly accounts for the
    asymmetric clock consumption between different action-outcome pairs:

    - Converting burns ~151s (retain possession, run more plays)
    - Failing burns ~48s (opponent gets ball)
    - Punting burns ~69s
    - FG make burns ~99s

    This means late-game situations are handled naturally:
    - When trailing with little time, the model correctly values keeping
      the ball (go for it) because burning clock hurts you.
    - When leading, the model correctly values possession because burning
      clock helps protect the lead.

    The previous hard-coded filter is no longer necessary and would actually
    hide interesting late-game decision analysis.
    """
    candidates = df[df['coach_wrong']].copy()

    if exclude_end_of_game:
        # Legacy filter - kept for backward compatibility but disabled by default
        # The clock model now handles this naturally
        end_of_game_mask = (
            (candidates['time_remaining'] < 150) &  # Under 2:30 left
            (candidates['score_diff'] < 0)  # Trailing
        )
        candidates = candidates[~end_of_game_mask]

    return candidates.nlargest(n, 'wp_cost')


def find_egregious_mistakes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find truly egregious mistakes (margin > 10% WP).
    """
    egregious = df[df['mistake_type'] == 'egregious'].copy()
    return egregious.sort_values('wp_cost', ascending=False)


def format_play_description(row) -> str:
    """Format a play for display."""
    # Game context
    season = row.get('season', '?')
    week = row.get('week', '?')
    posteam = row.get('posteam', '?')
    defteam = row.get('defteam', '?')

    # Situation
    field_pos = row['field_pos']
    if field_pos > 50:
        location = f"own {100 - field_pos}"
    else:
        location = f"opponent's {field_pos}"

    ytg = row['yards_to_go']
    score = row['score_diff']
    time_min = row['time_remaining'] // 60
    qtr = row.get('qtr', '?')

    # Decision
    actual = row['actual_decision']
    optimal = row['ex_ante_optimal']
    margin = row['ex_ante_margin'] * 100
    wp_cost = row.get('wp_cost', 0) * 100

    desc = f"""
{season} Week {week}: {posteam} vs {defteam}
  Situation: 4th & {ytg} at {location}, Q{qtr} {time_min}:{row['time_remaining'] % 60:02d} remaining
  Score: {'+' if score > 0 else ''}{score}
  Coach chose: {actual.upper()}
  Model said: {optimal.upper()} (margin: {margin:.1f}% WP)
  WP cost: {wp_cost:.1f}%"""

    if 'desc' in row and pd.notna(row['desc']):
        desc += f"\n  Play: {row['desc'][:100]}..."

    return desc


def create_latex_tables(df: pd.DataFrame, output_dir: Path):
    """Create LaTeX tables for the paper."""

    # Table 1: Mistake categorization summary
    total = len(df)
    total_wrong = df['coach_wrong'].sum()

    categories = [
        ('Correct', len(df[df['mistake_type'] == 'correct']), 'Coach agreed with model'),
        ('Difference of opinion', len(df[df['mistake_type'] == 'difference_of_opinion']),
         'Margin $<$ 2\\% WP'),
        ('Questionable', len(df[df['mistake_type'] == 'questionable']),
         'Margin 2--5\\% WP'),
        ('Clear mistake', len(df[df['mistake_type'] == 'clear_mistake']),
         'Margin 5--10\\% WP'),
        ('Egregious', len(df[df['mistake_type'] == 'egregious']),
         'Margin $>$ 10\\% WP'),
    ]

    latex = """\\begin{table}[H]
\\centering
\\caption{Categorization of Coach Decisions}
\\label{tab:mistake_categories}
\\begin{tabular}{lrrp{5cm}}
\\toprule
\\textbf{Category} & \\textbf{Count} & \\textbf{Percent} & \\textbf{Definition} \\\\
\\midrule
"""

    for name, count, defn in categories:
        pct = count / total * 100
        latex += f"{name} & {count:,} & {pct:.1f}\\% & {defn} \\\\\n"

    latex += """\\midrule
\\textbf{Total} & \\textbf{""" + f"{total:,}" + """} & \\textbf{100\\%} & \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_dir / 'mistake_categories_table.tex', 'w') as f:
        f.write(latex)

    # Table 2: WP cost by category
    latex2 = """\\begin{table}[H]
\\centering
\\caption{Win Probability Cost by Mistake Category}
\\label{tab:wp_cost}
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Category} & \\textbf{Count} & \\textbf{Total WP Cost} & \\textbf{Avg WP Cost} \\\\
\\midrule
"""

    for cat in ['difference_of_opinion', 'questionable', 'clear_mistake', 'egregious']:
        cat_df = df[df['mistake_type'] == cat]
        count = len(cat_df)
        total_cost = cat_df['wp_cost'].sum() * 100
        avg_cost = (cat_df['wp_cost'].mean() * 100) if count > 0 else 0
        name = cat.replace('_', ' ').title()
        latex2 += f"{name} & {count:,} & {total_cost:.1f}\\% & {avg_cost:.2f}\\% \\\\\n"

    latex2 += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_dir / 'wp_cost_table.tex', 'w') as f:
        f.write(latex2)

    print(f"LaTeX tables saved to {output_dir}")


def main():
    """Run the full decision categorization analysis."""
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and merging data...")
    df = load_and_merge_data()
    print(f"Loaded {len(df):,} plays")

    print("\nCategorizing decisions...")
    df = categorize_decisions(df)

    # Save categorized data
    df.to_parquet(output_dir / 'decision_categorization.parquet')
    df.to_csv(output_dir / 'decision_categorization.csv', index=False)

    # Generate summary statistics
    generate_summary_statistics(df)

    # Most controversial decisions
    print("\n" + "="*80)
    print("MOST CONTROVERSIAL DECISIONS OF ALL TIME")
    print("(Close calls where coach disagreed with model)")
    print("="*80)

    controversial = find_most_controversial(df, n=15)
    for _, row in controversial.iterrows():
        print(format_play_description(row))

    controversial.to_csv(output_dir / 'most_controversial_decisions.csv', index=False)

    # Worst decisions
    print("\n" + "="*80)
    print("WORST DECISIONS OF ALL TIME (BY WP COST)")
    print("="*80)

    worst = find_worst_decisions(df, n=15)
    for _, row in worst.iterrows():
        print(format_play_description(row))

    worst.to_csv(output_dir / 'worst_decisions.csv', index=False)

    # Egregious mistakes summary
    print("\n" + "="*80)
    print("EGREGIOUS MISTAKES (MARGIN > 10% WP)")
    print("="*80)

    egregious = find_egregious_mistakes(df)
    print(f"Total egregious mistakes: {len(egregious)}")

    # By actual decision
    print("\nBy what coach actually did:")
    for actual in ['punt', 'field_goal', 'go_for_it']:
        count = len(egregious[egregious['actual_decision'] == actual])
        if count > 0:
            print(f"  {actual}: {count}")

    # By what they should have done
    print("\nBy what they should have done:")
    for optimal in ['go_for_it', 'punt', 'field_goal']:
        count = len(egregious[egregious['ex_ante_optimal'] == optimal])
        if count > 0:
            print(f"  Should have {optimal}: {count}")

    egregious.to_csv(output_dir / 'egregious_mistakes.csv', index=False)

    # Create LaTeX tables
    create_latex_tables(df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to {output_dir}")

    return df


if __name__ == "__main__":
    main()
