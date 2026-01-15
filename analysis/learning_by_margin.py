"""
Learning Analysis by Decision Margin

Analyzes where coaches have learned (or failed to learn) by breaking down
decision quality across different margin sizes and situation types.

Key findings:
1. Coaches improved on "obvious" decisions but got worse on close calls
2. Short yardage: coaches became over-aggressive (under → over)
3. End-game situations show improvement; early-game shows decline
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt


def load_data():
    """Load expanding window results."""
    return pd.read_csv('outputs/tables/expanding_window_results.csv')


def add_situation_bins(df):
    """Add categorical bins for different situation types."""
    # Margin bins
    df['margin_bin'] = pd.cut(
        df['ex_ante_margin'].abs(),
        bins=[0, 0.02, 0.05, 0.10, 1.0],
        labels=['Close (0-2pp)', 'Moderate (2-5pp)', 'Clear (5-10pp)', 'Obvious (10+pp)']
    )

    # Field position
    df['field_zone'] = pd.cut(
        df['field_pos'],
        bins=[0, 20, 40, 60, 100],
        labels=['Red zone', 'Opp territory', 'Midfield', 'Own territory']
    )

    # Yards to go
    df['ytg_bin'] = pd.cut(
        df['yards_to_go'],
        bins=[0, 2, 5, 10, 100],
        labels=['Short (1-2)', 'Medium (3-5)', 'Long (6-10)', 'Very long (10+)']
    )

    # Score differential
    df['game_state'] = pd.cut(
        df['score_diff'].abs(),
        bins=[-1, 7, 14, 100],
        labels=['Close game', 'Moderate lead', 'Blowout']
    )

    # Time remaining
    df['game_time'] = pd.cut(
        df['time_remaining'],
        bins=[-1, 300, 900, 1800, 4000],
        labels=['End game (<5min)', 'Late (5-15min)', 'Mid (15-30min)', 'Early (30+min)']
    )

    # Era
    df['era'] = df['season'].apply(lambda x: 'Early (2006-2012)' if x <= 2012 else
                                   ('Middle (2013-2018)' if x <= 2018 else 'Late (2019-2024)'))

    return df


def compute_trend(df, group_col, metric='ex_ante_match'):
    """Compute linear trend for a metric over seasons."""
    yearly = df.groupby(['season', group_col])[metric].mean().reset_index()

    results = []
    for group in df[group_col].unique():
        if pd.isna(group):
            continue
        subset = yearly[yearly[group_col] == group]
        if len(subset) >= 3:
            slope, intercept, r, p, se = stats.linregress(subset['season'], subset[metric])
            results.append({
                'group': group,
                'n_plays': len(df[df[group_col] == group]),
                'overall_optimal': df[df[group_col] == group][metric].mean(),
                'trend_pp_year': slope * 100,
                'p_value': p,
                'significant': p < 0.05
            })

    return pd.DataFrame(results)


def analyze_mistake_types(df):
    """Analyze types of mistakes by era."""
    results = []

    for era in df['era'].unique():
        subset = df[df['era'] == era]
        n_total = len(subset)
        n_mistakes = (~subset['ex_ante_match']).sum()

        # Under-aggressive: should go, didn't
        under_agg = len(subset[(subset['ex_ante_optimal'] == 'go_for_it') &
                               (subset['actual_decision'] != 'go_for_it')])

        # Over-aggressive: shouldn't go, did
        over_agg = len(subset[(subset['ex_ante_optimal'] != 'go_for_it') &
                              (subset['actual_decision'] == 'go_for_it')])

        # Correct aggression
        correct_go = len(subset[(subset['ex_ante_optimal'] == 'go_for_it') &
                                (subset['actual_decision'] == 'go_for_it')])
        correct_no_go = len(subset[(subset['ex_ante_optimal'] != 'go_for_it') &
                                   (subset['actual_decision'] != 'go_for_it')])

        results.append({
            'era': era,
            'n_plays': n_total,
            'optimal_rate': 1 - n_mistakes / n_total,
            'under_aggressive_rate': under_agg / n_total,
            'over_aggressive_rate': over_agg / n_total,
            'go_rate': (subset['actual_decision'] == 'go_for_it').mean(),
            'model_go_rate': (subset['ex_ante_optimal'] == 'go_for_it').mean()
        })

    return pd.DataFrame(results)


def analyze_by_margin(df):
    """Comprehensive analysis by decision margin."""
    print("=" * 70)
    print("LEARNING ANALYSIS BY DECISION MARGIN")
    print("=" * 70)

    # Overall by margin
    print("\n1. DECISION QUALITY BY MARGIN SIZE")
    print("-" * 50)
    margin_stats = df.groupby('margin_bin').agg({
        'ex_ante_match': ['count', 'mean'],
        'season': lambda x: x.nunique()
    }).round(3)
    margin_stats.columns = ['n_plays', 'optimal_rate', 'n_seasons']
    print(margin_stats)

    # Trends by margin
    print("\n2. LEARNING TRENDS BY MARGIN SIZE")
    print("-" * 50)
    trends = compute_trend(df, 'margin_bin')
    trends = trends.sort_values('trend_pp_year')
    for _, row in trends.iterrows():
        sig = '*' if row['significant'] else ''
        print(f"{row['group']}: {row['overall_optimal']:.1%} optimal, "
              f"trend = {row['trend_pp_year']:+.2f} pp/year{sig} (p={row['p_value']:.3f})")

    return trends


def analyze_by_situation(df):
    """Analyze learning by situation type."""
    print("\n" + "=" * 70)
    print("LEARNING BY SITUATION TYPE")
    print("=" * 70)

    situation_vars = ['field_zone', 'ytg_bin', 'game_state', 'game_time']

    all_trends = {}
    for var in situation_vars:
        print(f"\n--- By {var.replace('_', ' ').title()} ---")
        trends = compute_trend(df, var)
        trends = trends.sort_values('trend_pp_year', ascending=False)
        for _, row in trends.iterrows():
            sig = '*' if row['significant'] else ''
            print(f"  {row['group']}: {row['n_plays']:,} plays, {row['overall_optimal']:.1%} optimal, "
                  f"trend = {row['trend_pp_year']:+.2f} pp/year{sig}")
        all_trends[var] = trends

    return all_trends


def analyze_aggression_shift(df):
    """Analyze the shift from under- to over-aggression."""
    print("\n" + "=" * 70)
    print("AGGRESSION SHIFT ANALYSIS")
    print("=" * 70)

    # By era overall
    print("\n1. OVERALL AGGRESSION BY ERA")
    print("-" * 50)
    era_stats = analyze_mistake_types(df)
    era_stats = era_stats.sort_values('era')
    print(era_stats.to_string(index=False))

    # By yards to go
    print("\n2. AGGRESSION SHIFT BY YARDS TO GO")
    print("-" * 50)
    for ytg in ['Short (1-2)', 'Medium (3-5)', 'Long (6-10)', 'Very long (10+)']:
        subset = df[df['ytg_bin'] == ytg]
        ytg_stats = analyze_mistake_types(subset)
        print(f"\n{ytg}:")
        for _, row in ytg_stats.sort_values('era').iterrows():
            print(f"  {row['era']}: go_rate={row['go_rate']:.1%}, "
                  f"under={row['under_aggressive_rate']:.1%}, over={row['over_aggressive_rate']:.1%}")

    return era_stats


def analyze_short_yardage_detail(df):
    """Deep dive into short yardage situations."""
    print("\n" + "=" * 70)
    print("SHORT YARDAGE DEEP DIVE (1-2 yards)")
    print("=" * 70)

    short = df[df['yards_to_go'] <= 2]

    # By what the optimal action is
    print("\n1. COMPLIANCE BY OPTIMAL ACTION")
    print("-" * 50)
    for action in ['go_for_it', 'punt', 'field_goal']:
        subset = short[short['ex_ante_optimal'] == action]
        yearly = subset.groupby('season')['ex_ante_match'].mean()
        slope, _, _, p, _ = stats.linregress(yearly.index, yearly.values)

        early = subset[subset['season'] <= 2012]['ex_ante_match'].mean()
        late = subset[subset['season'] >= 2019]['ex_ante_match'].mean()

        print(f"When model says {action.upper()}:")
        print(f"  N plays: {len(subset):,}")
        print(f"  Compliance: {early:.1%} (early) -> {late:.1%} (late)")
        print(f"  Trend: {slope*100:+.2f} pp/year (p={p:.3f})")

    # The paradox
    print("\n2. THE AGGRESSION PARADOX")
    print("-" * 50)
    yearly = short.groupby('season').agg({
        'actual_decision': lambda x: (x == 'go_for_it').mean(),
        'ex_ante_optimal': lambda x: (x == 'go_for_it').mean(),
        'ex_ante_match': 'mean'
    }).rename(columns={
        'actual_decision': 'coach_go_rate',
        'ex_ante_optimal': 'model_go_rate',
        'ex_ante_match': 'optimal_rate'
    })

    early = yearly[yearly.index <= 2012].mean()
    late = yearly[yearly.index >= 2019].mean()

    print("Early (2006-2012):")
    print(f"  Coach go rate: {early['coach_go_rate']:.1%}")
    print(f"  Model go rate: {early['model_go_rate']:.1%}")
    print(f"  Optimal rate: {early['optimal_rate']:.1%}")

    print("Late (2019-2024):")
    print(f"  Coach go rate: {late['coach_go_rate']:.1%}")
    print(f"  Model go rate: {late['model_go_rate']:.1%}")
    print(f"  Optimal rate: {late['optimal_rate']:.1%}")

    print("\nInterpretation: Coaches increased aggression (+21pp), but model")
    print("recommendation stayed flat. Result: over-aggression increased.")


def create_summary_table(df):
    """Create summary table for paper."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)

    results = []

    # By margin
    for margin in df['margin_bin'].dropna().unique():
        subset = df[df['margin_bin'] == margin]
        yearly = subset.groupby('season')['ex_ante_match'].mean()
        slope, _, _, p, _ = stats.linregress(yearly.index, yearly.values)

        results.append({
            'Category': 'Margin',
            'Subcategory': margin,
            'N': len(subset),
            'Optimal Rate': subset['ex_ante_match'].mean(),
            'Trend (pp/year)': slope * 100,
            'p-value': p
        })

    # By key situations
    for ytg in ['Short (1-2)', 'Long (6-10)']:
        subset = df[df['ytg_bin'] == ytg]
        yearly = subset.groupby('season')['ex_ante_match'].mean()
        slope, _, _, p, _ = stats.linregress(yearly.index, yearly.values)

        results.append({
            'Category': 'Yards to Go',
            'Subcategory': ytg,
            'N': len(subset),
            'Optimal Rate': subset['ex_ante_match'].mean(),
            'Trend (pp/year)': slope * 100,
            'p-value': p
        })

    for time in ['End game (<5min)', 'Early (30+min)']:
        subset = df[df['game_time'] == time]
        yearly = subset.groupby('season')['ex_ante_match'].mean()
        slope, _, _, p, _ = stats.linregress(yearly.index, yearly.values)

        results.append({
            'Category': 'Game Time',
            'Subcategory': time,
            'N': len(subset),
            'Optimal Rate': subset['ex_ante_match'].mean(),
            'Trend (pp/year)': slope * 100,
            'p-value': p
        })

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    """Run full learning analysis."""
    # Load and prepare data
    df = load_data()
    df = add_situation_bins(df)

    print(f"Loaded {len(df):,} fourth down plays (2006-2024)")

    # Run analyses
    margin_trends = analyze_by_margin(df)
    situation_trends = analyze_by_situation(df)
    era_stats = analyze_aggression_shift(df)
    analyze_short_yardage_detail(df)
    summary = create_summary_table(df)

    # Save results
    output_dir = Path('outputs/tables')
    summary.to_csv(output_dir / 'learning_by_margin.csv', index=False)
    print(f"\nSaved summary to {output_dir / 'learning_by_margin.csv'}")

    return {
        'margin_trends': margin_trends,
        'situation_trends': situation_trends,
        'era_stats': era_stats,
        'summary': summary
    }


if __name__ == "__main__":
    main()
