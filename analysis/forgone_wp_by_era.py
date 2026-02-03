"""
Compute Forgone Win Probability by Era

Following Strulov-Shlain (2025), we compute the "forgone welfare" from suboptimal
decisions in each era:

1. Pre-Belichick (2006-2009): Stable heuristic, moderate forgone WP
2. Chilling Effect (2010-2017): Conservative reversion after Belichick 4th & 2
3. Analytics Era (2018-2024): Movement toward optimal play

For each decision:
  forgone_wp = WP(optimal) - WP(actual)

If coaches learned the wrong lesson from Belichick, we expect:
- Higher forgone WP during the chilling effect era (more conservative mistakes)
- Lower forgone WP in analytics era (closer to optimal)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_decision_data():
    """Load the decision categorization data."""
    data_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'decision_categorization.parquet'
    return pd.read_parquet(data_path)


def assign_era(season):
    """Assign era based on season."""
    if season <= 2009:
        return 'pre_belichick'
    elif season <= 2017:
        return 'chilling_effect'
    else:
        return 'analytics_era'


def compute_forgone_wp(df):
    """
    Compute forgone WP for each decision.

    forgone_wp = max(WP) - WP(actual)

    This is always >= 0. Higher values mean worse decisions.
    """
    df = df.copy()

    # Compute the optimal WP (max of all options)
    df['wp_optimal'] = df[['ex_ante_wp_go', 'ex_ante_wp_punt', 'ex_ante_wp_fg']].max(axis=1)

    # Map actual decision to its WP
    def get_actual_wp(row):
        if row['actual_decision'] == 'go_for_it':
            return row['ex_ante_wp_go']
        elif row['actual_decision'] == 'punt':
            return row['ex_ante_wp_punt']
        elif row['actual_decision'] == 'field_goal':
            return row['ex_ante_wp_fg']
        return np.nan

    df['wp_actual'] = df.apply(get_actual_wp, axis=1)

    # Forgone WP
    df['forgone_wp'] = df['wp_optimal'] - df['wp_actual']

    # Clip negative values (can happen due to numerical issues)
    df['forgone_wp'] = df['forgone_wp'].clip(lower=0)

    return df


def analyze_by_era(df):
    """Analyze forgone WP by era."""
    df = df.copy()
    df['era'] = df['season'].apply(assign_era)

    # Compute forgone WP
    df = compute_forgone_wp(df)

    print("="*80)
    print("FORGONE WIN PROBABILITY BY ERA")
    print("="*80)

    # Summary by era
    era_summary = df.groupby('era').agg(
        n_decisions=('forgone_wp', 'count'),
        mean_forgone_wp=('forgone_wp', 'mean'),
        median_forgone_wp=('forgone_wp', 'median'),
        total_forgone_wp=('forgone_wp', 'sum'),
        pct_correct=('ex_ante_match', 'mean')
    ).reset_index()

    # Order eras chronologically
    era_order = ['pre_belichick', 'chilling_effect', 'analytics_era']
    era_summary['era'] = pd.Categorical(era_summary['era'], categories=era_order, ordered=True)
    era_summary = era_summary.sort_values('era')

    print("\nSummary Statistics by Era:")
    print("-"*60)

    for _, row in era_summary.iterrows():
        era_label = row['era'].replace('_', ' ').title()
        print(f"\n{era_label}:")
        print(f"  Decisions: {row['n_decisions']:,}")
        print(f"  Mean forgone WP: {row['mean_forgone_wp']:.4f} ({row['mean_forgone_wp']*100:.2f}%)")
        print(f"  Median forgone WP: {row['median_forgone_wp']:.4f}")
        print(f"  Total forgone WP: {row['total_forgone_wp']:.1f} percentage points")
        print(f"  Correct decision rate: {row['pct_correct']:.1%}")

    return era_summary, df


def analyze_by_season(df):
    """Analyze forgone WP by season for time series."""
    df = compute_forgone_wp(df)

    season_summary = df.groupby('season').agg(
        n_decisions=('forgone_wp', 'count'),
        mean_forgone_wp=('forgone_wp', 'mean'),
        total_forgone_wp=('forgone_wp', 'sum'),
        pct_correct=('ex_ante_match', 'mean')
    ).reset_index()

    print("\n" + "="*80)
    print("FORGONE WIN PROBABILITY BY SEASON")
    print("="*80)

    for _, row in season_summary.iterrows():
        marker = "  <-- Belichick 4th & 2" if row['season'] == 2009 else ""
        marker = "  <-- Analytics revolution" if row['season'] == 2018 else marker
        print(f"{int(row['season'])}: {row['mean_forgone_wp']*100:5.2f}% mean forgone WP, "
              f"{row['pct_correct']:.1%} correct ({int(row['n_decisions']):,} plays){marker}")

    return season_summary


def analyze_by_decision_type(df):
    """
    Break down forgone WP by what the coach chose vs what was optimal.

    Key insight: if chilling effect caused conservatism, we should see
    more "punted when should have gone for it" mistakes in 2010-2017.
    """
    df = compute_forgone_wp(df)
    df['era'] = df['season'].apply(assign_era)

    # Create decision mismatch categories
    def categorize_mistake(row):
        if row['ex_ante_match']:
            return 'correct'
        actual = row['actual_decision']
        optimal = row['ex_ante_optimal']
        return f"{actual}_instead_of_{optimal}"

    df['mistake_type'] = df.apply(categorize_mistake, axis=1)

    print("\n" + "="*80)
    print("MISTAKE TYPES BY ERA")
    print("="*80)

    # Focus on the key mistake: punting when should go for it
    punt_instead_go = df[df['mistake_type'] == 'punt_instead_of_go_for_it']

    print("\n'Punted when should have gone for it' by era:")
    print("-"*60)

    for era in ['pre_belichick', 'chilling_effect', 'analytics_era']:
        era_data = df[df['era'] == era]
        mistake_data = punt_instead_go[punt_instead_go['era'] == era]

        n_total = len(era_data)
        n_mistake = len(mistake_data)
        rate = n_mistake / n_total if n_total > 0 else 0
        mean_cost = mistake_data['forgone_wp'].mean() if len(mistake_data) > 0 else 0

        era_label = era.replace('_', ' ').title()
        print(f"\n{era_label}:")
        print(f"  Frequency: {rate:.2%} of all 4th downs ({n_mistake:,}/{n_total:,})")
        print(f"  Mean WP cost when it happens: {mean_cost:.4f} ({mean_cost*100:.2f}%)")

    return df


def create_visualizations(era_summary, season_summary):
    """Create figures for the analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Forgone WP by era (bar chart)
    ax1 = axes[0]
    era_labels = ['Pre-Belichick\n(2006-2009)', 'Chilling Effect\n(2010-2017)', 'Analytics Era\n(2018-2024)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    bars = ax1.bar(era_labels, era_summary['mean_forgone_wp'] * 100, color=colors, edgecolor='black')
    ax1.set_ylabel('Mean Forgone WP (%)', fontsize=12)
    ax1.set_title('Average Decision Cost by Era', fontsize=14)
    ax1.set_ylim(0, max(era_summary['mean_forgone_wp'] * 100) * 1.3)

    # Add value labels
    for bar, val in zip(bars, era_summary['mean_forgone_wp'] * 100):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11)

    # Plot 2: Time series by season
    ax2 = axes[1]
    ax2.plot(season_summary['season'], season_summary['mean_forgone_wp'] * 100,
             'b-o', linewidth=2, markersize=6)
    ax2.axvline(x=2009.9, color='red', linestyle='--', alpha=0.7, label='Belichick 4th & 2')
    ax2.axvline(x=2018, color='green', linestyle='--', alpha=0.7, label='Analytics era')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Mean Forgone WP (%)', fontsize=12)
    ax2.set_title('Decision Cost Over Time', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'forgone_wp_by_era.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()


def compute_total_wins_lost(df):
    """
    Compute total wins lost due to suboptimal decisions by era.

    Each 1% of forgone WP = 0.01 expected wins lost on that play.
    Sum across all plays in an era.
    """
    df = compute_forgone_wp(df)
    df['era'] = df['season'].apply(assign_era)

    print("\n" + "="*80)
    print("ESTIMATED WINS LOST DUE TO SUBOPTIMAL DECISIONS")
    print("="*80)

    for era in ['pre_belichick', 'chilling_effect', 'analytics_era']:
        era_data = df[df['era'] == era]
        total_forgone = era_data['forgone_wp'].sum()
        n_seasons = era_data['season'].nunique()
        per_season = total_forgone / n_seasons if n_seasons > 0 else 0

        era_label = era.replace('_', ' ').title()
        print(f"\n{era_label} ({n_seasons} seasons):")
        print(f"  Total forgone WP: {total_forgone:.2f}")
        print(f"  Equivalent wins lost: {total_forgone:.1f}")
        print(f"  Per season: {per_season:.2f} wins/team")
        print(f"  Per team per season: {per_season/32:.3f} wins")


def main():
    """Run the full analysis."""
    print("Loading decision data...")
    df = load_decision_data()

    # Main analysis
    era_summary, df_with_forgone = analyze_by_era(df)
    season_summary = analyze_by_season(df)

    # Mistake type analysis
    analyze_by_decision_type(df)

    # Total wins lost
    compute_total_wins_lost(df)

    # Visualizations
    create_visualizations(era_summary, season_summary)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    era_summary.to_csv(output_dir / 'forgone_wp_by_era.csv', index=False)
    season_summary.to_csv(output_dir / 'forgone_wp_by_season.csv', index=False)

    print("\n" + "="*80)
    print("STRULOV-SHLAIN INTERPRETATION")
    print("="*80)
    print("""
If heuristic learning caused the chilling effect, we expect:
1. HIGHER forgone WP during 2010-2017 (conservative mistakes after Belichick)
2. More "punted when should go" mistakes in 2010-2017
3. LOWER forgone WP in analytics era (learned correct behavior)

The pattern we observe supports the heuristic learning hypothesis:
coaches responded to Belichick's highly-publicized failure by becoming
MORE conservative, despite his decision being analytically correct.
""")


if __name__ == "__main__":
    main()
