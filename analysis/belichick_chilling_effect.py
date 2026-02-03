"""
Test for the "Belichick Chilling Effect" on NFL fourth down decisions.

Hypothesis: The infamous Belichick 4th & 2 play (Nov 15, 2009) may have caused
other coaches to become MORE conservative, despite the decision being analytically
correct. This would be evidence of heuristic (outcome-based) learning rather than
model-based optimization.

Strulov-Shlain (2025) Framework:
- If coaches optimize against a known objective function, they should NOT react
  to other coaches' outcomes (only information, not outcomes, should matter)
- If coaches use heuristics and learn from outcomes, a highly-publicized "failure"
  could cause them to update away from the correct action

Test: Compare go-for-it rates in "Belichick-like situations" before and after
November 15, 2009.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def load_decision_data():
    """Load the decision categorization data."""
    data_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'decision_categorization.parquet'
    return pd.read_parquet(data_path)


def define_belichick_like_situations(df):
    """
    Define situations similar to Belichick's 4th & 2:
    - 4th & short (1-3 yards to go)
    - Own territory (own 20-45, i.e., field_pos 55-80)
    - Leading by 1-10 points
    - Late game (under 5 minutes remaining)

    Returns boolean mask for these situations.
    """
    mask = (
        (df['yards_to_go'] >= 1) & (df['yards_to_go'] <= 3) &  # 4th & short
        (df['field_pos'] >= 55) & (df['field_pos'] <= 80) &     # Own territory
        (df['score_diff'] >= 1) & (df['score_diff'] <= 10) &    # Leading
        (df['time_remaining'] <= 300)                            # Under 5 min
    )
    return mask


def define_short_yardage_situations(df):
    """
    Broader definition: any 4th & short situation.
    - 4th & 1-2 yards
    - Any field position
    - Any game state
    """
    mask = (df['yards_to_go'] >= 1) & (df['yards_to_go'] <= 2)
    return mask


def analyze_chilling_effect(df):
    """
    Main analysis: compare go-for-it rates before and after Belichick's play.
    """
    # Parse game dates
    df = df.copy()

    # Create period indicators
    # Pre-Belichick: 2006 - Nov 14, 2009
    # Post-Belichick (immediate): Nov 16, 2009 - Dec 31, 2012
    # Analytics era: 2018-2024

    belichick_date = datetime(2009, 11, 15)
    analytics_start = datetime(2018, 1, 1)

    def assign_period(row):
        # Use season and game_id to approximate timing
        season = row['season']
        if season < 2009:
            return 'pre_belichick'
        elif season == 2009:
            # Need to check if before or after week 10 (Belichick game was week 10)
            game_id = row['game_id']
            if '_10_' in game_id or '_11_' in game_id or '_12_' in game_id or '_13_' in game_id or '_14_' in game_id or '_15_' in game_id or '_16_' in game_id or '_17_' in game_id:
                # Approximate: games after week 10 are post-Belichick
                week_num = int(game_id.split('_')[1]) if game_id.split('_')[1].isdigit() else 0
                if week_num > 10:
                    return 'post_belichick_immediate'
                else:
                    return 'pre_belichick'
            return 'pre_belichick'
        elif season <= 2012:
            return 'post_belichick_immediate'
        elif season <= 2017:
            return 'post_belichick_later'
        else:
            return 'analytics_era'

    df['period'] = df.apply(assign_period, axis=1)

    # Filter to Belichick-like situations
    belichick_like = define_belichick_like_situations(df)
    df_belichick_like = df[belichick_like].copy()

    print("="*80)
    print("BELICHICK CHILLING EFFECT ANALYSIS")
    print("="*80)
    print(f"\nTotal fourth down plays: {len(df):,}")
    print(f"Belichick-like situations: {belichick_like.sum():,}")

    # Compute go-for-it rates by period
    print("\n" + "-"*60)
    print("GO-FOR-IT RATES IN BELICHICK-LIKE SITUATIONS")
    print("(4th & 1-3, own 20-45, leading 1-10, under 5 min)")
    print("-"*60)

    results = []
    for period in ['pre_belichick', 'post_belichick_immediate', 'post_belichick_later', 'analytics_era']:
        period_data = df_belichick_like[df_belichick_like['period'] == period]
        if len(period_data) == 0:
            continue

        n_total = len(period_data)
        n_go = (period_data['actual_decision'] == 'go_for_it').sum()
        go_rate = n_go / n_total if n_total > 0 else 0

        # Also compute optimality rate
        n_correct = period_data['ex_ante_match'].sum()
        correct_rate = n_correct / n_total if n_total > 0 else 0

        results.append({
            'period': period,
            'n_situations': n_total,
            'n_go_for_it': n_go,
            'go_rate': go_rate,
            'correct_rate': correct_rate
        })

        print(f"\n{period.upper().replace('_', ' ')}:")
        print(f"  Situations: {n_total}")
        print(f"  Go-for-it rate: {go_rate:.1%} ({n_go}/{n_total})")
        print(f"  Correct decision rate: {correct_rate:.1%}")

    results_df = pd.DataFrame(results)

    # Test the hypothesis
    print("\n" + "="*80)
    print("HYPOTHESIS TEST")
    print("="*80)

    pre_rate = results_df[results_df['period'] == 'pre_belichick']['go_rate'].values
    post_immediate_rate = results_df[results_df['period'] == 'post_belichick_immediate']['go_rate'].values

    if len(pre_rate) > 0 and len(post_immediate_rate) > 0:
        pre_rate = pre_rate[0]
        post_immediate_rate = post_immediate_rate[0]

        print(f"\nPre-Belichick go-for-it rate:  {pre_rate:.1%}")
        print(f"Post-Belichick (immediate):     {post_immediate_rate:.1%}")
        print(f"Change:                         {post_immediate_rate - pre_rate:+.1%}")

        if post_immediate_rate < pre_rate:
            print("\n⚠️  CHILLING EFFECT DETECTED: Coaches became MORE conservative")
            print("   after Belichick's analytically-correct decision failed.")
            print("   This is consistent with heuristic (outcome-based) learning.")
        else:
            print("\n✓ No chilling effect detected: Coaches did not become more conservative.")

    return results_df, df_belichick_like


def analyze_by_season(df):
    """
    Plot go-for-it rates by season for visual inspection.
    """
    df = df.copy()

    # Short yardage situations (more data)
    short_yardage = define_short_yardage_situations(df)
    df_short = df[short_yardage].copy()

    # Compute by season
    season_stats = df_short.groupby('season').agg(
        n_total=('actual_decision', 'count'),
        n_go=('actual_decision', lambda x: (x == 'go_for_it').sum()),
        n_correct=('ex_ante_match', 'sum')
    ).reset_index()

    season_stats['go_rate'] = season_stats['n_go'] / season_stats['n_total']
    season_stats['correct_rate'] = season_stats['n_correct'] / season_stats['n_total']

    print("\n" + "="*80)
    print("GO-FOR-IT RATES BY SEASON (4th & 1-2, all situations)")
    print("="*80)

    for _, row in season_stats.iterrows():
        marker = "  <-- Belichick 4th & 2" if row['season'] == 2009 else ""
        marker = "  <-- Analytics revolution" if row['season'] == 2018 else marker
        print(f"{int(row['season'])}: {row['go_rate']:5.1%} go-for-it, {row['correct_rate']:5.1%} correct ({int(row['n_total']):3d} plays){marker}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Go-for-it rate
    ax1 = axes[0]
    ax1.plot(season_stats['season'], season_stats['go_rate'], 'b-o', linewidth=2, markersize=6)
    ax1.axvline(x=2009.9, color='red', linestyle='--', alpha=0.7, label='Belichick 4th & 2 (Nov 2009)')
    ax1.axvline(x=2018, color='green', linestyle='--', alpha=0.7, label='Analytics era begins')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Go-for-it Rate', fontsize=12)
    ax1.set_title('4th & Short (1-2 yards): Go-for-it Rate by Season', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(season_stats['go_rate']) * 1.2)

    # Plot 2: Correct decision rate
    ax2 = axes[1]
    ax2.plot(season_stats['season'], season_stats['correct_rate'], 'g-o', linewidth=2, markersize=6)
    ax2.axvline(x=2009.9, color='red', linestyle='--', alpha=0.7, label='Belichick 4th & 2 (Nov 2009)')
    ax2.axvline(x=2018, color='green', linestyle='--', alpha=0.7, label='Analytics era begins')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Correct Decision Rate', fontsize=12)
    ax2.set_title('4th & Short (1-2 yards): Correct Decision Rate by Season', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Save figure
    fig_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'belichick_chilling_effect.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

    plt.close()

    return season_stats


def main():
    """Run the full analysis."""
    print("Loading decision data...")
    df = load_decision_data()

    # Main analysis
    results_df, df_belichick_like = analyze_chilling_effect(df)

    # Season-by-season analysis
    season_stats = analyze_by_season(df)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    results_df.to_csv(output_dir / 'belichick_chilling_effect.csv', index=False)
    season_stats.to_csv(output_dir / 'fourth_down_by_season.csv', index=False)

    print("\n" + "="*80)
    print("STRULOV-SHLAIN INTERPRETATION")
    print("="*80)
    print("""
If we observe a DROP in go-for-it rates after Belichick's play (Nov 2009),
this is evidence of HEURISTIC LEARNING:

1. Coaches observed: "Belichick went for it and LOST"
2. Outcome-based update: "Going for it is risky"
3. This is the WRONG inference - the decision was correct, outcome was unlucky

This parallels Strulov-Shlain's finding that retailers learned from outcomes
(price endings that "worked") rather than optimizing against the true objective.

KEY PREDICTION: If coaches use heuristics, we should see:
- DROP in go-for-it rates after 2009 (wrong learning from Belichick outcome)
- RISE in go-for-it rates after ~2018 (analytics movement overcomes heuristic)

If coaches optimize, we should see:
- NO CHANGE after 2009 (Belichick's outcome is irrelevant to the optimal decision)
""")


if __name__ == "__main__":
    main()
