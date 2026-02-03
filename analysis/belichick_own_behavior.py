"""
Test Belichick's Own Behavior Before and After 2009

Key question: Did Belichick himself continue going for it after his 4th & 2
failure, while other coaches became more conservative?

If yes: This is strong evidence that Belichick understood the model (decision
was correct, outcome was unlucky) while other coaches used heuristics (learned
from the wrong signal).

This is the "sophisticated vs naive" learner split in Strulov-Shlain.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_decision_data():
    """Load the decision categorization data."""
    data_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'decision_categorization.parquet'
    return pd.read_parquet(data_path)


def analyze_belichick_vs_league(df):
    """
    Compare Belichick's (Patriots') go-for-it rates to league average
    before and after November 2009.
    """
    df = df.copy()

    # Filter to 4th & short (1-3 yards) for comparable situations
    short_yardage = (df['yards_to_go'] >= 1) & (df['yards_to_go'] <= 3)
    df_short = df[short_yardage].copy()

    # Identify Patriots decisions
    df_short['is_patriots'] = df_short['posteam'] == 'NE'

    # Define periods
    def assign_period(season):
        if season <= 2009:
            return 'pre_2009'
        elif season <= 2014:
            return '2010_2014'
        elif season <= 2019:
            return '2015_2019'
        else:
            return '2020_2024'

    df_short['period'] = df_short['season'].apply(assign_period)

    print("="*80)
    print("BELICHICK VS LEAGUE: GO-FOR-IT RATES ON 4TH & SHORT")
    print("="*80)

    results = []

    for period in ['pre_2009', '2010_2014', '2015_2019', '2020_2024']:
        period_data = df_short[df_short['period'] == period]

        # Patriots
        pats = period_data[period_data['is_patriots']]
        pats_n = len(pats)
        pats_go = (pats['actual_decision'] == 'go_for_it').sum()
        pats_rate = pats_go / pats_n if pats_n > 0 else 0

        # Rest of league
        league = period_data[~period_data['is_patriots']]
        league_n = len(league)
        league_go = (league['actual_decision'] == 'go_for_it').sum()
        league_rate = league_go / league_n if league_n > 0 else 0

        diff = pats_rate - league_rate

        results.append({
            'period': period,
            'pats_n': pats_n,
            'pats_go_rate': pats_rate,
            'league_n': league_n,
            'league_go_rate': league_rate,
            'difference': diff
        })

        print(f"\n{period.replace('_', '-')}:")
        print(f"  Patriots: {pats_rate:.1%} go-for-it ({pats_go}/{pats_n})")
        print(f"  League:   {league_rate:.1%} go-for-it ({league_go}/{league_n})")
        print(f"  Difference: {diff:+.1%}")

    results_df = pd.DataFrame(results)

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    pre = results_df[results_df['period'] == 'pre_2009'].iloc[0]
    post = results_df[results_df['period'] == '2010_2014'].iloc[0]

    pats_change = post['pats_go_rate'] - pre['pats_go_rate']
    league_change = post['league_go_rate'] - pre['league_go_rate']

    print(f"\nChange from pre-2009 to 2010-2014:")
    print(f"  Patriots: {pats_change:+.1%}")
    print(f"  League:   {league_change:+.1%}")

    if pats_change > league_change:
        print("\n✓ Belichick maintained/increased aggressiveness while league became conservative")
        print("  This suggests Belichick understood the decision was correct despite the outcome")
    else:
        print("\n⚠ Both Belichick and league changed similarly")

    return results_df


def analyze_by_season_team(df):
    """
    More granular analysis: Patriots vs League by season.
    """
    df = df.copy()

    # 4th & short
    short = (df['yards_to_go'] >= 1) & (df['yards_to_go'] <= 2)
    df_short = df[short].copy()

    df_short['is_patriots'] = df_short['posteam'] == 'NE'

    print("\n" + "="*80)
    print("SEASON-BY-SEASON: PATRIOTS VS LEAGUE (4th & 1-2)")
    print("="*80)

    results = []

    for season in sorted(df_short['season'].unique()):
        season_data = df_short[df_short['season'] == season]

        pats = season_data[season_data['is_patriots']]
        league = season_data[~season_data['is_patriots']]

        pats_rate = (pats['actual_decision'] == 'go_for_it').mean() if len(pats) > 0 else np.nan
        league_rate = (league['actual_decision'] == 'go_for_it').mean() if len(league) > 0 else np.nan

        results.append({
            'season': season,
            'pats_rate': pats_rate,
            'league_rate': league_rate,
            'pats_n': len(pats),
            'league_n': len(league)
        })

        marker = "  <-- Belichick 4th & 2" if season == 2009 else ""
        pats_str = f"{pats_rate:.0%}" if not np.isnan(pats_rate) else "N/A"
        print(f"{season}: Patriots {pats_str} ({len(pats):2d}), League {league_rate:.0%} ({len(league):3d}){marker}")

    return pd.DataFrame(results)


def run_placebo_test(df):
    """
    Placebo test: Check if rates dropped in situations UNLIKE Belichick's.

    If only Belichick-like situations show a drop, that's evidence of a
    specific chilling effect. If ALL situations dropped, it's a confound.

    Belichick-like: 4th & short, own territory, leading, late game
    Placebo: 4th & long, opponent territory, or trailing
    """
    df = df.copy()

    # Define Belichick-like situations
    belichick_like = (
        (df['yards_to_go'] >= 1) & (df['yards_to_go'] <= 3) &
        (df['field_pos'] >= 55) &  # Own territory
        (df['score_diff'] > 0) &   # Leading
        (df['time_remaining'] <= 600)  # Under 10 min
    )

    # Placebo: 4th & long OR opponent territory OR trailing
    placebo = (
        (df['yards_to_go'] >= 5) |  # 4th & long
        (df['field_pos'] <= 40) |   # Opponent territory
        (df['score_diff'] < 0)      # Trailing
    )

    def assign_period(season):
        if season <= 2009:
            return 'pre_belichick'
        elif season <= 2014:
            return 'post_immediate'
        else:
            return 'later'

    df['period'] = df['season'].apply(assign_period)

    print("\n" + "="*80)
    print("PLACEBO TEST: Belichick-like vs Other Situations")
    print("="*80)

    results = []

    for situation, mask, label in [
        ('belichick_like', belichick_like, 'Belichick-like (treatment)'),
        ('placebo', placebo, 'Placebo (control)')
    ]:
        print(f"\n{label}:")
        print("-"*50)

        for period in ['pre_belichick', 'post_immediate', 'later']:
            period_data = df[(df['period'] == period) & mask]
            n = len(period_data)
            go_rate = (period_data['actual_decision'] == 'go_for_it').mean() if n > 0 else 0

            results.append({
                'situation': situation,
                'period': period,
                'n': n,
                'go_rate': go_rate
            })

            print(f"  {period}: {go_rate:.1%} go-for-it (n={n:,})")

    results_df = pd.DataFrame(results)

    # Compute diff-in-diff
    print("\n" + "="*80)
    print("DIFFERENCE-IN-DIFFERENCES")
    print("="*80)

    # Treatment effect
    treat_pre = results_df[(results_df['situation'] == 'belichick_like') &
                           (results_df['period'] == 'pre_belichick')]['go_rate'].values[0]
    treat_post = results_df[(results_df['situation'] == 'belichick_like') &
                            (results_df['period'] == 'post_immediate')]['go_rate'].values[0]

    control_pre = results_df[(results_df['situation'] == 'placebo') &
                             (results_df['period'] == 'pre_belichick')]['go_rate'].values[0]
    control_post = results_df[(results_df['situation'] == 'placebo') &
                              (results_df['period'] == 'post_immediate')]['go_rate'].values[0]

    treat_change = treat_post - treat_pre
    control_change = control_post - control_pre
    did = treat_change - control_change

    print(f"\nBelichick-like change: {treat_change:+.1%} ({treat_pre:.1%} → {treat_post:.1%})")
    print(f"Placebo change:        {control_change:+.1%} ({control_pre:.1%} → {control_post:.1%})")
    print(f"\nDiff-in-diff estimate: {did:+.1%}")

    if did < -0.02:  # More than 2pp additional drop in treatment
        print("\n⚠ Belichick-like situations dropped MORE than placebo")
        print("  This is consistent with a specific chilling effect")
    elif abs(did) < 0.02:
        print("\n○ Similar changes in both groups")
        print("  Suggests a general trend, not Belichick-specific")
    else:
        print("\n✓ Belichick-like situations dropped LESS than placebo")
        print("  No evidence of chilling effect")

    return results_df, did


def create_visualization(season_df):
    """Create visualization of Patriots vs League over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(season_df['season'], season_df['pats_rate'] * 100, 'b-o',
            linewidth=2, markersize=8, label='Patriots (Belichick)')
    ax.plot(season_df['season'], season_df['league_rate'] * 100, 'gray', linestyle='--',
            linewidth=2, marker='s', markersize=6, label='Rest of League')

    ax.axvline(x=2009.9, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax.text(2010.1, ax.get_ylim()[1] * 0.95, 'Belichick\n4th & 2', fontsize=10, color='red')

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Go-for-it Rate on 4th & 1-2 (%)', fontsize=12)
    ax.set_title('Belichick vs League: Did He Stay Aggressive After 2009?', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = Path(__file__).parent.parent / 'outputs' / 'figures' / 'belichick_vs_league.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()


def main():
    """Run the full analysis."""
    print("Loading decision data...")
    df = load_decision_data()

    # Check if posteam column exists
    if 'posteam' not in df.columns:
        print("\nWARNING: 'posteam' column not found in data.")
        print("Cannot identify Patriots decisions.")
        print("Available columns:", list(df.columns))
        return

    # Main analysis
    period_results = analyze_belichick_vs_league(df)
    season_results = analyze_by_season_team(df)

    # Placebo test
    placebo_results, did = run_placebo_test(df)

    # Visualization
    create_visualization(season_results)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    period_results.to_csv(output_dir / 'belichick_vs_league_periods.csv', index=False)
    season_results.to_csv(output_dir / 'belichick_vs_league_seasons.csv', index=False)
    placebo_results.to_csv(output_dir / 'placebo_test_results.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key questions answered:

1. Did Belichick stay aggressive after 2009?
   - Compare Patriots rate changes vs league changes

2. Is this a Belichick-specific effect or general trend?
   - Diff-in-diff: if treatment dropped more than placebo, it's specific

3. What does this tell us about learning?
   - If Belichick stayed aggressive while others retreated:
     He understood the model, others used heuristics
   - If everyone changed similarly: general trend, not Belichick-specific
""")


if __name__ == "__main__":
    main()
