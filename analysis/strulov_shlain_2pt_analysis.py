"""
Strulov-Shlain (2025) Style Analysis: 2015 PAT Rule Change

The 2015 NFL PAT rule change provides a clean natural experiment analogous to
the Israeli pricing reform in Strulov-Shlain (2025):

BEFORE 2015: PAT from 2-yard line (~99% success)
AFTER 2015: PAT from 15-yard line (~94% success)

This 5 percentage point drop in PAT success substantially changes the
optimal 2-point conversion rate. If coaches optimize, they should:
1. Immediately increase 2pt attempts after 2015
2. Adjust their 2pt rate to match the new optimal

If coaches use heuristics (outcome-based learning), we might see:
1. Slow adjustment
2. Persistent gap between actual and optimal rates
3. Different adjustment speeds for different situation types

Key tests:
1. Immediate jump test: Did 2pt rate jump discontinuously in 2015?
2. Gap persistence: How large is the gap between actual and optimal?
3. Diff-in-diff: Treatment (situations where optimal changed) vs Control
4. Forgone WP: Welfare cost of suboptimal decisions by era
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Make matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def load_pbp_data():
    """Load play-by-play data."""
    data_dir = Path(__file__).parent.parent / 'data'
    return pd.read_parquet(data_dir / 'all_pbp_1999_2024.parquet')


def compute_pat_success_rates(pbp):
    """
    Task 1: Document PAT success rate pre/post 2015.
    """
    print("=" * 80)
    print("TASK 1: PAT SUCCESS RATES PRE/POST 2015 RULE CHANGE")
    print("=" * 80)

    pat_data = pbp[pbp['extra_point_attempt'] == 1].copy()
    pat_data = pat_data[pat_data['extra_point_result'].notna()]
    pat_data['success'] = (pat_data['extra_point_result'] == 'good').astype(int)

    # By season
    season_rates = pat_data.groupby('season').agg(
        attempts=('success', 'count'),
        successes=('success', 'sum'),
        rate=('success', 'mean')
    ).reset_index()

    print("\nPAT Success Rate by Season:")
    print("-" * 50)
    for _, row in season_rates.iterrows():
        marker = "  <-- Rule change" if row['season'] == 2015 else ""
        print(f"  {int(row['season'])}: {row['rate']:.1%} ({int(row['successes'])}/{int(row['attempts'])}){marker}")

    # Pre vs Post summary
    pre_2015 = pat_data[pat_data['season'] < 2015]
    post_2015 = pat_data[pat_data['season'] >= 2015]

    pre_rate = pre_2015['success'].mean()
    post_rate = post_2015['success'].mean()

    print("\n" + "-" * 50)
    print(f"Pre-2015 (2006-2014): {pre_rate:.2%} ({pre_2015['success'].sum()}/{len(pre_2015)})")
    print(f"Post-2015 (2015-2024): {post_rate:.2%} ({post_2015['success'].sum()}/{len(post_2015)})")
    print(f"Change: {(post_rate - pre_rate)*100:+.1f} percentage points")

    return season_rates


def compute_2pt_success_rates(pbp):
    """Compute 2pt conversion success rates by season."""
    two_pt = pbp[pbp['two_point_attempt'] == 1].copy()
    two_pt = two_pt[two_pt['two_point_conv_result'].notna()]
    two_pt['success'] = (two_pt['two_point_conv_result'] == 'success').astype(int)

    season_rates = two_pt.groupby('season').agg(
        attempts=('success', 'count'),
        successes=('success', 'sum'),
        rate=('success', 'mean')
    ).reset_index()

    print("\n2PT Conversion Success Rate by Season:")
    print("-" * 50)
    for _, row in season_rates.iterrows():
        if row['attempts'] >= 10:  # Only show seasons with enough data
            print(f"  {int(row['season'])}: {row['rate']:.1%} ({int(row['successes'])}/{int(row['attempts'])})")

    return season_rates


def compute_actual_2pt_attempt_rate(pbp):
    """
    Task 2: Time series of actual 2pt attempt rate.
    """
    print("\n" + "=" * 80)
    print("TASK 2: ACTUAL 2PT ATTEMPT RATE BY SEASON")
    print("=" * 80)

    # Get all post-TD situations
    pat_plays = pbp[pbp['extra_point_attempt'] == 1].copy()
    two_pt_plays = pbp[pbp['two_point_attempt'] == 1].copy()

    pat_plays['decision'] = 'pat'
    two_pt_plays['decision'] = '2pt'

    all_decisions = pd.concat([pat_plays, two_pt_plays])

    season_rates = all_decisions.groupby('season').agg(
        total=('decision', 'count'),
        n_2pt=('decision', lambda x: (x == '2pt').sum())
    ).reset_index()
    season_rates['actual_2pt_rate'] = season_rates['n_2pt'] / season_rates['total']

    print("\nActual 2PT Attempt Rate by Season:")
    print("-" * 50)
    for _, row in season_rates.iterrows():
        marker = "  <-- Rule change" if row['season'] == 2015 else ""
        print(f"  {int(row['season'])}: {row['actual_2pt_rate']:.1%} ({int(row['n_2pt'])}/{int(row['total'])} attempts){marker}")

    # Pre vs Post
    pre = season_rates[season_rates['season'] < 2015]
    post = season_rates[season_rates['season'] >= 2015]

    pre_rate = pre['n_2pt'].sum() / pre['total'].sum()
    post_rate = post['n_2pt'].sum() / post['total'].sum()

    print("\n" + "-" * 50)
    print(f"Pre-2015 average: {pre_rate:.2%}")
    print(f"Post-2015 average: {post_rate:.2%}")
    print(f"Change: {(post_rate - pre_rate)*100:+.1f} percentage points")

    # Immediate jump test (2014 vs 2015)
    rate_2014 = season_rates[season_rates['season'] == 2014]['actual_2pt_rate'].values
    rate_2015 = season_rates[season_rates['season'] == 2015]['actual_2pt_rate'].values

    if len(rate_2014) > 0 and len(rate_2015) > 0:
        jump = rate_2015[0] - rate_2014[0]
        print(f"\nImmediate jump (2014 → 2015): {jump*100:+.1f} pp")

    return season_rates


def compute_optimal_2pt_rate_simple(pbp):
    """
    Task 3: Compute optimal 2pt rate using simple expected value calculation.

    Optimal decision depends on:
    - P(PAT success)
    - P(2pt success)
    - Game state (score diff, time remaining)

    For simplicity, we'll compute the "indifference threshold" -
    the game states where 2pt becomes optimal.

    E[points | PAT] = 1 * P(PAT)
    E[points | 2pt] = 2 * P(2pt)

    In most situations, 2pt is better if: 2 * P(2pt) > 1 * P(PAT)
    => P(2pt) > P(PAT) / 2

    Pre-2015: P(PAT) ≈ 0.99, so need P(2pt) > 0.495
    Post-2015: P(PAT) ≈ 0.94, so need P(2pt) > 0.47

    With P(2pt) ≈ 0.48, this changes from "always PAT" to "sometimes 2pt"
    """
    print("\n" + "=" * 80)
    print("TASK 3: OPTIMAL 2PT RATE BY ERA")
    print("=" * 80)

    # Get success rates by era
    pat_data = pbp[pbp['extra_point_attempt'] == 1].copy()
    pat_data = pat_data[pat_data['extra_point_result'].notna()]
    pat_data['success'] = (pat_data['extra_point_result'] == 'good').astype(int)

    two_pt = pbp[pbp['two_point_attempt'] == 1].copy()
    two_pt = two_pt[two_pt['two_point_conv_result'].notna()]
    two_pt['success'] = (two_pt['two_point_conv_result'] == 'success').astype(int)

    results = []

    for era, start, end in [('pre_2015', 2006, 2014), ('post_2015', 2015, 2024)]:
        pat_era = pat_data[(pat_data['season'] >= start) & (pat_data['season'] <= end)]
        two_pt_era = two_pt[(two_pt['season'] >= start) & (two_pt['season'] <= end)]

        p_pat = pat_era['success'].mean()
        p_2pt = two_pt_era['success'].mean()

        # Expected points
        ev_pat = 1 * p_pat
        ev_2pt = 2 * p_2pt

        # Threshold for 2pt to be better (in expected points)
        threshold = p_pat / 2

        print(f"\n{era.upper()} ({start}-{end}):")
        print(f"  P(PAT success): {p_pat:.3f}")
        print(f"  P(2pt success): {p_2pt:.3f}")
        print(f"  E[points|PAT]: {ev_pat:.3f}")
        print(f"  E[points|2pt]: {ev_2pt:.3f}")
        print(f"  2pt threshold: P(2pt) > {threshold:.3f}")

        if ev_2pt > ev_pat:
            print(f"  => 2pt is better on average! (by {ev_2pt - ev_pat:.3f} expected points)")
        else:
            print(f"  => PAT is better on average (by {ev_pat - ev_2pt:.3f} expected points)")

        results.append({
            'era': era,
            'p_pat': p_pat,
            'p_2pt': p_2pt,
            'ev_pat': ev_pat,
            'ev_2pt': ev_2pt,
            'threshold': threshold
        })

    print("\n" + "-" * 50)
    print("KEY INSIGHT:")
    print("The rule change shifted the calculus. With P(PAT) dropping from ~99%")
    print("to ~94%, the threshold for 2pt being optimal dropped from P(2pt) > 0.495")
    print("to P(2pt) > 0.47. Since actual P(2pt) ≈ 0.48, this puts coaches right")
    print("at the indifference point post-2015, making 2pt much more attractive")
    print("in specific game situations (e.g., needing to score 8+ points).")

    return pd.DataFrame(results)


def identify_dominated_choices(pbp):
    """
    Task 4: Identify "dominated" choices - situations where 2pt is
    clearly optimal regardless of the rule change.

    Key dominated situations:
    1. Down 2 late in game (need 8 points = TD + 2pt)
    2. Down 5 late in game (need 8 points exactly)
    3. Certain score differentials where expected WP is clearly higher

    These serve as "control" for diff-in-diff.
    """
    print("\n" + "=" * 80)
    print("TASK 4: IDENTIFY DOMINATED CHOICES (ALWAYS SHOULD GO FOR 2)")
    print("=" * 80)

    pat_plays = pbp[pbp['extra_point_attempt'] == 1].copy()
    two_pt_plays = pbp[pbp['two_point_attempt'] == 1].copy()

    pat_plays['decision'] = 'pat'
    two_pt_plays['decision'] = '2pt'

    all_decisions = pd.concat([pat_plays, two_pt_plays])
    all_decisions['score_diff_pre_td'] = all_decisions['score_differential'] - 6
    all_decisions = all_decisions[all_decisions['game_seconds_remaining'].notna()]

    # Define dominated situations: late game (under 5 min) AND down by specific amounts
    # Down 2 pre-TD (now tied, need 8 to go up 2) => 2pt always better
    # Down 5 pre-TD (now +1, need 8 to be up 3) => 2pt often better

    dominated_mask = (
        (all_decisions['game_seconds_remaining'] <= 300) &  # Under 5 min
        (all_decisions['score_diff_pre_td'].isin([-2, -5, -9, -12]))  # Specific situations
    )

    dominated = all_decisions[dominated_mask]

    print(f"\nDominated situations (should always go for 2):")
    print(f"  Total plays: {len(dominated)}")
    print(f"  Actual 2pt rate: {(dominated['decision'] == '2pt').mean():.1%}")

    # Break down by score diff
    print("\nBy pre-TD score differential:")
    print("-" * 50)
    for score_diff in [-2, -5, -9, -12]:
        subset = dominated[dominated['score_diff_pre_td'] == score_diff]
        if len(subset) >= 5:
            rate = (subset['decision'] == '2pt').mean()
            print(f"  Down {abs(score_diff)} pre-TD: {rate:.1%} went for 2pt ({len(subset)} plays)")

    # By era
    print("\nBy era:")
    print("-" * 50)
    for era, label in [((2006, 2014), 'Pre-2015'), ((2015, 2024), 'Post-2015')]:
        era_data = dominated[(dominated['season'] >= era[0]) & (dominated['season'] <= era[1])]
        if len(era_data) >= 5:
            rate = (era_data['decision'] == '2pt').mean()
            print(f"  {label}: {rate:.1%} went for 2pt ({len(era_data)} plays)")

    return dominated


def compute_diff_in_diff(pbp):
    """
    Task 5: Diff-in-diff analysis.

    Treatment: Situations where optimal choice changed from PAT to 2pt after 2015
    Control: Dominated situations (always should be 2pt)

    If coaches optimize: Both should adjust proportionally
    If coaches use heuristics: Treatment may lag behind control
    """
    print("\n" + "=" * 80)
    print("TASK 5: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 80)

    pat_plays = pbp[pbp['extra_point_attempt'] == 1].copy()
    two_pt_plays = pbp[pbp['two_point_attempt'] == 1].copy()

    pat_plays['decision'] = 'pat'
    two_pt_plays['decision'] = '2pt'

    all_decisions = pd.concat([pat_plays, two_pt_plays])
    all_decisions['score_diff_pre_td'] = all_decisions['score_differential'] - 6
    all_decisions = all_decisions[all_decisions['game_seconds_remaining'].notna()]

    # CONTROL: Dominated situations (down 2, 5, 9, 12 pre-TD, late game)
    control_mask = (
        (all_decisions['game_seconds_remaining'] <= 300) &
        (all_decisions['score_diff_pre_td'].isin([-2, -5, -9, -12]))
    )

    # TREATMENT: Marginal situations that changed with rule change
    # Close games, moderate time remaining
    treatment_mask = (
        (all_decisions['game_seconds_remaining'] > 300) &
        (all_decisions['game_seconds_remaining'] <= 1200) &  # 5-20 min
        (all_decisions['score_diff_pre_td'].isin([0, -1, -4, -8]))  # Marginal score states
    )

    results = []

    for group, mask, label in [('control', control_mask, 'Dominated (always 2pt)'),
                                ('treatment', treatment_mask, 'Marginal (changed with rule)')]:
        subset = all_decisions[mask]

        pre = subset[subset['season'] < 2015]
        post = subset[subset['season'] >= 2015]

        pre_rate = (pre['decision'] == '2pt').mean() if len(pre) > 0 else np.nan
        post_rate = (post['decision'] == '2pt').mean() if len(post) > 0 else np.nan

        results.append({
            'group': group,
            'label': label,
            'pre_rate': pre_rate,
            'post_rate': post_rate,
            'change': post_rate - pre_rate if not (np.isnan(pre_rate) or np.isnan(post_rate)) else np.nan,
            'n_pre': len(pre),
            'n_post': len(post)
        })

        print(f"\n{label}:")
        print(f"  Pre-2015: {pre_rate:.1%} ({len(pre)} plays)")
        print(f"  Post-2015: {post_rate:.1%} ({len(post)} plays)")
        print(f"  Change: {(post_rate - pre_rate)*100:+.1f} pp")

    results_df = pd.DataFrame(results)

    # Diff-in-diff
    control_change = results_df[results_df['group'] == 'control']['change'].values[0]
    treatment_change = results_df[results_df['group'] == 'treatment']['change'].values[0]
    did = treatment_change - control_change

    print("\n" + "-" * 50)
    print("DIFF-IN-DIFF ESTIMATE:")
    print(f"  Control change: {control_change*100:+.1f} pp")
    print(f"  Treatment change: {treatment_change*100:+.1f} pp")
    print(f"  DiD: {did*100:+.1f} pp")

    if abs(did) < 0.02:
        print("\n  => Similar changes in both groups (coaches responding uniformly)")
    elif did > 0.02:
        print("\n  => Treatment increased MORE than control (coaches responding to new optimum)")
    else:
        print("\n  => Treatment increased LESS than control (sluggish adjustment)")

    return results_df


def compute_forgone_wp_all_years(pbp):
    """
    Compute forgone win probability for ALL years (2006-2024).

    For post-2015: Uses the existing two_point_decision_analysis.parquet
    For pre-2015: Computes wp_cost using the same methodology with era-appropriate PAT rates
    """
    print("\n" + "=" * 80)
    print("TASK 6: FORGONE WIN PROBABILITY BY SEASON (ALL YEARS)")
    print("=" * 80)

    # Load the post-2015 WP analysis data
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    try:
        post_2015_data = pd.read_parquet(output_dir / 'two_point_decision_analysis.parquet')
        print(f"Loaded post-2015 WP data: {len(post_2015_data)} decisions")
    except FileNotFoundError:
        print("ERROR: two_point_decision_analysis.parquet not found")
        return None

    # Compute post-2015 summary by season
    post_2015_summary = post_2015_data.groupby('season').agg(
        n_decisions=('wp_cost', 'count'),
        mean_wp_cost=('wp_cost', 'mean'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        optimal_2pt_rate=('optimal_decision', lambda x: (x == 'two_point').mean())
    ).reset_index()

    # Now compute pre-2015 using pbp data with same methodology
    # Get all PAT and 2pt decisions pre-2015
    pre_2015_pbp = pbp[pbp['season'] < 2015].copy()

    pat_plays = pre_2015_pbp[pre_2015_pbp['extra_point_attempt'] == 1].copy()
    two_pt_plays = pre_2015_pbp[pre_2015_pbp['two_point_attempt'] == 1].copy()

    pat_plays['actual_decision'] = 'pat'
    two_pt_plays['actual_decision'] = '2pt'

    pre_2015_decisions = pd.concat([pat_plays, two_pt_plays])
    pre_2015_decisions = pre_2015_decisions[pre_2015_decisions['game_seconds_remaining'].notna()]
    pre_2015_decisions = pre_2015_decisions[pre_2015_decisions['vegas_wp'].notna()]
    pre_2015_decisions['score_diff_pre_td'] = pre_2015_decisions['score_differential'] - 6

    print(f"Computing WP cost for {len(pre_2015_decisions)} pre-2015 decisions...")

    # Era-specific success rates
    p_pat_pre = 0.990  # Pre-2015 PAT success
    p_2pt = 0.48  # 2pt success rate

    def compute_wp_cost_pre2015(row):
        """
        Compute WP cost for pre-2015 decisions.

        Uses same methodology as two_point_analysis.py:
        - wp_pat = p_pat * WP(score+1) + (1-p_pat) * WP(score+0)
        - wp_2pt = p_2pt * WP(score+2) + (1-p_2pt) * WP(score+0)
        - wp_cost = |wp_optimal - wp_actual|

        We approximate WP changes using the vegas_wp field and empirical
        relationships between score differential and WP.
        """
        score_diff = row['score_diff_pre_td']  # Pre-TD score differential
        time_left = row['game_seconds_remaining']
        current_wp = row['vegas_wp']
        went_for_2 = row['actual_decision'] == '2pt'

        # Estimate WP impact per point based on time remaining
        # Empirically, 1 point ≈ 2-4% WP depending on time
        if time_left <= 300:  # Under 5 min
            wp_per_point = 0.04 * (1 + (300 - time_left) / 300)  # Up to 8% per point
        elif time_left <= 900:  # 5-15 min
            wp_per_point = 0.025
        else:  # Early game
            wp_per_point = 0.015

        # Compute expected WP for each choice
        # After TD, team is at score_diff + 6
        # PAT: +1 point with p_pat probability
        # 2pt: +2 points with p_2pt probability

        # WP if PAT
        wp_if_pat_make = current_wp + wp_per_point  # +1 point
        wp_if_pat_miss = current_wp  # +0 points
        wp_pat = p_pat_pre * wp_if_pat_make + (1 - p_pat_pre) * wp_if_pat_miss

        # WP if 2pt
        wp_if_2pt_success = current_wp + 2 * wp_per_point  # +2 points
        wp_if_2pt_fail = current_wp  # +0 points
        wp_2pt = p_2pt * wp_if_2pt_success + (1 - p_2pt) * wp_if_2pt_fail

        # But wait - this oversimplifies. The real value of 2pt depends on score situation.
        # For specific score differentials, 2pt has strategic value beyond just points.

        # Key situations where 2pt has extra strategic value:
        strategic_bonus = 0
        if time_left <= 600:  # Under 10 min
            if score_diff == -2:  # Down 2 pre-TD -> tied -> need 2 to go up by 2
                strategic_bonus = 0.03
            elif score_diff == -5:  # Down 5 pre-TD -> up 1 -> need 2 to go up by 3 (FG range)
                strategic_bonus = 0.025
            elif score_diff == -9:  # Down 9 pre-TD -> down 3 -> PAT still down 2, 2pt ties
                strategic_bonus = 0.02
            elif score_diff == -8:  # Down 8 pre-TD -> down 2 -> PAT down 1, 2pt ties
                strategic_bonus = 0.015

        wp_2pt += strategic_bonus * p_2pt

        # Determine optimal and compute cost
        optimal_is_2pt = wp_2pt > wp_pat
        wp_margin = abs(wp_2pt - wp_pat)

        if optimal_is_2pt:
            if went_for_2:
                wp_cost = 0
            else:
                wp_cost = wp_margin
        else:
            if went_for_2:
                wp_cost = wp_margin
            else:
                wp_cost = 0

        return pd.Series({
            'wp_cost': wp_cost,
            'optimal_is_2pt': optimal_is_2pt,
            'coach_correct': (optimal_is_2pt == went_for_2),
            'wp_pat': wp_pat,
            'wp_2pt': wp_2pt
        })

    # Apply WP cost calculation to pre-2015 data
    wp_results = pre_2015_decisions.apply(compute_wp_cost_pre2015, axis=1)
    pre_2015_decisions = pd.concat([pre_2015_decisions, wp_results], axis=1)

    # Summarize pre-2015 by season
    pre_2015_summary = pre_2015_decisions.groupby('season').agg(
        n_decisions=('wp_cost', 'count'),
        mean_wp_cost=('wp_cost', 'mean'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == '2pt').mean()),
        optimal_2pt_rate=('optimal_is_2pt', 'mean')
    ).reset_index()

    # Combine pre and post 2015 data
    all_seasons = pd.concat([pre_2015_summary, post_2015_summary], ignore_index=True)
    all_seasons = all_seasons.sort_values('season').reset_index(drop=True)

    print("\nForgone WP by Season (WP Model):")
    print("-" * 80)
    print(f"{'Season':<8} | {'Mean WP Cost':>14} | {'Correct %':>10} | {'Actual 2pt':>10} | {'Optimal 2pt':>11} | {'N':>6}")
    print("-" * 80)

    for _, row in all_seasons.iterrows():
        marker = " <--" if row['season'] == 2015 else ""
        # Display in same units as original (wp_cost * 1,000,000)
        wp_scaled = row['mean_wp_cost'] * 1000000
        print(f"{int(row['season']):<8} | {wp_scaled:>14.1f} | "
              f"{row['pct_correct']:>10.1%} | {row['actual_2pt_rate']:>10.1%} | "
              f"{row['optimal_2pt_rate']:>11.1%} | {int(row['n_decisions']):>6}{marker}")

    return all_seasons


def compute_forgone_wp_by_era(pbp):
    """
    Task 6: Compute forgone WP by era.

    Using the existing two_point_decision_analysis.parquet data which has
    wp_cost for each decision.
    """
    print("\n" + "=" * 80)
    print("TASK 6b: FORGONE WIN PROBABILITY BY ERA (WP MODEL)")
    print("=" * 80)

    # Try to load existing analysis data
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'

    try:
        df = pd.read_parquet(output_dir / 'two_point_decision_analysis.parquet')
    except FileNotFoundError:
        print("ERROR: two_point_decision_analysis.parquet not found")
        print("Run two_point_analysis.py first to generate this data")
        return None

    print(f"\nLoaded {len(df)} decisions with WP cost data")

    # Assign eras
    def assign_era(season):
        if season < 2015:
            return 'pre_2015'
        elif season <= 2017:
            return 'early_post'
        elif season <= 2020:
            return 'mid_post'
        else:
            return 'late_post'

    df['era'] = df['season'].apply(assign_era)

    # Summary by era
    era_summary = df.groupby('era').agg(
        n_decisions=('wp_cost', 'count'),
        mean_wp_cost=('wp_cost', 'mean'),
        median_wp_cost=('wp_cost', 'median'),
        total_wp_cost=('wp_cost', 'sum'),
        pct_correct=('coach_correct', 'mean')
    ).reset_index()

    era_order = ['pre_2015', 'early_post', 'mid_post', 'late_post']
    era_summary['era'] = pd.Categorical(era_summary['era'], categories=era_order, ordered=True)
    era_summary = era_summary.sort_values('era')

    print("\nForgone WP by Era:")
    print("-" * 60)

    for _, row in era_summary.iterrows():
        if pd.notna(row['mean_wp_cost']):
            print(f"\n{row['era'].upper()}:")
            print(f"  Decisions: {int(row['n_decisions']):,}")
            print(f"  Mean WP cost: {row['mean_wp_cost']*100:.3f}%")
            print(f"  Correct rate: {row['pct_correct']:.1%}")

    # By season for time series
    season_summary = df.groupby('season').agg(
        n_decisions=('wp_cost', 'count'),
        mean_wp_cost=('wp_cost', 'mean'),
        pct_correct=('coach_correct', 'mean')
    ).reset_index()

    print("\n\nForgone WP by Season (WP Model):")
    print("-" * 60)
    for _, row in season_summary.iterrows():
        marker = "  <-- Rule change" if row['season'] == 2015 else ""
        print(f"  {int(row['season'])}: {row['mean_wp_cost']*100:.3f}% mean cost, {row['pct_correct']:.1%} correct ({int(row['n_decisions'])} plays){marker}")

    return era_summary, season_summary


def create_strulov_shlain_figures(pbp, pat_rates, attempt_rates, decision_quality, forgone_wp=None):
    """
    Task 7: Create visualizations analogous to Strulov-Shlain (2025).

    Figure 1: PAT success rate discontinuity
    Figure 2: Actual vs optimal 2pt rate over time
    Figure 3: Gap between actual and optimal
    Figure 4: Decision quality over time (including pre-2015)
    """
    print("\n" + "=" * 80)
    print("TASK 7: CREATING STRULOV-SHLAIN STYLE VISUALIZATIONS")
    print("=" * 80)

    if not HAS_MATPLOTLIB:
        print("Skipping visualizations (matplotlib not available)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # =========================================================================
    # Panel A: PAT Success Rate (shows the "treatment" discontinuity)
    # =========================================================================
    ax = axes[0, 0]

    # Get PAT rates by season
    pat_data = pbp[pbp['extra_point_attempt'] == 1].copy()
    pat_data = pat_data[pat_data['extra_point_result'].notna()]
    pat_data['success'] = (pat_data['extra_point_result'] == 'good').astype(int)

    pat_by_season = pat_data.groupby('season')['success'].mean().reset_index()

    ax.plot(pat_by_season['season'], pat_by_season['success'] * 100, 'o-',
            color='#3498db', linewidth=2, markersize=8)
    ax.axvline(x=2014.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(2014.5, 2015.5, alpha=0.15, color='red')

    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('PAT Success Rate (%)', fontsize=11)
    ax.set_title('A. PAT Success Rate: The Rule Change', fontsize=12, fontweight='bold')
    ax.set_ylim(88, 101)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('2015 Rule\nChange', xy=(2014.5, 95), xytext=(2010, 91),
                fontsize=9, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    pre_avg = pat_by_season[pat_by_season['season'] < 2015]['success'].mean() * 100
    post_avg = pat_by_season[pat_by_season['season'] >= 2015]['success'].mean() * 100
    ax.text(2009, 99, f'Pre-2015: {pre_avg:.0f}%', fontsize=9, color='#3498db')
    ax.text(2019, 93.5, f'Post-2015: {post_avg:.0f}%', fontsize=9, color='#3498db')

    # =========================================================================
    # Panel B: Actual 2pt Attempt Rate (behavioral response)
    # =========================================================================
    ax = axes[0, 1]

    ax.plot(attempt_rates['season'], attempt_rates['actual_2pt_rate'] * 100, 'o-',
            color='#27ae60', linewidth=2, markersize=8, label='Actual 2pt rate')
    ax.axvline(x=2014.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(2014.5, 2015.5, alpha=0.15, color='red')

    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('2pt Attempt Rate (%)', fontsize=11)
    ax.set_title('B. Behavioral Response: 2pt Attempt Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(attempt_rates['actual_2pt_rate'] * 100) * 1.3)

    # =========================================================================
    # Panel C: Actual vs Optimal 2pt Rate (if we have optimal data)
    # =========================================================================
    ax = axes[1, 0]

    # Use decision_quality data for actual and optimal rates
    seasons = decision_quality['season'].values
    actual = decision_quality['actual_2pt_rate'].values * 100
    optimal = decision_quality['optimal_2pt_rate'].values * 100

    ax.plot(seasons, actual, 'o-', color='#27ae60', linewidth=2, markersize=8, label='Actual')
    ax.plot(seasons, optimal, 's--', color='#e74c3c', linewidth=2, markersize=6, alpha=0.7, label='Optimal')

    ax.fill_between(seasons, actual, optimal, where=(optimal > actual),
                    alpha=0.2, color='#e74c3c', label='Gap (undershoot)')
    ax.fill_between(seasons, actual, optimal, where=(actual > optimal),
                    alpha=0.2, color='#27ae60', label='Gap (overshoot)')
    ax.axvline(x=2014.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('2pt Rate (%)', fontsize=11)
    ax.set_title('C. Gap: Actual vs Optimal 2pt Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel D: Forgone Win Probability Over Time
    # =========================================================================
    ax = axes[1, 1]

    # Plot mean forgone WP by season (scaled by 1,000,000 to match original units)
    seasons = decision_quality['season'].values
    forgone_wp = decision_quality['mean_wp_cost'].values * 1000000  # Scale to original units

    ax.plot(seasons, forgone_wp, 'o-', color='#9b59b6', linewidth=2, markersize=8)
    ax.axvline(x=2014.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(2014.5, 2015.5, alpha=0.15, color='red')

    # Compute pre vs post averages
    pre_2015 = decision_quality[decision_quality['season'] < 2015]
    post_2015 = decision_quality[decision_quality['season'] >= 2015]

    if len(pre_2015) > 0 and len(post_2015) > 0:
        pre_avg = pre_2015['mean_wp_cost'].mean() * 1000000
        post_avg = post_2015['mean_wp_cost'].mean() * 1000000

        ax.axhline(y=pre_avg, color='#9b59b6', linestyle=':', alpha=0.5, xmin=0, xmax=0.45)
        ax.axhline(y=post_avg, color='#9b59b6', linestyle=':', alpha=0.5, xmin=0.55, xmax=1)

        ax.text(2008, pre_avg + 20, f'Pre-2015: {pre_avg:.0f}', fontsize=9, color='#9b59b6')
        ax.text(2019, post_avg + 20, f'Post-2015: {post_avg:.0f}', fontsize=9, color='#9b59b6')

    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Forgone WP (×10⁻⁶)', fontsize=11)
    ax.set_title('D. Decision Quality: Forgone Win Probability', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Set y-axis to show reasonable range
    if len(forgone_wp) > 0:
        ax.set_ylim(0, max(forgone_wp) * 1.2)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / 'strulov_shlain_2pt_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return fig


def main():
    """Run the full Strulov-Shlain style analysis."""
    print("=" * 80)
    print("STRULOV-SHLAIN (2025) STYLE ANALYSIS: 2015 PAT RULE CHANGE")
    print("=" * 80)
    print("""
This analysis applies the heuristic learning framework from Strulov-Shlain (2025)
to the 2015 NFL PAT rule change.

The key insight: When the NFL moved PATs from the 2-yard line to the 15-yard line,
PAT success dropped from ~99% to ~94%. This exogenous shock changed the optimal
2-point conversion rate.

If coaches optimize: They should immediately increase 2pt attempts
If coaches use heuristics: Adjustment should be slow and incomplete
""")

    # Load data
    print("\nLoading play-by-play data...")
    pbp = load_pbp_data()
    print(f"Loaded {len(pbp):,} plays from {pbp['season'].min()}-{pbp['season'].max()}")

    # Run all analyses
    pat_rates = compute_pat_success_rates(pbp)
    two_pt_rates = compute_2pt_success_rates(pbp)
    attempt_rates = compute_actual_2pt_attempt_rate(pbp)
    optimal_rates = compute_optimal_2pt_rate_simple(pbp)
    dominated = identify_dominated_choices(pbp)
    did_results = compute_diff_in_diff(pbp)
    decision_quality = compute_forgone_wp_all_years(pbp)  # Now includes forgone WP for all years
    forgone_wp = compute_forgone_wp_by_era(pbp)

    # Create visualizations
    create_strulov_shlain_figures(pbp, pat_rates, attempt_rates, decision_quality, forgone_wp)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    attempt_rates.to_csv(output_dir / 'strulov_shlain_2pt_attempt_rates.csv', index=False)
    did_results.to_csv(output_dir / 'strulov_shlain_diff_in_diff.csv', index=False)
    optimal_rates.to_csv(output_dir / 'strulov_shlain_optimal_rates.csv', index=False)

    print("\n" + "=" * 80)
    print("SUMMARY: STRULOV-SHLAIN INTERPRETATION")
    print("=" * 80)
    print("""
KEY FINDINGS:

1. RULE CHANGE EFFECT:
   - PAT success dropped from ~99% to ~94% after 2015 rule change
   - This is a clean, exogenous shock to the decision environment

2. BEHAVIORAL RESPONSE:
   - 2pt attempt rate [see output above]
   - Compare to optimal response

3. DIFF-IN-DIFF:
   - Treatment: Marginal situations where optimal changed
   - Control: Dominated situations (always should be 2pt)
   - DiD estimate: [see output above]

4. INTERPRETATION:
   - If coaches optimize: Immediate, full adjustment
   - If heuristic learning: Slow, incomplete adjustment
   - Gap persistence suggests heuristic behavior

This parallels Strulov-Shlain's finding that Israeli retailers used
heuristic 9-ending pricing rather than optimizing given the reform.
""")


if __name__ == "__main__":
    main()
