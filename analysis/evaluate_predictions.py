"""
Evaluate prediction accuracy: when each model recommends GO,
how often does the team actually convert?
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Load all results
bayesian = pd.read_csv(Path(__file__).parent.parent / 'outputs' / 'tables' / 'expanding_window_results.csv')
nflfastr_proper = pd.read_csv(Path(__file__).parent.parent / 'outputs' / 'tables' / 'nflfastr_wp_proper_results.csv')

# Load raw play-by-play to get actual outcomes
pbp = pd.read_parquet(Path(__file__).parent.parent / 'data' / 'all_pbp_1999_2024.parquet')

print("="*70)
print("EVALUATING PREDICTION ACCURACY")
print("="*70)

# Get actual conversion outcomes for fourth down attempts
fourth_down_outcomes = pbp[pbp['down'] == 4][['game_id', 'play_id', 'fourth_down_converted', 'fourth_down_failed',
                                               'field_goal_result', 'touchdown', 'yards_gained', 'play_type']].copy()

# Merge with Bayesian results
bayesian_merged = bayesian.merge(fourth_down_outcomes, on=['game_id', 'play_id'], how='left')

# Merge with nflfastR proper results
nflfastr_merged = nflfastr_proper.merge(fourth_down_outcomes, on=['game_id', 'play_id'], how='left')

print(f"\nBayesian plays: {len(bayesian_merged):,}")
print(f"nflfastR plays: {len(nflfastr_merged):,}")

# ============================================================
# 1. When model says GO and team actually went, did they convert?
# ============================================================
print("\n" + "="*70)
print("WHEN MODEL SAYS GO AND TEAM WENT FOR IT")
print("="*70)

# Bayesian
bayes_go_and_went = bayesian_merged[
    (bayesian_merged['ex_ante_optimal'] == 'go_for_it') &
    (bayesian_merged['actual_decision'] == 'go_for_it')
]
bayes_go_converted = bayes_go_and_went['fourth_down_converted'].sum()
bayes_go_total = bayes_go_and_went['fourth_down_converted'].notna().sum()
print(f"\nBayesian: {bayes_go_total:,} plays where model said GO and team went")
print(f"  Converted: {bayes_go_converted:,} ({bayes_go_converted/bayes_go_total*100:.1f}%)")

# nflfastR
nfl_go_and_went = nflfastr_merged[
    (nflfastr_merged['optimal_decision'] == 'go_for_it') &
    (nflfastr_merged['actual_decision'] == 'go_for_it')
]
nfl_go_converted = nfl_go_and_went['fourth_down_converted'].sum()
nfl_go_total = nfl_go_and_went['fourth_down_converted'].notna().sum()
print(f"\nnflfastR: {nfl_go_total:,} plays where model said GO and team went")
print(f"  Converted: {nfl_go_converted:,} ({nfl_go_converted/nfl_go_total*100:.1f}%)")

# ============================================================
# 2. When model says DON'T GO but team went anyway
# ============================================================
print("\n" + "="*70)
print("WHEN MODEL SAYS DON'T GO BUT TEAM WENT ANYWAY")
print("="*70)

# Bayesian
bayes_no_go_but_went = bayesian_merged[
    (bayesian_merged['ex_ante_optimal'] != 'go_for_it') &
    (bayesian_merged['actual_decision'] == 'go_for_it')
]
bayes_no_go_converted = bayes_no_go_but_went['fourth_down_converted'].sum()
bayes_no_go_total = bayes_no_go_but_went['fourth_down_converted'].notna().sum()
print(f"\nBayesian: {bayes_no_go_total:,} plays where model said NO but team went")
print(f"  Converted: {bayes_no_go_converted:,} ({bayes_no_go_converted/bayes_no_go_total*100:.1f}%)")

# nflfastR
nfl_no_go_but_went = nflfastr_merged[
    (nflfastr_merged['optimal_decision'] != 'go_for_it') &
    (nflfastr_merged['actual_decision'] == 'go_for_it')
]
nfl_no_go_converted = nfl_no_go_but_went['fourth_down_converted'].sum()
nfl_no_go_total = nfl_no_go_but_went['fourth_down_converted'].notna().sum()
print(f"\nnflfastR: {nfl_no_go_total:,} plays where model said NO but team went")
print(f"  Converted: {nfl_no_go_converted:,} ({nfl_no_go_converted/nfl_no_go_total*100:.1f}%)")

# ============================================================
# 3. Better metric: Did the DECISION lead to a better outcome?
# ============================================================
print("\n" + "="*70)
print("WIN RATE AFTER FOLLOWING VS IGNORING MODEL")
print("="*70)

# We need to track game outcomes - this is harder
# Let's use WPA as a proxy - did following the model lead to positive WPA?

if 'wpa' in fourth_down_outcomes.columns or 'wpa' in pbp.columns:
    # Add WPA to our data
    wpa_data = pbp[pbp['down'] == 4][['game_id', 'play_id', 'wpa']].copy()
    bayesian_merged = bayesian_merged.merge(wpa_data, on=['game_id', 'play_id'], how='left')
    nflfastr_merged = nflfastr_merged.merge(wpa_data, on=['game_id', 'play_id'], how='left')

    print("\nAverage WPA when following model recommendation:")

    # Bayesian - followed recommendation
    bayes_followed = bayesian_merged[bayesian_merged['ex_ante_match'] == True]
    bayes_followed_wpa = bayes_followed['wpa'].mean()
    print(f"  Bayesian (followed): {bayes_followed_wpa:.4f} WPA ({len(bayes_followed):,} plays)")

    # Bayesian - ignored recommendation
    bayes_ignored = bayesian_merged[bayesian_merged['ex_ante_match'] == False]
    bayes_ignored_wpa = bayes_ignored['wpa'].mean()
    print(f"  Bayesian (ignored): {bayes_ignored_wpa:.4f} WPA ({len(bayes_ignored):,} plays)")

    # nflfastR - followed
    nfl_followed = nflfastr_merged[nflfastr_merged['match'] == True]
    nfl_followed_wpa = nfl_followed['wpa'].mean()
    print(f"\n  nflfastR (followed): {nfl_followed_wpa:.4f} WPA ({len(nfl_followed):,} plays)")

    # nflfastR - ignored
    nfl_ignored = nflfastr_merged[nflfastr_merged['match'] == False]
    nfl_ignored_wpa = nfl_ignored['wpa'].mean()
    print(f"  nflfastR (ignored): {nfl_ignored_wpa:.4f} WPA ({len(nfl_ignored):,} plays)")

# ============================================================
# 4. Specific breakdown: GO recommendations
# ============================================================
print("\n" + "="*70)
print("WPA ANALYSIS FOR GO RECOMMENDATIONS")
print("="*70)

if 'wpa' in bayesian_merged.columns:
    # When Bayesian says GO
    bayes_says_go = bayesian_merged[bayesian_merged['ex_ante_optimal'] == 'go_for_it']
    went_for_it = bayes_says_go[bayes_says_go['actual_decision'] == 'go_for_it']
    didnt_go = bayes_says_go[bayes_says_go['actual_decision'] != 'go_for_it']

    print(f"\nBayesian says GO ({len(bayes_says_go):,} plays):")
    print(f"  Team went: {went_for_it['wpa'].mean():.4f} WPA ({len(went_for_it):,} plays)")
    print(f"  Team didn't: {didnt_go['wpa'].mean():.4f} WPA ({len(didnt_go):,} plays)")

    # When nflfastR says GO
    nfl_says_go = nflfastr_merged[nflfastr_merged['optimal_decision'] == 'go_for_it']
    went_for_it_nfl = nfl_says_go[nfl_says_go['actual_decision'] == 'go_for_it']
    didnt_go_nfl = nfl_says_go[nfl_says_go['actual_decision'] != 'go_for_it']

    print(f"\nnflfastR says GO ({len(nfl_says_go):,} plays):")
    print(f"  Team went: {went_for_it_nfl['wpa'].mean():.4f} WPA ({len(went_for_it_nfl):,} plays)")
    print(f"  Team didn't: {didnt_go_nfl['wpa'].mean():.4f} WPA ({len(didnt_go_nfl):,} plays)")

# ============================================================
# 5. 4th & 1 specific analysis
# ============================================================
print("\n" + "="*70)
print("4TH & 1 SPECIFIC ANALYSIS")
print("="*70)

bayes_4th1 = bayesian_merged[bayesian_merged['yards_to_go'] == 1]
nfl_4th1 = nflfastr_merged[nflfastr_merged['yards_to_go'] == 1]

print(f"\n4th & 1 plays: {len(bayes_4th1):,}")

# Conversion rate when going for it
went_4th1 = bayes_4th1[bayes_4th1['actual_decision'] == 'go_for_it']
conv_rate = went_4th1['fourth_down_converted'].mean()
print(f"  Actual conversion rate: {conv_rate*100:.1f}%")

# WPA analysis
if 'wpa' in bayes_4th1.columns:
    # Bayesian says GO on 4th & 1
    bayes_4th1_go = bayes_4th1[bayes_4th1['ex_ante_optimal'] == 'go_for_it']
    went = bayes_4th1_go[bayes_4th1_go['actual_decision'] == 'go_for_it']
    didnt = bayes_4th1_go[bayes_4th1_go['actual_decision'] != 'go_for_it']

    print(f"\nBayesian says GO on 4th & 1 ({len(bayes_4th1_go):,} plays, {len(bayes_4th1_go)/len(bayes_4th1)*100:.1f}%):")
    if len(went) > 0:
        print(f"  Team went: {went['wpa'].mean():.4f} WPA ({len(went):,} plays)")
    if len(didnt) > 0:
        print(f"  Team didn't: {didnt['wpa'].mean():.4f} WPA ({len(didnt):,} plays)")

    # nflfastR says GO on 4th & 1
    nfl_4th1_go = nfl_4th1[nfl_4th1['optimal_decision'] == 'go_for_it']
    went_nfl = nfl_4th1_go[nfl_4th1_go['actual_decision'] == 'go_for_it']
    didnt_nfl = nfl_4th1_go[nfl_4th1_go['actual_decision'] != 'go_for_it']

    print(f"\nnflfastR says GO on 4th & 1 ({len(nfl_4th1_go):,} plays, {len(nfl_4th1_go)/len(nfl_4th1)*100:.1f}%):")
    if len(went_nfl) > 0:
        print(f"  Team went: {went_nfl['wpa'].mean():.4f} WPA ({len(went_nfl):,} plays)")
    if len(didnt_nfl) > 0:
        print(f"  Team didn't: {didnt_nfl['wpa'].mean():.4f} WPA ({len(didnt_nfl):,} plays)")

# ============================================================
# 6. Summary comparison
# ============================================================
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

print(f"\n{'Metric':<40} {'Bayesian':>12} {'nflfastR':>12}")
print("-"*70)
print(f"{'Optimal GO rate':<40} {(bayesian['ex_ante_optimal']=='go_for_it').mean()*100:>11.1f}% {(nflfastr_proper['optimal_decision']=='go_for_it').mean()*100:>11.1f}%")
print(f"{'Match with actual decisions':<40} {bayesian['ex_ante_match'].mean()*100:>11.1f}% {nflfastr_proper['match'].mean()*100:>11.1f}%")
print(f"{'4th & 1 optimal GO rate':<40} {(bayes_4th1['ex_ante_optimal']=='go_for_it').mean()*100:>11.1f}% {(nfl_4th1['optimal_decision']=='go_for_it').mean()*100:>11.1f}%")

if 'wpa' in bayesian_merged.columns:
    print(f"{'Avg WPA when followed':<40} {bayes_followed_wpa:>12.4f} {nfl_followed_wpa:>12.4f}")
    print(f"{'Avg WPA when ignored':<40} {bayes_ignored_wpa:>12.4f} {nfl_ignored_wpa:>12.4f}")
