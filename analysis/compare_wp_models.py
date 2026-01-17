"""
Compare Bayesian WP model vs nflfastR WP model predictions.
Identify disagreements and evaluate which performs better.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Load both results
bayesian_results = pd.read_csv(Path(__file__).parent.parent / 'outputs' / 'tables' / 'expanding_window_results.csv')
nflfastr_results = pd.read_csv(Path(__file__).parent.parent / 'outputs' / 'tables' / 'nflfastr_wp_results.csv')

print("="*70)
print("COMPARING BAYESIAN VS NFLFASTR WP MODELS")
print("="*70)

# Rename columns for clarity before merge
bayesian_results = bayesian_results.rename(columns={
    'ex_ante_optimal': 'optimal_decision_bayes',
    'ex_ante_match': 'match_bayes',
    'ex_ante_wp_go': 'wp_go_bayes',
    'ex_ante_wp_punt': 'wp_punt_bayes',
    'ex_ante_wp_fg': 'wp_fg_bayes'
})

nflfastr_results = nflfastr_results.rename(columns={
    'optimal_decision': 'optimal_decision_nfl',
    'match': 'match_nfl',
    'wp_go': 'wp_go_nfl',
    'wp_punt': 'wp_punt_nfl',
    'wp_fg': 'wp_fg_nfl'
})

# Merge on game_id and play_id
merged = bayesian_results.merge(
    nflfastr_results[['game_id', 'play_id', 'optimal_decision_nfl', 'wp_go_nfl', 'wp_punt_nfl', 'wp_fg_nfl', 'nflfastr_wp', 'match_nfl']],
    on=['game_id', 'play_id']
)

print(f"\nTotal plays in both datasets: {len(merged):,}")

# Find disagreements
merged['disagree'] = merged['optimal_decision_bayes'] != merged['optimal_decision_nfl']
disagreements = merged[merged['disagree']]

print(f"Plays where models disagree: {len(disagreements):,} ({len(disagreements)/len(merged)*100:.1f}%)")

print("\n" + "="*70)
print("DISAGREEMENT BREAKDOWN")
print("="*70)

# Cross-tabulation of recommendations
print("\nCross-tab of optimal decisions:")
ct = pd.crosstab(merged['optimal_decision_bayes'], merged['optimal_decision_nfl'], margins=True)
print(ct)

# Specific disagreement patterns
print("\n\nDisagreement patterns:")
patterns = disagreements.groupby(['optimal_decision_bayes', 'optimal_decision_nfl']).size().sort_values(ascending=False)
for (bayes, nfl), count in patterns.items():
    print(f"  Bayesian says {bayes:12s}, nflfastR says {nfl:12s}: {count:,} plays ({count/len(disagreements)*100:.1f}%)")

print("\n" + "="*70)
print("DISAGREEMENTS BY SITUATION")
print("="*70)

# 4th & 1 disagreements
fourth_and_1 = merged[merged['yards_to_go'] == 1]
fourth_and_1_disagree = fourth_and_1[fourth_and_1['disagree']]
print(f"\n4th & 1 plays: {len(fourth_and_1):,}")
print(f"  Disagreements: {len(fourth_and_1_disagree):,} ({len(fourth_and_1_disagree)/len(fourth_and_1)*100:.1f}%)")
print(f"  Bayesian GO rate: {(fourth_and_1['optimal_decision_bayes'] == 'go_for_it').mean()*100:.1f}%")
print(f"  nflfastR GO rate: {(fourth_and_1['optimal_decision_nfl'] == 'go_for_it').mean()*100:.1f}%")

# By field position zones
print("\n\nBy field position (yards from opponent's endzone):")
for zone_name, (low, high) in [("Red zone (1-20)", (1, 20)),
                                ("Midfield (21-50)", (21, 50)),
                                ("Own territory (51-99)", (51, 99))]:
    zone = merged[(merged['field_pos'] >= low) & (merged['field_pos'] <= high)]
    zone_disagree = zone[zone['disagree']]
    print(f"\n  {zone_name}: {len(zone):,} plays")
    print(f"    Disagreements: {len(zone_disagree):,} ({len(zone_disagree)/len(zone)*100:.1f}%)")
    print(f"    Bayesian GO: {(zone['optimal_decision_bayes'] == 'go_for_it').mean()*100:.1f}%")
    print(f"    nflfastR GO: {(zone['optimal_decision_nfl'] == 'go_for_it').mean()*100:.1f}%")

# By score differential
print("\n\nBy score differential:")
for name, (low, high) in [("Losing big (< -14)", (-100, -14)),
                          ("Losing (-13 to -1)", (-13, -1)),
                          ("Tied (0)", (0, 0)),
                          ("Winning (1 to 13)", (1, 13)),
                          ("Winning big (> 14)", (14, 100))]:
    subset = merged[(merged['score_diff'] >= low) & (merged['score_diff'] <= high)]
    if len(subset) == 0:
        continue
    subset_disagree = subset[subset['disagree']]
    print(f"\n  {name}: {len(subset):,} plays")
    print(f"    Disagreements: {len(subset_disagree):,} ({len(subset_disagree)/len(subset)*100:.1f}%)")
    print(f"    Bayesian GO: {(subset['optimal_decision_bayes'] == 'go_for_it').mean()*100:.1f}%")
    print(f"    nflfastR GO: {(subset['optimal_decision_nfl'] == 'go_for_it').mean()*100:.1f}%")

# By time remaining
print("\n\nBy quarter:")
merged['quarter'] = pd.cut(merged['time_remaining'],
                           bins=[0, 900, 1800, 2700, 3600],
                           labels=['Q4', 'Q3', 'Q2', 'Q1'])
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = merged[merged['quarter'] == q]
    subset_disagree = subset[subset['disagree']]
    print(f"\n  {q}: {len(subset):,} plays")
    print(f"    Disagreements: {len(subset_disagree):,} ({len(subset_disagree)/len(subset)*100:.1f}%)")
    print(f"    Bayesian GO: {(subset['optimal_decision_bayes'] == 'go_for_it').mean()*100:.1f}%")
    print(f"    nflfastR GO: {(subset['optimal_decision_nfl'] == 'go_for_it').mean()*100:.1f}%")

print("\n" + "="*70)
print("PREDICTION ACCURACY (who is right more often?)")
print("="*70)

# Check which model matches actual decisions better
print(f"\nMatch with actual coach decisions:")
print(f"  Bayesian: {merged['match_bayes'].mean()*100:.1f}%")

# For nflfastR, compute match
merged['match_nfl'] = merged['actual_decision'] == merged['optimal_decision_nfl']
print(f"  nflfastR: {merged['match_nfl'].mean()*100:.1f}%")

# On disagreements specifically - who matches actual?
print(f"\nOn plays where models DISAGREE ({len(disagreements):,} plays):")
print(f"  Bayesian matches actual: {disagreements['match_bayes'].mean()*100:.1f}%")
print(f"  nflfastR matches actual: {disagreements['match_nfl'].mean()*100:.1f}%")

print("\n" + "="*70)
print("SAMPLE DISAGREEMENTS FOR INTUITION CHECK")
print("="*70)

# Show specific example plays where they disagree
print("\n--- 4th & 1 disagreements (Bayesian=GO, nflfastR=not GO) ---")
examples = disagreements[
    (disagreements['yards_to_go'] == 1) &
    (disagreements['optimal_decision_bayes'] == 'go_for_it') &
    (disagreements['optimal_decision_nfl'] != 'go_for_it')
].head(10)

for _, row in examples.iterrows():
    print(f"\n  Field: {int(row['field_pos'])} yds from endzone, Score diff: {int(row['score_diff'])}, Q{4-int(row['time_remaining']//900)}")
    print(f"    Bayesian: GO (WP_go={row['wp_go_bayes']:.3f})")
    print(f"    nflfastR: {row['optimal_decision_nfl']} (WP_go={row['wp_go_nfl']:.3f}, WP_punt={row['wp_punt_nfl']:.3f})")
    print(f"    Actual: {row['actual_decision']}")

print("\n--- 4th & 1 disagreements (nflfastR=GO, Bayesian=not GO) ---")
examples = disagreements[
    (disagreements['yards_to_go'] == 1) &
    (disagreements['optimal_decision_nfl'] == 'go_for_it') &
    (disagreements['optimal_decision_bayes'] != 'go_for_it')
].head(10)

for _, row in examples.iterrows():
    print(f"\n  Field: {int(row['field_pos'])} yds from endzone, Score diff: {int(row['score_diff'])}, Q{4-int(row['time_remaining']//900)}")
    print(f"    Bayesian: {row['optimal_decision_bayes']} (WP_go={row['wp_go_bayes']:.3f})")
    print(f"    nflfastR: GO (WP_go={row['wp_go_nfl']:.3f})")
    print(f"    Actual: {row['actual_decision']}")

print("\n--- Red zone disagreements ---")
examples = disagreements[
    (disagreements['field_pos'] <= 10)
].head(10)

for _, row in examples.iterrows():
    print(f"\n  4th & {int(row['yards_to_go'])} at {int(row['field_pos'])} yds, Score diff: {int(row['score_diff'])}, Time: {int(row['time_remaining'])}s")
    print(f"    Bayesian: {row['optimal_decision_bayes']}")
    print(f"    nflfastR: {row['optimal_decision_nfl']}")
    print(f"    Actual: {row['actual_decision']}")

print("\n--- Late game (Q4, close score) disagreements ---")
examples = disagreements[
    (disagreements['time_remaining'] <= 900) &
    (abs(disagreements['score_diff']) <= 7)
].head(10)

for _, row in examples.iterrows():
    print(f"\n  4th & {int(row['yards_to_go'])} at {int(row['field_pos'])} yds, Score diff: {int(row['score_diff'])}, Time: {int(row['time_remaining'])}s")
    print(f"    Bayesian: {row['optimal_decision_bayes']}")
    print(f"    nflfastR: {row['optimal_decision_nfl']}")
    print(f"    Actual: {row['actual_decision']}")

# Save disagreements for further analysis
disagreements.to_csv(Path(__file__).parent.parent / 'outputs' / 'tables' / 'model_disagreements.csv', index=False)
print(f"\n\nSaved disagreements to outputs/tables/model_disagreements.csv")
