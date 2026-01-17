"""
Fourth Down Analysis using nflfastR WP with proper state-transition lookups.

Instead of using crude heuristics, we build a lookup table of WP values
by game state (field position, score diff, time remaining, down, distance)
and use those to estimate expected WP for each decision outcome.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.bayesian_models import ConversionModel, PuntModel, FieldGoalModel
from data.acquire_data import (
    clean_pbp_data, extract_fourth_downs, extract_punt_plays,
    extract_field_goals, extract_fourth_down_attempts
)


def build_wp_lookup_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup table of average WP by game state.

    States are defined by:
    - Field position (binned)
    - Score differential (binned)
    - Time remaining (binned by quarter)
    - Possession (offense has ball)
    """
    # Use vegas_wp for more accurate WP
    wp_col = 'vegas_wp' if 'vegas_wp' in pbp.columns else 'wp'

    # Filter to valid plays with WP
    valid = pbp[pbp[wp_col].notna()].copy()

    # Create bins
    valid['field_pos_bin'] = pd.cut(valid['yardline_100'],
                                     bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     labels=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95])

    valid['score_diff_bin'] = pd.cut(valid['score_differential'],
                                      bins=[-100, -21, -14, -7, -3, 0, 3, 7, 14, 21, 100],
                                      labels=[-25, -17, -10, -5, -1, 1, 5, 10, 17, 25])

    valid['quarter'] = pd.cut(valid['game_seconds_remaining'],
                               bins=[0, 900, 1800, 2700, 3600],
                               labels=[4, 3, 2, 1])

    # Group and compute mean WP
    # For 1st and 10 situations (what you get after conversion or opponent after turnover)
    first_and_10 = valid[(valid['down'] == 1) & (valid['ydstogo'] >= 8) & (valid['ydstogo'] <= 12)]

    wp_table = first_and_10.groupby(['field_pos_bin', 'score_diff_bin', 'quarter'])[wp_col].mean().reset_index()
    wp_table.columns = ['field_pos_bin', 'score_diff_bin', 'quarter', 'wp']

    return wp_table


def lookup_wp(wp_table: pd.DataFrame, field_pos: int, score_diff: int,
              time_remaining: int, is_opponent: bool = False) -> float:
    """
    Look up WP for a given game state.

    If is_opponent=True, returns (1 - WP) since opponent having ball at X
    is equivalent to you having (100-X) WP inverted.
    """
    # Bin the inputs
    field_pos_bin = min(95, max(5, 10 * (field_pos // 10) + 5))

    if score_diff <= -21:
        score_diff_bin = -25
    elif score_diff <= -14:
        score_diff_bin = -17
    elif score_diff <= -7:
        score_diff_bin = -10
    elif score_diff <= -3:
        score_diff_bin = -5
    elif score_diff < 0:
        score_diff_bin = -1
    elif score_diff == 0:
        score_diff_bin = 1  # Slight positive for tie
    elif score_diff <= 3:
        score_diff_bin = 1
    elif score_diff <= 7:
        score_diff_bin = 5
    elif score_diff <= 14:
        score_diff_bin = 10
    elif score_diff <= 21:
        score_diff_bin = 17
    else:
        score_diff_bin = 25

    quarter = max(1, min(4, 4 - int(time_remaining // 900)))

    # Look up
    match = wp_table[
        (wp_table['field_pos_bin'] == field_pos_bin) &
        (wp_table['score_diff_bin'] == score_diff_bin) &
        (wp_table['quarter'] == quarter)
    ]

    if len(match) > 0:
        wp = match['wp'].values[0]
    else:
        # Fallback: use just field position and score
        match = wp_table[
            (wp_table['field_pos_bin'] == field_pos_bin) &
            (wp_table['score_diff_bin'] == score_diff_bin)
        ]
        if len(match) > 0:
            wp = match['wp'].mean()
        else:
            # Last resort fallback
            wp = 0.5 + score_diff / 50

    if is_opponent:
        # Opponent has ball, so invert and flip field position perspective
        return 1 - wp
    return wp


def analyze_fourth_downs_proper(
    test_year: int,
    all_pbp: pd.DataFrame,
    wp_table: pd.DataFrame,
    conversion_model: ConversionModel,
    punt_model: PuntModel,
    fg_model: FieldGoalModel
):
    """
    Analyze fourth down decisions using proper WP state transitions.
    """
    # Get test year data
    test_pbp = all_pbp[all_pbp['season'] == test_year].copy()
    cleaned = clean_pbp_data(test_pbp)
    fourth_downs = extract_fourth_downs(cleaned)

    # Use vegas_wp if available
    wp_col = 'vegas_wp' if 'vegas_wp' in fourth_downs.columns else 'wp'

    results = []

    for idx, row in tqdm(fourth_downs.iterrows(), total=len(fourth_downs), desc=f"Year {test_year}"):
        field_pos = row['yardline_100']
        yards_to_go = row['ydstogo']
        score_diff = row['score_differential']
        time_remaining = row['game_seconds_remaining']

        current_wp = row[wp_col]
        if pd.isna(current_wp):
            continue

        # Determine actual decision
        play_type = row['play_type']
        if play_type == 'punt':
            actual = 'punt'
        elif play_type == 'field_goal':
            actual = 'field_goal'
        else:
            actual = 'go_for_it'

        # === COMPUTE EXPECTED WP FOR EACH OPTION ===

        # 1. GO FOR IT
        conv_prob = conversion_model.get_conversion_prob(yards_to_go)

        # If convert: 1st down at (field_pos - yards_to_go), same score, same time (roughly)
        new_field_pos_convert = max(1, field_pos - yards_to_go)
        wp_if_convert = lookup_wp(wp_table, new_field_pos_convert, score_diff, time_remaining)

        # If fail: opponent gets ball at current field_pos (their 100 - field_pos)
        # This is equivalent to opponent having 1st and 10 at their (100 - field_pos)
        opp_field_pos = 100 - field_pos
        wp_if_fail = lookup_wp(wp_table, opp_field_pos, -score_diff, time_remaining)
        wp_if_fail = 1 - wp_if_fail  # Opponent's WP, so invert

        wp_go = conv_prob * wp_if_convert + (1 - conv_prob) * wp_if_fail

        # 2. PUNT
        # Expected net yards
        if punt_model is not None and punt_model.beta_mean is not None:
            exp_net = punt_model.beta_mean[0] + punt_model.beta_mean[1] * field_pos
        else:
            exp_net = 40

        # Opponent gets ball at their (100 - field_pos + exp_net), capped
        opp_start_after_punt = min(99, max(1, 100 - field_pos + exp_net))
        # From opponent's perspective, they're at opp_start_after_punt from their endzone
        # So their WP lookup is at that field position with inverted score
        wp_punt = lookup_wp(wp_table, opp_start_after_punt, -score_diff, time_remaining)
        wp_punt = 1 - wp_punt  # Invert since it's opponent's WP

        # 3. FIELD GOAL
        if field_pos <= 45:  # FG range
            fg_distance = field_pos + 17
            fg_prob = fg_model.get_make_prob(fg_distance)

            # If make: score +3, opponent gets ball at ~25 (touchback after kickoff)
            wp_if_make = lookup_wp(wp_table, 75, -score_diff - 3, time_remaining)
            wp_if_make = 1 - wp_if_make  # Opponent has ball

            # If miss: opponent gets ball at spot of kick (~field_pos + 7)
            opp_after_miss = min(99, max(1, 100 - field_pos - 7))
            wp_if_miss = lookup_wp(wp_table, opp_after_miss, -score_diff, time_remaining)
            wp_if_miss = 1 - wp_if_miss

            wp_fg = fg_prob * wp_if_make + (1 - fg_prob) * wp_if_miss
        else:
            wp_fg = 0

        # Determine optimal decision
        options = {'go_for_it': wp_go, 'punt': wp_punt}
        if wp_fg > 0:
            options['field_goal'] = wp_fg

        optimal = max(options, key=options.get)

        results.append({
            'season': test_year,
            'game_id': row['game_id'],
            'play_id': row['play_id'],
            'field_pos': field_pos,
            'yards_to_go': yards_to_go,
            'score_diff': score_diff,
            'time_remaining': time_remaining,
            'vegas_wp': current_wp,
            'actual_decision': actual,
            'optimal_decision': optimal,
            'match': actual == optimal,
            'wp_go': wp_go,
            'wp_punt': wp_punt,
            'wp_fg': wp_fg,
            'conv_prob': conv_prob
        })

    return pd.DataFrame(results)


def run_proper_nflfastr_analysis():
    """Run analysis using proper nflfastR WP state transitions."""

    print("Loading data...")
    all_pbp = pd.read_parquet(Path(__file__).parent.parent / 'data' / 'all_pbp_1999_2024.parquet')

    print("\nBuilding WP lookup table from historical data...")
    # Use all data to build lookup table (this is the "oracle" version)
    wp_table = build_wp_lookup_table(all_pbp)
    print(f"  WP table has {len(wp_table)} entries")

    print("\nFitting outcome models on all data...")
    cleaned = clean_pbp_data(all_pbp[all_pbp['season'] <= 2024])

    attempts = extract_fourth_down_attempts(cleaned)
    punts = extract_punt_plays(cleaned)
    fgs = extract_field_goals(cleaned)

    conversion_model = ConversionModel()
    conversion_model.fit(attempts, n_samples=1000)
    print(f"  Conversion model: {len(attempts)} attempts")

    punt_model = PuntModel()
    punt_model.fit(punts)
    print(f"  Punt model: {len(punts)} punts")

    fg_model = FieldGoalModel()
    fg_model.fit(fgs, n_samples=1000)
    print(f"  FG model: {len(fgs)} attempts")

    # Analyze each year
    all_results = []
    for year in range(2006, 2025):
        print(f"\nAnalyzing {year}...")
        results = analyze_fourth_downs_proper(
            year, all_pbp, wp_table, conversion_model, punt_model, fg_model
        )
        all_results.append(results)

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)

    output_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'nflfastr_wp_proper_results.csv'
    combined.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (using proper nflfastR WP state transitions)")
    print("="*60)
    print(f"Total plays: {len(combined):,}")
    print(f"\nOptimal decision rates:")
    print(f"  GO: {(combined['optimal_decision'] == 'go_for_it').mean()*100:.1f}%")
    print(f"  PUNT: {(combined['optimal_decision'] == 'punt').mean()*100:.1f}%")
    print(f"  FG: {(combined['optimal_decision'] == 'field_goal').mean()*100:.1f}%")
    print(f"\nActual GO rate: {(combined['actual_decision'] == 'go_for_it').mean()*100:.1f}%")
    print(f"Match rate: {combined['match'].mean()*100:.1f}%")

    # 4th & 1
    fourth_and_1 = combined[combined['yards_to_go'] == 1]
    print(f"\n4th & 1 optimal GO rate: {(fourth_and_1['optimal_decision'] == 'go_for_it').mean()*100:.1f}%")

    # By score situation
    print("\n\nBy score situation:")
    for name, (low, high) in [("Losing big (< -14)", (-100, -14)),
                              ("Losing (-13 to -1)", (-13, -1)),
                              ("Tied (0)", (0, 0)),
                              ("Winning (1 to 13)", (1, 13)),
                              ("Winning big (> 14)", (14, 100))]:
        subset = combined[(combined['score_diff'] >= low) & (combined['score_diff'] <= high)]
        if len(subset) > 0:
            go_rate = (subset['optimal_decision'] == 'go_for_it').mean() * 100
            print(f"  {name}: GO rate = {go_rate:.1f}%")

    return combined


if __name__ == "__main__":
    run_proper_nflfastr_analysis()
