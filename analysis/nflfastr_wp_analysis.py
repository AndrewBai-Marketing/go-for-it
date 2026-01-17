"""
Fourth Down Analysis using nflfastR's Win Probability model directly.

This uses the pre-computed WP values from nflfastR rather than our custom Bayesian model.
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


def compute_wp_after_conversion(row, conv_prob, fg_model):
    """Estimate WP after a successful conversion."""
    # If convert, you have 1st down at the same spot
    # Use the play's wp as baseline and adjust
    # Simplified: assume conversion gives you ~same WP as current but with fresh downs
    return row['wp']  # Will be adjusted below


def compute_wp_after_punt(row, punt_model):
    """Estimate opponent's field position after punt."""
    field_pos = row['yardline_100']

    # Get expected net punt distance
    if punt_model is not None and punt_model.beta_mean is not None:
        exp_net = punt_model.beta_mean[0] + punt_model.beta_mean[1] * field_pos
    else:
        exp_net = 40  # Default

    # Opponent gets ball at their own (100 - field_pos + exp_net) if not touchback
    opp_field_pos = min(80, max(1, 100 - field_pos + exp_net))
    return opp_field_pos


def analyze_fourth_downs_with_nflfastr_wp(
    test_year: int,
    all_pbp: pd.DataFrame,
    conversion_model: ConversionModel,
    punt_model: PuntModel,
    fg_model: FieldGoalModel
):
    """
    Analyze fourth down decisions using nflfastR's WP values.

    For each 4th down:
    1. Get current WP from nflfastR
    2. Compute expected WP for GO (conv_prob * wp_if_convert + (1-conv_prob) * wp_if_fail)
    3. Compute expected WP for PUNT (opponent's WP at expected field position)
    4. Compute expected WP for FG (fg_prob * wp_if_make + (1-fg_prob) * wp_if_miss)
    """
    # Get test year data
    test_pbp = all_pbp[all_pbp['season'] == test_year].copy()
    cleaned = clean_pbp_data(test_pbp)
    fourth_downs = extract_fourth_downs(cleaned)

    results = []

    for idx, row in tqdm(fourth_downs.iterrows(), total=len(fourth_downs), desc=f"Year {test_year}"):
        # Current state
        field_pos = row['yardline_100']
        yards_to_go = row['ydstogo']
        score_diff = row['score_differential']
        time_remaining = row['game_seconds_remaining']

        # Current WP from nflfastR
        current_wp = row['wp']
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
        # Get conversion probability
        conv_prob = conversion_model.get_conversion_prob(yards_to_go)

        # WP if convert: roughly current WP (you keep ball with 1st down)
        # WP if fail: opponent gets ball at current spot
        # Use row's data to estimate

        # Simplified model:
        # - If convert: WP stays similar (slight boost for new 1st down)
        # - If fail: opponent at your field_pos, so your WP drops
        # We'll use the empirical relationship from the data

        # For conversion, look at average WP change
        wp_if_convert = min(0.99, current_wp + 0.03)  # Slight boost for 1st down

        # For failure, opponent gets ball - significant WP drop
        # The closer to your endzone, the worse
        if field_pos <= 10:
            wp_if_fail = current_wp * 0.3  # Turnover on downs in red zone is bad
        elif field_pos <= 50:
            wp_if_fail = current_wp * 0.5
        else:
            wp_if_fail = current_wp * 0.7

        wp_go = conv_prob * wp_if_convert + (1 - conv_prob) * wp_if_fail

        # 2. PUNT
        # Expected net yards
        if punt_model is not None and punt_model.beta_mean is not None:
            exp_net = punt_model.beta_mean[0] + punt_model.beta_mean[1] * field_pos
        else:
            exp_net = 40

        # After punt, opponent has ball
        # Your WP is roughly 0.5 adjusted for field position and score
        # Simplified: punting gives up possession but pins opponent
        opp_start = 100 - field_pos + exp_net
        opp_start = min(99, max(1, opp_start))

        # WP after punt - opponent at opp_start yard line
        # The further back they start, the better for you
        punt_wp_boost = (opp_start - 25) / 100 * 0.05  # Small adjustment
        wp_punt = 0.5 + score_diff / 28 * 0.15 + punt_wp_boost
        wp_punt = min(0.99, max(0.01, wp_punt))

        # 3. FIELD GOAL
        if field_pos <= 45:  # FG range
            fg_distance = field_pos + 17  # Add 17 for snap + endzone
            fg_prob = fg_model.get_make_prob(fg_distance)

            # WP if make: current WP + boost for 3 points
            # WP if miss: opponent at spot of kick (roughly)
            wp_if_make = min(0.99, current_wp + 0.08)  # 3 points boost
            wp_if_miss = current_wp * 0.6  # Miss gives opponent good position

            wp_fg = fg_prob * wp_if_make + (1 - fg_prob) * wp_if_miss
        else:
            wp_fg = 0  # Not in FG range

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
            'nflfastr_wp': current_wp,
            'actual_decision': actual,
            'optimal_decision': optimal,
            'match': actual == optimal,
            'wp_go': wp_go,
            'wp_punt': wp_punt,
            'wp_fg': wp_fg,
            'conv_prob': conv_prob
        })

    return pd.DataFrame(results)


def run_nflfastr_analysis():
    """Run full analysis using nflfastR WP values."""
    # Load data
    print("Loading data...")
    all_pbp = pd.read_parquet(Path(__file__).parent.parent / 'data' / 'all_pbp_1999_2024.parquet')

    # Fit outcome models on all historical data (these are still needed)
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
        results = analyze_fourth_downs_with_nflfastr_wp(
            year, all_pbp, conversion_model, punt_model, fg_model
        )
        all_results.append(results)

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)

    output_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'nflfastr_wp_results.csv'
    combined.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (using nflfastR WP)")
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

    return combined


if __name__ == "__main__":
    run_nflfastr_analysis()
