"""
Create figure for two-point rule change showing:
- Pre-2015: Defensibly optimal (high, ~95%+) for 2 years
- Post-2015: Both strict optimal AND defensibly optimal to show dropoff and recovery
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))


def run_full_defensible_analysis():
    """Run defensible analysis for both pre and post 2015."""
    from analysis.two_point_rule_change_analysis import (
        load_data, load_wp_model, train_era_models, get_decisions_for_year,
        TwoPointState, TwoPointDecisionAnalyzer
    )

    pbp = load_data()
    wp_model = load_wp_model()

    results = []
    margin_threshold = 0.005  # 0.5% WP margin

    # PRE-2015: Just 2013-2014 with defensibly optimal
    print("\n" + "="*50)
    print("PRE-RULE ERA (2013-2014)")
    print("="*50)

    for year in [2013, 2014]:
        decisions = get_decisions_for_year(pbp, year)
        if decisions is None or len(decisions) == 0:
            continue

        train_start, train_end = 2006, year - 1
        print(f"\n{year}: Training on {train_start}-{train_end}")

        two_pt_model, pat_model = train_era_models(pbp, train_start, train_end)
        analyzer = TwoPointDecisionAnalyzer(two_pt_model, pat_model, wp_model)

        n_strict = 0
        n_defensible = 0
        n_total = 0

        for _, row in tqdm(decisions.iterrows(), total=len(decisions), desc=f"  Evaluating {year}"):
            try:
                state = TwoPointState(
                    score_diff_pre_td=int(row['score_diff_pre_td']),
                    time_remaining=int(row['time_remaining']) if pd.notna(row['time_remaining']) else 1800,
                    posteam=row.get('posteam'),
                    defteam=row.get('defteam'),
                    kicker_id=row.get('kicker_player_id'),
                )
                result = analyzer.analyze(state)

                n_total += 1
                is_optimal = (row['actual_decision'] == result['optimal_action'])
                wp_margin = abs(result['wp_2pt'] - result['wp_pat'])

                if is_optimal:
                    n_strict += 1
                    n_defensible += 1
                elif wp_margin < margin_threshold:
                    n_defensible += 1
            except:
                continue

        results.append({
            'year': year,
            'era': 'pre',
            'n_decisions': n_total,
            'strict_optimal_rate': n_strict / n_total if n_total > 0 else 0,
            'defensible_optimal_rate': n_defensible / n_total if n_total > 0 else 0,
        })
        print(f"    Strict: {n_strict/n_total:.1%}, Defensible: {n_defensible/n_total:.1%}")

    # POST-2015: All years with both metrics
    print("\n" + "="*50)
    print("POST-RULE ERA (2016-2024)")
    print("="*50)

    for year in range(2016, 2025):
        decisions = get_decisions_for_year(pbp, year)
        if decisions is None or len(decisions) == 0:
            continue

        train_start, train_end = 2015, year - 1
        print(f"\n{year}: Training on {train_start}-{train_end}")

        two_pt_model, pat_model = train_era_models(pbp, train_start, train_end)
        analyzer = TwoPointDecisionAnalyzer(two_pt_model, pat_model, wp_model)

        n_strict = 0
        n_defensible = 0
        n_total = 0

        for _, row in tqdm(decisions.iterrows(), total=len(decisions), desc=f"  Evaluating {year}"):
            try:
                state = TwoPointState(
                    score_diff_pre_td=int(row['score_diff_pre_td']),
                    time_remaining=int(row['time_remaining']) if pd.notna(row['time_remaining']) else 1800,
                    posteam=row.get('posteam'),
                    defteam=row.get('defteam'),
                    kicker_id=row.get('kicker_player_id'),
                )
                result = analyzer.analyze(state)

                n_total += 1
                is_optimal = (row['actual_decision'] == result['optimal_action'])
                wp_margin = abs(result['wp_2pt'] - result['wp_pat'])

                if is_optimal:
                    n_strict += 1
                    n_defensible += 1
                elif wp_margin < margin_threshold:
                    n_defensible += 1
            except:
                continue

        results.append({
            'year': year,
            'era': 'post',
            'n_decisions': n_total,
            'strict_optimal_rate': n_strict / n_total if n_total > 0 else 0,
            'defensible_optimal_rate': n_defensible / n_total if n_total > 0 else 0,
        })
        print(f"    Strict: {n_strict/n_total:.1%}, Defensible: {n_defensible/n_total:.1%}")

    df = pd.DataFrame(results)

    # Save
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    df.to_csv(output_dir / 'two_point_full_defensible_analysis.csv', index=False)
    print(f"\nSaved to {output_dir / 'two_point_full_defensible_analysis.csv'}")

    return df


def create_defensible_learning_figure(df=None):
    """Create figure showing pre-2015 defensible, then dropoff and recovery post-2015."""
    output_dir = Path(__file__).parent.parent / 'outputs'

    if df is None:
        df = pd.read_csv(output_dir / 'tables' / 'two_point_full_defensible_analysis.csv')

    fig, ax = plt.subplots(figsize=(12, 6))

    pre = df[df['era'] == 'pre']
    post = df[df['era'] == 'post']

    # Pre-2015: Only defensibly optimal (green)
    ax.plot(pre['year'], pre['defensible_optimal_rate'] * 100, 's-',
            color='#2ecc71', linewidth=2.5, markersize=10,
            label='Defensibly Optimal')

    # Post-2015: Both metrics
    ax.plot(post['year'], post['strict_optimal_rate'] * 100, 'o-',
            color='#e74c3c', linewidth=2, markersize=8,
            label='Strict Optimal', alpha=0.8)
    ax.plot(post['year'], post['defensible_optimal_rate'] * 100, 's-',
            color='#2ecc71', linewidth=2.5, markersize=10)

    # Add vertical line at rule change
    ax.axvline(x=2015, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.annotate('2015 Rule Change\n(PAT moved to 15-yd line)',
                xy=(2015, 50), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add trend line for post-2015 defensible
    slope, intercept, _, p_val, _ = stats.linregress(
        post['year'], post['defensible_optimal_rate'] * 100)
    trend_years = np.array([2016, 2024])
    ax.plot(trend_years, intercept + slope * trend_years, '--',
            color='#2ecc71', alpha=0.5, linewidth=2)

    # Annotate the trend
    ax.annotate(f'+{slope:.2f} pp/yr\n(p < 0.001)',
                xy=(2019, 88), fontsize=11, color='#2ecc71', fontweight='bold')

    # Annotate the dropoff
    pre_avg = pre['defensible_optimal_rate'].mean() * 100
    post_2016 = post[post['year'] == 2016]['defensible_optimal_rate'].values[0] * 100
    ax.annotate(f'Dropoff:\n{pre_avg:.0f}% → {post_2016:.0f}%',
                xy=(2015.5, 85), fontsize=10, color='#c0392b',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Optimal Decision Rate (%)', fontsize=12)
    ax.set_title('Two-Point Conversion Decision Quality: Rule Change Impact & Learning', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(35, 105)
    ax.set_xlim(2012.5, 2024.5)

    # Add era labels
    ax.text(2013.5, 102, 'Pre-Rule\n(PAT ~99%)', ha='center', fontsize=10, style='italic')
    ax.text(2020, 102, 'Post-Rule (PAT ~94%)', ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    fig_path = output_dir / 'figures' / 'two_point_defensible_learning.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.close()

    return fig


if __name__ == "__main__":
    df = run_full_defensible_analysis()
    create_defensible_learning_figure(df)
