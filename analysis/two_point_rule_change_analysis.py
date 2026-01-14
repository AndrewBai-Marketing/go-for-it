"""
Two-Point Conversion Analysis: Pre vs Post 2015 Rule Change

The 2015 rule change moved the PAT from the 2-yard line to the 15-yard line,
making PATs harder (~94% success vs ~99% before).

This analysis uses the FULL Bayesian WP model:
1. Train separate 2pt/PAT models for pre-2015 and post-2015
2. Evaluate each decision using the appropriate era's model
3. Show how coach decision quality changed around the rule change
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from analysis.two_point_analysis import (
    TwoPointState,
    TwoPointConversionModel,
    TwoPointDecisionAnalyzer,
    HierarchicalOffDefTwoPointModel,
    HierarchicalPATModel,
)


class SimplePATModel:
    """Simple PAT success model for an era."""
    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        self.success_rate_samples = None

    def fit(self, pat_data):
        """Fit using beta-binomial."""
        successes = (pat_data['extra_point_result'] == 'good').sum()
        attempts = len(pat_data)

        alpha = successes + 1
        beta = (attempts - successes) + 1
        self.success_rate_samples = np.random.beta(alpha, beta, self.n_samples)
        self.overall_rate = successes / attempts

        print(f"  PAT model: {successes}/{attempts} = {self.overall_rate:.1%}")

    def get_posterior_samples(self, *args, **kwargs):
        return self.success_rate_samples


class Simple2ptModel:
    """Simple 2pt success model for an era."""
    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        self.success_rate_samples = None

    def fit(self, two_pt_data):
        """Fit using beta-binomial."""
        successes = (two_pt_data['two_point_conv_result'] == 'success').sum()
        attempts = len(two_pt_data)

        alpha = successes + 1
        beta = (attempts - successes) + 1
        self.success_rate_samples = np.random.beta(alpha, beta, self.n_samples)
        self.overall_rate = successes / attempts

        print(f"  2pt model: {successes}/{attempts} = {self.overall_rate:.1%}")

    def get_posterior_samples(self, *args, **kwargs):
        return self.success_rate_samples


def load_data():
    """Load play-by-play data."""
    data_dir = Path(__file__).parent.parent / 'data'
    pbp = pd.read_parquet(data_dir / 'all_pbp_1999_2024.parquet')
    return pbp


def load_wp_model():
    """Load the win probability model."""
    models_dir = Path(__file__).parent.parent / 'models'
    from models.bayesian_models import WinProbabilityModel
    wp_model = WinProbabilityModel()
    wp_model = wp_model.load(models_dir / 'wp_model.pkl')
    return wp_model


def train_era_models(pbp, start_year, end_year, n_samples=2000, use_hierarchical=True):
    """Train 2pt and PAT models for a specific era."""
    era_data = pbp[(pbp['season'] >= start_year) & (pbp['season'] <= end_year)]

    # PAT data
    pat = era_data[era_data['extra_point_attempt'] == 1].copy()
    pat = pat[pat['extra_point_result'].notna()]

    # 2pt data
    two_pt = era_data[era_data['two_point_attempt'] == 1].copy()
    two_pt = two_pt[two_pt['two_point_conv_result'].notna()]

    if use_hierarchical and len(two_pt) >= 100:
        # Use full hierarchical model with off/def effects
        print(f"  Using hierarchical model with off/def effects")
        two_pt_model = HierarchicalOffDefTwoPointModel(n_samples=n_samples)
        two_pt_model.fit(two_pt, min_attempts=3)

        pat_model = HierarchicalPATModel(n_samples=n_samples)
        pat_model.fit(pat)
    else:
        # Fall back to simple models
        print(f"  Using simple population-level models")
        pat_model = SimplePATModel(n_samples)
        pat_model.fit(pat)

        two_pt_model = Simple2ptModel(n_samples)
        two_pt_model.fit(two_pt)

    return two_pt_model, pat_model


def analyze_decision(row, analyzer):
    """Analyze a single decision using the full WP model."""
    try:
        state = TwoPointState(
            score_diff_pre_td=int(row['score_diff_pre_td']),
            time_remaining=int(row['time_remaining']) if pd.notna(row['time_remaining']) else 1800,
            posteam=row.get('posteam'),
            defteam=row.get('defteam'),
        )
        result = analyzer.analyze(state)
        return {
            'optimal_action': result['optimal_action'],
            'wp_margin': result['wp_margin'],
        }
    except Exception as e:
        return None


def get_decisions_for_year(pbp, year):
    """Get all PAT/2pt decisions for a year."""
    year_data = pbp[pbp['season'] == year]

    pat_plays = year_data[year_data['extra_point_attempt'] == 1].copy()
    pat_plays = pat_plays[pat_plays['extra_point_result'].notna()]
    pat_plays['actual_decision'] = 'pat'

    two_pt_plays = year_data[year_data['two_point_attempt'] == 1].copy()
    two_pt_plays = two_pt_plays[two_pt_plays['two_point_conv_result'].notna()]
    two_pt_plays['actual_decision'] = 'two_point'

    all_decisions = pd.concat([pat_plays, two_pt_plays], ignore_index=True)

    if len(all_decisions) == 0:
        return None

    all_decisions['score_diff_pre_td'] = all_decisions['score_differential'] - 6
    all_decisions['time_remaining'] = all_decisions['game_seconds_remaining']

    return all_decisions


def run_analysis():
    """Run the full pre/post rule change analysis."""
    print("=" * 70)
    print("TWO-POINT RULE CHANGE ANALYSIS (FULL MODEL)")
    print("=" * 70)

    pbp = load_data()
    wp_model = load_wp_model()

    # Train era-specific models
    print("\nTraining pre-rule models (2006-2014)...")
    pre_2pt_model, pre_pat_model = train_era_models(pbp, 2006, 2014)

    print("\nTraining post-rule models (2015-2024)...")
    post_2pt_model, post_pat_model = train_era_models(pbp, 2015, 2024)

    # Create analyzers
    pre_analyzer = TwoPointDecisionAnalyzer(pre_2pt_model, pre_pat_model, wp_model)
    post_analyzer = TwoPointDecisionAnalyzer(post_2pt_model, post_pat_model, wp_model)

    results = []

    # Analyze pre-rule years (2006-2014)
    print("\nAnalyzing pre-rule years (2006-2014)...")
    for year in range(2006, 2015):
        decisions = get_decisions_for_year(pbp, year)
        if decisions is None or len(decisions) == 0:
            continue

        n_optimal = 0
        n_total = 0

        for _, row in tqdm(decisions.iterrows(), total=len(decisions), desc=f"  {year}"):
            analysis = analyze_decision(row, pre_analyzer)
            if analysis:
                n_total += 1
                if row['actual_decision'] == analysis['optimal_action']:
                    n_optimal += 1

        if n_total > 0:
            results.append({
                'year': year,
                'era': 'pre',
                'n_decisions': n_total,
                'optimal_rate': n_optimal / n_total,
                'actual_2pt_rate': (decisions['actual_decision'] == 'two_point').mean(),
                'pat_success_rate': pre_pat_model.overall_rate,
            })
            print(f"    {year}: {n_optimal/n_total:.1%} optimal ({n_total} decisions)")

    # Analyze post-rule years (2015-2024)
    print("\nAnalyzing post-rule years (2015-2024)...")
    for year in range(2015, 2025):
        decisions = get_decisions_for_year(pbp, year)
        if decisions is None or len(decisions) == 0:
            continue

        n_optimal = 0
        n_total = 0

        for _, row in tqdm(decisions.iterrows(), total=len(decisions), desc=f"  {year}"):
            analysis = analyze_decision(row, post_analyzer)
            if analysis:
                n_total += 1
                if row['actual_decision'] == analysis['optimal_action']:
                    n_optimal += 1

        if n_total > 0:
            results.append({
                'year': year,
                'era': 'post',
                'n_decisions': n_total,
                'optimal_rate': n_optimal / n_total,
                'actual_2pt_rate': (decisions['actual_decision'] == 'two_point').mean(),
                'pat_success_rate': post_pat_model.overall_rate,
            })
            print(f"    {year}: {n_optimal/n_total:.1%} optimal ({n_total} decisions)")

    df = pd.DataFrame(results)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    df.to_csv(output_dir / 'two_point_rule_change_analysis.csv', index=False)
    print(f"\nSaved to {output_dir / 'two_point_rule_change_analysis.csv'}")

    return df


def create_rule_change_figure(df=None):
    """Create the rule change visualization showing pre and post trends."""
    if df is None:
        output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
        df = pd.read_csv(output_dir / 'two_point_rule_change_analysis.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Optimal rate over time with rule change marked
    ax = axes[0]

    pre = df[df['era'] == 'pre']
    post = df[df['era'] == 'post']

    # Plot pre-rule data
    ax.plot(pre['year'], pre['optimal_rate'] * 100, 'o-',
            color='#27ae60', linewidth=2.5, markersize=8, label='Pre-rule change (2006-2014)')

    # Plot post-rule data
    ax.plot(post['year'], post['optimal_rate'] * 100, 's-',
            color='#8e44ad', linewidth=2.5, markersize=8, label='Post-rule change (2015-2024)')

    # Mark rule change
    ax.axvline(x=2014.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(2014.5, 2015.5, alpha=0.15, color='red')

    # Add trend lines
    if len(pre) >= 3:
        slope_pre, intercept_pre, _, _, _ = stats.linregress(pre['year'], pre['optimal_rate'])
        trend_pre = intercept_pre + slope_pre * pre['year']
        ax.plot(pre['year'], trend_pre * 100, '--', color='#27ae60', alpha=0.5, linewidth=2)

    if len(post) >= 3:
        slope_post, intercept_post, _, _, _ = stats.linregress(post['year'], post['optimal_rate'])
        trend_post = intercept_post + slope_post * post['year']
        ax.plot(post['year'], trend_post * 100, '--', color='#8e44ad', alpha=0.5, linewidth=2)
        print(f"Post-rule trend: {slope_post*100:.2f} pp/year")

    # Annotations
    pre_avg = pre['optimal_rate'].mean() * 100
    post_avg = post['optimal_rate'].mean() * 100

    ax.annotate('2015 PAT\nrule change',
                xy=(2014.5, 62), xytext=(2011, 52),
                fontsize=10, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Show averages for each era
    ax.annotate(f'Pre-2015 avg:\n{pre_avg:.0f}%',
                xy=(2010, pre_avg), xytext=(2007.5, pre_avg + 8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#27ae60', alpha=0.5))

    ax.annotate(f'Post-2015 avg:\n{post_avg:.0f}%',
                xy=(2020, post_avg), xytext=(2021, post_avg - 10),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#8e44ad', alpha=0.5))

    ax.set_xlabel('Season')
    ax.set_ylabel('Optimal Decision Rate (%)')
    ax.set_title('Two-Point Decision Quality Around 2015 Rule Change\n(Each era evaluated with its own model)')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)

    # Right panel: PAT success rate showing the rule change effect
    ax = axes[1]

    # Get actual PAT success rates by year
    pbp = load_data()
    pat_rates = []
    for year in range(2006, 2025):
        year_data = pbp[pbp['season'] == year]
        pat = year_data[year_data['extra_point_attempt'] == 1]
        pat = pat[pat['extra_point_result'].notna()]
        if len(pat) > 0:
            pat_rates.append({
                'year': year,
                'pat_rate': (pat['extra_point_result'] == 'good').mean()
            })

    pat_df = pd.DataFrame(pat_rates)

    ax.plot(pat_df['year'], pat_df['pat_rate'] * 100, 'o-',
            color='#3498db', linewidth=2.5, markersize=8)

    ax.axvline(x=2014.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(2014.5, 2015.5, alpha=0.15, color='red')

    ax.annotate('PAT moved\nto 15-yard line',
                xy=(2014.5, 96), xytext=(2010, 92),
                fontsize=10, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    pre_pat_avg = pat_df[pat_df['year'] < 2015]['pat_rate'].mean() * 100
    post_pat_avg = pat_df[pat_df['year'] >= 2015]['pat_rate'].mean() * 100
    ax.annotate(f'~{pre_pat_avg:.0f}%',
                xy=(2010, pre_pat_avg), fontsize=10, ha='center', color='#3498db')
    ax.annotate(f'~{post_pat_avg:.0f}%',
                xy=(2019, post_pat_avg), fontsize=10, ha='center', color='#3498db')

    ax.set_xlabel('Season')
    ax.set_ylabel('PAT Success Rate (%)')
    ax.set_title('PAT Success Rate: The Rule Change Effect\n(Moved from 2-yard to 15-yard line)')
    ax.set_ylim(88, 101)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_path = output_dir / 'two_point_rule_change.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()

    return fig


if __name__ == "__main__":
    df = run_analysis()
    create_rule_change_figure(df)
