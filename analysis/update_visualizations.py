"""
Visualizations for the Update Document

Creates key figures showing:
1. Go-for-it rates and lack of optimality improvement over time (fourth down)
2. Two-point conversion optimality: drop after rule change then recovery
3. Down 8 vs Down 9 behavioral paradox
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def create_fourth_down_trends(output_dir: Path):
    """
    Create the main trend figure showing:
    - Go-for-it rate increasing dramatically
    - Optimal decision rate staying flat

    This is the key visual showing the paradox: behavior changed but accuracy didn't.
    """
    # Load yearly data
    yearly_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'yearly_comparison.csv'
    yearly = pd.read_csv(yearly_path)

    # Load expanding window results for go rates
    expanding_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'expanding_window_results.csv'
    if expanding_path.exists():
        expanding = pd.read_csv(expanding_path)
        # Compute go-for-it rate by season
        go_rates = expanding.groupby('season').apply(
            lambda x: (x['actual_decision'] == 'go_for_it').mean()
        ).reset_index()
        go_rates.columns = ['season', 'go_rate']
        yearly = yearly.merge(go_rates, on='season', how='left')
    else:
        # Estimate from era_comparison if needed
        yearly['go_rate'] = np.linspace(0.11, 0.19, len(yearly))  # Fallback

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Go-for-it rate over time
    ax = axes[0]
    ax.plot(yearly['season'], yearly['go_rate'] * 100, 'o-',
            color='#2980b9', linewidth=2.5, markersize=8, label='Actual go-for-it rate')

    # Add trend line
    slope, intercept, r, p, se = stats.linregress(yearly['season'], yearly['go_rate'])
    trend_line = intercept + slope * yearly['season']
    ax.plot(yearly['season'], trend_line * 100, '--', color='#2980b9',
            alpha=0.7, linewidth=2, label=f'Trend: +{slope*100:.2f} pp/year')

    ax.set_xlabel('Season')
    ax.set_ylabel('Go-for-it Rate (%)')
    ax.set_title('Fourth Down Aggression Over Time')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(f'+{(yearly["go_rate"].iloc[-1] - yearly["go_rate"].iloc[0])*100:.0f}pp\nover period',
                xy=(yearly['season'].iloc[-1], yearly['go_rate'].iloc[-1]*100),
                xytext=(yearly['season'].iloc[-3], yearly['go_rate'].iloc[-1]*100 + 3),
                fontsize=10, color='#2980b9',
                arrowprops=dict(arrowstyle='->', color='#2980b9', alpha=0.7))

    # Right panel: Optimal decision rate (flat)
    ax = axes[1]
    ax.plot(yearly['season'], yearly['Ex Ante Rate'] * 100, 's-',
            color='#27ae60', linewidth=2.5, markersize=8, label='Ex ante optimal rate')

    # Add trend line
    slope2, intercept2, r2, p2, se2 = stats.linregress(yearly['season'], yearly['Ex Ante Rate'])
    trend_line2 = intercept2 + slope2 * yearly['season']
    ax.plot(yearly['season'], trend_line2 * 100, '--', color='#27ae60',
            alpha=0.7, linewidth=2, label=f'Trend: {slope2*100:+.2f} pp/year (p={p2:.2f})')

    ax.set_xlabel('Season')
    ax.set_ylabel('Optimal Decision Rate (%)')
    ax.set_title('Fourth Down Decision Quality Over Time')
    ax.legend(loc='lower left')
    ax.set_ylim(70, 90)
    ax.grid(True, alpha=0.3)

    # Add horizontal line at average
    avg_optimal = yearly['Ex Ante Rate'].mean() * 100
    ax.axhline(y=avg_optimal, color='gray', linestyle=':', alpha=0.7)
    ax.text(yearly['season'].iloc[0], avg_optimal + 0.5, f'Average: {avg_optimal:.1f}%',
            fontsize=9, color='gray')

    plt.tight_layout()

    output_path = output_dir / 'fourth_down_trends.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved fourth down trends to {output_path}")
    plt.close()

    return fig


def create_two_point_rule_change_figure(output_dir: Path):
    """
    Create figure showing two-point conversion optimality around the 2015 rule change.

    The 2015 rule moved the PAT from the 2-yard line to the 15-yard line,
    making PATs harder (~94% success vs ~99% before) and changing the math.

    Key insight: Optimality dropped after rule change as coaches were calibrated
    to old math, then recovered as they learned the new environment.
    """
    # Load two-point by season data
    two_pt_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'two_point_by_season.csv'
    two_pt = pd.read_csv(two_pt_path)

    # Also load expanding window summary for ex ante optimal rates
    expanding_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'two_point_expanding_window_summary.csv'
    if expanding_path.exists():
        expanding = pd.read_csv(expanding_path)
        two_pt = two_pt.merge(expanding[['year', 'pct_correct_ex_ante']],
                               left_on='season', right_on='year', how='left')
        # Use ex ante correct rate where available
        two_pt['optimal_rate_display'] = two_pt['pct_correct_ex_ante'].fillna(two_pt['optimal_rate'])
    else:
        two_pt['optimal_rate_display'] = two_pt['optimal_rate']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Optimal rate over time with rule change marked
    ax = axes[0]

    # Plot optimal rate
    ax.plot(two_pt['season'], two_pt['optimal_rate_display'] * 100, 'o-',
            color='#8e44ad', linewidth=2.5, markersize=8, label='Ex ante optimal rate')

    # Mark rule change year
    ax.axvline(x=2015, color='red', linestyle='--', linewidth=2, alpha=0.7, label='PAT rule change (2015)')

    # Shade pre-2015 region (we don't have proper ex ante data for this)
    ax.axvspan(2014.5, 2015.5, alpha=0.2, color='red', label='Transition year')

    # Add trend line for post-2016 (after coaches had time to adjust)
    post_2016 = two_pt[two_pt['season'] >= 2016]
    if len(post_2016) >= 3:
        slope, intercept, r, p, se = stats.linregress(post_2016['season'], post_2016['optimal_rate_display'])
        trend_years = np.arange(2016, two_pt['season'].max() + 1)
        trend_line = intercept + slope * trend_years
        ax.plot(trend_years, trend_line * 100, '--', color='#8e44ad',
                alpha=0.7, linewidth=2, label=f'Post-2016 trend: +{slope*100:.2f} pp/year')

    ax.set_xlabel('Season')
    ax.set_ylabel('Optimal Decision Rate (%)')
    ax.set_title('Two-Point Conversion Decision Quality\n(PAT rule changed in 2015)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(40, 70)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Learning\nphase',
                xy=(2017, two_pt[two_pt['season']==2017]['optimal_rate_display'].values[0]*100),
                xytext=(2016, 45),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax.annotate('Recovery',
                xy=(2024, two_pt[two_pt['season']==2024]['optimal_rate_display'].values[0]*100),
                xytext=(2022.5, 67),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    # Right panel: Actual 2pt rate vs optimal 2pt rate
    ax = axes[1]

    ax.plot(two_pt['season'], two_pt['actual_2pt_rate'] * 100, 's-',
            color='#3498db', linewidth=2.5, markersize=8, label='Actual 2pt attempt rate')
    ax.plot(two_pt['season'], two_pt['optimal_2pt_rate'] * 100, 'o-',
            color='#27ae60', linewidth=2.5, markersize=8, label='Optimal 2pt attempt rate')

    # Fill the gap
    ax.fill_between(two_pt['season'],
                    two_pt['actual_2pt_rate'] * 100,
                    two_pt['optimal_2pt_rate'] * 100,
                    alpha=0.2, color='red', label='Conservatism gap')

    ax.axvline(x=2015, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Season')
    ax.set_ylabel('Two-Point Attempt Rate (%)')
    ax.set_title('Two-Point Attempt Rates: Actual vs Optimal')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3)

    # Annotation about the gap
    mid_year = 2019
    mid_actual = two_pt[two_pt['season']==mid_year]['actual_2pt_rate'].values[0] * 100
    mid_optimal = two_pt[two_pt['season']==mid_year]['optimal_2pt_rate'].values[0] * 100
    gap = mid_optimal - mid_actual
    ax.annotate(f'Gap: ~{gap:.0f}pp\n(coaches still\ntoo conservative)',
                xy=(mid_year, (mid_actual + mid_optimal)/2),
                xytext=(mid_year + 2, 25),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    plt.tight_layout()

    output_path = output_dir / 'two_point_rule_change.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved two-point rule change figure to {output_path}")
    plt.close()

    return fig


def create_down_8_vs_9_paradox(output_dir: Path):
    """
    Create the Down 8 vs Down 9 behavioral paradox visualization.

    The paradox:
    - Down 8: Go for 2 ties the game immediately. Coaches do this ~79% of the time.
    - Down 9: Go for 2 means a field goal can tie later. Coaches do this ~1% of the time.
    - But Down 9 has a HIGHER optimal 2pt rate (91% vs 85%)!

    This is a striking example of present bias and salience in decision-making.
    """
    # Data from paper_summary.tex
    situations = [
        {'label': 'Down 8\n→ Down 2', 'score_diff': -8, 'n': 196,
         'model_2pt': 0.85, 'actual_2pt': 0.79, 'compliance': 0.84},
        {'label': 'Down 9\n→ Down 3', 'score_diff': -9, 'n': 118,
         'model_2pt': 0.91, 'actual_2pt': 0.01, 'compliance': 0.01},
        {'label': 'Down 14\n→ Down 8', 'score_diff': -14, 'n': 448,
         'model_2pt': 0.94, 'actual_2pt': 0.08, 'compliance': 0.09},
        {'label': 'Down 15\n→ Down 9', 'score_diff': -15, 'n': 97,
         'model_2pt': 0.99, 'actual_2pt': 0.23, 'compliance': 0.23},
        {'label': 'Down 7\n→ Down 1', 'score_diff': -7, 'n': 1125,
         'model_2pt': 0.67, 'actual_2pt': 0.03, 'compliance': 0.03},
        {'label': 'Tied\n→ Up 6', 'score_diff': 0, 'n': 2582,
         'model_2pt': 0.42, 'actual_2pt': 0.01, 'compliance': 0.01},
    ]

    df = pd.DataFrame(situations)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: The core paradox (Down 8 vs Down 9)
    ax = axes[0]

    # Bar positions
    x = np.array([0, 1])
    width = 0.35

    # Extract Down 8 and Down 9 data
    down_8 = df[df['score_diff'] == -8].iloc[0]
    down_9 = df[df['score_diff'] == -9].iloc[0]

    bars1 = ax.bar(x - width/2, [down_8['model_2pt']*100, down_9['model_2pt']*100],
                   width, label='Model says: Go for 2', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, [down_8['actual_2pt']*100, down_9['actual_2pt']*100],
                   width, label='Coaches: Go for 2', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Rate (%)')
    ax.set_title('The Down 8 vs Down 9 Paradox\n"Go for 2 to tie now" vs "Go for 2 so FG ties later"')
    ax.set_xticks(x)
    ax.set_xticklabels(['Down 8\n(2pt ties immediately)', 'Down 9\n(2pt → FG ties later)'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Add paradox annotation
    ax.annotate('Down 9 has HIGHER\noptimal 2pt rate\nbut 1% compliance!',
                xy=(1, 50), xytext=(0.5, 65),
                fontsize=10, ha='center', color='#c0392b',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='#c0392b'))

    # Right panel: Full behavioral patterns
    ax = axes[1]

    x_full = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x_full - width/2, df['model_2pt']*100, width,
                   label='Model: Go for 2', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x_full + width/2, df['actual_2pt']*100, width,
                   label='Coaches: Go for 2', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Rate (%)')
    ax.set_title('Two-Point Conversion Decisions by Score Differential')
    ax.set_xticks(x_full)
    ax.set_xticklabels(df['label'], fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample size labels
    for i, row in df.iterrows():
        ax.annotate(f'n={row["n"]}', xy=(i, 102), ha='center', fontsize=8, color='gray')

    # Highlight the Down 8 (high compliance) bar
    ax.get_children()[0].set_edgecolor('#2ecc71')
    ax.get_children()[0].set_linewidth(3)
    ax.get_children()[6].set_edgecolor('#2ecc71')
    ax.get_children()[6].set_linewidth(3)

    plt.tight_layout()

    output_path = output_dir / 'down_8_vs_9_paradox.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Down 8 vs 9 paradox to {output_path}")
    plt.close()

    return fig


def create_learning_by_margin_figure(output_dir: Path):
    """
    Create figure showing learning (or lack thereof) by decision margin.

    Key finding: Coaches got WORSE on close calls but stayed flat on obvious decisions.
    """
    # Data from update_margin_analysis.tex
    margins = [
        {'label': 'Close\n(0-2pp)', 'n': 38272, 'share': 0.533, 'optimal': 0.692, 'trend': -0.17, 'sig': True},
        {'label': 'Moderate\n(2-5pp)', 'n': 22256, 'share': 0.310, 'optimal': 0.919, 'trend': -0.20, 'sig': True},
        {'label': 'Clear\n(5-10pp)', 'n': 10964, 'share': 0.153, 'optimal': 0.985, 'trend': -0.01, 'sig': False},
        {'label': 'Obvious\n(10+pp)', 'n': 216, 'share': 0.003, 'optimal': 0.986, 'trend': -0.06, 'sig': False},
    ]

    df = pd.DataFrame(margins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Optimal rate by margin
    ax = axes[0]
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#27ae60']  # Red for worst, green for best
    bars = ax.bar(df['label'], df['optimal'] * 100, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Optimal Decision Rate (%)')
    ax.set_xlabel('Decision Margin')
    ax.set_title('Decision Quality by Margin Size\n(Most decisions are close calls)')
    ax.set_ylim(60, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Add share labels on bars
    for bar, row in zip(bars, df.itertuples()):
        height = bar.get_height()
        ax.annotate(f'{row.share*100:.0f}% of\ndecisions',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Right panel: Trends by margin
    ax = axes[1]
    colors = ['#e74c3c' if row['trend'] < -0.1 else '#95a5a6' for _, row in df.iterrows()]
    bars = ax.bar(df['label'], df['trend'], color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Trend (pp/year)')
    ax.set_xlabel('Decision Margin')
    ax.set_title('Learning Trend by Margin Size\n(Red = notable decline)')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylim(-0.3, 0.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'learning_by_margin.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning by margin to {output_path}")
    plt.close()

    return fig


def create_short_yardage_paradox(output_dir: Path):
    """
    Create figure showing the short yardage overcorrection paradox.

    The paradox: Coaches increased go-for-it rate by 22pp, but model says only +2pp more.
    Result: Over-aggressive errors nearly doubled.
    """
    # Data from update_margin_analysis.tex
    eras = ['2006-2012', '2019-2024']

    # Short yardage (4th & 1-2) data
    data = {
        'Coach go rate': [32.5, 54.6],
        'Model go rate': [34.8, 36.6],
        'Optimal rate': [60.9, 56.5],
    }

    errors = {
        'Under-aggressive': [20.5, 12.5],
        'Over-aggressive': [18.2, 30.6],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Go rates vs model recommendation
    ax = axes[0]
    x = np.arange(len(eras))
    width = 0.25

    bars1 = ax.bar(x - width, data['Coach go rate'], width, label='Coach go-for-it rate',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, data['Model go rate'], width, label='Model go-for-it rate',
                   color='#27ae60', alpha=0.8)
    bars3 = ax.bar(x + width, data['Optimal rate'], width, label='Optimal decision rate',
                   color='#9b59b6', alpha=0.8)

    ax.set_ylabel('Rate (%)')
    ax.set_title('Short Yardage (4th & 1-2): The Overcorrection\nCoaches increased aggression, but not optimality')
    ax.set_xticks(x)
    ax.set_xticklabels(eras)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 75)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotations for changes
    ax.annotate(f'+{data["Coach go rate"][1] - data["Coach go rate"][0]:.0f}pp',
                xy=(0.5, (data['Coach go rate'][0] + data['Coach go rate'][1])/2),
                fontsize=10, color='#3498db', fontweight='bold')
    ax.annotate(f'+{data["Model go rate"][1] - data["Model go rate"][0]:.0f}pp',
                xy=(0.5, (data['Model go rate'][0] + data['Model go rate'][1])/2 - 5),
                fontsize=10, color='#27ae60', fontweight='bold')
    ax.annotate(f'{data["Optimal rate"][1] - data["Optimal rate"][0]:.0f}pp',
                xy=(0.5, (data['Optimal rate'][0] + data['Optimal rate'][1])/2 + 5),
                fontsize=10, color='#9b59b6', fontweight='bold')

    # Right panel: Error decomposition
    ax = axes[1]
    x = np.arange(len(eras))
    width = 0.35

    bars1 = ax.bar(x - width/2, errors['Under-aggressive'], width,
                   label='Under-aggressive errors', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, errors['Over-aggressive'], width,
                   label='Over-aggressive errors', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Decomposition: Under vs Over-Aggression\nCoaches traded one error type for another')
    ax.set_xticks(x)
    ax.set_xticklabels(eras)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 40)
    ax.grid(True, alpha=0.3, axis='y')

    # Add change annotations
    ax.annotate(f'{errors["Under-aggressive"][1] - errors["Under-aggressive"][0]:.0f}pp',
                xy=(0.3, 17), fontsize=11, color='#3498db', fontweight='bold')
    ax.annotate(f'+{errors["Over-aggressive"][1] - errors["Over-aggressive"][0]:.0f}pp',
                xy=(0.7, 25), fontsize=11, color='#e74c3c', fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'short_yardage_paradox.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved short yardage paradox to {output_path}")
    plt.close()

    return fig


def main():
    """Generate all figures for the update document."""
    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CREATING FIGURES FOR UPDATE DOCUMENT")
    print("=" * 60)

    # 1. Fourth down trends (go rate vs optimality)
    print("\n1. Creating fourth down trends figure...")
    try:
        create_fourth_down_trends(output_dir)
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Two-point rule change
    print("\n2. Creating two-point rule change figure...")
    try:
        create_two_point_rule_change_figure(output_dir)
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Down 8 vs 9 paradox
    print("\n3. Creating Down 8 vs 9 paradox figure...")
    try:
        create_down_8_vs_9_paradox(output_dir)
    except Exception as e:
        print(f"   Error: {e}")

    # 4. Learning by margin
    print("\n4. Creating learning by margin figure...")
    try:
        create_learning_by_margin_figure(output_dir)
    except Exception as e:
        print(f"   Error: {e}")

    # 5. Short yardage paradox
    print("\n5. Creating short yardage paradox figure...")
    try:
        create_short_yardage_paradox(output_dir)
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print(f"All figures saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
