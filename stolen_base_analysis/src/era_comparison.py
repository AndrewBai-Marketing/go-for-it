"""
Era Comparison and Visualization for Stolen Base Analysis

Generates figures and summary statistics for the paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def create_attempt_rate_figure(year_df: pd.DataFrame, output_dir: Path):
    """
    Create figure showing steal attempt metrics over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Attempt Rate over time
    ax1 = axes[0, 0]
    ax1.plot(year_df['season'], year_df['attempt_rate'] * 100,
             'b-o', label='Steal Attempt Rate', linewidth=2, markersize=6)
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Attempt Rate (%)')
    ax1.set_title('Stolen Base Attempt Rate Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success Rate vs Break-Even
    ax2 = axes[0, 1]
    ax2.plot(year_df['season'], year_df['success_rate'] * 100,
             'g-o', label='Actual Success Rate', linewidth=2, markersize=6)
    ax2.plot(year_df['season'], year_df['avg_break_even'] * 100,
             'r--s', label='Break-Even Threshold', linewidth=2, markersize=6)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Success Rate vs Break-Even Requirement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Success Margin (Success Rate - Break-Even)
    ax3 = axes[1, 0]
    ax3.bar(year_df['season'], year_df['success_margin'] * 100,
            color=['green' if x > 0 else 'red' for x in year_df['success_margin']], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Success Margin (pp)')
    ax3.set_title('Success Margin: Actual - Break-Even (positive = good)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Percent of Attempts Above Break-Even
    ax4 = axes[1, 1]
    ax4.plot(year_df['season'], year_df['pct_above_break_even'] * 100,
             'purple', marker='o', linewidth=2, markersize=6)
    ax4.axhline(y=50, color='gray', linestyle='--', label='50% threshold')
    ax4.set_xlabel('Season')
    ax4.set_ylabel('% of Attempts')
    ax4.set_title('% of Steal Attempts Above Break-Even')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'attempt_rate_by_era.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_dir / 'attempt_rate_by_era.png'}")


def create_break_even_heatmap(break_even_df: pd.DataFrame, output_dir: Path):
    """
    Create heatmap showing break-even probability by game state.
    """
    # Create pivot for heatmap
    # Focus on key states: outs x score_bucket

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, base in enumerate(['1', '2']):
        ax = axes[idx]

        subset = break_even_df[break_even_df['base_state'] == base]

        if len(subset) == 0:
            continue

        # Create pivot table
        pivot = subset.pivot_table(
            values='break_even_mean',
            index='outs',
            columns='score',
            aggfunc='mean'
        )

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', vmin=0.65, vmax=0.85)

        # Add labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(o)} outs" for o in pivot.index])
        ax.set_xlabel('Score Situation')
        ax.set_title(f'Break-Even % for Stealing {"2nd" if base == "1" else "3rd"}')

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=axes, label='Break-Even Success Rate')
    plt.tight_layout()
    plt.savefig(output_dir / 'break_even_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_dir / 'break_even_heatmap.png'}")


def create_wp_lost_figure(year_df: pd.DataFrame, output_dir: Path):
    """
    Create figure showing success margin over time (replaces WP lost figure).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart of success margin
    ax.bar(year_df['season'], year_df['success_margin'] * 100,
           color=['green' if x > 0 else 'red' for x in year_df['success_margin']], alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Season')
    ax.set_ylabel('Success Margin (pp)')
    ax.set_title('Success Margin by Season (Success Rate - Break-Even)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add trend line if multiple years
    if len(year_df) > 1:
        z = np.polyfit(year_df['season'], year_df['success_margin'] * 100, 1)
        p = np.poly1d(z)
        ax.plot(year_df['season'], p(year_df['season']), 'b--',
                linewidth=2, label=f'Trend: {z[0]:+.3f} pp/year')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'wp_lost_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_dir / 'wp_lost_over_time.png'}")


def create_success_vs_break_even_scatter(df: pd.DataFrame, output_dir: Path, n_sample: int = 5000):
    """
    Create scatter plot of predicted success prob vs break-even.
    Shows which quadrant decisions fall into.
    """
    # Sample for visualization
    if len(df) > n_sample:
        sample = df.sample(n_sample, random_state=42)
    else:
        sample = df

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by decision
    colors = np.where(sample['steal_attempt'], 'blue', 'gray')
    alphas = np.where(sample['steal_attempt'], 0.6, 0.2)

    # Plot
    for i, (_, row) in enumerate(sample.iterrows()):
        ax.scatter(row['predicted_success_prob'], row['break_even'],
                   c=colors[i], alpha=alphas[i], s=20)

    # Add diagonal line (optimal threshold)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='π = π* (threshold)')

    # Add quadrant labels
    ax.text(0.25, 0.85, 'Should NOT steal\n(π < π*)', ha='center', fontsize=10)
    ax.text(0.75, 0.25, 'SHOULD steal\n(π > π*)', ha='center', fontsize=10)

    ax.set_xlabel('Predicted Success Probability (π)')
    ax.set_ylabel('Break-Even Probability (π*)')
    ax.set_title('Steal Decisions: Success Prob vs Break-Even')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add legend for colors
    ax.scatter([], [], c='blue', alpha=0.6, s=50, label='Attempted')
    ax.scatter([], [], c='gray', alpha=0.3, s=50, label='Not Attempted')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'success_prob_vs_break_even.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_dir / 'success_prob_vs_break_even.png'}")


def compute_trend_statistics(year_df: pd.DataFrame) -> dict:
    """
    Compute trend statistics for the paper.
    """
    if len(year_df) < 2:
        # Not enough data for trends
        return {
            'margin_slope': 0,
            'margin_p_value': 1.0,
            'margin_r_squared': 0,
            'above_be_slope': 0,
            'above_be_p_value': 1.0,
            'above_be_r_squared': 0,
            'first_year_margin': year_df.iloc[0]['success_margin'],
            'last_year_margin': year_df.iloc[-1]['success_margin'],
            'first_year_above_be': year_df.iloc[0]['pct_above_break_even'],
            'last_year_above_be': year_df.iloc[-1]['pct_above_break_even']
        }

    # Linear regression for success margin trend
    slope_margin, intercept_margin, r_margin, p_margin, se_margin = stats.linregress(
        year_df['season'], year_df['success_margin'] * 100
    )

    # Linear regression for pct above break-even trend
    slope_above, intercept_above, r_above, p_above, se_above = stats.linregress(
        year_df['season'], year_df['pct_above_break_even'] * 100
    )

    return {
        'margin_slope': slope_margin,
        'margin_p_value': p_margin,
        'margin_r_squared': r_margin ** 2,
        'above_be_slope': slope_above,
        'above_be_p_value': p_above,
        'above_be_r_squared': r_above ** 2,
        'first_year_margin': year_df.iloc[0]['success_margin'],
        'last_year_margin': year_df.iloc[-1]['success_margin'],
        'first_year_above_be': year_df.iloc[0]['pct_above_break_even'],
        'last_year_above_be': year_df.iloc[-1]['pct_above_break_even']
    }


def generate_paper_summary(df: pd.DataFrame, year_df: pd.DataFrame, era_df: pd.DataFrame) -> str:
    """
    Generate summary text for the paper.
    """
    # Overall statistics from attempts
    attempts = df[df['steal_attempt']]
    overall_attempt_rate = df['steal_attempt'].mean()
    overall_success_rate = attempts['steal_success'].mean()
    overall_break_even = attempts['break_even'].mean()
    overall_margin = overall_success_rate - overall_break_even
    n_above_be = (attempts['predicted_success_prob'] > attempts['break_even']).sum()
    pct_above_be = n_above_be / len(attempts)

    # Trend statistics
    trends = compute_trend_statistics(year_df)

    # Era comparison
    if len(era_df) >= 1:
        last_era = era_df.iloc[-1]
    else:
        last_era = year_df.iloc[-1]

    summary = f"""
================================================================================
STOLEN BASE DECISION ANALYSIS: SUMMARY FOR PAPER
================================================================================

1. OVERALL STATISTICS (2023-2024 seasons)
   - Total opportunities analyzed: {len(df):,}
   - Total steal attempts: {len(attempts):,}
   - Attempt rate: {overall_attempt_rate:.2%}
   - Success rate (when attempted): {overall_success_rate:.1%}
   - Average break-even threshold: {overall_break_even:.1%}
   - Success margin: {overall_margin*100:+.1f} pp

2. MAIN FINDING
   Teams succeed at stealing at {overall_success_rate:.1%}, which is
   {abs(overall_margin)*100:.1f} pp {'above' if overall_margin > 0 else 'below'}
   the break-even threshold of ~{overall_break_even:.1%}.

   {pct_above_be:.1%} of steal attempts have predicted success > break-even.

   This suggests teams are {'appropriately selective' if overall_margin > 0 else 'over-aggressive'}
   in choosing when to attempt steals.

3. INTERPRETATION
   The high success rate ({overall_success_rate:.1%}) is EXPECTED because teams
   only attempt steals when conditions are favorable. This reflects GOOD
   decision-making: teams correctly identify high-probability situations.

   The key question is not "should teams steal more?" but rather
   "are teams correctly identifying when to steal?"

   Evidence: {pct_above_be:.1%} of attempts are above break-even, and the
   observed success rate exceeds the break-even by {overall_margin*100:.1f} pp.

4. YEAR-BY-YEAR METRICS
   2023: Attempt rate = {year_df.iloc[0]['attempt_rate']:.2%}, Success = {year_df.iloc[0]['success_rate']:.1%}, Margin = {year_df.iloc[0]['success_margin']*100:+.1f} pp
   2024: Attempt rate = {year_df.iloc[-1]['attempt_rate']:.2%}, Success = {year_df.iloc[-1]['success_rate']:.1%}, Margin = {year_df.iloc[-1]['success_margin']*100:+.1f} pp

5. LIMITATIONS
   - Only 2 seasons of data (2023-2024) due to data availability
   - Cannot assess counterfactual: what would success be if teams stole more?
   - Selection bias: teams only attempt when they believe success is likely
   - Model does not capture unobserved factors (runner jump, catcher arm, etc.)

================================================================================
"""
    return summary


def main():
    """Generate all visualizations and summary statistics."""
    output_dir = Path('stolen_base_analysis/outputs')
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'
    results_dir = output_dir / 'results'

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load processed results
    print("Loading processed results...")

    results_file = results_dir / 'opportunity_level_results.parquet'
    if not results_file.exists():
        print("Results not found. Run decision_analysis.py first.")
        return

    df = pd.read_parquet(results_file)
    year_df = pd.read_csv(tables_dir / 'decision_analysis_by_year.csv')
    era_df = pd.read_csv(tables_dir / 'decision_analysis_by_era.csv')
    break_even_df = pd.read_csv(tables_dir / 'break_even_by_state.csv')

    print(f"Loaded {len(df):,} opportunities")

    # Create visualizations
    print("\nCreating visualizations...")

    print("  1. Attempt rate figure...")
    create_attempt_rate_figure(year_df, figures_dir)

    print("  2. Break-even heatmap...")
    create_break_even_heatmap(break_even_df, figures_dir)

    print("  3. WP lost figure...")
    create_wp_lost_figure(year_df, figures_dir)

    print("  4. Success vs break-even scatter...")
    create_success_vs_break_even_scatter(df, figures_dir)

    # Generate summary
    print("\nGenerating paper summary...")
    summary = generate_paper_summary(df, year_df, era_df)
    print(summary)

    # Save summary
    with open(output_dir / 'paper_summary.txt', 'w') as f:
        f.write(summary)

    print(f"\nSaved summary to {output_dir / 'paper_summary.txt'}")

    # Trend statistics
    trends = compute_trend_statistics(year_df)
    print("\nTrend Statistics:")
    for k, v in trends.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
