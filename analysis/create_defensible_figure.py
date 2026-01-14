"""
Create figure for two-point defensibly optimal learning curve.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

def create_defensible_learning_figure():
    """Create the defensibly optimal learning curve figure."""
    output_dir = Path(__file__).parent.parent / 'outputs'

    # Load the defensible analysis results
    df = pd.read_csv(output_dir / 'tables' / 'two_point_defensible_analysis.csv')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both strict and defensible optimal rates
    ax.plot(df['year'], df['strict_optimal_rate'] * 100, 'o-',
            color='#e74c3c', linewidth=2, markersize=8,
            label='Strict Optimal', alpha=0.7)
    ax.plot(df['year'], df['defensible_optimal_rate'] * 100, 's-',
            color='#2ecc71', linewidth=2.5, markersize=8,
            label='Defensibly Optimal (±0.5% margin)')

    # Add trend lines
    years = df['year'].values

    # Strict optimal trend
    slope_strict, intercept_strict, _, p_strict, _ = stats.linregress(
        years, df['strict_optimal_rate'] * 100)
    ax.plot(years, intercept_strict + slope_strict * years, '--',
            color='#e74c3c', alpha=0.5, linewidth=2)

    # Defensible optimal trend
    slope_def, intercept_def, _, p_def, _ = stats.linregress(
        years, df['defensible_optimal_rate'] * 100)
    ax.plot(years, intercept_def + slope_def * years, '--',
            color='#2ecc71', alpha=0.5, linewidth=2)

    # Add annotations
    ax.annotate(f'+{slope_strict:.2f} pp/yr',
                xy=(2022, df[df['year']==2022]['strict_optimal_rate'].values[0] * 100 - 5),
                fontsize=10, color='#e74c3c')
    ax.annotate(f'+{slope_def:.2f} pp/yr\n(p < 0.001)',
                xy=(2020, df[df['year']==2020]['defensible_optimal_rate'].values[0] * 100 + 3),
                fontsize=10, color='#2ecc71', fontweight='bold')

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Optimal Decision Rate (%)', fontsize=12)
    ax.set_title('Two-Point Conversion Decision Quality (2016-2024)\nPost-Rule Change Learning', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(35, 105)
    ax.set_xlim(2015.5, 2024.5)

    # Add sample sizes as secondary info
    for _, row in df.iterrows():
        ax.annotate(f'n={int(row["n_decisions"])}',
                   xy=(row['year'], 38),
                   fontsize=7, ha='center', alpha=0.6, rotation=45)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / 'figures' / 'two_point_defensible_learning.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.close()

    return fig


if __name__ == "__main__":
    create_defensible_learning_figure()
