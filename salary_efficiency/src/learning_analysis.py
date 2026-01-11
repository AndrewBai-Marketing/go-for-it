"""
Learning Analysis: Has the MLB Salary Market Become More Efficient?

Tests whether pricing inefficiencies have diminished over time by:
1. Computing residual variance by era (has it decreased?)
2. Testing whether mispricing patterns have weakened
3. Examining convergence of position/skill premiums toward theoretical values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def compute_residual_variance_by_era(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute residual variance from salary ~ WAR regression by era.

    If market is becoming more efficient, residual variance should decrease
    (salaries become more aligned with productivity).
    """
    # Focus on free agents (market-determined salaries)
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # Define eras
    eras = [
        (2000, 2004, "2000-2004"),
        (2005, 2009, "2005-2009"),
        (2010, 2014, "2010-2014"),
        (2015, 2019, "2015-2019"),
        (2020, 2024, "2020-2024")
    ]

    results = []
    for start, end, name in eras:
        era_df = fa[(fa['season'] >= start) & (fa['season'] <= end)]
        if len(era_df) < 50:
            continue

        # Log-log regression: log(salary) ~ log(WAR)
        # Use WAR + 1 to handle zeros
        log_salary = np.log(era_df['salary'])
        log_war = np.log(era_df['war'] + 0.1)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_war, log_salary)

        # Compute residuals
        predicted = intercept + slope * log_war
        residuals = log_salary - predicted
        rmse = np.sqrt(np.mean(residuals**2))

        results.append({
            'era': name,
            'n_players': len(era_df),
            'r_squared': r_value**2,
            'rmse': rmse,
            'residual_var': np.var(residuals),
            'slope': slope,
            'intercept': intercept
        })

    return pd.DataFrame(results)


def test_position_premium_convergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether position premiums have converged over time.

    In an efficient market, $/WAR should be similar across positions
    (adjusting for risk/scarcity).
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # Standardize positions
    position_map = {
        'C': 'C', 'Catcher': 'C',
        '1B': '1B', 'First Base': '1B',
        '2B': '2B', 'Second Base': '2B',
        '3B': '3B', 'Third Base': '3B',
        'SS': 'SS', 'Shortstop': 'SS',
        'LF': 'OF', 'CF': 'OF', 'RF': 'OF', 'OF': 'OF',
        'DH': 'DH',
        'P': 'P', 'SP': 'P', 'RP': 'P', 'Pitcher': 'P'
    }

    fa['pos_group'] = fa['position'].map(lambda x: position_map.get(str(x), 'Other'))
    fa = fa[fa['pos_group'].isin(['C', '1B', '2B', '3B', 'SS', 'OF', 'DH', 'P'])]

    # Define eras
    eras = [
        (2000, 2009, "2000-2009"),
        (2010, 2019, "2010-2019"),
        (2020, 2024, "2020-2024")
    ]

    results = []
    for start, end, era_name in eras:
        era_df = fa[(fa['season'] >= start) & (fa['season'] <= end)]

        # Compute $/WAR by position
        pos_stats = era_df.groupby('pos_group').agg({
            'salary': 'sum',
            'war': 'sum',
            'season': 'count'
        }).rename(columns={'season': 'n_players'})

        pos_stats['dollars_per_war'] = pos_stats['salary'] / pos_stats['war']

        # Compute coefficient of variation across positions
        cv = pos_stats['dollars_per_war'].std() / pos_stats['dollars_per_war'].mean()

        results.append({
            'era': era_name,
            'n_players': era_df['season'].count(),
            'cv_position_pricing': cv,
            'max_min_ratio': pos_stats['dollars_per_war'].max() / pos_stats['dollars_per_war'].min()
        })

    return pd.DataFrame(results)


def test_star_premium_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether star premium has become more consistent.

    Examine $/WAR for different WAR buckets over time.
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # WAR buckets
    fa['war_bucket'] = pd.cut(fa['war'],
                               bins=[0, 1, 2, 3, 5, 15],
                               labels=['0-1', '1-2', '2-3', '3-5', '5+'])

    eras = [
        (2000, 2009, "2000-2009"),
        (2010, 2019, "2010-2019"),
        (2020, 2024, "2020-2024")
    ]

    all_results = []

    for start, end, era_name in eras:
        era_df = fa[(fa['season'] >= start) & (fa['season'] <= end)]

        bucket_stats = era_df.groupby('war_bucket', observed=True).agg({
            'salary': 'sum',
            'war': 'sum'
        })
        bucket_stats['dollars_per_war'] = bucket_stats['salary'] / bucket_stats['war']

        # Normalize to 1-2 WAR bucket
        baseline = bucket_stats.loc['1-2', 'dollars_per_war'] if '1-2' in bucket_stats.index else bucket_stats['dollars_per_war'].median()

        for bucket in bucket_stats.index:
            all_results.append({
                'era': era_name,
                'war_bucket': bucket,
                'dollars_per_war': bucket_stats.loc[bucket, 'dollars_per_war'],
                'relative_to_baseline': bucket_stats.loc[bucket, 'dollars_per_war'] / baseline
            })

    return pd.DataFrame(all_results)


def test_market_efficiency_trend(df: pd.DataFrame) -> dict:
    """
    Formal test: Is the market becoming more efficient over time?

    Regress residual variance on year to test for trend.
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # Compute year-by-year R-squared
    yearly_results = []
    for year in sorted(fa['season'].unique()):
        year_df = fa[fa['season'] == year]
        if len(year_df) < 30:
            continue

        log_salary = np.log(year_df['salary'])
        log_war = np.log(year_df['war'] + 0.1)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_war, log_salary)

        predicted = intercept + slope * log_war
        residuals = log_salary - predicted

        yearly_results.append({
            'year': year,
            'r_squared': r_value**2,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'n': len(year_df)
        })

    yearly_df = pd.DataFrame(yearly_results)

    # Test trend in R-squared (higher = more efficient)
    r2_slope, r2_intercept, r2_r, r2_p, r2_se = stats.linregress(
        yearly_df['year'], yearly_df['r_squared']
    )

    # Test trend in RMSE (lower = more efficient)
    rmse_slope, rmse_intercept, rmse_r, rmse_p, rmse_se = stats.linregress(
        yearly_df['year'], yearly_df['rmse']
    )

    return {
        'yearly_data': yearly_df,
        'r2_trend_slope': r2_slope,
        'r2_trend_pvalue': r2_p,
        'r2_trend_direction': 'improving' if r2_slope > 0 else 'declining',
        'rmse_trend_slope': rmse_slope,
        'rmse_trend_pvalue': rmse_p,
        'rmse_trend_direction': 'improving' if rmse_slope < 0 else 'declining',
        'conclusion': 'Market efficiency is improving' if (r2_slope > 0 and rmse_slope < 0) else 'No clear efficiency improvement'
    }


def create_learning_figures(yearly_df: pd.DataFrame, era_df: pd.DataFrame,
                            position_df: pd.DataFrame, output_dir: Path):
    """Create figures for learning analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. R-squared over time
    ax1 = axes[0, 0]
    ax1.plot(yearly_df['year'], yearly_df['r_squared'], 'b-o', linewidth=2, markersize=6)
    z = np.polyfit(yearly_df['year'], yearly_df['r_squared'], 1)
    p = np.poly1d(z)
    ax1.plot(yearly_df['year'], p(yearly_df['year']), 'r--',
             label=f'Trend: {z[0]*1000:.3f}/decade')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('R-squared (salary ~ WAR)')
    ax1.set_title('Model Fit Over Time (Higher = More Efficient)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RMSE over time
    ax2 = axes[0, 1]
    ax2.plot(yearly_df['year'], yearly_df['rmse'], 'g-o', linewidth=2, markersize=6)
    z = np.polyfit(yearly_df['year'], yearly_df['rmse'], 1)
    p = np.poly1d(z)
    ax2.plot(yearly_df['year'], p(yearly_df['year']), 'r--',
             label=f'Trend: {z[0]*10:.3f}/decade')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('RMSE (log scale)')
    ax2.set_title('Prediction Error Over Time (Lower = More Efficient)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residual variance by era
    ax3 = axes[1, 0]
    ax3.bar(era_df['era'], era_df['residual_var'], color='steelblue', alpha=0.7)
    ax3.set_xlabel('Era')
    ax3.set_ylabel('Residual Variance')
    ax3.set_title('Salary-WAR Residual Variance by Era')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Position pricing convergence
    ax4 = axes[1, 1]
    ax4.bar(position_df['era'], position_df['cv_position_pricing'], color='coral', alpha=0.7)
    ax4.set_xlabel('Era')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Position Pricing Dispersion (Lower = More Efficient)')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'market_efficiency_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {output_dir / 'market_efficiency_trends.png'}")


def main():
    """Run learning analysis."""
    data_dir = Path('salary_efficiency/data/processed')
    output_dir = Path('salary_efficiency/outputs')
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_dir / 'player_seasons.parquet')
    print(f"Loaded {len(df):,} player-seasons")

    # 1. Residual variance by era
    print("\n1. Computing residual variance by era...")
    era_results = compute_residual_variance_by_era(df)
    era_results.to_csv(tables_dir / 'residual_variance_by_era.csv', index=False)
    print(era_results.to_string(index=False))

    # 2. Position premium convergence
    print("\n2. Testing position pricing convergence...")
    position_results = test_position_premium_convergence(df)
    position_results.to_csv(tables_dir / 'position_premium_convergence.csv', index=False)
    print(position_results.to_string(index=False))

    # 3. Star premium stability
    print("\n3. Testing star premium stability...")
    star_results = test_star_premium_stability(df)
    star_results.to_csv(tables_dir / 'star_premium_by_era.csv', index=False)
    print(star_results.to_string(index=False))

    # 4. Formal efficiency trend test
    print("\n4. Testing market efficiency trend...")
    trend_results = test_market_efficiency_trend(df)

    print(f"\n  R-squared trend: {trend_results['r2_trend_slope']:.6f} per year (p={trend_results['r2_trend_pvalue']:.4f})")
    print(f"  RMSE trend: {trend_results['rmse_trend_slope']:.6f} per year (p={trend_results['rmse_trend_pvalue']:.4f})")
    print(f"\n  CONCLUSION: {trend_results['conclusion']}")

    yearly_df = trend_results['yearly_data']
    yearly_df.to_csv(tables_dir / 'yearly_efficiency_metrics.csv', index=False)

    # Create figures
    print("\nCreating figures...")
    create_learning_figures(yearly_df, era_results, position_results, figures_dir)

    # Summary
    print("\n" + "="*60)
    print("LEARNING ANALYSIS SUMMARY")
    print("="*60)

    print(f"""
Key Findings:

1. RESIDUAL VARIANCE TREND
   First era ({era_results.iloc[0]['era']}): Residual var = {era_results.iloc[0]['residual_var']:.4f}
   Last era ({era_results.iloc[-1]['era']}): Residual var = {era_results.iloc[-1]['residual_var']:.4f}
   Change: {((era_results.iloc[-1]['residual_var'] / era_results.iloc[0]['residual_var']) - 1) * 100:+.1f}%

2. R-SQUARED TREND (Salary ~ WAR fit)
   Slope: {trend_results['r2_trend_slope']:.6f} per year
   p-value: {trend_results['r2_trend_pvalue']:.4f}
   Direction: {trend_results['r2_trend_direction']}

3. POSITION PRICING DISPERSION
   First era: CV = {position_results.iloc[0]['cv_position_pricing']:.4f}
   Last era: CV = {position_results.iloc[-1]['cv_position_pricing']:.4f}

4. INTERPRETATION
   {trend_results['conclusion']}
""")

    # Save summary
    with open(output_dir / 'learning_analysis_summary.txt', 'w') as f:
        f.write("LEARNING ANALYSIS: HAS MLB SALARY MARKET BECOME MORE EFFICIENT?\n")
        f.write("="*60 + "\n\n")
        f.write(f"R-squared trend: {trend_results['r2_trend_slope']:.6f}/year (p={trend_results['r2_trend_pvalue']:.4f})\n")
        f.write(f"RMSE trend: {trend_results['rmse_trend_slope']:.6f}/year (p={trend_results['rmse_trend_pvalue']:.4f})\n")
        f.write(f"\nConclusion: {trend_results['conclusion']}\n")

    return trend_results, era_results, position_results


if __name__ == "__main__":
    trend_results, era_results, position_results = main()
