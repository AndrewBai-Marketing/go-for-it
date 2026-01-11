"""
Descriptive Analysis v2: Using Real Salary Data

Computes:
1. $/WAR by year (market rate trend)
2. Position pricing analysis
3. Star premium analysis
4. Learning over time (has market become more efficient?)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def compute_market_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute $/WAR by year."""
    yearly = df.groupby('season').agg({
        'salary': ['sum', 'mean', 'count'],
        'war': ['sum', 'mean']
    })
    yearly.columns = ['total_salary', 'avg_salary', 'n_players', 'total_war', 'avg_war']
    yearly['dollars_per_war'] = yearly['total_salary'] / yearly['total_war']
    yearly = yearly.reset_index()
    return yearly


def compute_position_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute $/WAR by position."""
    # Use position_type (batter vs pitcher)
    pos_stats = df.groupby('position').agg({
        'salary': 'sum',
        'war': 'sum',
        'season': 'count'
    }).rename(columns={'season': 'n_players'})

    pos_stats['dollars_per_war'] = pos_stats['salary'] / pos_stats['war']
    pos_stats['avg_salary'] = pos_stats['salary'] / pos_stats['n_players']
    pos_stats['avg_war'] = pos_stats['war'] / pos_stats['n_players']

    # Compute deviation from mean
    mean_dpw = pos_stats['dollars_per_war'].mean()
    pos_stats['deviation_from_mean'] = pos_stats['dollars_per_war'] - mean_dpw
    pos_stats['pct_deviation'] = (pos_stats['deviation_from_mean'] / mean_dpw) * 100

    return pos_stats.reset_index().sort_values('dollars_per_war', ascending=False)


def compute_star_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Compute $/WAR by WAR bucket to measure star premium."""
    df = df.copy()
    df['war_bucket'] = pd.cut(df['war'],
                               bins=[0, 1, 2, 3, 4, 5, 15],
                               labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5+'])

    bucket_stats = df.groupby('war_bucket', observed=True).agg({
        'salary': ['sum', 'mean'],
        'war': ['sum', 'mean'],
        'season': 'count'
    })
    bucket_stats.columns = ['total_salary', 'avg_salary', 'total_war', 'avg_war', 'n_players']
    bucket_stats['dollars_per_war'] = bucket_stats['total_salary'] / bucket_stats['total_war']

    return bucket_stats.reset_index()


def compute_efficiency_by_era(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether market efficiency has improved over time.

    Efficiency = R² of salary ~ WAR regression.
    Higher R² means salaries are more aligned with productivity.
    """
    results = []

    # Define eras based on available data (2000-2016)
    eras = [
        (2000, 2004, "2000-2004"),
        (2005, 2008, "2005-2008"),
        (2009, 2012, "2009-2012"),
        (2013, 2016, "2013-2016")
    ]

    for start, end, name in eras:
        era_df = df[(df['season'] >= start) & (df['season'] <= end)]
        if len(era_df) < 100:
            continue

        # Log-log regression
        log_salary = np.log(era_df['salary'])
        log_war = np.log(era_df['war'] + 0.1)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_war, log_salary)

        # Residuals
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
            'avg_salary': era_df['salary'].mean(),
            'avg_war': era_df['war'].mean()
        })

    return pd.DataFrame(results)


def compute_yearly_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R² by year to test efficiency trend."""
    results = []

    for year in sorted(df['season'].unique()):
        year_df = df[df['season'] == year]
        if len(year_df) < 50:
            continue

        log_salary = np.log(year_df['salary'])
        log_war = np.log(year_df['war'] + 0.1)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_war, log_salary)

        predicted = intercept + slope * log_war
        residuals = log_salary - predicted

        results.append({
            'year': year,
            'n_players': len(year_df),
            'r_squared': r_value**2,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'residual_var': np.var(residuals),
            'slope': slope
        })

    return pd.DataFrame(results)


def test_efficiency_trend(yearly_df: pd.DataFrame) -> dict:
    """Formal test: is the market becoming more efficient over time?"""
    # Trend in R²
    r2_slope, r2_int, r2_r, r2_p, r2_se = stats.linregress(
        yearly_df['year'], yearly_df['r_squared']
    )

    # Trend in RMSE
    rmse_slope, rmse_int, rmse_r, rmse_p, rmse_se = stats.linregress(
        yearly_df['year'], yearly_df['rmse']
    )

    return {
        'r2_trend_slope': r2_slope,
        'r2_trend_pvalue': r2_p,
        'r2_improving': r2_slope > 0,
        'rmse_trend_slope': rmse_slope,
        'rmse_trend_pvalue': rmse_p,
        'rmse_improving': rmse_slope < 0,
        'conclusion': 'Market efficiency is improving' if (r2_slope > 0 and rmse_slope < 0) else 'No clear improvement'
    }


def create_figures(yearly_df: pd.DataFrame, era_df: pd.DataFrame,
                   market_rate_df: pd.DataFrame, star_df: pd.DataFrame,
                   output_dir: Path):
    """Create visualization figures."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. $/WAR over time
    ax1 = axes[0, 0]
    ax1.plot(market_rate_df['season'], market_rate_df['dollars_per_war'] / 1e6,
             'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('$/WAR (Millions)')
    ax1.set_title('Market Rate for Wins Over Time')
    ax1.grid(True, alpha=0.3)

    # 2. R² over time (efficiency)
    ax2 = axes[0, 1]
    ax2.plot(yearly_df['year'], yearly_df['r_squared'], 'g-o', linewidth=2, markersize=6)
    z = np.polyfit(yearly_df['year'], yearly_df['r_squared'], 1)
    p = np.poly1d(z)
    ax2.plot(yearly_df['year'], p(yearly_df['year']), 'r--',
             label=f'Trend: {z[0]*10:.4f}/decade')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('R² (Salary ~ WAR)')
    ax2.set_title('Market Efficiency Over Time (Higher = Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residual variance by era
    ax3 = axes[1, 0]
    ax3.bar(era_df['era'], era_df['residual_var'], color='steelblue', alpha=0.7)
    ax3.set_xlabel('Era')
    ax3.set_ylabel('Residual Variance')
    ax3.set_title('Salary-WAR Residual Variance by Era (Lower = More Efficient)')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Star premium
    ax4 = axes[1, 1]
    ax4.bar(star_df['war_bucket'].astype(str), star_df['dollars_per_war'] / 1e6,
            color='coral', alpha=0.7)
    ax4.set_xlabel('WAR Bucket')
    ax4.set_ylabel('$/WAR (Millions)')
    ax4.set_title('Star Premium: $/WAR by Performance Tier')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'descriptive_analysis_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_dir / 'descriptive_analysis_v2.png'}")


def main():
    """Run descriptive analysis on real salary data."""
    data_dir = Path('salary_efficiency/data/processed')
    output_dir = Path('salary_efficiency/outputs')
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load real salary data
    print("Loading real salary data...")
    df = pd.read_parquet(data_dir / 'player_seasons_real_salary.parquet')
    print(f"Loaded {len(df):,} player-seasons")
    print(f"Years: {df['season'].min()}-{df['season'].max()}")

    # 1. Market rate by year
    print("\n1. Computing market rate by year...")
    market_rate = compute_market_rate_by_year(df)
    market_rate.to_csv(tables_dir / 'market_rate_by_year_v2.csv', index=False)
    print(market_rate.to_string(index=False))

    # 2. Position pricing
    print("\n2. Computing position pricing...")
    position_pricing = compute_position_pricing(df)
    position_pricing.to_csv(tables_dir / 'position_pricing_v2.csv', index=False)
    print(position_pricing.to_string(index=False))

    # 3. Star premium
    print("\n3. Computing star premium...")
    star_premium = compute_star_premium(df)
    star_premium.to_csv(tables_dir / 'star_premium_v2.csv', index=False)
    print(star_premium.to_string(index=False))

    # 4. Efficiency by era
    print("\n4. Computing efficiency by era...")
    era_efficiency = compute_efficiency_by_era(df)
    era_efficiency.to_csv(tables_dir / 'efficiency_by_era_v2.csv', index=False)
    print(era_efficiency.to_string(index=False))

    # 5. Yearly efficiency
    print("\n5. Computing yearly efficiency...")
    yearly_efficiency = compute_yearly_efficiency(df)
    yearly_efficiency.to_csv(tables_dir / 'yearly_efficiency_v2.csv', index=False)

    # 6. Test efficiency trend
    print("\n6. Testing efficiency trend...")
    trend_results = test_efficiency_trend(yearly_efficiency)
    print(f"  R² trend: {trend_results['r2_trend_slope']:.6f}/year (p={trend_results['r2_trend_pvalue']:.4f})")
    print(f"  RMSE trend: {trend_results['rmse_trend_slope']:.6f}/year (p={trend_results['rmse_trend_pvalue']:.4f})")
    print(f"  Conclusion: {trend_results['conclusion']}")

    # Create figures
    print("\n7. Creating figures...")
    create_figures(yearly_efficiency, era_efficiency, market_rate, star_premium, figures_dir)

    # Summary
    print("\n" + "="*70)
    print("DESCRIPTIVE ANALYSIS SUMMARY (Real Salary Data)")
    print("="*70)

    first_year = market_rate.iloc[0]
    last_year = market_rate.iloc[-1]
    print(f"""
1. MARKET RATE FOR WINS
   {int(first_year['season'])}: ${first_year['dollars_per_war']/1e6:.2f}M/WAR
   {int(last_year['season'])}: ${last_year['dollars_per_war']/1e6:.2f}M/WAR
   Growth: {((last_year['dollars_per_war']/first_year['dollars_per_war'])-1)*100:.1f}%

2. POSITION PRICING (Batters vs Pitchers)
   Batters: ${position_pricing[position_pricing['position']=='batter']['dollars_per_war'].values[0]/1e6:.2f}M/WAR
   Pitchers: ${position_pricing[position_pricing['position']=='pitcher']['dollars_per_war'].values[0]/1e6:.2f}M/WAR

3. STAR PREMIUM
   0-1 WAR: ${star_premium[star_premium['war_bucket']=='0-1']['dollars_per_war'].values[0]/1e6:.2f}M/WAR
   5+ WAR: ${star_premium[star_premium['war_bucket']=='5+']['dollars_per_war'].values[0]/1e6:.2f}M/WAR
   Premium ratio: {star_premium[star_premium['war_bucket']=='5+']['dollars_per_war'].values[0] / star_premium[star_premium['war_bucket']=='0-1']['dollars_per_war'].values[0]:.2f}x

4. MARKET EFFICIENCY OVER TIME
   First era R²: {era_efficiency.iloc[0]['r_squared']:.4f}
   Last era R²: {era_efficiency.iloc[-1]['r_squared']:.4f}
   Residual var change: {((era_efficiency.iloc[-1]['residual_var']/era_efficiency.iloc[0]['residual_var'])-1)*100:.1f}%
   Trend: {trend_results['conclusion']}
""")

    return market_rate, position_pricing, star_premium, era_efficiency, yearly_efficiency, trend_results


if __name__ == "__main__":
    results = main()
