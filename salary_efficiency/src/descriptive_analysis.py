"""
Descriptive Analysis for MLB Salary Efficiency

Computes $/WAR by year, position, and skill type.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')


def compute_market_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market rate ($/WAR) for free agents by year.
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    results = []
    for year in sorted(fa['season'].unique()):
        year_data = fa[fa['season'] == year]
        total_salary = year_data['salary'].sum()
        total_war = year_data['war'].sum()
        n_players = len(year_data)

        if total_war > 0:
            dollars_per_war = total_salary / total_war
        else:
            dollars_per_war = np.nan

        results.append({
            'season': year,
            'n_players': n_players,
            'total_salary': total_salary,
            'total_war': total_war,
            'dollars_per_war': dollars_per_war,
            'avg_war': year_data['war'].mean(),
            'avg_salary': year_data['salary'].mean()
        })

    return pd.DataFrame(results)


def compute_surplus_by_service_class(df: pd.DataFrame, market_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute surplus value by service time class.

    Surplus = Market value (WAR * $/WAR) - Actual salary
    """
    df = df.copy()

    # Merge market rate
    rate_dict = dict(zip(market_rates['season'], market_rates['dollars_per_war']))
    df['market_rate'] = df['season'].map(rate_dict)
    df['market_value'] = df['war'] * df['market_rate']
    df['surplus_value'] = df['market_value'] - df['salary']

    # Summarize by class and year
    results = df.groupby(['season', 'service_time_class']).agg({
        'surplus_value': 'mean',
        'salary': 'mean',
        'market_value': 'mean',
        'war': 'mean',
        'player_id': 'count'
    }).rename(columns={'player_id': 'n_players'}).reset_index()

    return results


def compute_position_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute $/WAR by position for free agents.
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # Clean up position codes (Fangraphs uses numeric position codes)
    position_map = {
        'P': 'Pitcher',
        '1': '1B', '1.0': '1B', '1B': '1B',
        '2': 'C', '2.0': 'C', 'C': 'C',
        '3': '2B', '3.0': '2B', '2B': '2B',
        '4': 'SS', '4.0': 'SS', 'SS': 'SS',
        '5': '3B', '5.0': '3B', '3B': '3B',
        '6': 'LF', '6.0': 'LF', 'LF': 'LF',
        '7': 'CF', '7.0': 'CF', 'CF': 'CF',
        '8': 'RF', '8.0': 'RF', 'RF': 'RF',
        '9': 'DH', '9.0': 'DH', 'DH': 'DH',
    }

    # Simplify positions - many are decimal (weighted average)
    def clean_position(pos):
        pos_str = str(pos)
        if pos_str == 'P' or pos_str == 'Pitcher':
            return 'Pitcher'
        try:
            pos_num = float(pos_str)
            if pos_num < 1.5:
                return '1B'
            elif pos_num < 2.5:
                return 'C'
            elif pos_num < 3.5:
                return '2B'
            elif pos_num < 4.5:
                return 'SS'
            elif pos_num < 5.5:
                return '3B'
            elif pos_num < 6.5:
                return 'LF'
            elif pos_num < 7.5:
                return 'CF'
            elif pos_num < 8.5:
                return 'RF'
            else:
                return 'DH'
        except:
            return 'Unknown'

    fa['position_clean'] = fa['position'].apply(clean_position)

    results = fa.groupby('position_clean').agg({
        'salary': 'sum',
        'war': 'sum',
        'player_id': 'count'
    }).rename(columns={'player_id': 'n_players'}).reset_index()

    results = results[results['war'] > 0].copy()
    results['dollars_per_war'] = results['salary'] / results['war']

    # Overall average
    overall = results['salary'].sum() / results['war'].sum()
    results['deviation_from_mean'] = results['dollars_per_war'] - overall
    results['pct_deviation'] = (results['dollars_per_war'] / overall - 1) * 100

    return results.sort_values('dollars_per_war', ascending=False)


def compute_star_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute $/WAR by WAR bucket to test for nonlinearity.
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # Create WAR buckets
    bins = [0, 1, 2, 3, 4, 5, 15]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']
    fa['war_bucket'] = pd.cut(fa['war'], bins=bins, labels=labels)

    results = fa.groupby('war_bucket', observed=True).agg({
        'salary': ['sum', 'mean'],
        'war': ['sum', 'mean', 'count']
    })
    results.columns = ['total_salary', 'avg_salary', 'total_war', 'avg_war', 'n_players']
    results = results.reset_index()
    results['dollars_per_war'] = results['total_salary'] / results['total_war']

    return results


def test_skill_pricing(df: pd.DataFrame) -> dict:
    """
    Test whether different skills (offense vs defense) are priced differently.

    Requires off_runs (offensive runs) and def_runs (defensive runs) columns.
    """
    fa = df[(df['service_time_class'] == 'free_agent') & (~df['is_pitcher'].astype(bool))].copy()

    results = {}

    # Check if we have component columns
    if 'off_runs' in fa.columns and 'def_runs' in fa.columns:
        # Clean data
        fa = fa.dropna(subset=['off_runs', 'def_runs', 'salary'])

        if len(fa) > 100:
            # Regression: salary ~ off_runs + def_runs
            model = smf.ols('salary ~ off_runs + def_runs', data=fa).fit()

            results['offense_coefficient'] = model.params['off_runs']
            results['defense_coefficient'] = model.params['def_runs']
            results['offense_pvalue'] = model.pvalues['off_runs']
            results['defense_pvalue'] = model.pvalues['def_runs']
            results['r_squared'] = model.rsquared
            results['n_obs'] = len(fa)

            # Compute implied $/run for each
            results['dollars_per_off_run'] = model.params['off_runs']
            results['dollars_per_def_run'] = model.params['def_runs']
            results['off_def_ratio'] = model.params['off_runs'] / model.params['def_runs'] if model.params['def_runs'] != 0 else np.nan

    return results


def test_age_effects(df: pd.DataFrame) -> dict:
    """
    Test whether teams correctly price aging.
    """
    fa = df[df['service_time_class'] == 'free_agent'].copy()

    # Regression: salary ~ WAR + age + age^2
    fa['age_sq'] = fa['age'] ** 2
    model = smf.ols('salary ~ war + age + age_sq', data=fa).fit()

    results = {
        'war_coefficient': model.params['war'],
        'age_coefficient': model.params['age'],
        'age_sq_coefficient': model.params['age_sq'],
        'war_pvalue': model.pvalues['war'],
        'age_pvalue': model.pvalues['age'],
        'age_sq_pvalue': model.pvalues['age_sq'],
        'r_squared': model.rsquared,
        'n_obs': len(fa)
    }

    # Compute residuals by age
    fa['residual'] = model.resid
    age_residuals = fa.groupby('age')['residual'].mean()
    results['residuals_by_age'] = age_residuals.to_dict()

    return results


def create_figures(df: pd.DataFrame, market_rates: pd.DataFrame,
                   position_pricing: pd.DataFrame, star_premium: pd.DataFrame,
                   output_dir: Path):
    """Create visualization figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. $/WAR over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(market_rates['season'], market_rates['dollars_per_war'] / 1e6,
            'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Season')
    ax.set_ylabel('$/WAR (millions)')
    ax.set_title('Market Rate for Wins: $/WAR Over Time (Free Agents)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'dollars_per_war_over_time.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'dollars_per_war_over_time.png'}")

    # 2. Salary vs WAR scatter
    fa = df[df['service_time_class'] == 'free_agent'].copy()
    sample = fa.sample(min(2000, len(fa)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(sample['war'], sample['salary'] / 1e6, alpha=0.3, s=20)
    ax.set_xlabel('WAR')
    ax.set_ylabel('Salary (millions)')
    ax.set_title('Salary vs WAR (Free Agents)')
    ax.grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(fa['war'], fa['salary'] / 1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, fa['war'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Linear: ${z[0]:.1f}M/WAR')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'salary_vs_war_scatter.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'salary_vs_war_scatter.png'}")

    # 3. Position pricing bars
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x < 0 else 'red' for x in position_pricing['deviation_from_mean']]
    bars = ax.barh(position_pricing['position_clean'],
                   position_pricing['dollars_per_war'] / 1e6,
                   color=colors, alpha=0.7)
    ax.axvline(x=position_pricing['salary'].sum() / position_pricing['war'].sum() / 1e6,
               color='black', linestyle='--', linewidth=2, label='Average')
    ax.set_xlabel('$/WAR (millions)')
    ax.set_title('$/WAR by Position (Free Agents)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'position_pricing_bars.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'position_pricing_bars.png'}")

    # 4. Star premium curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(star_premium)), star_premium['dollars_per_war'] / 1e6, alpha=0.7)
    ax.set_xticks(range(len(star_premium)))
    ax.set_xticklabels(star_premium['war_bucket'])
    ax.set_xlabel('WAR Range')
    ax.set_ylabel('$/WAR (millions)')
    ax.set_title('Star Premium: $/WAR by Performance Level')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'star_premium_curve.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'star_premium_curve.png'}")


def main():
    """Run all descriptive analyses."""
    data_dir = Path('salary_efficiency/data/processed')
    output_dir = Path('salary_efficiency/outputs')
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_dir / 'player_seasons.parquet')
    print(f"Loaded {len(df):,} player-seasons")

    # 1. Market rate by year
    print("\nComputing market rate by year...")
    market_rates = compute_market_rate_by_year(df)
    market_rates.to_csv(tables_dir / 'market_rate_by_year.csv', index=False)
    print(market_rates.tail(10).to_string(index=False))

    # 2. Surplus by service class
    print("\nComputing surplus by service class...")
    surplus = compute_surplus_by_service_class(df, market_rates)
    surplus.to_csv(tables_dir / 'surplus_by_service_class.csv', index=False)

    # Print summary
    recent_surplus = surplus[surplus['season'] >= 2020]
    print("\nRecent (2020+) average surplus by class:")
    summary = recent_surplus.groupby('service_time_class')['surplus_value'].mean() / 1e6
    print(summary)

    # 3. Position pricing
    print("\nComputing position pricing...")
    position_pricing = compute_position_pricing(df)
    position_pricing.to_csv(tables_dir / 'position_pricing.csv', index=False)
    print(position_pricing.to_string(index=False))

    # 4. Star premium
    print("\nComputing star premium...")
    star_premium = compute_star_premium(df)
    star_premium.to_csv(tables_dir / 'star_premium.csv', index=False)
    print(star_premium.to_string(index=False))

    # 5. Skill pricing test
    print("\nTesting skill pricing (offense vs defense)...")
    skill_results = test_skill_pricing(df)
    if skill_results:
        print(f"  Offense coefficient: ${skill_results.get('offense_coefficient', 0):,.0f} per run")
        print(f"  Defense coefficient: ${skill_results.get('defense_coefficient', 0):,.0f} per run")
        print(f"  Offense/Defense ratio: {skill_results.get('off_def_ratio', 0):.2f}")
        pd.DataFrame([skill_results]).to_csv(tables_dir / 'skill_pricing.csv', index=False)

    # 6. Age effects
    print("\nTesting age effects...")
    age_results = test_age_effects(df)
    print(f"  Age coefficient: ${age_results['age_coefficient']:,.0f}")
    print(f"  Age^2 coefficient: ${age_results['age_sq_coefficient']:,.0f}")
    pd.DataFrame([{k: v for k, v in age_results.items() if k != 'residuals_by_age'}]).to_csv(
        tables_dir / 'age_effects.csv', index=False)

    # Create figures
    print("\nCreating figures...")
    create_figures(df, market_rates, position_pricing, star_premium, figures_dir)

    # Summary
    print("\n" + "=" * 60)
    print("DESCRIPTIVE ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\n1. MARKET RATE FOR WINS")
    print(f"   2000: ${market_rates[market_rates['season'] == 2000]['dollars_per_war'].values[0] / 1e6:.1f}M/WAR")
    latest_year = market_rates['season'].max()
    print(f"   {latest_year}: ${market_rates[market_rates['season'] == latest_year]['dollars_per_war'].values[0] / 1e6:.1f}M/WAR")

    print(f"\n2. POSITION EFFECTS")
    most_expensive = position_pricing.iloc[0]
    least_expensive = position_pricing.iloc[-1]
    print(f"   Most expensive: {most_expensive['position_clean']} (${most_expensive['dollars_per_war']/1e6:.1f}M/WAR)")
    print(f"   Least expensive: {least_expensive['position_clean']} (${least_expensive['dollars_per_war']/1e6:.1f}M/WAR)")

    print(f"\n3. STAR PREMIUM")
    low_war = star_premium[star_premium['war_bucket'] == '0-1']['dollars_per_war'].values[0]
    high_war = star_premium[star_premium['war_bucket'] == '5+']['dollars_per_war'].values[0]
    print(f"   0-1 WAR players: ${low_war/1e6:.1f}M/WAR")
    print(f"   5+ WAR players: ${high_war/1e6:.1f}M/WAR")
    print(f"   Premium ratio: {high_war/low_war:.2f}x")

    return df, market_rates, position_pricing, star_premium


if __name__ == "__main__":
    df, market_rates, position_pricing, star_premium = main()
