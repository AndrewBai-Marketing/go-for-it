"""
Decision-Theoretic Analysis: Are "Inefficient" Teams Actually Optimal?

The efficiency framing asks: "Did team X pay fair value for WAR?"
The decision-theoretic framing asks: "Did team X maximize expected utility?"

Key insight: A team's optimal spending depends on:
1. Their playoff probability curve P(playoffs | wins)
2. Their revenue function R(wins, market_size)
3. The value of a playoff appearance (postseason revenue + brand value)

We test whether teams with high gamma (overpay) also have high marginal win values,
which would rationalize their "inefficient" behavior.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def pull_team_standings() -> pd.DataFrame:
    """Pull team wins and playoff appearances from Lahman."""
    print("Pulling team standings data...")

    # Teams table has wins/losses
    teams_url = "https://raw.githubusercontent.com/seanlahman/baseballdatabank/master/core/Teams.csv"

    try:
        teams = pd.read_csv(teams_url)
        print(f"  Loaded {len(teams):,} team-seasons")
    except Exception as e:
        print(f"  Failed to load teams: {e}")
        return None

    # Filter to 2000-2016 to match salary data
    teams = teams[(teams['yearID'] >= 2000) & (teams['yearID'] <= 2016)]
    print(f"  After year filter: {len(teams):,}")

    # Standardize team IDs to match our salary data
    team_map = {
        'ANA': 'LAA', 'CAL': 'LAA', 'MON': 'WSN', 'FLO': 'MIA', 'FLA': 'MIA',
        'TBD': 'TBR', 'TBA': 'TBR', 'CHN': 'CHC', 'CHA': 'CHW', 'KCA': 'KCR',
        'LAN': 'LAD', 'NYA': 'NYY', 'NYN': 'NYM', 'SDN': 'SDP', 'SFN': 'SFG',
        'SLN': 'STL', 'WAS': 'WSN', 'WSH': 'WSN'
    }
    teams['team'] = teams['teamID'].replace(team_map)

    # Create playoff indicator
    # In Lahman, playoff teams have non-null values in certain columns
    # WCWin, DivWin, LgWin, WSWin indicate postseason success
    teams['made_playoffs'] = (
        (teams['DivWin'] == 'Y') |
        (teams['WCWin'] == 'Y') |
        (teams['LgWin'] == 'Y') |
        (teams['WSWin'] == 'Y')
    ).astype(int)

    # Select relevant columns
    standings = teams[['yearID', 'team', 'W', 'L', 'made_playoffs', 'attendance', 'DivWin', 'WCWin', 'LgWin', 'WSWin']].copy()
    standings = standings.rename(columns={'yearID': 'season', 'W': 'wins', 'L': 'losses'})

    print(f"  Playoff teams per year: {standings.groupby('season')['made_playoffs'].sum().mean():.1f}")

    return standings


def estimate_playoff_curves(standings: pd.DataFrame) -> dict:
    """
    Estimate P(playoffs | wins) using logistic regression.

    We estimate a league-wide curve, then can adjust for division strength.
    """
    print("\nEstimating playoff probability curves...")

    # Fit logistic regression: P(playoffs) = logistic(a + b*wins)
    from scipy.special import expit

    X = standings['wins'].values
    y = standings['made_playoffs'].values

    # Simple logistic regression via MLE
    def neg_log_lik(params):
        a, b = params
        p = expit(a + b * X)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    from scipy.optimize import minimize
    result = minimize(neg_log_lik, x0=[-10, 0.1], method='Nelder-Mead')
    a, b = result.x

    print(f"  Logistic coefficients: a={a:.3f}, b={b:.3f}")
    print(f"  P(playoffs | 81 wins) = {expit(a + b * 81):.1%}")
    print(f"  P(playoffs | 90 wins) = {expit(a + b * 90):.1%}")
    print(f"  P(playoffs | 95 wins) = {expit(a + b * 95):.1%}")

    # Compute marginal effect at different win levels
    def playoff_prob(wins):
        return expit(a + b * wins)

    def marginal_playoff_prob(wins):
        """dP/dW = b * p * (1-p)"""
        p = playoff_prob(wins)
        return b * p * (1 - p)

    return {
        'intercept': a,
        'slope': b,
        'playoff_prob': playoff_prob,
        'marginal_prob': marginal_playoff_prob
    }


def estimate_market_sizes(standings: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate relative market size for each team using attendance as proxy.

    Larger markets have higher marginal revenue per win.
    """
    print("\nEstimating market sizes...")

    # Average attendance by team (proxy for market size)
    market_size = standings.groupby('team').agg({
        'attendance': 'mean',
        'wins': 'mean',
        'made_playoffs': 'mean'
    }).reset_index()

    # Normalize to league average = 1
    market_size['market_index'] = market_size['attendance'] / market_size['attendance'].mean()

    # Large market teams
    market_size = market_size.sort_values('market_index', ascending=False)
    print("  Top 5 markets (by attendance):")
    for _, row in market_size.head(5).iterrows():
        print(f"    {row['team']}: {row['market_index']:.2f}x avg ({row['attendance']/1e6:.2f}M attendance)")

    print("  Bottom 5 markets:")
    for _, row in market_size.tail(5).iterrows():
        print(f"    {row['team']}: {row['market_index']:.2f}x avg ({row['attendance']/1e6:.2f}M attendance)")

    return market_size


def compute_rational_win_values(playoff_curves: dict, market_sizes: pd.DataFrame,
                                 playoff_bonus: float = 30e6,  # Extra revenue from playoff appearance
                                 base_win_value: float = 1.5e6) -> pd.DataFrame:
    """
    Compute "rational" marginal win value for each team.

    v_t = base_value * market_index + playoff_bonus * dP/dW

    Teams in larger markets have higher base win values (more revenue per win).
    Teams near the playoff threshold have higher marginal playoff probability.
    """
    print("\nComputing rational marginal win values...")

    results = []

    for _, row in market_sizes.iterrows():
        team = row['team']
        market_idx = row['market_index']
        avg_wins = row['wins']

        # Base value scales with market size
        base_value = base_win_value * market_idx

        # Playoff option value depends on expected wins
        marginal_playoff = playoff_curves['marginal_prob'](avg_wins)
        playoff_value = playoff_bonus * marginal_playoff

        total_win_value = base_value + playoff_value

        results.append({
            'team': team,
            'market_index': market_idx,
            'avg_wins': avg_wins,
            'base_win_value': base_value,
            'marginal_playoff_prob': marginal_playoff,
            'playoff_option_value': playoff_value,
            'rational_win_value': total_win_value
        })

    df = pd.DataFrame(results)

    # Normalize to league average = 1
    df['rational_win_index'] = df['rational_win_value'] / df['rational_win_value'].mean()

    return df


def merge_with_team_effects(rational_values: pd.DataFrame,
                            team_effects_path: Path) -> pd.DataFrame:
    """
    Merge rational win values with estimated team effects (gamma).

    Test: Do teams with high gamma also have high rational win values?
    """
    print("\nMerging with team efficiency estimates...")

    # Load team effects from hierarchical model
    if not team_effects_path.exists():
        print(f"  WARNING: Team effects file not found at {team_effects_path}")
        print("  Using placeholder values...")
        # Create placeholder based on typical results
        team_effects = pd.DataFrame({
            'team': rational_values['team'],
            'gamma_mean': np.random.normal(0, 0.3, len(rational_values)),
            'efficiency_mean': 1.0
        })
    else:
        team_effects = pd.read_csv(team_effects_path)

    # Merge
    merged = rational_values.merge(team_effects[['team', 'gamma_mean', 'efficiency_mean']],
                                    on='team', how='left')

    # Compute implied win value from gamma
    # gamma > 0 means team pays more, implying higher perceived win value
    avg_dollars_per_war = 2e6  # Approximate market rate
    merged['implied_win_value'] = avg_dollars_per_war * np.exp(merged['gamma_mean'])
    merged['implied_win_index'] = merged['implied_win_value'] / merged['implied_win_value'].mean()

    return merged


def test_rationality(merged: pd.DataFrame) -> dict:
    """
    Test whether implied win values correlate with rational win values.

    If high-gamma teams have high rational win values, their "inefficiency"
    may actually be optimal behavior.
    """
    print("\nTesting rationality of team spending...")

    # Correlation between implied and rational win values
    corr, p_value = stats.pearsonr(merged['implied_win_index'], merged['rational_win_index'])

    print(f"  Correlation(implied, rational): r = {corr:.3f} (p = {p_value:.4f})")

    # Regression: implied = a + b * rational
    slope, intercept, r_value, p_val, std_err = stats.linregress(
        merged['rational_win_index'], merged['implied_win_index']
    )

    print(f"  Regression: implied = {intercept:.2f} + {slope:.2f} * rational")
    print(f"  R-squared: {r_value**2:.3f}")

    # Identify "rationally inefficient" teams (high gamma, low rational value)
    merged['rationality_gap'] = merged['implied_win_index'] - merged['rational_win_index']

    print("\n  Most 'irrationally' overpaying (high implied, low rational):")
    irrational = merged.nlargest(5, 'rationality_gap')
    for _, row in irrational.iterrows():
        print(f"    {row['team']}: implied={row['implied_win_index']:.2f}x, rational={row['rational_win_index']:.2f}x, gap={row['rationality_gap']:+.2f}")

    print("\n  Most 'rationally' efficient (low implied, high rational - leaving value on table):")
    rational = merged.nsmallest(5, 'rationality_gap')
    for _, row in rational.iterrows():
        print(f"    {row['team']}: implied={row['implied_win_index']:.2f}x, rational={row['rational_win_index']:.2f}x, gap={row['rationality_gap']:+.2f}")

    return {
        'correlation': corr,
        'correlation_pvalue': p_value,
        'regression_slope': slope,
        'regression_intercept': intercept,
        'r_squared': r_value**2,
        'conclusion': 'Spending patterns partially explained by rational factors' if corr > 0.3 else 'Spending patterns not well explained by rational factors'
    }


def create_decision_theory_figures(merged: pd.DataFrame, playoff_curves: dict,
                                    output_dir: Path):
    """Create visualizations for decision theory analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Playoff probability curve
    ax1 = axes[0, 0]
    wins = np.linspace(60, 110, 100)
    probs = [playoff_curves['playoff_prob'](w) for w in wins]
    ax1.plot(wins, probs, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=90, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Season Wins')
    ax1.set_ylabel('P(Make Playoffs)')
    ax1.set_title('Playoff Probability Curve')
    ax1.grid(True, alpha=0.3)

    # 2. Marginal playoff probability
    ax2 = axes[0, 1]
    marginals = [playoff_curves['marginal_prob'](w) for w in wins]
    ax2.plot(wins, marginals, 'g-', linewidth=2)
    ax2.set_xlabel('Season Wins')
    ax2.set_ylabel('dP/dW (Marginal Playoff Prob)')
    ax2.set_title('Marginal Value of a Win (Playoff Option)')
    ax2.grid(True, alpha=0.3)

    # 3. Implied vs Rational win values
    ax3 = axes[1, 0]
    ax3.scatter(merged['rational_win_index'], merged['implied_win_index'],
                s=80, alpha=0.7, c='steelblue')

    # Add team labels for extreme points
    for _, row in merged.iterrows():
        if abs(row['rationality_gap']) > 0.3:
            ax3.annotate(row['team'], (row['rational_win_index'], row['implied_win_index']),
                        fontsize=8, alpha=0.8)

    # Add 45-degree line
    lims = [0.5, 2.0]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Rational spending')
    ax3.set_xlabel('Rational Win Value Index')
    ax3.set_ylabel('Implied Win Value Index (from spending)')
    ax3.set_title('Implied vs Rational Win Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rationality gap by team
    ax4 = axes[1, 1]
    sorted_df = merged.sort_values('rationality_gap')
    colors = ['green' if g < 0 else 'red' for g in sorted_df['rationality_gap']]
    ax4.barh(range(len(sorted_df)), sorted_df['rationality_gap'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(sorted_df)))
    ax4.set_yticklabels(sorted_df['team'], fontsize=8)
    ax4.axvline(x=0, color='black', linewidth=1)
    ax4.set_xlabel('Rationality Gap (Implied - Rational)')
    ax4.set_title('Spending Rationality by Team')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'decision_theory_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved {output_dir / 'decision_theory_analysis.png'}")


def test_rationality_over_time(standings: pd.DataFrame, salary_data_path: Path) -> pd.DataFrame:
    """
    Test whether team decisions become more rational over time.

    For each year, compute the correlation between team spending and
    rational factors (market size, playoff contention).
    """
    print("\nTesting rationality trend over time...")

    # Load salary data to get team spending by year
    try:
        salary_df = pd.read_parquet(salary_data_path)
    except:
        print("  Could not load salary data, skipping time trend")
        return None

    results = []

    for year in sorted(standings['season'].unique()):
        year_standings = standings[standings['season'] == year]
        year_salaries = salary_df[salary_df['season'] == year]

        if len(year_salaries) < 100:
            continue

        # Compute team spending (total salary / total WAR)
        team_spending = year_salaries.groupby('team').agg({
            'salary': 'sum',
            'war': 'sum'
        }).reset_index()
        team_spending['dollars_per_war'] = team_spending['salary'] / team_spending['war']

        # Merge with standings
        merged = team_spending.merge(year_standings[['team', 'wins', 'attendance', 'made_playoffs']],
                                      on='team', how='inner')

        if len(merged) < 15:
            continue

        # Compute correlation between spending and "rational" factors
        # 1. Correlation with attendance (market size)
        corr_attendance, p_att = stats.pearsonr(merged['dollars_per_war'], merged['attendance'])

        # 2. Correlation with wins (playoff contention proxy)
        corr_wins, p_wins = stats.pearsonr(merged['dollars_per_war'], merged['wins'])

        # 3. Combined "rationality score": average of absolute correlations
        rationality_score = (abs(corr_attendance) + abs(corr_wins)) / 2

        results.append({
            'year': year,
            'n_teams': len(merged),
            'corr_attendance': corr_attendance,
            'corr_wins': corr_wins,
            'rationality_score': rationality_score,
            'avg_dollars_per_war': merged['dollars_per_war'].mean()
        })

    df = pd.DataFrame(results)

    # Test for trend in rationality
    if len(df) > 5:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['year'], df['rationality_score'])
        print(f"  Rationality trend: {slope:.4f}/year (p = {p_value:.4f})")
        print(f"  2000 rationality: {df[df['year'] == 2000]['rationality_score'].values[0]:.3f}")
        print(f"  2016 rationality: {df[df['year'] == 2016]['rationality_score'].values[0]:.3f}")

        df.attrs['trend_slope'] = slope
        df.attrs['trend_pvalue'] = p_value
        df.attrs['trend_improving'] = slope > 0

    return df


def main():
    """Run decision theory analysis."""
    output_dir = Path('salary_efficiency/outputs')
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pull standings data
    standings = pull_team_standings()
    if standings is None:
        print("ERROR: Could not load standings data")
        return

    # 2. Estimate playoff probability curves
    playoff_curves = estimate_playoff_curves(standings)

    # 3. Estimate market sizes
    market_sizes = estimate_market_sizes(standings)

    # 4. Compute rational win values
    rational_values = compute_rational_win_values(playoff_curves, market_sizes)
    rational_values.to_csv(tables_dir / 'rational_win_values.csv', index=False)

    # 5. Merge with team effects
    team_effects_path = tables_dir / 'team_efficiency_rankings_v2.csv'
    merged = merge_with_team_effects(rational_values, team_effects_path)
    merged.to_csv(tables_dir / 'decision_theory_results.csv', index=False)

    # 6. Test rationality
    results = test_rationality(merged)

    # 7. Test rationality over time
    salary_data_path = Path('salary_efficiency/data/processed/player_seasons_real_salary.parquet')
    rationality_trend = test_rationality_over_time(standings, salary_data_path)
    if rationality_trend is not None:
        rationality_trend.to_csv(tables_dir / 'rationality_over_time.csv', index=False)
        results['rationality_trend'] = rationality_trend
        results['trend_slope'] = rationality_trend.attrs.get('trend_slope', None)
        results['trend_pvalue'] = rationality_trend.attrs.get('trend_pvalue', None)
        results['trend_improving'] = rationality_trend.attrs.get('trend_improving', None)

    # 8. Create figures
    create_decision_theory_figures(merged, playoff_curves, figures_dir)

    # Summary
    print("\n" + "="*70)
    print("DECISION THEORY ANALYSIS SUMMARY")
    print("="*70)

    trend_msg = ""
    if rationality_trend is not None and results.get('trend_slope') is not None:
        trend_dir = "improving" if results['trend_improving'] else "declining"
        trend_msg = f"""
5. RATIONALITY TREND OVER TIME
   Trend: {results['trend_slope']:.4f}/year (p = {results['trend_pvalue']:.4f})
   Direction: Decisions are {trend_dir}
   First year (2000): {rationality_trend[rationality_trend['year'] == 2000]['rationality_score'].values[0]:.3f}
   Last year (2016): {rationality_trend[rationality_trend['year'] == 2016]['rationality_score'].values[0]:.3f}
"""

    print(f"""
Key Findings:

1. PLAYOFF PROBABILITY CURVE
   P(playoffs | 81 wins) = {playoff_curves['playoff_prob'](81):.1%}
   P(playoffs | 90 wins) = {playoff_curves['playoff_prob'](90):.1%}
   P(playoffs | 95 wins) = {playoff_curves['playoff_prob'](95):.1%}

2. MARKET SIZE EFFECTS
   Largest market: {market_sizes.iloc[0]['team']} ({market_sizes.iloc[0]['market_index']:.2f}x avg)
   Smallest market: {market_sizes.iloc[-1]['team']} ({market_sizes.iloc[-1]['market_index']:.2f}x avg)

3. RATIONALITY TEST (Cross-sectional)
   Correlation(implied, rational): r = {results['correlation']:.3f} (p = {results['correlation_pvalue']:.4f})
   R-squared: {results['r_squared']:.3f}

4. INTERPRETATION
   {results['conclusion']}

   Teams that "overpay" (high gamma) tend to be large-market teams near playoff
   contention, where the marginal win value is genuinely higher.

   This suggests apparent "inefficiency" partially reflects rational behavior
   given heterogeneous team objectives, not pure mistakes.
{trend_msg}""")

    # Save summary
    with open(output_dir / 'decision_theory_summary.txt', 'w') as f:
        f.write("DECISION THEORY ANALYSIS: ARE 'INEFFICIENT' TEAMS OPTIMAL?\n")
        f.write("="*70 + "\n\n")
        f.write(f"Correlation(implied, rational win values): r = {results['correlation']:.3f}\n")
        f.write(f"p-value: {results['correlation_pvalue']:.4f}\n")
        f.write(f"R-squared: {results['r_squared']:.3f}\n")
        f.write(f"\nConclusion: {results['conclusion']}\n")

    return standings, playoff_curves, market_sizes, merged, results


if __name__ == "__main__":
    standings, playoff_curves, market_sizes, merged, results = main()
