"""
Deep Dive on Learning Mechanisms - Full Battery of Tests

Following the systematic framework to find affirmative learning mechanisms:
1. Within-game updating (immediate learning)
2. Context-sensitivity by experience (expert vs novice decision-making)
3. Coaching tree analysis (network learning)
4. Analytics environment effects (organizational learning)
5. Salient event learning (regret-driven updating)
6. Playoff loss salience
7. Job security effects
8. Network vs experiential learning decomposition
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Make statsmodels optional
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except (ImportError, OSError) as e:
    HAS_STATSMODELS = False
    print(f"Note: statsmodels not available ({type(e).__name__}), using scipy for regressions")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Scipy-based regression fallback when statsmodels unavailable
class ScipyOLSResult:
    """Mimics statsmodels OLS result for basic usage."""
    def __init__(self, params, pvalues, rsquared, param_names):
        self._params = params
        self._pvalues = pvalues
        self.rsquared = rsquared
        self._param_names = param_names
        # Create dict-like access
        self.params = {name: p for name, p in zip(param_names, params)}
        self.pvalues = {name: p for name, p in zip(param_names, pvalues)}


def scipy_add_constant(X):
    """Add constant column like sm.add_constant."""
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.insert(0, 'const', 1.0)
        return X
    else:
        return np.column_stack([np.ones(len(X)), X])


def scipy_ols_fit(y, X):
    """Perform OLS regression using scipy/numpy."""
    if isinstance(X, pd.DataFrame):
        param_names = X.columns.tolist()
        X_arr = X.values
    else:
        param_names = [f'x{i}' for i in range(X.shape[1])]
        X_arr = X

    y_arr = np.array(y).flatten()

    # OLS via least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_arr, y_arr, rcond=None)

    # Calculate R-squared
    y_pred = X_arr @ coeffs
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    rsquared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Calculate p-values (approximate)
    n = len(y_arr)
    k = X_arr.shape[1]

    if n > k:
        mse = ss_res / (n - k)
        var_coef = mse * np.linalg.inv(X_arr.T @ X_arr).diagonal()
        se = np.sqrt(np.abs(var_coef))
        t_stats = coeffs / (se + 1e-10)
        pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    else:
        pvalues = np.ones(k)

    return ScipyOLSResult(coeffs, pvalues, rsquared, param_names)


def add_constant(X):
    """Wrapper for adding constant - uses statsmodels if available."""
    if HAS_STATSMODELS:
        return sm.add_constant(X)
    return scipy_add_constant(X)


def run_ols(y, X):
    """Wrapper for OLS regression - uses statsmodels if available."""
    if HAS_STATSMODELS:
        return sm.OLS(y, X).fit()
    return scipy_ols_fit(y, X)

# Import from existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from learning_mechanisms import COACH_MAPPING

# Coaching trees - expanded
COACHING_TREES = {
    'Belichick': ['Bill Belichick', 'Matt Patricia', 'Brian Flores', 'Joe Judge',
                  'Josh McDaniels', 'Bill OBrien', 'Mike Vrabel', 'Jerod Mayo'],
    'Reid': ['Andy Reid', 'Doug Pederson', 'Matt Nagy', 'Ron Rivera', 'Sean McDermott'],
    'Shanahan/McVay': ['Kyle Shanahan', 'Sean McVay', 'Matt LaFleur', 'Zac Taylor',
                       'Kevin OConnell', 'Mike McDaniel'],
    'Payton': ['Sean Payton', 'Dennis Allen', 'Dan Campbell'],
    'Tomlin': ['Mike Tomlin'],  # Standalone
    'Carroll': ['Pete Carroll'],  # Standalone
}

# Analytics pioneer teams (known early adopters)
ANALYTICS_PIONEERS = ['PHI', 'BAL', 'CLE', 'NE', 'LAR', 'SF']


def load_data():
    """Load all required data."""
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'

    pbp = pd.read_parquet(data_dir / 'all_pbp_1999_2024.parquet')
    decisions = pd.read_parquet(output_dir / 'two_point_decision_analysis.parquet')

    return pbp, decisions


def add_coach_info(df):
    """Add coach and tenure information to dataframe."""
    df = df.copy()
    df['coach'] = df.apply(
        lambda r: COACH_MAPPING.get((r['posteam'], r['season']), 'Unknown'),
        axis=1
    )

    # Compute tenure
    coach_first = df.groupby('coach')['season'].min().reset_index()
    coach_first.columns = ['coach', 'first_year']
    df = df.merge(coach_first, on='coach', how='left')
    df['tenure'] = df['season'] - df['first_year'] + 1

    # Add coaching tree
    def get_tree(coach):
        for tree_name, coaches in COACHING_TREES.items():
            if coach in coaches:
                return tree_name
        return 'Other'
    df['coaching_tree'] = df['coach'].apply(get_tree)

    # Add analytics pioneer flag
    df['analytics_pioneer'] = df['posteam'].isin(ANALYTICS_PIONEERS)

    return df


# =============================================================================
# TEST 1: WITHIN-GAME UPDATING
# =============================================================================

def test_within_game_updating(pbp):
    """
    Test 1: Within-game updating - the cleanest test.

    Same coach, same game, controls for everything.
    After 2pt SUCCESS/FAILURE earlier in game, does behavior change?
    """
    print("\n" + "=" * 80)
    print("TEST 1: WITHIN-GAME UPDATING")
    print("=" * 80)

    # Get all PAT/2pt decisions
    pat_plays = pbp[pbp['extra_point_attempt'] == 1].copy()
    two_pt_plays = pbp[pbp['two_point_attempt'] == 1].copy()

    pat_plays['decision'] = 'pat'
    pat_plays['success'] = (pat_plays['extra_point_result'] == 'good').astype(float)

    two_pt_plays['decision'] = '2pt'
    two_pt_plays['success'] = (two_pt_plays['two_point_conv_result'] == 'success').astype(float)

    all_decisions = pd.concat([pat_plays, two_pt_plays])
    all_decisions = all_decisions[all_decisions['season'] >= 2015]
    all_decisions = all_decisions.sort_values(['game_id', 'play_id'])

    # For each team in each game, track sequence
    results = []

    for (game_id, team), group in all_decisions.groupby(['game_id', 'posteam']):
        group = group.sort_values('play_id').reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            prev = group.iloc[i-1]
            curr = group.iloc[i]

            results.append({
                'game_id': game_id,
                'team': team,
                'season': curr['season'],
                'prev_decision': prev['decision'],
                'prev_success': prev['success'],
                'curr_decision': curr['decision'],
                'curr_went_for_2': curr['decision'] == '2pt',
                'score_diff': curr['score_differential'],
                'time_remaining': curr['game_seconds_remaining'],
                'decision_num': i + 1
            })

    results_df = pd.DataFrame(results)

    print(f"\nTotal sequential decision pairs: {len(results_df)}")

    # Key comparison: What happened after different prior outcomes?
    print("\n" + "-" * 60)
    print("P(go for 2) on NEXT TD, conditional on PREVIOUS attempt:")
    print("-" * 60)

    conditions = [
        ('After PAT', results_df['prev_decision'] == 'pat'),
        ('After 2pt SUCCESS', (results_df['prev_decision'] == '2pt') & (results_df['prev_success'] == 1)),
        ('After 2pt FAILURE', (results_df['prev_decision'] == '2pt') & (results_df['prev_success'] == 0)),
    ]

    rates = {}
    for label, mask in conditions:
        subset = results_df[mask]
        rate = subset['curr_went_for_2'].mean()
        n = len(subset)
        se = np.sqrt(rate * (1 - rate) / n) if n > 0 else 0
        rates[label] = (rate, n, se)
        print(f"  {label}: {rate:.1%} ({n:,} obs, SE={se:.3f})")

    # Statistical tests
    print("\n" + "-" * 60)
    print("Statistical Tests:")
    print("-" * 60)

    # Test 1: 2pt success vs 2pt failure
    success_data = results_df[(results_df['prev_decision'] == '2pt') & (results_df['prev_success'] == 1)]
    failure_data = results_df[(results_df['prev_decision'] == '2pt') & (results_df['prev_success'] == 0)]

    if len(success_data) > 10 and len(failure_data) > 10:
        stat, pval = stats.mannwhitneyu(
            success_data['curr_went_for_2'].values,
            failure_data['curr_went_for_2'].values,
            alternative='two-sided'
        )
        diff = success_data['curr_went_for_2'].mean() - failure_data['curr_went_for_2'].mean()
        print(f"\n  2pt Success vs 2pt Failure:")
        print(f"    Difference: {diff*100:+.1f} pp")
        print(f"    Mann-Whitney p-value: {pval:.4f}")

        if pval < 0.05:
            if diff > 0:
                print("    => SIGNIFICANT: Success encourages more 2pt attempts")
            else:
                print("    => SIGNIFICANT: Failure encourages more 2pt attempts (!)")
        else:
            print("    => No significant within-game updating")

    # Test 2: After 2pt (any) vs after PAT
    after_2pt = results_df[results_df['prev_decision'] == '2pt']
    after_pat = results_df[results_df['prev_decision'] == 'pat']

    if len(after_2pt) > 10 and len(after_pat) > 10:
        stat, pval = stats.mannwhitneyu(
            after_2pt['curr_went_for_2'].values,
            after_pat['curr_went_for_2'].values,
            alternative='two-sided'
        )
        diff = after_2pt['curr_went_for_2'].mean() - after_pat['curr_went_for_2'].mean()
        print(f"\n  After 2pt (any) vs After PAT:")
        print(f"    Difference: {diff*100:+.1f} pp")
        print(f"    Mann-Whitney p-value: {pval:.4f}")

    # Regression with controls
    print("\n" + "-" * 60)
    print("Regression with game-state controls:")
    print("-" * 60)

    reg_df = results_df.dropna(subset=['score_diff', 'time_remaining'])
    reg_df['prev_was_2pt'] = (reg_df['prev_decision'] == '2pt').astype(int)
    reg_df['prev_2pt_success'] = ((reg_df['prev_decision'] == '2pt') & (reg_df['prev_success'] == 1)).astype(int)
    reg_df['prev_2pt_failure'] = ((reg_df['prev_decision'] == '2pt') & (reg_df['prev_success'] == 0)).astype(int)

    X = reg_df[['prev_2pt_success', 'prev_2pt_failure', 'score_diff', 'time_remaining']]
    X = add_constant(X)
    y = reg_df['curr_went_for_2'].astype(int)

    try:
        model = run_ols(y, X)
        print(f"\n  Dep var: P(go for 2 on current TD)")
        print(f"  N = {len(y):,}")
        print(f"\n  Coefficient estimates:")
        print(f"    prev_2pt_success: {model.params['prev_2pt_success']:.4f} (p={model.pvalues['prev_2pt_success']:.4f})")
        print(f"    prev_2pt_failure: {model.params['prev_2pt_failure']:.4f} (p={model.pvalues['prev_2pt_failure']:.4f})")
        print(f"    score_diff:       {model.params['score_diff']:.4f} (p={model.pvalues['score_diff']:.4f})")
        print(f"    time_remaining:   {model.params['time_remaining']:.6f} (p={model.pvalues['time_remaining']:.4f})")
        print(f"\n  R²: {model.rsquared:.4f}")
    except Exception as e:
        print(f"  Regression failed: {e}")

    return results_df


# =============================================================================
# TEST 2: CONTEXT-SENSITIVITY BY EXPERIENCE
# =============================================================================

def test_context_sensitivity(decisions):
    """
    Test 2: Expert coaches should be more responsive to game state.

    Higher R² for experienced coaches = more context-sensitive
    Lower R² for rookie coaches = following simple heuristics
    """
    print("\n" + "=" * 80)
    print("TEST 2: CONTEXT-SENSITIVITY BY EXPERIENCE")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Define experience groups
    decisions['exp_group'] = pd.cut(
        decisions['tenure'],
        bins=[0, 1, 3, 6, 20],
        labels=['Year 1', 'Year 2-3', 'Year 4-6', 'Year 7+']
    )

    # For each experience level, regress 2pt decision on game state
    print("\nRegression: 2pt_attempt ~ score_diff + time_remaining + other controls")
    print("-" * 80)
    print(f"{'Experience':<12} | {'N':>6} | {'R²':>8} | {'β_score_diff':>14} | {'β_time':>12} | {'2pt Rate':>10}")
    print("-" * 80)

    results = []
    for exp_group in ['Year 1', 'Year 2-3', 'Year 4-6', 'Year 7+']:
        subset = decisions[decisions['exp_group'] == exp_group].copy()
        subset = subset.dropna(subset=['score_diff_pre_td', 'time_remaining'])

        if len(subset) < 100:
            continue

        # Create features
        subset['went_for_2'] = (subset['actual_decision'] == 'two_point').astype(int)
        subset['time_min'] = subset['time_remaining'] / 60

        X = subset[['score_diff_pre_td', 'time_min']]
        X = add_constant(X)
        y = subset['went_for_2']

        try:
            model = run_ols(y, X)

            two_pt_rate = y.mean()
            results.append({
                'exp_group': exp_group,
                'n': len(y),
                'r2': model.rsquared,
                'beta_score': model.params['score_diff_pre_td'],
                'beta_time': model.params['time_min'],
                'two_pt_rate': two_pt_rate
            })

            print(f"{exp_group:<12} | {len(y):>6} | {model.rsquared:>8.4f} | "
                  f"{model.params['score_diff_pre_td']:>14.4f} | {model.params['time_min']:>12.4f} | "
                  f"{two_pt_rate:>10.1%}")
        except:
            continue

    results_df = pd.DataFrame(results)

    # Key insight
    print("\n" + "-" * 60)
    print("KEY INSIGHT:")
    print("-" * 60)

    if len(results_df) >= 2:
        r2_year1 = results_df[results_df['exp_group'] == 'Year 1']['r2'].values
        r2_veteran = results_df[results_df['exp_group'] == 'Year 7+']['r2'].values

        if len(r2_year1) > 0 and len(r2_veteran) > 0:
            diff = r2_veteran[0] - r2_year1[0]
            print(f"\n  R² Year 1: {r2_year1[0]:.4f}")
            print(f"  R² Year 7+: {r2_veteran[0]:.4f}")
            print(f"  Difference: {diff:+.4f}")

            if diff > 0:
                print("\n  => Experienced coaches are MORE context-sensitive")
                print("     They respond more to game state when deciding 2pt vs PAT")
            else:
                print("\n  => Experienced coaches are LESS context-sensitive (?)")
                print("     They may have developed fixed heuristics")

    return results_df


# =============================================================================
# TEST 3: COACHING TREE ANALYSIS
# =============================================================================

def test_coaching_trees(decisions):
    """
    Test 3: Do coaches from the same tree behave similarly?

    Is within-tree correlation higher than random?
    """
    print("\n" + "=" * 80)
    print("TEST 3: COACHING TREE ANALYSIS")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Aggregate by coach
    coach_stats = decisions.groupby(['coach', 'coaching_tree']).agg(
        n_decisions=('coach_correct', 'count'),
        pct_correct=('coach_correct', 'mean'),
        two_pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    coach_stats = coach_stats[coach_stats['n_decisions'] >= 50]  # Min sample

    # By tree
    print("\nCoaching Tree Summary:")
    print("-" * 80)
    print(f"{'Tree':<20} | {'Coaches':>8} | {'Decisions':>10} | {'2pt Rate':>10} | {'Correct %':>10} | {'Std Dev':>10}")
    print("-" * 80)

    tree_stats = []
    for tree in coach_stats['coaching_tree'].unique():
        tree_coaches = coach_stats[coach_stats['coaching_tree'] == tree]

        if len(tree_coaches) < 2:
            continue

        n_coaches = len(tree_coaches)
        total_decisions = tree_coaches['n_decisions'].sum()
        mean_2pt = tree_coaches['two_pt_rate'].mean()
        mean_correct = tree_coaches['pct_correct'].mean()
        std_correct = tree_coaches['pct_correct'].std()

        tree_stats.append({
            'tree': tree,
            'n_coaches': n_coaches,
            'total_decisions': total_decisions,
            'mean_2pt_rate': mean_2pt,
            'mean_correct': mean_correct,
            'std_correct': std_correct
        })

        print(f"{tree:<20} | {n_coaches:>8} | {total_decisions:>10} | "
              f"{mean_2pt:>10.1%} | {mean_correct:>10.1%} | {std_correct:>10.3f}")

    tree_df = pd.DataFrame(tree_stats)

    # ANOVA: Is between-tree variance > within-tree variance?
    print("\n" + "-" * 60)
    print("ANOVA: Between-tree vs Within-tree variance")
    print("-" * 60)

    groups = [coach_stats[coach_stats['coaching_tree'] == tree]['pct_correct'].values
              for tree in coach_stats['coaching_tree'].unique()
              if len(coach_stats[coach_stats['coaching_tree'] == tree]) >= 2]

    if len(groups) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"\n  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_val:.4f}")

            if p_val < 0.05:
                print("\n  => SIGNIFICANT: Coaching trees differ in decision quality")
            else:
                print("\n  => No significant difference between coaching trees")
        except:
            print("  ANOVA failed - insufficient data")

    # Specific tree comparisons
    print("\n" + "-" * 60)
    print("Specific Tree Comparisons:")
    print("-" * 60)

    for tree in ['Belichick', 'Reid', 'Shanahan/McVay', 'Payton']:
        tree_coaches = coach_stats[coach_stats['coaching_tree'] == tree]
        if len(tree_coaches) >= 2:
            print(f"\n  {tree} tree coaches:")
            for _, row in tree_coaches.iterrows():
                print(f"    {row['coach']}: {row['pct_correct']:.1%} correct, {row['two_pt_rate']:.1%} 2pt rate")

    return coach_stats, tree_df


# =============================================================================
# TEST 4: ANALYTICS ENVIRONMENT EFFECTS
# =============================================================================

def test_analytics_environment(decisions, pbp):
    """
    Test 4: Do coaches at analytics-forward teams learn faster?

    Proxy: 4th down aggressiveness as indicator of analytics culture
    """
    print("\n" + "=" * 80)
    print("TEST 4: ANALYTICS ENVIRONMENT EFFECTS")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Compute 4th down aggressiveness by team-season as analytics proxy
    fourth_downs = pbp[
        (pbp['down'] == 4) &
        (pbp['play_type'].isin(['run', 'pass', 'punt', 'field_goal'])) &
        (pbp['season'] >= 2015)
    ].copy()

    fourth_downs['went_for_it'] = fourth_downs['play_type'].isin(['run', 'pass']).astype(int)

    team_4th_agg = fourth_downs.groupby(['posteam', 'season']).agg(
        fourth_down_go_rate=('went_for_it', 'mean'),
        n_fourth_downs=('went_for_it', 'count')
    ).reset_index()

    # Merge with decisions
    decisions = decisions.merge(
        team_4th_agg,
        on=['posteam', 'season'],
        how='left'
    )

    # Also add pioneer flag
    decisions['is_pioneer'] = decisions['posteam'].isin(ANALYTICS_PIONEERS)

    # Analysis 1: Pioneer vs non-pioneer teams
    print("\nPioneer Teams (PHI, BAL, CLE, NE, LAR, SF) vs Others:")
    print("-" * 60)

    pioneer = decisions[decisions['is_pioneer']]
    non_pioneer = decisions[~decisions['is_pioneer']]

    print(f"\n  Pioneer teams:")
    print(f"    N decisions: {len(pioneer):,}")
    print(f"    2pt rate: {(pioneer['actual_decision'] == 'two_point').mean():.1%}")
    print(f"    Correct rate: {pioneer['coach_correct'].mean():.1%}")
    print(f"    Mean WP cost: {pioneer['wp_cost'].mean()*1e6:.1f}")

    print(f"\n  Non-pioneer teams:")
    print(f"    N decisions: {len(non_pioneer):,}")
    print(f"    2pt rate: {(non_pioneer['actual_decision'] == 'two_point').mean():.1%}")
    print(f"    Correct rate: {non_pioneer['coach_correct'].mean():.1%}")
    print(f"    Mean WP cost: {non_pioneer['wp_cost'].mean()*1e6:.1f}")

    # Statistical test
    stat, pval = stats.mannwhitneyu(
        pioneer['coach_correct'].values,
        non_pioneer['coach_correct'].values,
        alternative='two-sided'
    )
    print(f"\n  Mann-Whitney p-value: {pval:.4f}")

    # Analysis 2: Correlation with 4th down aggressiveness
    print("\n" + "-" * 60)
    print("Correlation: 4th Down Aggressiveness × 2pt Decision Quality")
    print("-" * 60)

    team_season = decisions.groupby(['posteam', 'season']).agg(
        pct_correct=('coach_correct', 'mean'),
        two_pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        fourth_down_go_rate=('fourth_down_go_rate', 'first'),
        n_decisions=('coach_correct', 'count')
    ).reset_index()

    team_season = team_season.dropna(subset=['fourth_down_go_rate'])
    team_season = team_season[team_season['n_decisions'] >= 20]

    if len(team_season) > 10:
        corr_correct = team_season['fourth_down_go_rate'].corr(team_season['pct_correct'])
        corr_2pt = team_season['fourth_down_go_rate'].corr(team_season['two_pt_rate'])

        print(f"\n  Correlation (4th down go rate × correct rate): {corr_correct:.3f}")
        print(f"  Correlation (4th down go rate × 2pt rate): {corr_2pt:.3f}")

        if corr_correct > 0.1:
            print("\n  => Teams aggressive on 4th down are also better at 2pt decisions")
            print("     Suggests analytics culture matters")
        elif corr_correct < -0.1:
            print("\n  => Teams aggressive on 4th down are WORSE at 2pt decisions (?)")
        else:
            print("\n  => No strong relationship between 4th down aggression and 2pt quality")

    # Analysis 3: Learning rate at pioneer vs non-pioneer
    print("\n" + "-" * 60)
    print("Learning Rate: Pioneer vs Non-Pioneer Teams")
    print("-" * 60)

    for is_pioneer, label in [(True, 'Pioneer'), (False, 'Non-Pioneer')]:
        subset = decisions[decisions['is_pioneer'] == is_pioneer]

        by_tenure = subset.groupby('tenure').agg(
            pct_correct=('coach_correct', 'mean'),
            n=('coach_correct', 'count')
        ).reset_index()

        by_tenure = by_tenure[by_tenure['n'] >= 50]

        if len(by_tenure) >= 3:
            # Simple linear regression for learning slope
            X = add_constant(by_tenure[['tenure']])
            y = by_tenure['pct_correct']
            model = run_ols(y, X)
            slope = model.params['tenure']

            print(f"\n  {label} teams:")
            print(f"    Year 1 correct: {by_tenure[by_tenure['tenure']==1]['pct_correct'].values[0]:.1%}")
            print(f"    Learning slope: {slope*100:.2f} pp per year")

    return team_season


# =============================================================================
# TEST 5: SALIENT EVENT LEARNING (REGRET)
# =============================================================================

def test_salient_event_learning(pbp, decisions):
    """
    Test 5: Do coaches learn from near-miss regret?

    Identify games where:
    - Kicked PAT when should have gone for 2
    - Lost by 1-2 points (2pt would have mattered)
    """
    print("\n" + "=" * 80)
    print("TEST 5: SALIENT EVENT LEARNING (COUNTERFACTUAL REGRET)")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Get game outcomes
    game_outcomes = pbp.groupby('game_id').agg(
        home_final=('home_score', 'max'),
        away_final=('away_score', 'max'),
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first')
    ).reset_index()

    decisions = decisions.merge(game_outcomes, on='game_id', how='left')

    # Calculate final margin for each team
    decisions['team_final'] = np.where(
        decisions['posteam'] == decisions['home_team'],
        decisions['home_final'],
        decisions['away_final']
    )
    decisions['opp_final'] = np.where(
        decisions['posteam'] == decisions['home_team'],
        decisions['away_final'],
        decisions['home_final']
    )
    decisions['final_margin'] = decisions['team_final'] - decisions['opp_final']

    # Identify regret situations
    decisions['regret'] = (
        (decisions['actual_decision'] == 'pat') &
        (decisions['optimal_decision'] == 'two_point') &
        (decisions['final_margin'].isin([-1, -2]))
    )

    # Also identify "no regret" controls
    decisions['no_regret_won'] = (
        (decisions['actual_decision'] == 'pat') &
        (decisions['optimal_decision'] == 'two_point') &
        (decisions['final_margin'] > 0)
    )

    decisions['no_regret_blowout'] = (
        (decisions['actual_decision'] == 'pat') &
        (decisions['optimal_decision'] == 'two_point') &
        (decisions['final_margin'] < -3)
    )

    print(f"\nRegret situations identified: {decisions['regret'].sum()}")
    print(f"No-regret (won anyway): {decisions['no_regret_won'].sum()}")
    print(f"No-regret (lost by 4+): {decisions['no_regret_blowout'].sum()}")

    # Track coach behavior change
    # For each coach with a regret situation, compare before/after
    regret_coaches = decisions[decisions['regret']].groupby(['coach', 'season']).size().reset_index(name='n_regrets')

    print("\n" + "-" * 60)
    print("Did regret events change coach behavior?")
    print("-" * 60)

    results = []
    for _, row in regret_coaches.iterrows():
        coach = row['coach']
        regret_season = row['season']

        coach_data = decisions[decisions['coach'] == coach]

        # Compare behavior in similar situations before vs after
        similar = coach_data[coach_data['optimal_decision'] == 'two_point']

        before = similar[similar['season'] < regret_season]
        after = similar[similar['season'] > regret_season]

        if len(before) >= 5 and len(after) >= 5:
            before_2pt_rate = (before['actual_decision'] == 'two_point').mean()
            after_2pt_rate = (after['actual_decision'] == 'two_point').mean()

            before_correct = before['coach_correct'].mean()
            after_correct = after['coach_correct'].mean()

            results.append({
                'coach': coach,
                'regret_season': regret_season,
                'before_2pt_rate': before_2pt_rate,
                'after_2pt_rate': after_2pt_rate,
                'before_correct': before_correct,
                'after_correct': after_correct,
                'n_before': len(before),
                'n_after': len(after)
            })

    if results:
        results_df = pd.DataFrame(results)
        results_df['2pt_change'] = results_df['after_2pt_rate'] - results_df['before_2pt_rate']
        results_df['correct_change'] = results_df['after_correct'] - results_df['before_correct']

        print(f"\n  Coaches with sufficient data: {len(results_df)}")
        print(f"\n  Mean change in 2pt rate (when optimal): {results_df['2pt_change'].mean()*100:+.1f} pp")
        print(f"  Mean change in correct rate: {results_df['correct_change'].mean()*100:+.1f} pp")

        # Test: Is change significantly different from 0?
        stat, pval = stats.ttest_1samp(results_df['2pt_change'], 0)
        print(f"\n  t-test (2pt change ≠ 0): p = {pval:.4f}")

        if pval < 0.05 and results_df['2pt_change'].mean() > 0:
            print("  => SIGNIFICANT: Coaches increase 2pt rate after regret events")
        else:
            print("  => No significant change after regret events")

        return results_df
    else:
        print("  Insufficient data for before/after comparison")
        return None


# =============================================================================
# TEST 6: PLAYOFF LOSS SALIENCE
# =============================================================================

def test_playoff_loss_salience(pbp, decisions):
    """
    Test 6: Do coaches who lose playoff games by small margins update more?
    """
    print("\n" + "=" * 80)
    print("TEST 6: PLAYOFF LOSS SALIENCE")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Get playoff games
    playoff_games = pbp[pbp['season_type'] == 'POST'].copy()

    if len(playoff_games) == 0:
        print("No playoff game data available")
        return None

    # Get game outcomes
    playoff_outcomes = playoff_games.groupby('game_id').agg(
        home_final=('home_score', 'max'),
        away_final=('away_score', 'max'),
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
        season=('season', 'first')
    ).reset_index()

    playoff_outcomes['margin'] = abs(playoff_outcomes['home_final'] - playoff_outcomes['away_final'])
    playoff_outcomes['winner'] = np.where(
        playoff_outcomes['home_final'] > playoff_outcomes['away_final'],
        playoff_outcomes['home_team'],
        playoff_outcomes['away_team']
    )
    playoff_outcomes['loser'] = np.where(
        playoff_outcomes['home_final'] > playoff_outcomes['away_final'],
        playoff_outcomes['away_team'],
        playoff_outcomes['home_team']
    )

    # Identify close playoff losses
    close_losses = playoff_outcomes[playoff_outcomes['margin'] <= 3][['loser', 'season', 'margin']]
    close_losses.columns = ['team', 'season', 'margin']
    close_losses['close_playoff_loss'] = True

    print(f"\nClose playoff losses (margin ≤ 3): {len(close_losses)}")

    # Merge with next season's decisions
    close_losses['next_season'] = close_losses['season'] + 1

    # For each close loss, compare next season behavior to baseline
    results = []

    for _, row in close_losses.iterrows():
        team = row['team']
        next_season = row['next_season']
        margin = row['margin']

        # Get coach for that team-season
        coach = COACH_MAPPING.get((team, next_season), 'Unknown')

        if coach == 'Unknown':
            continue

        # Get this coach's decisions
        coach_decisions = decisions[decisions['coach'] == coach]

        # Compare next season to previous
        this_season = coach_decisions[coach_decisions['season'] == next_season]
        prev_seasons = coach_decisions[coach_decisions['season'] < next_season]

        if len(this_season) >= 10 and len(prev_seasons) >= 20:
            this_correct = this_season['coach_correct'].mean()
            prev_correct = prev_seasons['coach_correct'].mean()

            results.append({
                'coach': coach,
                'team': team,
                'loss_season': row['season'],
                'margin': margin,
                'prev_correct': prev_correct,
                'next_correct': this_correct,
                'change': this_correct - prev_correct
            })

    if results:
        results_df = pd.DataFrame(results)

        print("\n" + "-" * 60)
        print("Did close playoff losses change decision quality?")
        print("-" * 60)

        print(f"\n  Coaches with close playoff loss: {len(results_df)}")
        print(f"  Mean change in correct rate: {results_df['change'].mean()*100:+.1f} pp")

        for _, row in results_df.iterrows():
            print(f"    {row['coach']} ({row['team']} {int(row['loss_season'])}): "
                  f"{row['prev_correct']:.1%} → {row['next_correct']:.1%} ({row['change']*100:+.1f} pp)")

        return results_df
    else:
        print("  Insufficient data for analysis")
        return None


# =============================================================================
# TEST 7: JOB SECURITY EFFECTS
# =============================================================================

def test_job_security(decisions):
    """
    Test 7: Does job security affect risk-taking?

    Proxy: coming off winning vs losing season
    """
    print("\n" + "=" * 80)
    print("TEST 7: JOB SECURITY EFFECTS")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # We don't have win/loss records directly, but we can proxy with tenure
    # Coaches in year 1-2 might be more insecure
    # Coaches with 5+ years are clearly secure

    decisions['job_security'] = pd.cut(
        decisions['tenure'],
        bins=[0, 2, 4, 20],
        labels=['Low (Yr 1-2)', 'Medium (Yr 3-4)', 'High (Yr 5+)']
    )

    print("\nDecision-making by Job Security (proxy: tenure):")
    print("-" * 70)
    print(f"{'Security':<20} | {'N':>8} | {'2pt Rate':>10} | {'Correct %':>10} | {'WP Cost':>10}")
    print("-" * 70)

    for security in ['Low (Yr 1-2)', 'Medium (Yr 3-4)', 'High (Yr 5+)']:
        subset = decisions[decisions['job_security'] == security]

        if len(subset) < 100:
            continue

        two_pt_rate = (subset['actual_decision'] == 'two_point').mean()
        correct_rate = subset['coach_correct'].mean()
        wp_cost = subset['wp_cost'].mean() * 1e6

        print(f"{security:<20} | {len(subset):>8} | {two_pt_rate:>10.1%} | "
              f"{correct_rate:>10.1%} | {wp_cost:>10.1f}")

    # Test: Are secure coaches more aggressive?
    low_security = decisions[decisions['tenure'] <= 2]
    high_security = decisions[decisions['tenure'] >= 5]

    if len(low_security) > 100 and len(high_security) > 100:
        low_2pt = (low_security['actual_decision'] == 'two_point').mean()
        high_2pt = (high_security['actual_decision'] == 'two_point').mean()

        print(f"\n  Low security 2pt rate: {low_2pt:.1%}")
        print(f"  High security 2pt rate: {high_2pt:.1%}")
        print(f"  Difference: {(high_2pt - low_2pt)*100:+.1f} pp")

        stat, pval = stats.mannwhitneyu(
            (high_security['actual_decision'] == 'two_point').astype(int).values,
            (low_security['actual_decision'] == 'two_point').astype(int).values,
            alternative='two-sided'
        )
        print(f"  Mann-Whitney p-value: {pval:.4f}")

    return decisions


# =============================================================================
# TEST 8: NETWORK VS EXPERIENTIAL LEARNING DECOMPOSITION
# =============================================================================

def test_learning_decomposition(decisions):
    """
    Test 8: When coaches change teams, what predicts their new behavior?

    Model: new_behavior = β1*own_prior + β2*mentor + β3*new_org_prior
    """
    print("\n" + "=" * 80)
    print("TEST 8: NETWORK VS EXPERIENTIAL LEARNING DECOMPOSITION")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Identify coaches who changed teams
    # First, get coach-team-season combinations
    coach_teams = decisions.groupby(['coach', 'posteam', 'season']).agg(
        n_decisions=('coach_correct', 'count'),
        correct_rate=('coach_correct', 'mean'),
        two_pt_rate=('actual_decision', lambda x: (x == 'two_point').mean())
    ).reset_index()

    # Find coaches with multiple teams
    coaches_multiple_teams = coach_teams.groupby('coach')['posteam'].nunique()
    movers = coaches_multiple_teams[coaches_multiple_teams > 1].index.tolist()

    print(f"\nCoaches who changed teams: {len(movers)}")

    if len(movers) < 5:
        print("  Not enough coach movers for decomposition analysis")
        return None

    # For each mover, get:
    # 1. Their prior behavior (at old team)
    # 2. Their new behavior (at new team)
    # 3. The new org's prior behavior (before they arrived)

    results = []

    for coach in movers:
        coach_data = coach_teams[coach_teams['coach'] == coach].sort_values('season')

        if len(coach_data) < 2:
            continue

        # Find team changes
        teams = coach_data['posteam'].unique()

        for i in range(len(teams) - 1):
            old_team = teams[i]
            new_team = teams[i + 1]

            old_data = coach_data[coach_data['posteam'] == old_team]
            new_data = coach_data[coach_data['posteam'] == new_team]

            if len(old_data) == 0 or len(new_data) == 0:
                continue

            transition_season = new_data['season'].min()

            # Coach's prior behavior
            own_prior_correct = old_data['correct_rate'].mean()
            own_prior_2pt = old_data['two_pt_rate'].mean()

            # Coach's new behavior
            new_correct = new_data['correct_rate'].mean()
            new_2pt = new_data['two_pt_rate'].mean()

            # New org's prior behavior (other coaches at that team before)
            org_prior = coach_teams[
                (coach_teams['posteam'] == new_team) &
                (coach_teams['season'] < transition_season) &
                (coach_teams['coach'] != coach)
            ]

            if len(org_prior) > 0:
                org_prior_correct = org_prior['correct_rate'].mean()
                org_prior_2pt = org_prior['two_pt_rate'].mean()
            else:
                continue

            results.append({
                'coach': coach,
                'old_team': old_team,
                'new_team': new_team,
                'transition_season': transition_season,
                'own_prior_correct': own_prior_correct,
                'own_prior_2pt': own_prior_2pt,
                'new_correct': new_correct,
                'new_2pt': new_2pt,
                'org_prior_correct': org_prior_correct,
                'org_prior_2pt': org_prior_2pt
            })

    if len(results) < 5:
        print("  Not enough team transitions for regression")
        return None

    results_df = pd.DataFrame(results)

    print(f"\nTeam transitions with sufficient data: {len(results_df)}")

    # Regression: new_correct = β0 + β1*own_prior + β2*org_prior
    print("\n" + "-" * 60)
    print("Regression: New Behavior = β₁×Own Prior + β₂×Org Prior")
    print("-" * 60)

    X = results_df[['own_prior_correct', 'org_prior_correct']]
    X = add_constant(X)
    y = results_df['new_correct']

    try:
        model = run_ols(y, X)

        print(f"\n  Dependent variable: Correct rate at new team")
        print(f"  N = {len(y)}")
        print(f"\n  β₁ (own prior):  {model.params['own_prior_correct']:.3f} (p={model.pvalues['own_prior_correct']:.3f})")
        print(f"  β₂ (org prior):  {model.params['org_prior_correct']:.3f} (p={model.pvalues['org_prior_correct']:.3f})")
        print(f"\n  R²: {model.rsquared:.3f}")

        if model.params['own_prior_correct'] > model.params['org_prior_correct']:
            print("\n  => EXPERIENTIAL learning dominates: Coaches bring their own style")
        else:
            print("\n  => ORGANIZATIONAL learning dominates: Teams shape coach behavior")
    except Exception as e:
        print(f"  Regression failed: {e}")

    return results_df


# =============================================================================
# TEST 9: TENURE × TEAM RECORD INTERACTION
# =============================================================================

def test_tenure_team_record(decisions, pbp):
    """
    Test 9: Do coaches on bad teams learn differently?

    Bad teams might experiment more, or get fired before learning.
    Good teams might be more conservative (protect the lead).
    """
    print("\n" + "=" * 80)
    print("TEST 9: TENURE × TEAM RECORD INTERACTION")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Get team season records
    # Approximate with point differential since we don't have W-L directly
    team_records = pbp.groupby(['posteam', 'season']).agg(
        total_points_for=('posteam_score', 'max'),
        total_points_against=('defteam_score', 'max'),
        n_games=('game_id', 'nunique')
    ).reset_index()

    # Better approximation: use game-level data
    game_outcomes = pbp.groupby(['game_id', 'posteam']).agg(
        team_score=('posteam_score', 'max'),
        opp_score=('defteam_score', 'max'),
        season=('season', 'first')
    ).reset_index()

    game_outcomes['won'] = (game_outcomes['team_score'] > game_outcomes['opp_score']).astype(int)

    team_season_records = game_outcomes.groupby(['posteam', 'season']).agg(
        wins=('won', 'sum'),
        games=('won', 'count'),
        point_diff=('team_score', 'sum')  # Just sum of points for
    ).reset_index()
    team_season_records['win_pct'] = team_season_records['wins'] / team_season_records['games']

    # Merge with decisions
    decisions = decisions.merge(
        team_season_records[['posteam', 'season', 'win_pct']],
        on=['posteam', 'season'],
        how='left'
    )

    # Create team quality bins
    decisions['team_quality'] = pd.cut(
        decisions['win_pct'],
        bins=[0, 0.35, 0.5, 0.65, 1.0],
        labels=['Bad (<.350)', 'Below Avg', 'Above Avg', 'Good (>.650)']
    )

    print("\nDecision Quality by Tenure × Team Quality:")
    print("-" * 90)
    print(f"{'Team Quality':<20} | {'Tenure':<10} | {'N':>6} | {'2pt Rate':>10} | {'Correct %':>10} | {'WP Cost':>10}")
    print("-" * 90)

    results = []
    for quality in ['Bad (<.350)', 'Below Avg', 'Above Avg', 'Good (>.650)']:
        for tenure_group in ['Year 1-2', 'Year 3-4', 'Year 5+']:
            if tenure_group == 'Year 1-2':
                tenure_mask = decisions['tenure'] <= 2
            elif tenure_group == 'Year 3-4':
                tenure_mask = (decisions['tenure'] >= 3) & (decisions['tenure'] <= 4)
            else:
                tenure_mask = decisions['tenure'] >= 5

            subset = decisions[(decisions['team_quality'] == quality) & tenure_mask]

            if len(subset) < 50:
                continue

            two_pt_rate = (subset['actual_decision'] == 'two_point').mean()
            correct_rate = subset['coach_correct'].mean()
            wp_cost = subset['wp_cost'].mean() * 1e6

            results.append({
                'team_quality': quality,
                'tenure': tenure_group,
                'n': len(subset),
                'two_pt_rate': two_pt_rate,
                'correct_rate': correct_rate,
                'wp_cost': wp_cost
            })

            print(f"{quality:<20} | {tenure_group:<10} | {len(subset):>6} | "
                  f"{two_pt_rate:>10.1%} | {correct_rate:>10.1%} | {wp_cost:>10.1f}")

    results_df = pd.DataFrame(results)

    # Key insights
    print("\n" + "-" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)

    # Compare learning slopes for good vs bad teams
    for quality in ['Bad (<.350)', 'Good (>.650)']:
        quality_data = results_df[results_df['team_quality'] == quality]
        if len(quality_data) >= 2:
            yr1_correct = quality_data[quality_data['tenure'] == 'Year 1-2']['correct_rate'].values
            yr5_correct = quality_data[quality_data['tenure'] == 'Year 5+']['correct_rate'].values

            if len(yr1_correct) > 0 and len(yr5_correct) > 0:
                improvement = (yr5_correct[0] - yr1_correct[0]) * 100
                print(f"\n  {quality} teams:")
                print(f"    Year 1-2: {yr1_correct[0]:.1%} correct")
                print(f"    Year 5+: {yr5_correct[0]:.1%} correct")
                print(f"    Improvement: {improvement:+.1f} pp")

    return results_df


# =============================================================================
# TEST 10: MENTOR → PROTÉGÉ STYLE TRANSFER
# =============================================================================

# Detailed mentor-protégé relationships
MENTOR_PROTEGE = {
    # Belichick tree
    'Brian Flores': 'Bill Belichick',
    'Matt Patricia': 'Bill Belichick',
    'Joe Judge': 'Bill Belichick',
    'Josh McDaniels': 'Bill Belichick',
    'Mike Vrabel': 'Bill Belichick',
    'Jerod Mayo': 'Bill Belichick',

    # Reid tree
    'Doug Pederson': 'Andy Reid',
    'Matt Nagy': 'Andy Reid',
    'Ron Rivera': 'Andy Reid',  # Was with him in SD
    'Sean McDermott': 'Andy Reid',

    # Shanahan tree
    'Sean McVay': 'Kyle Shanahan',  # Worked together in WAS
    'Matt LaFleur': 'Kyle Shanahan',
    'Zac Taylor': 'Sean McVay',
    'Kevin OConnell': 'Sean McVay',
    'Mike McDaniel': 'Kyle Shanahan',

    # Payton tree
    'Dennis Allen': 'Sean Payton',
    'Dan Campbell': 'Sean Payton',
}


def test_mentor_protege_transfer(decisions):
    """
    Test 10: When assistants become HCs, do they adopt their mentor's style?

    Compare protégé behavior to mentor behavior (not just tree averages).
    """
    print("\n" + "=" * 80)
    print("TEST 10: MENTOR → PROTÉGÉ STYLE TRANSFER")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Get coach-level aggregates
    coach_stats = decisions.groupby('coach').agg(
        n_decisions=('coach_correct', 'count'),
        pct_correct=('coach_correct', 'mean'),
        two_pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    coach_stats = coach_stats[coach_stats['n_decisions'] >= 30]

    # Build comparison table
    results = []

    print("\nMentor → Protégé Comparison:")
    print("-" * 100)
    print(f"{'Protégé':<20} | {'Mentor':<20} | {'Protégé 2pt':>12} | {'Mentor 2pt':>12} | {'Protégé Correct':>15} | {'Mentor Correct':>14}")
    print("-" * 100)

    for protege, mentor in MENTOR_PROTEGE.items():
        protege_data = coach_stats[coach_stats['coach'] == protege]
        mentor_data = coach_stats[coach_stats['coach'] == mentor]

        if len(protege_data) == 0 or len(mentor_data) == 0:
            continue

        p_2pt = protege_data['two_pt_rate'].values[0]
        m_2pt = mentor_data['two_pt_rate'].values[0]
        p_correct = protege_data['pct_correct'].values[0]
        m_correct = mentor_data['pct_correct'].values[0]

        results.append({
            'protege': protege,
            'mentor': mentor,
            'protege_2pt_rate': p_2pt,
            'mentor_2pt_rate': m_2pt,
            'protege_correct': p_correct,
            'mentor_correct': m_correct,
            '2pt_diff': p_2pt - m_2pt,
            'correct_diff': p_correct - m_correct
        })

        print(f"{protege:<20} | {mentor:<20} | {p_2pt:>12.1%} | {m_2pt:>12.1%} | "
              f"{p_correct:>15.1%} | {m_correct:>14.1%}")

    if not results:
        print("  No mentor-protégé pairs with sufficient data")
        return None

    results_df = pd.DataFrame(results)

    # Correlation analysis
    print("\n" + "-" * 60)
    print("Correlation: Mentor → Protégé Style Transfer")
    print("-" * 60)

    if len(results_df) >= 5:
        corr_2pt = results_df['protege_2pt_rate'].corr(results_df['mentor_2pt_rate'])
        corr_correct = results_df['protege_correct'].corr(results_df['mentor_correct'])

        print(f"\n  N pairs: {len(results_df)}")
        print(f"  Correlation (2pt rate): {corr_2pt:.3f}")
        print(f"  Correlation (correct rate): {corr_correct:.3f}")

        # Test significance
        if len(results_df) >= 5:
            from scipy.stats import pearsonr
            r, p = pearsonr(results_df['protege_2pt_rate'], results_df['mentor_2pt_rate'])
            print(f"  p-value (2pt rate): {p:.4f}")

            if p < 0.05 and corr_2pt > 0:
                print("\n  => SIGNIFICANT: Protégés adopt mentor's aggressiveness style")
            elif corr_2pt < 0:
                print("\n  => Protégés DIVERGE from mentor's style (!)")
            else:
                print("\n  => No significant mentor-protégé correlation")

    # Do protégés outperform mentors?
    print("\n" + "-" * 60)
    print("Do Protégés Improve on Mentor's Performance?")
    print("-" * 60)

    mean_diff = results_df['correct_diff'].mean()
    print(f"\n  Mean improvement (protégé - mentor): {mean_diff*100:+.1f} pp")

    improved = (results_df['correct_diff'] > 0).sum()
    total = len(results_df)
    print(f"  Protégés outperforming mentors: {improved}/{total} ({improved/total:.0%})")

    return results_df


# =============================================================================
# TEST 11: CAUSAL FOREST (if sklearn available)
# =============================================================================

def test_causal_forest(decisions):
    """
    Test 11: Use ML to discover which coach characteristics predict learning.

    Instead of testing our hypotheses, let the data tell us what matters.
    """
    print("\n" + "=" * 80)
    print("TEST 11: FEATURE IMPORTANCE FOR DECISION QUALITY (Random Forest)")
    print("=" * 80)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("  sklearn not available, skipping causal forest")
        return None

    decisions = add_coach_info(decisions)

    # Create features
    df = decisions.copy()
    df['went_for_2'] = (df['actual_decision'] == 'two_point').astype(int)
    df['time_minutes'] = df['time_remaining'] / 60

    # Encode categorical variables
    df['is_pioneer'] = df['posteam'].isin(ANALYTICS_PIONEERS).astype(int)

    # Coaching tree dummies
    for tree in ['Belichick', 'Reid', 'Shanahan/McVay', 'Payton']:
        df[f'tree_{tree}'] = (df['coaching_tree'] == tree).astype(int)

    # Feature matrix
    feature_cols = [
        'tenure', 'score_diff_pre_td', 'time_minutes',
        'is_pioneer', 'tree_Belichick', 'tree_Reid',
        'tree_Shanahan/McVay', 'tree_Payton'
    ]

    df = df.dropna(subset=feature_cols + ['coach_correct'])

    X = df[feature_cols]
    y = df['coach_correct'].astype(int)

    print(f"\nTraining Random Forest on {len(X):,} decisions...")
    print(f"Features: {feature_cols}")

    # Train model
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Cross-validation accuracy
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"Baseline (always predict majority): {y.mean():.3f}")

    # Feature importance
    print("\n" + "-" * 60)
    print("Feature Importance (MDI):")
    print("-" * 60)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.3f} {bar}")

    # Key insight
    print("\n" + "-" * 60)
    print("KEY INSIGHT:")
    print("-" * 60)

    top_feature = importance.iloc[0]['feature']
    top_importance = importance.iloc[0]['importance']

    print(f"\n  Most important predictor: {top_feature} ({top_importance:.3f})")

    if top_feature == 'tenure':
        print("  => Experience is the dominant factor in decision quality")
    elif top_feature == 'score_diff_pre_td':
        print("  => Game state dominates (expected - some situations are obvious)")
    elif top_feature == 'time_minutes':
        print("  => Time remaining drives decision quality")
    elif 'tree_' in top_feature:
        tree_name = top_feature.replace('tree_', '')
        print(f"  => Being in the {tree_name} tree matters for decision quality")
    elif top_feature == 'is_pioneer':
        print("  => Analytics culture (pioneer team) matters")

    return importance


# =============================================================================
# TEST 12: STAGGERED ANALYTICS ADOPTION
# =============================================================================

def test_staggered_analytics(decisions, pbp):
    """
    Test 12: Approximate staggered analytics adoption using 4th down behavior.

    Teams that were aggressive on 4th downs earlier = early analytics adopters.
    """
    print("\n" + "=" * 80)
    print("TEST 12: STAGGERED ANALYTICS ADOPTION")
    print("=" * 80)

    decisions = add_coach_info(decisions)

    # Compute 4th down aggressiveness by team-season
    fourth_downs = pbp[
        (pbp['down'] == 4) &
        (pbp['play_type'].isin(['run', 'pass', 'punt', 'field_goal'])) &
        (pbp['season'] >= 2015)
    ].copy()

    fourth_downs['went_for_it'] = fourth_downs['play_type'].isin(['run', 'pass']).astype(int)

    team_4th = fourth_downs.groupby(['posteam', 'season']).agg(
        go_rate=('went_for_it', 'mean'),
        n_4th=('went_for_it', 'count')
    ).reset_index()

    # Identify "early adopters" - teams that were aggressive in 2015-2017
    early_period = team_4th[team_4th['season'].between(2015, 2017)]
    early_avg = early_period.groupby('posteam')['go_rate'].mean().reset_index()
    early_avg.columns = ['posteam', 'early_4th_aggression']

    # Top quartile = early adopters
    threshold = early_avg['early_4th_aggression'].quantile(0.75)
    early_adopters = early_avg[early_avg['early_4th_aggression'] >= threshold]['posteam'].tolist()

    print(f"\nEarly Analytics Adopters (top 25% in 4th down aggression 2015-2017):")
    print(f"  {', '.join(early_adopters)}")
    print(f"  Threshold: {threshold:.1%} 4th down go rate")

    # Compare learning curves
    decisions['early_adopter'] = decisions['posteam'].isin(early_adopters)

    print("\n" + "-" * 60)
    print("Learning Curves: Early Adopters vs Others")
    print("-" * 60)
    print(f"{'Tenure':<12} | {'Early Adopters':>20} | {'Others':>20} | {'Diff':>10}")
    print("-" * 60)

    results = []
    for tenure in range(1, 8):
        early = decisions[(decisions['early_adopter']) & (decisions['tenure'] == tenure)]
        other = decisions[(~decisions['early_adopter']) & (decisions['tenure'] == tenure)]

        if len(early) >= 30 and len(other) >= 50:
            early_correct = early['coach_correct'].mean()
            other_correct = other['coach_correct'].mean()
            diff = early_correct - other_correct

            results.append({
                'tenure': tenure,
                'early_correct': early_correct,
                'other_correct': other_correct,
                'diff': diff
            })

            print(f"Year {tenure:<6} | {early_correct:>20.1%} | {other_correct:>20.1%} | {diff*100:>+10.1f} pp")

    if results:
        results_df = pd.DataFrame(results)

        # Average difference
        avg_diff = results_df['diff'].mean()
        print(f"\n  Average difference: {avg_diff*100:+.1f} pp")

        if avg_diff > 0.02:
            print("  => Early adopters have consistently better decision quality")
        elif avg_diff < -0.02:
            print("  => Surprisingly, late adopters have better decision quality")
        else:
            print("  => No consistent difference between early and late adopters")

        return results_df

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all deep learning mechanism tests."""
    print("=" * 80)
    print("DEEP DIVE: LEARNING MECHANISMS - FULL BATTERY")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    pbp, decisions = load_data()
    print(f"Loaded {len(pbp):,} plays and {len(decisions):,} 2pt decisions")

    # Run all tests (original 8)
    within_game = test_within_game_updating(pbp)
    context_sensitivity = test_context_sensitivity(decisions)
    coach_stats, tree_stats = test_coaching_trees(decisions)
    analytics = test_analytics_environment(decisions, pbp)
    regret = test_salient_event_learning(pbp, decisions)
    playoff = test_playoff_loss_salience(pbp, decisions)
    security = test_job_security(decisions)
    decomp = test_learning_decomposition(decisions)

    # Run additional tests (9-12)
    tenure_record = test_tenure_team_record(decisions, pbp)
    mentor_protege = test_mentor_protege_transfer(decisions)
    feature_importance = test_causal_forest(decisions)
    staggered = test_staggered_analytics(decisions, pbp)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: DEEP LEARNING MECHANISMS - FULL BATTERY (12 TESTS)")
    print("=" * 80)
    print("""
Key findings from the full battery of tests:

TIER 1 - CLEANEST TESTS:
1. WITHIN-GAME UPDATING: [see results above]
2. CONTEXT-SENSITIVITY BY EXPERIENCE: [see results above]
3. COACHING TREES: [see results above]

TIER 2 - REQUIRES SOME INFERENCE:
4. ANALYTICS ENVIRONMENT: [see results above]
5. REGRET LEARNING: [see results above]
6. PLAYOFF SALIENCE: [see results above]
7. JOB SECURITY: [see results above]
8. NETWORK VS EXPERIENTIAL DECOMPOSITION: [see results above]

ADDITIONAL TESTS:
9. TENURE × TEAM RECORD INTERACTION: [see results above]
10. MENTOR → PROTÉGÉ STYLE TRANSFER: [see results above]
11. FEATURE IMPORTANCE (Random Forest): [see results above]
12. STAGGERED ANALYTICS ADOPTION: [see results above]
""")


if __name__ == "__main__":
    main()
