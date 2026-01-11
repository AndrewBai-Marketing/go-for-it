"""
Decision Analysis for Stolen Base Attempts

Computes break-even success probabilities and evaluates whether teams
make optimal steal decisions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import pickle
from success_model import StealSuccessModel


def compute_break_even(wp_current: float, wp_success: float, wp_fail: float) -> float:
    """
    Compute break-even success probability for a steal to be +WP.

    A steal is +WP if:
        π * WP_success + (1-π) * WP_fail > WP_current

    Solving for break-even π*:
        π* = (WP_current - WP_fail) / (WP_success - WP_fail)
    """
    denom = wp_success - wp_fail

    if denom <= 0:
        return 1.0  # Never profitable to steal

    pi_star = (wp_current - wp_fail) / denom

    # Clamp to [0, 1]
    return np.clip(pi_star, 0.0, 1.0)


def compute_break_even_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Compute break-even for all rows using RE24-based values.

    Standard break-even from run expectancy tables:
    - Stealing 2nd, 0 outs: ~72%
    - Stealing 2nd, 1 out: ~73%
    - Stealing 3rd, 0 outs: ~75%
    - Stealing 3rd, 1 out: ~72%
    """
    df = df.copy()

    # Default break-even based on outs and base being stolen
    # These are derived from standard RE24 tables
    pi_star = pd.Series(0.72, index=df.index)  # Default

    # Adjust for outs
    mask_0_outs = df['outs_when_up'] == 0
    mask_1_out = df['outs_when_up'] == 1

    # Adjust for base being stolen (when available)
    if 'steal_base' in df.columns:
        mask_3b = df['steal_base'] == '3B'
        # Stealing 3rd with 0 outs is harder (need higher success rate)
        pi_star.loc[mask_0_outs & mask_3b] = 0.75
        # Stealing 2nd with 0 outs
        pi_star.loc[mask_0_outs & ~mask_3b] = 0.715
        # Stealing with 1 out
        pi_star.loc[mask_1_out & mask_3b] = 0.72
        pi_star.loc[mask_1_out & ~mask_3b] = 0.726
    else:
        # Without steal_base info, use runner_on_1b/2b
        if 'runner_on_1b' in df.columns:
            mask_from_1st = df['runner_on_1b'] == True
            pi_star.loc[mask_0_outs & mask_from_1st] = 0.715
            pi_star.loc[mask_1_out & mask_from_1st] = 0.726
        if 'runner_on_2b' in df.columns:
            mask_from_2nd = df['runner_on_2b'] == True
            pi_star.loc[mask_0_outs & mask_from_2nd] = 0.75
            pi_star.loc[mask_1_out & mask_from_2nd] = 0.72

    return pi_star


def compute_run_expectancy_break_even(outs: int, base_from: str, base_to: str,
                                      re_matrix: Optional[Dict] = None) -> float:
    """
    Compute break-even using run expectancy (RE24) approach.

    Standard break-even formula from run expectancy:
    π* = (RE_current - RE_fail) / (RE_success - RE_fail)

    For typical steal of 2nd with runner on 1st:
    - RE_success: runner on 2nd, same outs
    - RE_fail: bases empty, outs + 1
    """
    # Default RE24 matrix (approximate values)
    if re_matrix is None:
        re_matrix = {
            # (runners_string, outs): run expectancy
            ('_', 0): 0.481, ('_', 1): 0.254, ('_', 2): 0.098,
            ('1', 0): 0.859, ('1', 1): 0.509, ('1', 2): 0.224,
            ('2', 0): 1.100, ('2', 1): 0.664, ('2', 2): 0.319,
            ('1-2', 0): 1.437, ('1-2', 1): 0.884, ('1-2', 2): 0.429,
            ('3', 0): 1.359, ('3', 1): 0.983, ('3', 2): 0.353,
            ('1-3', 0): 1.798, ('1-3', 1): 1.211, ('1-3', 2): 0.478,
            ('2-3', 0): 1.964, ('2-3', 1): 1.376, ('2-3', 2): 0.580,
            ('1-2-3', 0): 2.292, ('1-2-3', 1): 1.541, ('1-2-3', 2): 0.752,
        }

    # Map steal attempt to state transitions
    if base_from == '1' and base_to == '2':
        # Stealing 2nd with runner on 1st only
        re_current = re_matrix.get(('1', outs), 0.5)
        re_success = re_matrix.get(('2', outs), 0.7)
        re_fail = re_matrix.get(('_', min(outs + 1, 2)), 0.2)
    elif base_from == '2' and base_to == '3':
        # Stealing 3rd with runner on 2nd only
        re_current = re_matrix.get(('2', outs), 0.7)
        re_success = re_matrix.get(('3', outs), 1.0)
        re_fail = re_matrix.get(('_', min(outs + 1, 2)), 0.2)
    else:
        # Default
        return 0.75

    denom = re_success - re_fail
    if denom <= 0:
        return 1.0

    return (re_current - re_fail) / denom


def evaluate_decision(row: pd.Series, model: StealSuccessModel,
                      n_samples: int = 1000) -> Dict:
    """
    Evaluate whether the steal decision was optimal.

    Returns probability that stealing was optimal given the data.
    """
    # Get break-even for this state
    pi_star = row['break_even']

    # Get predicted success probability
    pi_hat = row['predicted_success_prob']

    # Simple decision rule (using point estimate)
    steal_is_optimal = pi_hat > pi_star

    # More sophisticated: account for parameter uncertainty
    # For now, use a simple threshold
    confidence = abs(pi_hat - pi_star)  # How far from break-even

    # Actual decision
    actual_decision = row['steal_attempt']

    # Was decision correct?
    if actual_decision:
        # Attempted steal - was it optimal?
        decision_correct = steal_is_optimal
        wp_impact = (pi_hat - pi_star) * (row['wp_success'] - row['wp_fail'])
    else:
        # No steal - was staying correct?
        decision_correct = not steal_is_optimal
        if steal_is_optimal:
            # Missed opportunity
            wp_impact = -(pi_hat - pi_star) * (row['wp_success'] - row['wp_fail'])
        else:
            wp_impact = 0

    return {
        'break_even': pi_star,
        'predicted_success_prob': pi_hat,
        'steal_is_optimal': steal_is_optimal,
        'actual_decision': actual_decision,
        'decision_correct': decision_correct,
        'confidence': confidence,
        'wp_impact': wp_impact
    }


def analyze_decisions(df: pd.DataFrame, model: StealSuccessModel) -> pd.DataFrame:
    """
    Analyze all steal decisions in the dataset.

    KEY APPROACH:
    - For ATTEMPTS: Evaluate whether the predicted success prob > break-even
    - For NON-ATTEMPTS: Due to selection bias, we cannot reliably evaluate
      whether they "should have" stolen. We mark these as "correctly not stolen"
      unless strong evidence suggests otherwise.

    The analysis focuses on:
    1. Whether actual attempts were above break-even (good vs bad steals)
    2. The observed success rate vs required break-even
    3. Trend analysis of steal rates and success rates over time
    """
    print("Analyzing decisions...")

    df = df.copy()

    # Compute break-even probabilities
    print("  Computing break-even probabilities...")
    df['break_even'] = compute_break_even_vectorized(df)

    # Get predictions from model for all rows
    print("  Predicting success probabilities...")
    df['predicted_success_prob'] = model.predict_proba(df)

    # For evaluation, use model prediction for attempts, 0 for non-attempts
    # (We can't evaluate counterfactual success for non-attempts)
    df['eval_success_prob'] = np.where(
        df['steal_attempt'],
        df['predicted_success_prob'],
        0.0  # Non-attempts: not evaluable, assume correctly not stolen
    )

    # An attempt is "optimal" if predicted success > break-even
    # A non-attempt is considered "optimal" (can't prove otherwise due to selection bias)
    df['steal_is_optimal'] = np.where(
        df['steal_attempt'],
        df['predicted_success_prob'] > df['break_even'],
        True  # Non-attempts assumed correct due to selection bias
    )

    # Evaluate decisions
    df['decision_correct'] = (
            (df['steal_attempt'] & df['steal_is_optimal']) |
            (~df['steal_attempt'] & ~df['steal_is_optimal'])
    )

    # Compute WP impact
    # For attempts: actual outcome
    # For non-attempts: what was left on the table (if should have stolen)
    df['wp_impact'] = 0.0

    # Correct attempts (stole when should have)
    mask_correct_attempt = df['steal_attempt'] & df['steal_is_optimal']
    df.loc[mask_correct_attempt, 'wp_impact'] = (
            df.loc[mask_correct_attempt, 'eval_success_prob'] -
            df.loc[mask_correct_attempt, 'break_even']
    ) * (df.loc[mask_correct_attempt, 'wp_success'] - df.loc[mask_correct_attempt, 'wp_fail'])

    # Incorrect attempts (stole when shouldn't have)
    mask_incorrect_attempt = df['steal_attempt'] & ~df['steal_is_optimal']
    df.loc[mask_incorrect_attempt, 'wp_impact'] = (
            df.loc[mask_incorrect_attempt, 'eval_success_prob'] -
            df.loc[mask_incorrect_attempt, 'break_even']
    ) * (df.loc[mask_incorrect_attempt, 'wp_success'] - df.loc[mask_incorrect_attempt, 'wp_fail'])

    # Missed opportunities (didn't steal when should have)
    mask_missed = ~df['steal_attempt'] & df['steal_is_optimal']
    df.loc[mask_missed, 'wp_impact'] = -(
            df.loc[mask_missed, 'eval_success_prob'] -
            df.loc[mask_missed, 'break_even']
    ) * (df.loc[mask_missed, 'wp_success'] - df.loc[mask_missed, 'wp_fail'])

    # Focus summary on actual attempts
    attempts = df[df['steal_attempt']]
    n_attempts = len(attempts)
    n_above_break_even = (attempts['predicted_success_prob'] > attempts['break_even']).sum()
    actual_success_rate = attempts['steal_success'].mean()
    avg_break_even = attempts['break_even'].mean()

    print(f"\nDecision Analysis Summary:")
    print(f"  Total opportunities: {len(df):,}")
    print(f"  Steal attempts: {n_attempts:,} ({df['steal_attempt'].mean():.1%})")
    print(f"  Attempts above break-even: {n_above_break_even:,} ({n_above_break_even/n_attempts:.1%})")
    print(f"  Actual success rate: {actual_success_rate:.1%}")
    print(f"  Avg break-even for attempts: {avg_break_even:.1%}")
    print(f"  Success rate margin: {(actual_success_rate - avg_break_even)*100:+.1f} pp")

    return df


def compute_break_even_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average break-even probability by game state.
    """
    results = []

    for outs in [0, 1]:
        for base_state in ['1', '2', '1-2']:
            for inning_bucket in ['early', 'middle', 'late']:
                for score_bucket in ['losing', 'close', 'winning']:
                    # Filter data
                    mask = (
                            (df['outs_when_up'] == outs) &
                            (df['base_state'] == base_state)
                    )

                    if inning_bucket == 'early':
                        mask &= df['inning'] <= 3
                    elif inning_bucket == 'middle':
                        mask &= (df['inning'] > 3) & (df['inning'] <= 6)
                    else:
                        mask &= df['inning'] > 6

                    if score_bucket == 'losing':
                        mask &= df['score_diff'] < -2
                    elif score_bucket == 'close':
                        mask &= (df['score_diff'] >= -2) & (df['score_diff'] <= 2)
                    else:
                        mask &= df['score_diff'] > 2

                    subset = df[mask]

                    if len(subset) < 100:
                        continue

                    results.append({
                        'outs': outs,
                        'base_state': base_state,
                        'inning': inning_bucket,
                        'score': score_bucket,
                        'n_opportunities': len(subset),
                        'break_even_mean': subset['break_even'].mean(),
                        'break_even_std': subset['break_even'].std(),
                        'actual_attempt_rate': subset['steal_attempt'].mean(),
                        'optimal_attempt_rate': subset['steal_is_optimal'].mean(),
                        'gap': subset['steal_is_optimal'].mean() - subset['steal_attempt'].mean()
                    })

    return pd.DataFrame(results)


def analyze_by_era(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze steal decisions by era.

    Focus on:
    - Attempt rate (how often teams try to steal)
    - Success rate (how often attempts succeed)
    - Success margin (actual success rate - break-even)
    - Quality of attempts (% above break-even threshold)
    """
    eras = {
        'pre_statcast': (2008, 2014),
        'early_statcast': (2015, 2018),
        'late_statcast': (2019, 2022),
        'new_rules': (2023, 2024)
    }

    results = []

    for era_name, (start, end) in eras.items():
        era_data = df[(df['season'] >= start) & (df['season'] <= end)]

        if len(era_data) == 0:
            continue

        attempts = era_data[era_data['steal_attempt']]
        n_attempts = len(attempts)

        if n_attempts == 0:
            continue

        # Attempt quality metrics
        n_above_be = (attempts['predicted_success_prob'] > attempts['break_even']).sum()
        actual_success = attempts['steal_success'].mean()
        avg_be = attempts['break_even'].mean()

        results.append({
            'era': era_name,
            'years': f"{start}-{end}",
            'n_opportunities': len(era_data),
            'n_attempts': n_attempts,
            'attempt_rate': n_attempts / len(era_data),
            'success_rate': actual_success,
            'avg_break_even': avg_be,
            'success_margin': actual_success - avg_be,
            'pct_above_break_even': n_above_be / n_attempts,
            'wp_per_attempt': attempts['wp_impact'].mean()
        })

    return pd.DataFrame(results)


def analyze_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze steal decisions by year for trend analysis.
    """
    results = []

    for year in sorted(df['season'].unique()):
        year_data = df[df['season'] == year]

        if len(year_data) == 0:
            continue

        attempts = year_data[year_data['steal_attempt']]
        n_opps = len(year_data)
        n_attempts = len(attempts)

        if n_attempts == 0:
            continue

        # Quality metrics
        n_above_be = (attempts['predicted_success_prob'] > attempts['break_even']).sum()
        actual_success = attempts['steal_success'].mean()
        avg_be = attempts['break_even'].mean()

        results.append({
            'season': year,
            'n_opportunities': n_opps,
            'n_attempts': n_attempts,
            'attempt_rate': n_attempts / n_opps,
            'success_rate': actual_success,
            'avg_break_even': avg_be,
            'success_margin': actual_success - avg_be,
            'pct_above_break_even': n_above_be / n_attempts
        })

    return pd.DataFrame(results)


def main():
    """Run full decision analysis."""
    data_dir = Path('stolen_base_analysis/data/processed')
    model_dir = Path('stolen_base_analysis/models')
    output_dir = Path('stolen_base_analysis/outputs')

    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_dir / 'steal_opportunities.parquet')
    print(f"Loaded {len(df):,} steal opportunities")

    # Load or fit model
    model_path = model_dir / 'success_model.pkl'
    if model_path.exists():
        print("Loading trained model...")
        model = StealSuccessModel.load(model_path)
    else:
        print("Model not found. Fitting new model...")
        model = StealSuccessModel()
        model.fit(df)
        model.save(model_path)

    # Analyze decisions
    df = analyze_decisions(df, model)

    # Save full results
    df.to_parquet(output_dir / 'results' / 'opportunity_level_results.parquet')
    print(f"\nSaved opportunity-level results")

    # Break-even by state
    print("\nComputing break-even by game state...")
    break_even_df = compute_break_even_by_state(df)
    break_even_df.to_csv(output_dir / 'tables' / 'break_even_by_state.csv', index=False)
    print(break_even_df.to_string(index=False))

    # Era analysis
    print("\n" + "=" * 60)
    print("ANALYSIS BY ERA")
    print("=" * 60)
    era_df = analyze_by_era(df)
    era_df.to_csv(output_dir / 'tables' / 'decision_analysis_by_era.csv', index=False)
    print(era_df.to_string(index=False))

    # Year analysis
    print("\n" + "=" * 60)
    print("ANALYSIS BY YEAR")
    print("=" * 60)
    year_df = analyze_by_year(df)
    year_df.to_csv(output_dir / 'tables' / 'decision_analysis_by_year.csv', index=False)
    print(year_df.to_string(index=False))

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    attempts = df[df['steal_attempt']]
    overall_attempt_rate = df['steal_attempt'].mean()
    success_rate = attempts['steal_success'].mean()
    avg_break_even = attempts['break_even'].mean()
    success_margin = success_rate - avg_break_even
    n_above_be = (attempts['predicted_success_prob'] > attempts['break_even']).sum()
    pct_above_be = n_above_be / len(attempts)

    print(f"""
1. Steal attempt rate: {overall_attempt_rate:.2%} ({len(attempts):,} attempts)
2. Observed success rate: {success_rate:.1%}
3. Average break-even requirement: {avg_break_even:.1%}
4. Success margin: {success_margin*100:+.1f} pp (success rate - break-even)
5. Attempts above break-even: {pct_above_be:.1%}

Interpretation:
- Teams succeed at {success_rate:.1%}, which is {abs(success_margin)*100:.1f} pp
  {'above' if success_margin > 0 else 'below'} the ~{avg_break_even:.1%} break-even threshold
- {pct_above_be:.1%} of steal attempts have predicted success > break-even
- This suggests teams are {'appropriately selective' if success_margin > 0 else 'over-aggressive'}
  when choosing when to steal
""")

    return df, model


if __name__ == "__main__":
    df, model = main()
