"""
Check if team is correlated with being down 2 vs 3 after a TD.

Professor's suggestion: see if certain teams systematically end up in
particular deficit situations more often, which could indicate confounding.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path

def analyze_team_deficit_correlation():
    """Analyze whether teams are correlated with specific deficits."""

    # Load the data
    data_path = Path(__file__).parent.parent / 'data' / 'all_pbp_1999_2024.parquet'
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Filter to post-2015 (post PAT rule change)
    df = df[df['season'] >= 2015].copy()
    print(f"Post-2015 plays: {len(df):,}")

    # Get situations right after a TD where PAT/2pt decision happens
    # These are extra_point_attempt or two_point_attempt plays
    conversion_plays = df[(df['extra_point_attempt'] == 1) | (df['two_point_attempt'] == 1)].copy()
    print(f"Conversion attempts (PAT or 2pt): {len(conversion_plays):,}")

    # Calculate score differential BEFORE the conversion (after TD, before PAT/2pt)
    # At this point, the TD has been scored but not the conversion
    # posteam_score includes the TD (6 pts), so score_diff = posteam_score - defteam_score
    conversion_plays['score_diff'] = conversion_plays['posteam_score'] - conversion_plays['defteam_score']

    print("\n" + "="*60)
    print("DISTRIBUTION OF SCORE DIFFERENTIALS (After TD, Before Conversion)")
    print("="*60)
    print(conversion_plays['score_diff'].value_counts().sort_index().head(20))

    # Focus on the interesting deficits for the Down 9 paradox
    # Down 9 before TD -> Down 3 after TD (score_diff = -3)
    # Down 8 before TD -> Down 2 after TD (score_diff = -2)
    # Down 7 before TD -> Down 1 after TD (score_diff = -1)
    interesting_deficits = [-3, -2, -1]
    interesting = conversion_plays[conversion_plays['score_diff'].isin(interesting_deficits)].copy()

    print(f"\nPlays with score diff in {interesting_deficits}: {len(interesting):,}")
    print("\nBreakdown:")
    for diff in interesting_deficits:
        n = (interesting['score_diff'] == diff).sum()
        print(f"  Down {abs(diff)}: {n:,} plays")

    # Create team x deficit contingency table
    print("\n" + "="*60)
    print("TEAM x DEFICIT CONTINGENCY TABLE")
    print("="*60)

    team_deficit = pd.crosstab(interesting['posteam'], interesting['score_diff'])
    team_deficit.columns = [f'Down_{abs(c)}' for c in team_deficit.columns]
    print(team_deficit.to_string())

    # Chi-square test for independence
    print("\n" + "="*60)
    print("CHI-SQUARE TEST FOR INDEPENDENCE")
    print("="*60)

    chi2, p, dof, expected = chi2_contingency(team_deficit)
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print("\n*** SIGNIFICANT: Team IS correlated with deficit type (p < 0.05) ***")
    else:
        print("\n*** NOT SIGNIFICANT: Team is NOT correlated with deficit type (p >= 0.05) ***")

    # Calculate residuals to see which teams deviate most
    print("\n" + "="*60)
    print("STANDARDIZED RESIDUALS (Observed - Expected) / sqrt(Expected)")
    print("="*60)

    residuals = (team_deficit.values - expected) / np.sqrt(expected)
    residuals_df = pd.DataFrame(residuals, index=team_deficit.index, columns=team_deficit.columns)

    # Find biggest deviations
    print("\nTeams with largest deviations from expected:")
    for col in residuals_df.columns:
        print(f"\n{col}:")
        sorted_res = residuals_df[col].sort_values()
        print(f"  Under-represented: {sorted_res.head(3).to_dict()}")
        print(f"  Over-represented: {sorted_res.tail(3).to_dict()}")

    # Calculate proportions
    print("\n" + "="*60)
    print("PROPORTION OF EACH DEFICIT BY TEAM")
    print("="*60)

    team_props = team_deficit.div(team_deficit.sum(axis=1), axis=0)
    overall_props = team_deficit.sum() / team_deficit.sum().sum()

    print(f"\nOverall league proportions:")
    for col in team_props.columns:
        print(f"  {col}: {overall_props[col]:.1%}")

    print("\nTeams with most extreme proportions vs league average:")
    for col in team_props.columns:
        diff_from_avg = team_props[col] - overall_props[col]
        print(f"\n{col}:")
        print(f"  Lowest (under-represented):  {diff_from_avg.nsmallest(3).to_dict()}")
        print(f"  Highest (over-represented): {diff_from_avg.nlargest(3).to_dict()}")

    # Effect size: Cramer's V
    n = team_deficit.sum().sum()
    min_dim = min(team_deficit.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    print("\n" + "="*60)
    print("EFFECT SIZE")
    print("="*60)
    print(f"Cramer's V: {cramers_v:.4f}")
    print("(0.1 = small, 0.3 = medium, 0.5 = large)")

    if cramers_v < 0.1:
        print("Effect size: NEGLIGIBLE")
    elif cramers_v < 0.3:
        print("Effect size: SMALL")
    elif cramers_v < 0.5:
        print("Effect size: MEDIUM")
    else:
        print("Effect size: LARGE")

    return team_deficit, chi2, p, cramers_v


if __name__ == "__main__":
    analyze_team_deficit_correlation()
