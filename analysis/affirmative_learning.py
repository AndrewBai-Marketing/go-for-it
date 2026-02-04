"""
Affirmative Learning Mechanisms for 2-Point Conversion Decisions

We have strong null results on standard learning channels (no social learning,
no own-experience learning). But aggregate improvement IS happening.

This analysis digs deeper to find what's actually driving learning:
1. High-stakes success effects
2. Within-game learning
3. Coaching tree/network effects
4. Permission structure (Pederson Super Bowl effect)
5. Counterfactual regret learning
6. Generational shift
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import coach mapping from learning_mechanisms
from learning_mechanisms import COACH_MAPPING

# Coaching trees - who trained under whom
COACHING_TREES = {
    'Reid Tree': ['Doug Pederson', 'Matt Nagy', 'Ron Rivera'],  # Reid disciples
    'Belichick Tree': ['Bill OBrien', 'Matt Patricia', 'Joe Judge', 'Jerod Mayo', 'Mike Vrabel'],
    'McVay/Shanahan Tree': ['Matt LaFleur', 'Zac Taylor', 'Kevin OConnell'],
    'Payton Tree': ['Dennis Allen', 'Dan Campbell'],
}

# Coach backgrounds (offensive vs defensive)
COACH_BACKGROUND = {
    # Offensive background
    'Andy Reid': 'offensive', 'Doug Pederson': 'offensive', 'Sean Payton': 'offensive',
    'Kyle Shanahan': 'offensive', 'Sean McVay': 'offensive', 'Matt LaFleur': 'offensive',
    'Kevin OConnell': 'offensive', 'Zac Taylor': 'offensive', 'Mike McCarthy': 'offensive',
    'Matt Nagy': 'offensive', 'Kliff Kingsbury': 'offensive', 'Kevin Stefanski': 'offensive',
    'Arthur Smith': 'offensive', 'Frank Reich': 'offensive', 'Josh McDaniels': 'offensive',
    'Brian Daboll': 'offensive', 'Todd Bowles': 'defensive',
    # Defensive background
    'Bill Belichick': 'defensive', 'Mike Tomlin': 'defensive', 'John Harbaugh': 'special_teams',
    'Ron Rivera': 'defensive', 'Dan Quinn': 'defensive', 'Matt Patricia': 'defensive',
    'Vic Fangio': 'defensive', 'Robert Saleh': 'defensive', 'Brian Flores': 'defensive',
    'DeMeco Ryans': 'defensive', 'Dennis Allen': 'defensive', 'Lovie Smith': 'defensive',
    'Mike Zimmer': 'defensive', 'Dan Campbell': 'offensive',  # Former TE
    'Nick Sirianni': 'offensive', 'Mike McDaniel': 'offensive',
}


def load_data():
    """Load all required data."""
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'

    pbp = pd.read_parquet(data_dir / 'all_pbp_1999_2024.parquet')
    decisions = pd.read_parquet(output_dir / 'two_point_decision_analysis.parquet')

    return pbp, decisions


def analyze_high_stakes_success(pbp, decisions):
    """
    Test 1: Does HIGH-STAKES SUCCESS have a bigger effect than routine success?

    High-stakes = game-winning, game-tying, or high WP impact 2pt conversions
    """
    print("\n" + "=" * 80)
    print("TEST 1: HIGH-STAKES SUCCESS EFFECTS")
    print("=" * 80)

    # Get all 2pt attempts
    two_pt = pbp[pbp['two_point_attempt'] == 1].copy()
    two_pt = two_pt[two_pt['season'] >= 2015]
    two_pt['success'] = (two_pt['two_point_conv_result'] == 'success').astype(int)
    two_pt['score_diff_pre_td'] = two_pt['score_differential'] - 6

    # Define high-stakes: game-winning or game-tying in final 5 minutes
    two_pt['late_game'] = two_pt['game_seconds_remaining'] <= 300
    two_pt['game_tying'] = (two_pt['score_diff_pre_td'] == -8) | (two_pt['score_diff_pre_td'] == -2)
    two_pt['game_winning'] = two_pt['score_diff_pre_td'].isin([-2, -5])  # Would put team ahead
    two_pt['high_stakes'] = two_pt['late_game'] & (two_pt['game_tying'] | two_pt['game_winning'])

    print(f"\nTotal 2pt attempts (2015-2024): {len(two_pt)}")
    print(f"High-stakes attempts: {two_pt['high_stakes'].sum()}")
    print(f"High-stakes success rate: {two_pt[two_pt['high_stakes']]['success'].mean():.1%}")
    print(f"Regular success rate: {two_pt[~two_pt['high_stakes']]['success'].mean():.1%}")

    # Find the most impactful 2pt conversions
    print("\n" + "-" * 60)
    print("Most impactful 2pt conversions (high-stakes successes):")
    print("-" * 60)

    high_stakes_success = two_pt[two_pt['high_stakes'] & (two_pt['success'] == 1)].copy()
    high_stakes_success = high_stakes_success.sort_values('game_seconds_remaining')

    for _, row in high_stakes_success.head(10).iterrows():
        print(f"  {int(row['season'])} Week {int(row['week'])}: {row['posteam']} vs {row['defteam']}, "
              f"{row['game_seconds_remaining']:.0f}s left, score diff {row['score_diff_pre_td']:+.0f}")

    # Event study: League behavior after high-stakes successes
    # Group by season and see if high-stakes successes predict next-season changes
    high_stakes_by_season = two_pt[two_pt['high_stakes']].groupby('season').agg(
        n_high_stakes=('success', 'count'),
        n_success=('success', 'sum'),
        success_rate=('success', 'mean')
    ).reset_index()

    print("\nHigh-stakes 2pt by season:")
    for _, row in high_stakes_by_season.iterrows():
        print(f"  {int(row['season'])}: {int(row['n_success'])}/{int(row['n_high_stakes'])} successes ({row['success_rate']:.1%})")

    return two_pt, high_stakes_by_season


def analyze_within_game_learning(pbp):
    """
    Test 2: Do coaches adapt during a single game?

    For games with multiple 2pt opportunities:
    - If first 2pt attempt SUCCEEDS → P(second 2pt attempt)?
    - If first 2pt attempt FAILS → P(second 2pt attempt)?
    """
    print("\n" + "=" * 80)
    print("TEST 2: WITHIN-GAME LEARNING")
    print("=" * 80)

    # Get all PAT/2pt decisions
    pat_plays = pbp[pbp['extra_point_attempt'] == 1].copy()
    two_pt_plays = pbp[pbp['two_point_attempt'] == 1].copy()

    pat_plays['decision'] = 'pat'
    two_pt_plays['decision'] = '2pt'
    two_pt_plays['success'] = (two_pt_plays['two_point_conv_result'] == 'success').astype(int)

    all_decisions = pd.concat([pat_plays, two_pt_plays])
    all_decisions = all_decisions[all_decisions['season'] >= 2015]
    all_decisions = all_decisions.sort_values(['game_id', 'play_id'])

    # For each game, track sequence of decisions
    games_with_multiple = []

    for game_id, game_df in all_decisions.groupby('game_id'):
        team_decisions = game_df.groupby('posteam')

        for team, team_df in team_decisions:
            if len(team_df) < 2:
                continue

            team_df = team_df.sort_values('play_id').reset_index(drop=True)

            for i in range(len(team_df) - 1):
                first = team_df.iloc[i]
                second = team_df.iloc[i + 1]

                if first['decision'] == '2pt':
                    first_success = first.get('success', np.nan)
                    second_went_for_2 = second['decision'] == '2pt'

                    games_with_multiple.append({
                        'game_id': game_id,
                        'team': team,
                        'season': first['season'],
                        'first_decision': first['decision'],
                        'first_success': first_success,
                        'second_decision': second['decision'],
                        'second_went_for_2': second_went_for_2
                    })

    multi_df = pd.DataFrame(games_with_multiple)

    if len(multi_df) == 0:
        print("No games with multiple 2pt opportunities found")
        return None

    print(f"\nGames where first attempt was 2pt: {len(multi_df)}")

    # Key test: Does first 2pt success predict second 2pt attempt?
    first_success = multi_df[multi_df['first_success'] == 1]
    first_fail = multi_df[multi_df['first_success'] == 0]

    print(f"\nAfter FIRST 2pt SUCCESS ({len(first_success)} cases):")
    print(f"  P(second attempt is 2pt): {first_success['second_went_for_2'].mean():.1%}")

    print(f"\nAfter FIRST 2pt FAILURE ({len(first_fail)} cases):")
    print(f"  P(second attempt is 2pt): {first_fail['second_went_for_2'].mean():.1%}")

    # Statistical test
    if len(first_success) > 5 and len(first_fail) > 5:
        from scipy import stats
        stat, pval = stats.fisher_exact([
            [first_success['second_went_for_2'].sum(), len(first_success) - first_success['second_went_for_2'].sum()],
            [first_fail['second_went_for_2'].sum(), len(first_fail) - first_fail['second_went_for_2'].sum()]
        ])
        print(f"\nFisher exact test p-value: {pval:.4f}")

        if pval < 0.05:
            print("  => SIGNIFICANT within-game learning!")
        else:
            print("  => No significant within-game learning")

    return multi_df


def analyze_coaching_trees(decisions):
    """
    Test 3: Do coaches from the same tree have similar 2pt behavior?
    """
    print("\n" + "=" * 80)
    print("TEST 3: COACHING TREE EFFECTS")
    print("=" * 80)

    # Add coach to decisions
    decisions = decisions.copy()
    decisions['coach'] = decisions.apply(
        lambda r: COACH_MAPPING.get((r['posteam'], r['season']), 'Unknown'),
        axis=1
    )

    # Add coaching tree
    def get_tree(coach):
        for tree_name, coaches in COACHING_TREES.items():
            if coach in coaches:
                return tree_name
        return 'Other'

    decisions['coaching_tree'] = decisions['coach'].apply(get_tree)

    # Analyze by tree
    tree_stats = decisions.groupby('coaching_tree').agg(
        n_decisions=('coach_correct', 'count'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    print("\n2pt Behavior by Coaching Tree:")
    print("-" * 70)
    print(f"{'Tree':<20} | {'N':>6} | {'2pt Rate':>10} | {'Correct %':>10} | {'WP Cost':>10}")
    print("-" * 70)

    for _, row in tree_stats.sort_values('pct_correct', ascending=False).iterrows():
        print(f"{row['coaching_tree']:<20} | {int(row['n_decisions']):>6} | "
              f"{row['actual_2pt_rate']:>10.1%} | {row['pct_correct']:>10.1%} | "
              f"{row['mean_wp_cost']*1e6:>10.1f}")

    return tree_stats


def analyze_coach_background(decisions):
    """
    Test 4: Do offensive vs defensive coaches differ in 2pt accuracy?
    """
    print("\n" + "=" * 80)
    print("TEST 4: COACH BACKGROUND (OFFENSIVE vs DEFENSIVE)")
    print("=" * 80)

    decisions = decisions.copy()
    decisions['coach'] = decisions.apply(
        lambda r: COACH_MAPPING.get((r['posteam'], r['season']), 'Unknown'),
        axis=1
    )

    decisions['background'] = decisions['coach'].apply(
        lambda c: COACH_BACKGROUND.get(c, 'Unknown')
    )

    # Analyze by background
    bg_stats = decisions[decisions['background'] != 'Unknown'].groupby('background').agg(
        n_decisions=('coach_correct', 'count'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    print("\n2pt Behavior by Coach Background:")
    print("-" * 70)
    print(f"{'Background':<15} | {'N':>6} | {'2pt Rate':>10} | {'Correct %':>10} | {'WP Cost':>10}")
    print("-" * 70)

    for _, row in bg_stats.sort_values('pct_correct', ascending=False).iterrows():
        print(f"{row['background']:<15} | {int(row['n_decisions']):>6} | "
              f"{row['actual_2pt_rate']:>10.1%} | {row['pct_correct']:>10.1%} | "
              f"{row['mean_wp_cost']*1e6:>10.1f}")

    return bg_stats


def analyze_pederson_effect(pbp, decisions):
    """
    Test 5: The Pederson Super Bowl Effect (Feb 2018)

    Doug Pederson won Super Bowl LII with famously aggressive play-calling.
    Did this create a permission structure for other coaches?
    """
    print("\n" + "=" * 80)
    print("TEST 5: THE PEDERSON SUPER BOWL EFFECT")
    print("=" * 80)

    # Super Bowl LII was Feb 4, 2018
    # Compare league 2pt behavior before and after

    decisions = decisions.copy()

    # Pre-Pederson: 2015-2017
    # Post-Pederson: 2018-2024
    decisions['era'] = np.where(decisions['season'] <= 2017, 'Pre-Pederson (2015-17)', 'Post-Pederson (2018-24)')

    era_stats = decisions.groupby('era').agg(
        n_decisions=('coach_correct', 'count'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    print("\nLeague-wide 2pt behavior before/after Pederson Super Bowl:")
    print("-" * 70)
    for _, row in era_stats.iterrows():
        print(f"\n{row['era']}:")
        print(f"  Decisions: {int(row['n_decisions']):,}")
        print(f"  2pt rate: {row['actual_2pt_rate']:.1%}")
        print(f"  Correct rate: {row['pct_correct']:.1%}")
        print(f"  Mean WP cost: {row['mean_wp_cost']*1e6:.1f}")

    # Did teams who PLAYED the Eagles in 2017 change more?
    # Eagles 2017 opponents
    eagles_opponents_2017 = ['WAS', 'KC', 'NYG', 'LAC', 'ARI', 'CAR', 'WAS',
                             'SF', 'DEN', 'DAL', 'CHI', 'SEA', 'LAR', 'NYG', 'OAK', 'DAL']

    decisions['played_eagles_2017'] = decisions['posteam'].isin(eagles_opponents_2017)

    # Compare change for Eagles opponents vs non-opponents
    for played in [True, False]:
        subset = decisions[decisions['played_eagles_2017'] == played]
        pre = subset[subset['season'] <= 2017]
        post = subset[subset['season'] >= 2018]

        label = "Teams who played Eagles in 2017" if played else "Other teams"
        pre_rate = (pre['actual_decision'] == 'two_point').mean()
        post_rate = (post['actual_decision'] == 'two_point').mean()

        print(f"\n{label}:")
        print(f"  Pre-SB 2pt rate: {pre_rate:.1%}")
        print(f"  Post-SB 2pt rate: {post_rate:.1%}")
        print(f"  Change: {(post_rate - pre_rate)*100:+.1f} pp")

    return era_stats


def analyze_counterfactual_regret(pbp, decisions):
    """
    Test 6: Does a coach who LOSES because they DIDN'T go for 2 learn?

    Identify games where:
    - Coach kicked PAT when model said go for 2
    - Team lost by 1-2 points (the 2pt would have mattered)
    """
    print("\n" + "=" * 80)
    print("TEST 6: COUNTERFACTUAL REGRET LEARNING")
    print("=" * 80)

    decisions = decisions.copy()
    decisions['coach'] = decisions.apply(
        lambda r: COACH_MAPPING.get((r['posteam'], r['season']), 'Unknown'),
        axis=1
    )

    # Get game outcomes
    game_outcomes = pbp.groupby('game_id').agg(
        home_final=('home_score', 'max'),
        away_final=('away_score', 'max'),
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first')
    ).reset_index()

    # Merge with decisions
    decisions = decisions.merge(game_outcomes, on='game_id', how='left')

    # Calculate margin
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

    # Find "regret" situations: kicked PAT, should have gone for 2, lost by 1-2
    regret_situations = decisions[
        (decisions['actual_decision'] == 'pat') &
        (decisions['optimal_decision'] == 'two_point') &
        (decisions['final_margin'].isin([-1, -2]))
    ].copy()

    print(f"\nRegret situations (kicked PAT, should go for 2, lost by 1-2): {len(regret_situations)}")

    if len(regret_situations) == 0:
        print("No regret situations found")
        return None

    # For each regret situation, check if coach improved in next season
    coach_regrets = regret_situations.groupby(['coach', 'season']).size().reset_index(name='n_regrets')

    print("\nCoaches with regret situations:")
    for _, row in coach_regrets.iterrows():
        print(f"  {row['coach']} ({int(row['season'])}): {int(row['n_regrets'])} situations")

    # Check if these coaches improved
    # Compare their correct rate before and after regret season
    results = []
    for _, regret in coach_regrets.iterrows():
        coach = regret['coach']
        regret_season = regret['season']

        coach_decisions = decisions[decisions['coach'] == coach]
        before = coach_decisions[coach_decisions['season'] < regret_season]
        after = coach_decisions[coach_decisions['season'] > regret_season]

        if len(before) >= 10 and len(after) >= 10:
            before_correct = before['coach_correct'].mean()
            after_correct = after['coach_correct'].mean()

            results.append({
                'coach': coach,
                'regret_season': regret_season,
                'before_correct': before_correct,
                'after_correct': after_correct,
                'improvement': after_correct - before_correct
            })

    if results:
        results_df = pd.DataFrame(results)
        print("\nDid regret lead to improvement?")
        print("-" * 60)
        for _, row in results_df.iterrows():
            print(f"  {row['coach']}: {row['before_correct']:.1%} → {row['after_correct']:.1%} "
                  f"({row['improvement']*100:+.1f} pp)")

        print(f"\nMean improvement after regret: {results_df['improvement'].mean()*100:+.1f} pp")

    return regret_situations


def analyze_generational_shift(decisions):
    """
    Test 7: Are newer-generation coaches better from the start?
    """
    print("\n" + "=" * 80)
    print("TEST 7: GENERATIONAL SHIFT")
    print("=" * 80)

    decisions = decisions.copy()
    decisions['coach'] = decisions.apply(
        lambda r: COACH_MAPPING.get((r['posteam'], r['season']), 'Unknown'),
        axis=1
    )

    # Find first year for each coach
    coach_first_year = decisions.groupby('coach')['season'].min().reset_index()
    coach_first_year.columns = ['coach', 'first_year']

    # Classify by generation
    def classify_generation(year):
        if year <= 2015:
            return 'Pre-2015'
        elif year <= 2019:
            return '2015-2019'
        else:
            return '2020+'

    coach_first_year['generation'] = coach_first_year['first_year'].apply(classify_generation)

    # Merge back
    decisions = decisions.merge(coach_first_year, on='coach', how='left')

    # Analyze by generation
    gen_stats = decisions[decisions['generation'].notna()].groupby('generation').agg(
        n_decisions=('coach_correct', 'count'),
        n_coaches=('coach', 'nunique'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    print("\n2pt Behavior by Coach Generation:")
    print("-" * 80)
    print(f"{'Generation':<15} | {'Coaches':>8} | {'Decisions':>10} | {'2pt Rate':>10} | {'Correct %':>10}")
    print("-" * 80)

    for _, row in gen_stats.iterrows():
        print(f"{row['generation']:<15} | {int(row['n_coaches']):>8} | {int(row['n_decisions']):>10} | "
              f"{row['actual_2pt_rate']:>10.1%} | {row['pct_correct']:>10.1%}")

    return gen_stats


def analyze_new_coach_honeymoon(decisions):
    """
    Test 8: Do new coaches experiment more in year 1?
    """
    print("\n" + "=" * 80)
    print("TEST 8: NEW COACH HONEYMOON EFFECT")
    print("=" * 80)

    decisions = decisions.copy()
    decisions['coach'] = decisions.apply(
        lambda r: COACH_MAPPING.get((r['posteam'], r['season']), 'Unknown'),
        axis=1
    )

    # Find coach tenure
    coach_first = decisions.groupby('coach')['season'].min().reset_index()
    coach_first.columns = ['coach', 'first_year']
    decisions = decisions.merge(coach_first, on='coach', how='left')
    decisions['tenure'] = decisions['season'] - decisions['first_year'] + 1

    # Analyze by tenure
    tenure_stats = decisions[decisions['tenure'] <= 5].groupby('tenure').agg(
        n_decisions=('coach_correct', 'count'),
        pct_correct=('coach_correct', 'mean'),
        actual_2pt_rate=('actual_decision', lambda x: (x == 'two_point').mean()),
        mean_wp_cost=('wp_cost', 'mean')
    ).reset_index()

    print("\n2pt Behavior by Coach Tenure:")
    print("-" * 60)
    print(f"{'Tenure':<10} | {'Decisions':>10} | {'2pt Rate':>10} | {'Correct %':>10}")
    print("-" * 60)

    for _, row in tenure_stats.iterrows():
        print(f"Year {int(row['tenure']):<5} | {int(row['n_decisions']):>10} | "
              f"{row['actual_2pt_rate']:>10.1%} | {row['pct_correct']:>10.1%}")

    # Is year 1 different?
    year1 = decisions[decisions['tenure'] == 1]
    year3plus = decisions[decisions['tenure'] >= 3]

    print(f"\nYear 1 vs Year 3+ comparison:")
    print(f"  Year 1 2pt rate: {(year1['actual_decision'] == 'two_point').mean():.1%}")
    print(f"  Year 3+ 2pt rate: {(year3plus['actual_decision'] == 'two_point').mean():.1%}")
    print(f"  Year 1 correct rate: {year1['coach_correct'].mean():.1%}")
    print(f"  Year 3+ correct rate: {year3plus['coach_correct'].mean():.1%}")

    return tenure_stats


def main():
    """Run all affirmative learning mechanism tests."""
    print("=" * 80)
    print("AFFIRMATIVE LEARNING MECHANISMS ANALYSIS")
    print("=" * 80)
    print("""
We have strong null results on standard learning (no social learning, no
own-experience learning). But aggregate improvement IS happening.

This analysis tests specific mechanisms that might explain HOW learning occurs:
1. High-stakes success creates permission (legitimacy)
2. Within-game adaptation (real-time learning)
3. Coaching networks transmit knowledge
4. Coach background matters
5. Pederson Super Bowl effect (permission structure)
6. Counterfactual regret learning
7. Generational replacement with better priors
8. New coach honeymoon effect
""")

    # Load data
    print("\nLoading data...")
    pbp, decisions = load_data()
    print(f"Loaded {len(pbp):,} plays and {len(decisions):,} 2pt decisions")

    # Run all tests
    high_stakes, hs_by_season = analyze_high_stakes_success(pbp, decisions)
    within_game = analyze_within_game_learning(pbp)
    tree_stats = analyze_coaching_trees(decisions)
    background_stats = analyze_coach_background(decisions)
    pederson_stats = analyze_pederson_effect(pbp, decisions)
    regret = analyze_counterfactual_regret(pbp, decisions)
    gen_stats = analyze_generational_shift(decisions)
    tenure_stats = analyze_new_coach_honeymoon(decisions)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: AFFIRMATIVE LEARNING MECHANISMS")
    print("=" * 80)
    print("""
Key findings will be summarized here after running all tests.
Look for mechanisms with statistically significant and economically meaningful effects.
""")


if __name__ == "__main__":
    main()
