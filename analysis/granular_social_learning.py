"""
Granular Social Learning Tests

Our aggregate tests found no social learning. But learning might happen through
specific, salient exposures rather than gradual absorption of league trends.

Tests:
1. Direct in-game exposure: opponent went for 2 in game you just played
2. First exposure effect: staggered adoption design
3. Recency effects: what happened last week specifically
4. High-salience games: primetime, playoffs, divisional
5. "It happened TO me" effect: being on receiving end
6. Copycat effect: successful plays in visible games
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from learning_mechanisms import COACH_MAPPING, TEAM_TO_DIVISION


def load_data():
    """Load two-point decision data with game context."""
    data_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'two_point_decision_analysis.parquet'
    df = pd.read_parquet(data_path)

    df['coach'] = df.apply(lambda row: COACH_MAPPING.get((row['posteam'], row['season']), 'Unknown'), axis=1)
    df['division'] = df['posteam'].map(TEAM_TO_DIVISION)
    df['went_for_2'] = (df['actual_decision'] == 'two_point').astype(int)
    df['should_go_for_2'] = (df['optimal_decision'] == 'two_point').astype(int)

    return df


def test_direct_ingame_exposure(df):
    """
    Test 1: Direct In-Game Exposure

    When your opponent goes for 2 in the game you're playing,
    does it affect your future behavior?
    """
    print("=" * 80)
    print("TEST 1: DIRECT IN-GAME EXPOSURE")
    print("=" * 80)
    print("\nDid seeing opponent go for 2 IN YOUR GAME affect your next game?\n")

    # Aggregate to game level: for each team-game, did they go for 2?
    game_level = df.groupby(['posteam', 'season', 'week', 'game_id', 'defteam']).agg({
        'went_for_2': ['sum', 'max'],  # count and any
        'should_go_for_2': 'max',
        'decision_success': 'mean',
        'coach': 'first'
    }).reset_index()

    game_level.columns = ['team', 'season', 'week', 'game_id', 'opponent',
                         'n_2pt_attempts', 'any_2pt', 'should_go', 'success_rate', 'coach']

    # For each game, we need to know what the OPPONENT did (from their perspective)
    # Create opponent's 2pt behavior in this game
    opponent_behavior = game_level[['game_id', 'team', 'any_2pt', 'n_2pt_attempts', 'success_rate']].copy()
    opponent_behavior.columns = ['game_id', 'opponent', 'opp_went_for_2', 'opp_n_2pt', 'opp_success_rate']

    # Merge: for each team-game, what did their opponent do?
    game_with_opp = game_level.merge(opponent_behavior, on=['game_id', 'opponent'], how='left')

    # Sort and get NEXT game for each team
    game_with_opp = game_with_opp.sort_values(['team', 'season', 'week'])

    # Lag: what happened in your PREVIOUS game?
    game_with_opp['prev_opp_went_for_2'] = game_with_opp.groupby('team')['opp_went_for_2'].shift(1)
    game_with_opp['prev_opp_success'] = game_with_opp.groupby('team')['opp_success_rate'].shift(1)

    # Filter to valid observations (have previous game)
    valid = game_with_opp.dropna(subset=['prev_opp_went_for_2'])

    # Test: P(go for 2) ~ f(prev opponent went for 2)
    exposed = valid[valid['prev_opp_went_for_2'] == 1]
    not_exposed = valid[valid['prev_opp_went_for_2'] == 0]

    print(f"Games after opponent went for 2:     {exposed['any_2pt'].mean():.1%} went for 2 (n={len(exposed)})")
    print(f"Games after opponent did NOT go for 2: {not_exposed['any_2pt'].mean():.1%} went for 2 (n={len(not_exposed)})")

    diff = exposed['any_2pt'].mean() - not_exposed['any_2pt'].mean()
    print(f"\nDifference: {diff:+.1%}")

    if diff > 0.02:
        print("→ Evidence of DIRECT EXPOSURE learning: seeing opponent do it increases your rate")
    else:
        print("→ No strong evidence of direct exposure learning")

    # Conditional on opponent success
    print("\n" + "-" * 50)
    print("Conditional on opponent's 2pt SUCCESS:")
    print("-" * 50)

    exposed_success = valid[(valid['prev_opp_went_for_2'] == 1) & (valid['prev_opp_success'] == 1)]
    exposed_fail = valid[(valid['prev_opp_went_for_2'] == 1) & (valid['prev_opp_success'] == 0)]

    if len(exposed_success) > 10 and len(exposed_fail) > 10:
        print(f"After opponent SUCCESS: {exposed_success['any_2pt'].mean():.1%} (n={len(exposed_success)})")
        print(f"After opponent FAILURE: {exposed_fail['any_2pt'].mean():.1%} (n={len(exposed_fail)})")
        print(f"Difference: {exposed_success['any_2pt'].mean() - exposed_fail['any_2pt'].mean():+.1%}")
    else:
        print("Insufficient data for success/failure breakdown")

    return game_with_opp


def test_first_exposure(df):
    """
    Test 2: First Exposure Effect (Staggered Adoption Design)

    The first time a team sees an opponent go for 2 in a "should go" situation.
    """
    print("\n" + "=" * 80)
    print("TEST 2: FIRST EXPOSURE EFFECT")
    print("=" * 80)
    print("\nDoes first exposure to opponent's 2pt attempt trigger adoption?\n")

    # Get game-level data
    game_level = df.groupby(['posteam', 'season', 'week', 'game_id', 'defteam']).agg({
        'went_for_2': 'max',
        'should_go_for_2': 'max',
        'coach_correct': 'mean'
    }).reset_index()

    game_level.columns = ['team', 'season', 'week', 'game_id', 'opponent',
                         'any_2pt', 'should_go', 'correct_rate']

    # Get opponent's behavior
    opp_behavior = game_level[['game_id', 'team', 'any_2pt']].copy()
    opp_behavior.columns = ['game_id', 'opponent', 'opp_went_for_2']

    game_level = game_level.merge(opp_behavior, on=['game_id', 'opponent'], how='left')
    game_level = game_level.sort_values(['team', 'season', 'week'])

    # For each team, find their FIRST exposure (first game where opponent went for 2)
    first_exposures = []

    for team in game_level['team'].unique():
        team_games = game_level[game_level['team'] == team].copy()

        # Find first game where opponent went for 2
        exposed_games = team_games[team_games['opp_went_for_2'] == 1]

        if len(exposed_games) > 0:
            first_exposure = exposed_games.iloc[0]
            first_exposures.append({
                'team': team,
                'exposure_season': first_exposure['season'],
                'exposure_week': first_exposure['week'],
                'exposure_game_id': first_exposure['game_id']
            })

    exposure_df = pd.DataFrame(first_exposures)
    print(f"Teams with identified first exposure: {len(exposure_df)}")

    # Merge back to get pre/post exposure behavior
    game_level = game_level.merge(exposure_df, on='team', how='left')

    # Tag pre/post exposure
    def get_period(row):
        if pd.isna(row['exposure_season']):
            return 'never_exposed'
        elif (row['season'] < row['exposure_season']) or \
             (row['season'] == row['exposure_season'] and row['week'] < row['exposure_week']):
            return 'pre_exposure'
        else:
            return 'post_exposure'

    game_level['exposure_period'] = game_level.apply(get_period, axis=1)

    # Compare pre vs post exposure
    pre = game_level[game_level['exposure_period'] == 'pre_exposure']
    post = game_level[game_level['exposure_period'] == 'post_exposure']

    print(f"\nPre-exposure 2pt rate:  {pre['any_2pt'].mean():.1%} (n={len(pre)} team-games)")
    print(f"Post-exposure 2pt rate: {post['any_2pt'].mean():.1%} (n={len(post)} team-games)")
    print(f"Difference: {post['any_2pt'].mean() - pre['any_2pt'].mean():+.1%}")

    # Window around exposure
    print("\n" + "-" * 50)
    print("Behavior in narrow window around exposure:")
    print("-" * 50)

    # Get behavior in 4 games before vs 4 games after exposure
    results = []

    for team in exposure_df['team'].unique():
        team_games = game_level[game_level['team'] == team].sort_values(['season', 'week']).copy()
        team_games['game_order'] = range(len(team_games))

        exposure_info = exposure_df[exposure_df['team'] == team].iloc[0]

        # Find exposure game index
        exp_mask = (team_games['season'] == exposure_info['exposure_season']) & \
                   (team_games['week'] == exposure_info['exposure_week'])

        if exp_mask.sum() > 0:
            exp_idx = team_games[exp_mask]['game_order'].iloc[0]

            # Get 4 games before and after
            pre_games = team_games[(team_games['game_order'] >= exp_idx - 4) &
                                   (team_games['game_order'] < exp_idx)]
            post_games = team_games[(team_games['game_order'] > exp_idx) &
                                    (team_games['game_order'] <= exp_idx + 4)]

            if len(pre_games) >= 2 and len(post_games) >= 2:
                results.append({
                    'team': team,
                    'pre_rate': pre_games['any_2pt'].mean(),
                    'post_rate': post_games['any_2pt'].mean(),
                    'change': post_games['any_2pt'].mean() - pre_games['any_2pt'].mean()
                })

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        print(f"\nTeams with valid pre/post window: {len(results_df)}")
        print(f"Average pre-exposure rate:  {results_df['pre_rate'].mean():.1%}")
        print(f"Average post-exposure rate: {results_df['post_rate'].mean():.1%}")
        print(f"Average change: {results_df['change'].mean():+.1%}")

        # How many teams increased?
        increased = (results_df['change'] > 0).sum()
        decreased = (results_df['change'] < 0).sum()
        print(f"\nTeams that increased after exposure: {increased}/{len(results_df)}")
        print(f"Teams that decreased after exposure: {decreased}/{len(results_df)}")

    return game_level


def test_it_happened_to_me(df):
    """
    Test 5: "It Happened TO Me" Effect

    Being on the receiving end of a 2pt attempt might be more memorable.
    Does opponent going for 2 AGAINST YOU change YOUR behavior?
    """
    print("\n" + "=" * 80)
    print("TEST 5: 'IT HAPPENED TO ME' EFFECT")
    print("=" * 80)
    print("\nDoes being on receiving end of 2pt attempt change your behavior?\n")

    # Game level aggregation
    game_level = df.groupby(['posteam', 'defteam', 'season', 'week', 'game_id']).agg({
        'went_for_2': ['sum', 'max'],
        'decision_success': 'sum',  # number of successful 2pt
        'should_go_for_2': 'max'
    }).reset_index()

    game_level.columns = ['team', 'opponent', 'season', 'week', 'game_id',
                         'n_2pt', 'any_2pt', 'n_success', 'should_go']

    # "It happened to me" = opponent went for 2 against me
    # We need to flip perspective: what did my opponent (as offense) do?

    # Get offensive behavior per game
    off_behavior = game_level[['game_id', 'team', 'any_2pt', 'n_success', 'n_2pt']].copy()
    off_behavior.columns = ['game_id', 'opponent', 'opp_went_for_2_vs_me',
                           'opp_successes_vs_me', 'opp_attempts_vs_me']

    # Merge: for each team-game, what did opponent do against them?
    game_level = game_level.merge(off_behavior, on=['game_id', 'opponent'], how='left')

    # Sort and lag
    game_level = game_level.sort_values(['team', 'season', 'week'])
    game_level['prev_opp_2pt_vs_me'] = game_level.groupby('team')['opp_went_for_2_vs_me'].shift(1)
    game_level['prev_opp_success_vs_me'] = game_level.groupby('team')['opp_successes_vs_me'].shift(1)
    game_level['prev_opp_attempts_vs_me'] = game_level.groupby('team')['opp_attempts_vs_me'].shift(1)

    valid = game_level.dropna(subset=['prev_opp_2pt_vs_me'])

    # Did opponent succeed vs me?
    valid['prev_opp_succeeded'] = (valid['prev_opp_success_vs_me'] > 0).astype(int)

    # Test
    happened_to_me = valid[valid['prev_opp_2pt_vs_me'] == 1]
    not_happened = valid[valid['prev_opp_2pt_vs_me'] == 0]

    print(f"After opponent went for 2 AGAINST ME: {happened_to_me['any_2pt'].mean():.1%} (n={len(happened_to_me)})")
    print(f"After opponent did NOT go for 2 vs me: {not_happened['any_2pt'].mean():.1%} (n={len(not_happened)})")
    print(f"Difference: {happened_to_me['any_2pt'].mean() - not_happened['any_2pt'].mean():+.1%}")

    # Conditional on success against me
    print("\n" + "-" * 50)
    print("Conditional on whether opponent SUCCEEDED against me:")
    print("-" * 50)

    succeeded_vs_me = happened_to_me[happened_to_me['prev_opp_succeeded'] == 1]
    failed_vs_me = happened_to_me[happened_to_me['prev_opp_succeeded'] == 0]

    if len(succeeded_vs_me) > 5 and len(failed_vs_me) > 5:
        print(f"Opponent SUCCEEDED vs me: {succeeded_vs_me['any_2pt'].mean():.1%} (n={len(succeeded_vs_me)})")
        print(f"Opponent FAILED vs me:    {failed_vs_me['any_2pt'].mean():.1%} (n={len(failed_vs_me)})")
        diff = succeeded_vs_me['any_2pt'].mean() - failed_vs_me['any_2pt'].mean()
        print(f"Difference: {diff:+.1%}")

        if diff > 0.05:
            print("\n→ Teams ADOPT when opponent succeeds against them")
        elif diff < -0.05:
            print("\n→ Teams DISMISS when opponent succeeds (counter-imitation?)")
        else:
            print("\n→ No differential effect based on success")

    return valid


def test_high_salience_games(df):
    """
    Test 4: High-Salience Games

    Learning might be stronger from high-visibility games (primetime, playoffs).
    """
    print("\n" + "=" * 80)
    print("TEST 4: HIGH-SALIENCE GAMES")
    print("=" * 80)
    print("\nDo 2pt attempts in primetime games have larger effects?\n")

    # We don't have primetime flags directly, but we can proxy:
    # - Week 18+ = playoffs
    # - Thanksgiving = week 12, Thursday
    # - Division games are higher salience

    # Tag salience
    df = df.copy()
    df['is_playoff'] = (df['week'] > 18).astype(int)
    df['is_division'] = df.apply(
        lambda r: 1 if TEAM_TO_DIVISION.get(r['posteam']) == TEAM_TO_DIVISION.get(r['defteam']) else 0,
        axis=1
    )

    # Aggregate to game level with salience tags
    game_level = df.groupby(['posteam', 'season', 'week', 'game_id', 'defteam']).agg({
        'went_for_2': 'max',
        'is_playoff': 'first',
        'is_division': 'first'
    }).reset_index()

    game_level.columns = ['team', 'season', 'week', 'game_id', 'opponent',
                         'any_2pt', 'is_playoff', 'is_division']

    # High salience = playoff OR division
    game_level['high_salience'] = ((game_level['is_playoff'] == 1) |
                                   (game_level['is_division'] == 1)).astype(int)

    print(f"High-salience games: {game_level['high_salience'].sum()} ({game_level['high_salience'].mean():.1%})")
    print(f"Playoff games: {game_level['is_playoff'].sum()}")
    print(f"Division games: {game_level['is_division'].sum()}")

    # Get opponent behavior
    opp_behavior = game_level[['game_id', 'team', 'any_2pt', 'high_salience']].copy()
    opp_behavior.columns = ['game_id', 'opponent', 'opp_went_for_2', 'opp_high_salience']

    game_level = game_level.merge(opp_behavior, on=['game_id', 'opponent'], how='left')
    game_level = game_level.sort_values(['team', 'season', 'week'])

    # Lag
    game_level['prev_opp_2pt'] = game_level.groupby('team')['opp_went_for_2'].shift(1)
    game_level['prev_high_salience'] = game_level.groupby('team')['opp_high_salience'].shift(1)

    valid = game_level.dropna(subset=['prev_opp_2pt'])

    # Compare effect in high vs low salience games
    print("\n" + "-" * 50)
    print("Effect of exposure, by salience of exposure game:")
    print("-" * 50)

    high_sal_exp = valid[(valid['prev_opp_2pt'] == 1) & (valid['prev_high_salience'] == 1)]
    low_sal_exp = valid[(valid['prev_opp_2pt'] == 1) & (valid['prev_high_salience'] == 0)]
    no_exposure = valid[valid['prev_opp_2pt'] == 0]

    print(f"\nNo exposure:                       {no_exposure['any_2pt'].mean():.1%} (n={len(no_exposure)})")
    print(f"Exposure in LOW salience game:     {low_sal_exp['any_2pt'].mean():.1%} (n={len(low_sal_exp)})")
    print(f"Exposure in HIGH salience game:    {high_sal_exp['any_2pt'].mean():.1%} (n={len(high_sal_exp)})")

    low_effect = low_sal_exp['any_2pt'].mean() - no_exposure['any_2pt'].mean()
    high_effect = high_sal_exp['any_2pt'].mean() - no_exposure['any_2pt'].mean()

    print(f"\nLow salience exposure effect:  {low_effect:+.1%}")
    print(f"High salience exposure effect: {high_effect:+.1%}")
    print(f"Difference (high - low):       {high_effect - low_effect:+.1%}")

    if high_effect > low_effect + 0.03:
        print("\n→ HIGH salience games have LARGER learning effect")
    else:
        print("\n→ No evidence that salience matters for learning")

    return valid


def test_copycat_effect(df):
    """
    Test 7: Copycat Effect

    Does a successful 2pt attempt in a visible game predict league-wide increase?
    Event study around high-profile conversions.
    """
    print("\n" + "=" * 80)
    print("TEST 7: COPYCAT EFFECT (EVENT STUDY)")
    print("=" * 80)
    print("\nDoes league 2pt rate increase after high-profile successful conversions?\n")

    # Find high-profile successful 2pt conversions
    # High-profile = late game (clutch), successful, in a close game

    clutch_2pt = df[
        (df['actual_decision'] == 'two_point') &
        (df['decision_success'] == 1) &
        (df['time_remaining'] <= 300) &  # Last 5 minutes
        (abs(df['score_diff_pre_td']) <= 8)  # Close game
    ].copy()

    print(f"Clutch successful 2pt conversions: {len(clutch_2pt)}")

    # Get weekly league 2pt rate
    weekly = df.groupby(['season', 'week']).agg({
        'went_for_2': ['sum', 'count', 'mean']
    }).reset_index()
    weekly.columns = ['season', 'week', 'n_2pt', 'n_total', 'rate']

    # For each clutch conversion, look at league rate in subsequent weeks
    print("\n" + "-" * 50)
    print("League 2pt rate before vs after clutch conversions:")
    print("-" * 50)

    effects = []

    for _, event in clutch_2pt.iterrows():
        event_season = event['season']
        event_week = event['week']

        # Get rate 2 weeks before and 2 weeks after
        pre_weeks = weekly[(weekly['season'] == event_season) &
                           (weekly['week'] >= event_week - 2) &
                           (weekly['week'] < event_week)]
        post_weeks = weekly[(weekly['season'] == event_season) &
                            (weekly['week'] > event_week) &
                            (weekly['week'] <= event_week + 2)]

        if len(pre_weeks) >= 1 and len(post_weeks) >= 1:
            pre_rate = pre_weeks['rate'].mean()
            post_rate = post_weeks['rate'].mean()
            effects.append({
                'season': event_season,
                'week': event_week,
                'team': event['posteam'],
                'pre_rate': pre_rate,
                'post_rate': post_rate,
                'change': post_rate - pre_rate
            })

    if len(effects) > 0:
        effects_df = pd.DataFrame(effects)
        print(f"\nEvents with valid pre/post data: {len(effects_df)}")
        print(f"Average pre-event rate:  {effects_df['pre_rate'].mean():.1%}")
        print(f"Average post-event rate: {effects_df['post_rate'].mean():.1%}")
        print(f"Average change:          {effects_df['change'].mean():+.1%}")

        # How many show increase?
        increased = (effects_df['change'] > 0).sum()
        print(f"\nEvents followed by increase: {increased}/{len(effects_df)}")

    # Compare to random weeks (placebo)
    print("\n" + "-" * 50)
    print("Placebo: Random week changes:")
    print("-" * 50)

    random_changes = []
    for season in weekly['season'].unique():
        season_data = weekly[weekly['season'] == season].sort_values('week')
        if len(season_data) >= 4:
            for i in range(1, len(season_data) - 2):
                pre = season_data.iloc[i-1:i+1]['rate'].mean()
                post = season_data.iloc[i+1:i+3]['rate'].mean()
                random_changes.append(post - pre)

    if len(random_changes) > 0:
        avg_random = np.mean(random_changes)
        print(f"Average week-to-week change (baseline): {avg_random:+.1%}")

        if len(effects) > 0:
            event_avg = effects_df['change'].mean()
            diff = event_avg - avg_random
            print(f"Clutch conversion effect vs baseline: {diff:+.1%}")

    return effects_df if len(effects) > 0 else None


def test_division_rivalry_learning(df):
    """
    Test: Division Rivalry Learning

    Do teams learn more from division rivals (play 2x/year, pay more attention)?
    """
    print("\n" + "=" * 80)
    print("TEST: DIVISION RIVALRY LEARNING")
    print("=" * 80)
    print("\nDo teams learn more from division rivals than non-division opponents?\n")

    df = df.copy()
    df['is_division_game'] = df.apply(
        lambda r: 1 if TEAM_TO_DIVISION.get(r['posteam']) == TEAM_TO_DIVISION.get(r['defteam']) else 0,
        axis=1
    )

    # Game level
    game_level = df.groupby(['posteam', 'season', 'week', 'game_id', 'defteam']).agg({
        'went_for_2': 'max',
        'is_division_game': 'first'
    }).reset_index()

    game_level.columns = ['team', 'season', 'week', 'game_id', 'opponent', 'any_2pt', 'is_div']

    # Get opponent behavior
    opp_behavior = game_level[['game_id', 'team', 'any_2pt']].copy()
    opp_behavior.columns = ['game_id', 'opponent', 'opp_went_for_2']

    game_level = game_level.merge(opp_behavior, on=['game_id', 'opponent'], how='left')
    game_level = game_level.sort_values(['team', 'season', 'week'])

    # Lag
    game_level['prev_opp_2pt'] = game_level.groupby('team')['opp_went_for_2'].shift(1)
    game_level['prev_was_div'] = game_level.groupby('team')['is_div'].shift(1)

    valid = game_level.dropna(subset=['prev_opp_2pt', 'prev_was_div'])

    # Effect from division vs non-division exposure
    div_exposure = valid[(valid['prev_opp_2pt'] == 1) & (valid['prev_was_div'] == 1)]
    nondiv_exposure = valid[(valid['prev_opp_2pt'] == 1) & (valid['prev_was_div'] == 0)]
    no_exposure = valid[valid['prev_opp_2pt'] == 0]

    print(f"No exposure:                      {no_exposure['any_2pt'].mean():.1%} (n={len(no_exposure)})")
    print(f"Exposure from NON-DIVISION opp:   {nondiv_exposure['any_2pt'].mean():.1%} (n={len(nondiv_exposure)})")
    print(f"Exposure from DIVISION opp:       {div_exposure['any_2pt'].mean():.1%} (n={len(div_exposure)})")

    nondiv_effect = nondiv_exposure['any_2pt'].mean() - no_exposure['any_2pt'].mean()
    div_effect = div_exposure['any_2pt'].mean() - no_exposure['any_2pt'].mean()

    print(f"\nNon-division exposure effect: {nondiv_effect:+.1%}")
    print(f"Division exposure effect:     {div_effect:+.1%}")
    print(f"Difference (div - nondiv):    {div_effect - nondiv_effect:+.1%}")

    if div_effect > nondiv_effect + 0.03:
        print("\n→ Teams learn MORE from division rivals")
    else:
        print("\n→ No evidence of stronger learning from division rivals")

    return valid


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} decisions")

    # Run all granular tests
    game_data = test_direct_ingame_exposure(df)
    test_first_exposure(df)
    test_it_happened_to_me(df)
    test_high_salience_games(df)
    test_copycat_effect(df)
    test_division_rivalry_learning(df)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: GRANULAR SOCIAL LEARNING TESTS")
    print("=" * 80)
    print("""
We tested finer-grained social learning mechanisms:

1. DIRECT IN-GAME EXPOSURE
   Did seeing opponent go for 2 in your game affect next game?

2. FIRST EXPOSURE EFFECT
   Staggered adoption: did first exposure trigger change?

3. "IT HAPPENED TO ME" EFFECT
   Does opponent's 2pt attempt AGAINST YOU change your behavior?

4. HIGH-SALIENCE GAMES
   Do playoff/division games have larger learning effects?

5. COPYCAT EFFECT
   Event study: does league rate increase after clutch conversions?

6. DIVISION RIVALRY LEARNING
   Do teams learn more from division rivals?

Key insight: If all these are null, learning may happen through:
- Analytics departments (not game observation)
- Media narratives (not direct experience)
- Front office pressure (not coach learning)
- Selection (new coaches, not same coaches learning)
""")


if __name__ == "__main__":
    main()
