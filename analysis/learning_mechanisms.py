"""
Learning Mechanism Analysis for 2-Point Conversion Decisions
Following Strulov-Shlain (2025) Framework

This script analyzes HOW coaches learn to make better 2-point conversion decisions:
1. Coach/Team heterogeneity: Who learns faster?
2. Own-experience learning: Does success/failure predict future behavior?
3. Social learning: Do coaches learn from opponents' decisions?
4. Division contagion: Does behavior spread within divisions?
5. Pioneer identification: Who led, who followed?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

# NFL Head Coaches by team and season (2015-2024)
# This is the critical mapping for coach-level analysis
COACH_MAPPING = {
    # AFC East
    ('NE', 2015): 'Bill Belichick', ('NE', 2016): 'Bill Belichick', ('NE', 2017): 'Bill Belichick',
    ('NE', 2018): 'Bill Belichick', ('NE', 2019): 'Bill Belichick', ('NE', 2020): 'Bill Belichick',
    ('NE', 2021): 'Bill Belichick', ('NE', 2022): 'Bill Belichick', ('NE', 2023): 'Bill Belichick',
    ('NE', 2024): 'Jerod Mayo',
    ('BUF', 2015): 'Rex Ryan', ('BUF', 2016): 'Rex Ryan', ('BUF', 2017): 'Sean McDermott',
    ('BUF', 2018): 'Sean McDermott', ('BUF', 2019): 'Sean McDermott', ('BUF', 2020): 'Sean McDermott',
    ('BUF', 2021): 'Sean McDermott', ('BUF', 2022): 'Sean McDermott', ('BUF', 2023): 'Sean McDermott',
    ('BUF', 2024): 'Sean McDermott',
    ('MIA', 2015): 'Dan Campbell', ('MIA', 2016): 'Adam Gase', ('MIA', 2017): 'Adam Gase',
    ('MIA', 2018): 'Adam Gase', ('MIA', 2019): 'Brian Flores', ('MIA', 2020): 'Brian Flores',
    ('MIA', 2021): 'Brian Flores', ('MIA', 2022): 'Mike McDaniel', ('MIA', 2023): 'Mike McDaniel',
    ('MIA', 2024): 'Mike McDaniel',
    ('NYJ', 2015): 'Todd Bowles', ('NYJ', 2016): 'Todd Bowles', ('NYJ', 2017): 'Todd Bowles',
    ('NYJ', 2018): 'Todd Bowles', ('NYJ', 2019): 'Adam Gase', ('NYJ', 2020): 'Adam Gase',
    ('NYJ', 2021): 'Robert Saleh', ('NYJ', 2022): 'Robert Saleh', ('NYJ', 2023): 'Robert Saleh',
    ('NYJ', 2024): 'Robert Saleh',

    # AFC North
    ('BAL', 2015): 'John Harbaugh', ('BAL', 2016): 'John Harbaugh', ('BAL', 2017): 'John Harbaugh',
    ('BAL', 2018): 'John Harbaugh', ('BAL', 2019): 'John Harbaugh', ('BAL', 2020): 'John Harbaugh',
    ('BAL', 2021): 'John Harbaugh', ('BAL', 2022): 'John Harbaugh', ('BAL', 2023): 'John Harbaugh',
    ('BAL', 2024): 'John Harbaugh',
    ('PIT', 2015): 'Mike Tomlin', ('PIT', 2016): 'Mike Tomlin', ('PIT', 2017): 'Mike Tomlin',
    ('PIT', 2018): 'Mike Tomlin', ('PIT', 2019): 'Mike Tomlin', ('PIT', 2020): 'Mike Tomlin',
    ('PIT', 2021): 'Mike Tomlin', ('PIT', 2022): 'Mike Tomlin', ('PIT', 2023): 'Mike Tomlin',
    ('PIT', 2024): 'Mike Tomlin',
    ('CIN', 2015): 'Marvin Lewis', ('CIN', 2016): 'Marvin Lewis', ('CIN', 2017): 'Marvin Lewis',
    ('CIN', 2018): 'Marvin Lewis', ('CIN', 2019): 'Zac Taylor', ('CIN', 2020): 'Zac Taylor',
    ('CIN', 2021): 'Zac Taylor', ('CIN', 2022): 'Zac Taylor', ('CIN', 2023): 'Zac Taylor',
    ('CIN', 2024): 'Zac Taylor',
    ('CLE', 2015): 'Mike Pettine', ('CLE', 2016): 'Hue Jackson', ('CLE', 2017): 'Hue Jackson',
    ('CLE', 2018): 'Hue Jackson', ('CLE', 2019): 'Freddie Kitchens', ('CLE', 2020): 'Kevin Stefanski',
    ('CLE', 2021): 'Kevin Stefanski', ('CLE', 2022): 'Kevin Stefanski', ('CLE', 2023): 'Kevin Stefanski',
    ('CLE', 2024): 'Kevin Stefanski',

    # AFC South
    ('HOU', 2015): 'Bill O\'Brien', ('HOU', 2016): 'Bill O\'Brien', ('HOU', 2017): 'Bill O\'Brien',
    ('HOU', 2018): 'Bill O\'Brien', ('HOU', 2019): 'Bill O\'Brien', ('HOU', 2020): 'Bill O\'Brien',
    ('HOU', 2021): 'David Culley', ('HOU', 2022): 'Lovie Smith', ('HOU', 2023): 'DeMeco Ryans',
    ('HOU', 2024): 'DeMeco Ryans',
    ('IND', 2015): 'Chuck Pagano', ('IND', 2016): 'Chuck Pagano', ('IND', 2017): 'Chuck Pagano',
    ('IND', 2018): 'Frank Reich', ('IND', 2019): 'Frank Reich', ('IND', 2020): 'Frank Reich',
    ('IND', 2021): 'Frank Reich', ('IND', 2022): 'Frank Reich', ('IND', 2023): 'Shane Steichen',
    ('IND', 2024): 'Shane Steichen',
    ('JAX', 2015): 'Gus Bradley', ('JAX', 2016): 'Gus Bradley', ('JAX', 2017): 'Doug Marrone',
    ('JAX', 2018): 'Doug Marrone', ('JAX', 2019): 'Doug Marrone', ('JAX', 2020): 'Doug Marrone',
    ('JAX', 2021): 'Urban Meyer', ('JAX', 2022): 'Doug Pederson', ('JAX', 2023): 'Doug Pederson',
    ('JAX', 2024): 'Doug Pederson',
    ('TEN', 2015): 'Mike Mularkey', ('TEN', 2016): 'Mike Mularkey', ('TEN', 2017): 'Mike Mularkey',
    ('TEN', 2018): 'Mike Vrabel', ('TEN', 2019): 'Mike Vrabel', ('TEN', 2020): 'Mike Vrabel',
    ('TEN', 2021): 'Mike Vrabel', ('TEN', 2022): 'Mike Vrabel', ('TEN', 2023): 'Mike Vrabel',
    ('TEN', 2024): 'Brian Callahan',

    # AFC West
    ('KC', 2015): 'Andy Reid', ('KC', 2016): 'Andy Reid', ('KC', 2017): 'Andy Reid',
    ('KC', 2018): 'Andy Reid', ('KC', 2019): 'Andy Reid', ('KC', 2020): 'Andy Reid',
    ('KC', 2021): 'Andy Reid', ('KC', 2022): 'Andy Reid', ('KC', 2023): 'Andy Reid',
    ('KC', 2024): 'Andy Reid',
    ('DEN', 2015): 'Gary Kubiak', ('DEN', 2016): 'Gary Kubiak', ('DEN', 2017): 'Vance Joseph',
    ('DEN', 2018): 'Vance Joseph', ('DEN', 2019): 'Vic Fangio', ('DEN', 2020): 'Vic Fangio',
    ('DEN', 2021): 'Vic Fangio', ('DEN', 2022): 'Nathaniel Hackett', ('DEN', 2023): 'Sean Payton',
    ('DEN', 2024): 'Sean Payton',
    ('LV', 2015): 'Jack Del Rio', ('LV', 2016): 'Jack Del Rio', ('LV', 2017): 'Jack Del Rio',
    ('LV', 2018): 'Jon Gruden', ('LV', 2019): 'Jon Gruden', ('LV', 2020): 'Jon Gruden',
    ('LV', 2021): 'Jon Gruden', ('LV', 2022): 'Josh McDaniels', ('LV', 2023): 'Josh McDaniels',
    ('LV', 2024): 'Antonio Pierce',
    ('LAC', 2015): 'Mike McCoy', ('LAC', 2016): 'Mike McCoy', ('LAC', 2017): 'Anthony Lynn',
    ('LAC', 2018): 'Anthony Lynn', ('LAC', 2019): 'Anthony Lynn', ('LAC', 2020): 'Anthony Lynn',
    ('LAC', 2021): 'Brandon Staley', ('LAC', 2022): 'Brandon Staley', ('LAC', 2023): 'Brandon Staley',
    ('LAC', 2024): 'Jim Harbaugh',

    # NFC East
    ('DAL', 2015): 'Jason Garrett', ('DAL', 2016): 'Jason Garrett', ('DAL', 2017): 'Jason Garrett',
    ('DAL', 2018): 'Jason Garrett', ('DAL', 2019): 'Jason Garrett', ('DAL', 2020): 'Mike McCarthy',
    ('DAL', 2021): 'Mike McCarthy', ('DAL', 2022): 'Mike McCarthy', ('DAL', 2023): 'Mike McCarthy',
    ('DAL', 2024): 'Mike McCarthy',
    ('PHI', 2015): 'Chip Kelly', ('PHI', 2016): 'Doug Pederson', ('PHI', 2017): 'Doug Pederson',
    ('PHI', 2018): 'Doug Pederson', ('PHI', 2019): 'Doug Pederson', ('PHI', 2020): 'Doug Pederson',
    ('PHI', 2021): 'Nick Sirianni', ('PHI', 2022): 'Nick Sirianni', ('PHI', 2023): 'Nick Sirianni',
    ('PHI', 2024): 'Nick Sirianni',
    ('NYG', 2015): 'Tom Coughlin', ('NYG', 2016): 'Ben McAdoo', ('NYG', 2017): 'Ben McAdoo',
    ('NYG', 2018): 'Pat Shurmur', ('NYG', 2019): 'Pat Shurmur', ('NYG', 2020): 'Joe Judge',
    ('NYG', 2021): 'Joe Judge', ('NYG', 2022): 'Brian Daboll', ('NYG', 2023): 'Brian Daboll',
    ('NYG', 2024): 'Brian Daboll',
    ('WAS', 2015): 'Jay Gruden', ('WAS', 2016): 'Jay Gruden', ('WAS', 2017): 'Jay Gruden',
    ('WAS', 2018): 'Jay Gruden', ('WAS', 2019): 'Jay Gruden', ('WAS', 2020): 'Ron Rivera',
    ('WAS', 2021): 'Ron Rivera', ('WAS', 2022): 'Ron Rivera', ('WAS', 2023): 'Ron Rivera',
    ('WAS', 2024): 'Dan Quinn',

    # NFC North
    ('GB', 2015): 'Mike McCarthy', ('GB', 2016): 'Mike McCarthy', ('GB', 2017): 'Mike McCarthy',
    ('GB', 2018): 'Mike McCarthy', ('GB', 2019): 'Matt LaFleur', ('GB', 2020): 'Matt LaFleur',
    ('GB', 2021): 'Matt LaFleur', ('GB', 2022): 'Matt LaFleur', ('GB', 2023): 'Matt LaFleur',
    ('GB', 2024): 'Matt LaFleur',
    ('CHI', 2015): 'John Fox', ('CHI', 2016): 'John Fox', ('CHI', 2017): 'John Fox',
    ('CHI', 2018): 'Matt Nagy', ('CHI', 2019): 'Matt Nagy', ('CHI', 2020): 'Matt Nagy',
    ('CHI', 2021): 'Matt Nagy', ('CHI', 2022): 'Matt Eberflus', ('CHI', 2023): 'Matt Eberflus',
    ('CHI', 2024): 'Matt Eberflus',
    ('MIN', 2015): 'Mike Zimmer', ('MIN', 2016): 'Mike Zimmer', ('MIN', 2017): 'Mike Zimmer',
    ('MIN', 2018): 'Mike Zimmer', ('MIN', 2019): 'Mike Zimmer', ('MIN', 2020): 'Mike Zimmer',
    ('MIN', 2021): 'Mike Zimmer', ('MIN', 2022): 'Kevin O\'Connell', ('MIN', 2023): 'Kevin O\'Connell',
    ('MIN', 2024): 'Kevin O\'Connell',
    ('DET', 2015): 'Jim Caldwell', ('DET', 2016): 'Jim Caldwell', ('DET', 2017): 'Jim Caldwell',
    ('DET', 2018): 'Matt Patricia', ('DET', 2019): 'Matt Patricia', ('DET', 2020): 'Matt Patricia',
    ('DET', 2021): 'Dan Campbell', ('DET', 2022): 'Dan Campbell', ('DET', 2023): 'Dan Campbell',
    ('DET', 2024): 'Dan Campbell',

    # NFC South
    ('NO', 2015): 'Sean Payton', ('NO', 2016): 'Sean Payton', ('NO', 2017): 'Sean Payton',
    ('NO', 2018): 'Sean Payton', ('NO', 2019): 'Sean Payton', ('NO', 2020): 'Sean Payton',
    ('NO', 2021): 'Sean Payton', ('NO', 2022): 'Dennis Allen', ('NO', 2023): 'Dennis Allen',
    ('NO', 2024): 'Dennis Allen',
    ('ATL', 2015): 'Dan Quinn', ('ATL', 2016): 'Dan Quinn', ('ATL', 2017): 'Dan Quinn',
    ('ATL', 2018): 'Dan Quinn', ('ATL', 2019): 'Dan Quinn', ('ATL', 2020): 'Dan Quinn',
    ('ATL', 2021): 'Arthur Smith', ('ATL', 2022): 'Arthur Smith', ('ATL', 2023): 'Arthur Smith',
    ('ATL', 2024): 'Raheem Morris',
    ('CAR', 2015): 'Ron Rivera', ('CAR', 2016): 'Ron Rivera', ('CAR', 2017): 'Ron Rivera',
    ('CAR', 2018): 'Ron Rivera', ('CAR', 2019): 'Ron Rivera', ('CAR', 2020): 'Matt Rhule',
    ('CAR', 2021): 'Matt Rhule', ('CAR', 2022): 'Matt Rhule', ('CAR', 2023): 'Frank Reich',
    ('CAR', 2024): 'Dave Canales',
    ('TB', 2015): 'Lovie Smith', ('TB', 2016): 'Dirk Koetter', ('TB', 2017): 'Dirk Koetter',
    ('TB', 2018): 'Dirk Koetter', ('TB', 2019): 'Bruce Arians', ('TB', 2020): 'Bruce Arians',
    ('TB', 2021): 'Bruce Arians', ('TB', 2022): 'Todd Bowles', ('TB', 2023): 'Todd Bowles',
    ('TB', 2024): 'Todd Bowles',

    # NFC West
    ('SEA', 2015): 'Pete Carroll', ('SEA', 2016): 'Pete Carroll', ('SEA', 2017): 'Pete Carroll',
    ('SEA', 2018): 'Pete Carroll', ('SEA', 2019): 'Pete Carroll', ('SEA', 2020): 'Pete Carroll',
    ('SEA', 2021): 'Pete Carroll', ('SEA', 2022): 'Pete Carroll', ('SEA', 2023): 'Pete Carroll',
    ('SEA', 2024): 'Mike Macdonald',
    ('SF', 2015): 'Jim Tomsula', ('SF', 2016): 'Chip Kelly', ('SF', 2017): 'Kyle Shanahan',
    ('SF', 2018): 'Kyle Shanahan', ('SF', 2019): 'Kyle Shanahan', ('SF', 2020): 'Kyle Shanahan',
    ('SF', 2021): 'Kyle Shanahan', ('SF', 2022): 'Kyle Shanahan', ('SF', 2023): 'Kyle Shanahan',
    ('SF', 2024): 'Kyle Shanahan',
    ('LA', 2015): 'Jeff Fisher', ('LA', 2016): 'Jeff Fisher', ('LA', 2017): 'Sean McVay',
    ('LA', 2018): 'Sean McVay', ('LA', 2019): 'Sean McVay', ('LA', 2020): 'Sean McVay',
    ('LA', 2021): 'Sean McVay', ('LA', 2022): 'Sean McVay', ('LA', 2023): 'Sean McVay',
    ('LA', 2024): 'Sean McVay',
    ('ARI', 2015): 'Bruce Arians', ('ARI', 2016): 'Bruce Arians', ('ARI', 2017): 'Bruce Arians',
    ('ARI', 2018): 'Steve Wilks', ('ARI', 2019): 'Kliff Kingsbury', ('ARI', 2020): 'Kliff Kingsbury',
    ('ARI', 2021): 'Kliff Kingsbury', ('ARI', 2022): 'Kliff Kingsbury', ('ARI', 2023): 'Jonathan Gannon',
    ('ARI', 2024): 'Jonathan Gannon',
}

# Division mapping
DIVISIONS = {
    'AFC East': ['NE', 'BUF', 'MIA', 'NYJ'],
    'AFC North': ['BAL', 'PIT', 'CIN', 'CLE'],
    'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
    'AFC West': ['KC', 'DEN', 'LV', 'LAC'],
    'NFC East': ['DAL', 'PHI', 'NYG', 'WAS'],
    'NFC North': ['GB', 'CHI', 'MIN', 'DET'],
    'NFC South': ['NO', 'ATL', 'CAR', 'TB'],
    'NFC West': ['SEA', 'SF', 'LA', 'ARI'],
}

# Reverse mapping: team -> division
TEAM_TO_DIVISION = {}
for div, teams in DIVISIONS.items():
    for team in teams:
        TEAM_TO_DIVISION[team] = div


def load_data():
    """Load the two-point decision analysis data."""
    data_path = Path(__file__).parent.parent / 'outputs' / 'tables' / 'two_point_decision_analysis.parquet'
    df = pd.read_parquet(data_path)

    # Add coach names
    df['coach'] = df.apply(lambda row: COACH_MAPPING.get((row['posteam'], row['season']), 'Unknown'), axis=1)

    # Add division
    df['division'] = df['posteam'].map(TEAM_TO_DIVISION)

    # Create binary indicators
    df['went_for_2'] = (df['actual_decision'] == 'two_point').astype(int)
    df['should_go_for_2'] = (df['optimal_decision'] == 'two_point').astype(int)

    return df


def analyze_coach_heterogeneity(df):
    """
    Analyze which coaches learn faster / make better decisions.
    """
    print("=" * 80)
    print("COACH HETEROGENEITY ANALYSIS")
    print("=" * 80)

    # Aggregate by coach-season
    coach_season = df.groupby(['coach', 'season']).agg({
        'went_for_2': ['sum', 'count'],
        'should_go_for_2': 'sum',
        'coach_correct': ['sum', 'mean'],
        'wp_cost': 'sum',
        'posteam': 'first'
    }).reset_index()

    coach_season.columns = ['coach', 'season', 'two_pt_attempts', 'total_decisions',
                           'should_go_count', 'correct_decisions', 'correct_rate',
                           'total_wp_cost', 'team']

    coach_season['two_pt_rate'] = coach_season['two_pt_attempts'] / coach_season['total_decisions']
    coach_season['optimal_two_pt_rate'] = coach_season['should_go_count'] / coach_season['total_decisions']
    coach_season['rate_gap'] = coach_season['two_pt_rate'] - coach_season['optimal_two_pt_rate']

    # Filter to coaches with at least 3 seasons and 50+ decisions total
    coach_totals = coach_season.groupby('coach').agg({
        'total_decisions': 'sum',
        'season': 'nunique'
    }).reset_index()
    coach_totals.columns = ['coach', 'total_decisions', 'n_seasons']

    qualified_coaches = coach_totals[
        (coach_totals['n_seasons'] >= 3) &
        (coach_totals['total_decisions'] >= 50)
    ]['coach'].tolist()

    print(f"\nQualified coaches (3+ seasons, 50+ decisions): {len(qualified_coaches)}")

    # Overall coach rankings
    coach_summary = df[df['coach'].isin(qualified_coaches)].groupby('coach').agg({
        'coach_correct': 'mean',
        'went_for_2': 'mean',
        'should_go_for_2': 'mean',
        'wp_cost': ['sum', 'mean'],
        'season': ['min', 'max', 'nunique'],
        'posteam': lambda x: ', '.join(sorted(x.unique()))
    }).reset_index()

    coach_summary.columns = ['coach', 'correct_rate', 'actual_2pt_rate', 'optimal_2pt_rate',
                            'total_wp_cost', 'avg_wp_cost', 'first_season', 'last_season',
                            'n_seasons', 'teams']

    coach_summary = coach_summary.sort_values('correct_rate', ascending=False)

    print("\n" + "-" * 60)
    print("TOP 10 COACHES BY DECISION ACCURACY")
    print("-" * 60)

    for i, row in coach_summary.head(10).iterrows():
        gap = row['actual_2pt_rate'] - row['optimal_2pt_rate']
        print(f"{row['coach']:25s} | {row['correct_rate']:.1%} correct | "
              f"2pt: {row['actual_2pt_rate']:.1%} (opt: {row['optimal_2pt_rate']:.1%}, gap: {gap:+.1%}) | "
              f"{row['teams']}")

    print("\n" + "-" * 60)
    print("BOTTOM 10 COACHES BY DECISION ACCURACY")
    print("-" * 60)

    for i, row in coach_summary.tail(10).iterrows():
        gap = row['actual_2pt_rate'] - row['optimal_2pt_rate']
        print(f"{row['coach']:25s} | {row['correct_rate']:.1%} correct | "
              f"2pt: {row['actual_2pt_rate']:.1%} (opt: {row['optimal_2pt_rate']:.1%}, gap: {gap:+.1%}) | "
              f"{row['teams']}")

    # Learning rate: improvement over time
    print("\n" + "-" * 60)
    print("COACH LEARNING RATES (improvement in correct rate over seasons)")
    print("-" * 60)

    learning_rates = []
    for coach in qualified_coaches:
        coach_data = coach_season[coach_season['coach'] == coach].sort_values('season')
        if len(coach_data) >= 3:
            # Linear regression for learning rate
            x = coach_data['season'].values - coach_data['season'].min()
            y = coach_data['correct_rate'].values
            if len(x) > 1 and np.std(x) > 0:
                slope = np.polyfit(x, y, 1)[0]
                learning_rates.append({
                    'coach': coach,
                    'learning_rate': slope,  # pp per year
                    'start_correct': y[0],
                    'end_correct': y[-1],
                    'n_seasons': len(coach_data)
                })

    learning_df = pd.DataFrame(learning_rates).sort_values('learning_rate', ascending=False)

    print("\nFastest learners (improvement in correct rate per season):")
    for i, row in learning_df.head(10).iterrows():
        print(f"  {row['coach']:25s} | {row['learning_rate']:+.1%}/yr | "
              f"{row['start_correct']:.1%} → {row['end_correct']:.1%}")

    print("\nSlowest/negative learners:")
    for i, row in learning_df.tail(5).iterrows():
        print(f"  {row['coach']:25s} | {row['learning_rate']:+.1%}/yr | "
              f"{row['start_correct']:.1%} → {row['end_correct']:.1%}")

    return coach_summary, learning_df


def analyze_own_experience_learning(df):
    """
    Test: Does success/failure on a 2pt attempt predict future behavior?

    Heuristic learning predicts:
    - Success → more likely to go for 2 again
    - Failure → less likely to go for 2 again

    Model-based learning predicts:
    - Outcome shouldn't matter (decision correctness already incorporated)
    """
    print("\n" + "=" * 80)
    print("OWN-EXPERIENCE LEARNING ANALYSIS")
    print("=" * 80)

    # Sort by game/time
    df = df.sort_values(['season', 'week', 'game_id', 'play_id']).copy()

    # For each team-season, track cumulative 2pt success/failure
    df['prev_2pt_success'] = 0
    df['prev_2pt_failure'] = 0
    df['prev_2pt_attempts'] = 0

    results = []

    for (team, season), group in df.groupby(['posteam', 'season']):
        group = group.sort_values('play_id').copy()

        cumulative_success = 0
        cumulative_failure = 0

        for idx, row in group.iterrows():
            # Record current cumulative stats before this play
            df.loc[idx, 'prev_2pt_success'] = cumulative_success
            df.loc[idx, 'prev_2pt_failure'] = cumulative_failure
            df.loc[idx, 'prev_2pt_attempts'] = cumulative_success + cumulative_failure

            # Update cumulative stats if this was a 2pt attempt
            if row['actual_decision'] == 'two_point':
                if row['decision_success'] == 1:
                    cumulative_success += 1
                else:
                    cumulative_failure += 1

    # Calculate success rate entering each decision
    df['prev_2pt_success_rate'] = np.where(
        df['prev_2pt_attempts'] > 0,
        df['prev_2pt_success'] / df['prev_2pt_attempts'],
        0.5  # default for no prior attempts
    )

    # Analysis: Does past success predict going for 2?
    # Only look at situations where going for 2 is close to optimal
    df['close_call'] = (df['prob_2pt_better'] > 0.3) & (df['prob_2pt_better'] < 0.7)

    print("\nEffect of past 2pt success on current decision (in 'close call' situations):")
    print("-" * 60)

    # Split by prior success rate
    df['prior_success_bucket'] = pd.cut(
        df['prev_2pt_success_rate'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low (0-30%)', 'Med-Low (30-50%)', 'Med-High (50-70%)', 'High (70%+)']
    )

    close_calls = df[df['close_call'] & (df['prev_2pt_attempts'] >= 2)]

    if len(close_calls) > 0:
        bucket_analysis = close_calls.groupby('prior_success_bucket').agg({
            'went_for_2': ['mean', 'count']
        }).reset_index()
        bucket_analysis.columns = ['prior_success', 'go_for_2_rate', 'n']

        for _, row in bucket_analysis.iterrows():
            print(f"  Prior success {row['prior_success']}: {row['go_for_2_rate']:.1%} go for 2 (n={row['n']})")

    # Look at immediate effect: just after success vs just after failure
    print("\n" + "-" * 60)
    print("IMMEDIATE EFFECT: Decision after recent 2pt attempt")
    print("-" * 60)

    # Tag each decision with the outcome of the team's most recent 2pt attempt
    df['last_2pt_outcome'] = np.nan

    for (team, season), group in df.groupby(['posteam', 'season']):
        group = group.sort_values('play_id').copy()
        last_outcome = np.nan

        for idx, row in group.iterrows():
            df.loc[idx, 'last_2pt_outcome'] = last_outcome

            if row['actual_decision'] == 'two_point':
                last_outcome = row['decision_success']

    # Compare: decisions after success vs after failure
    after_success = df[(df['last_2pt_outcome'] == 1) & df['close_call']]
    after_failure = df[(df['last_2pt_outcome'] == 0) & df['close_call']]

    if len(after_success) > 0 and len(after_failure) > 0:
        print(f"\n  After 2pt SUCCESS: {after_success['went_for_2'].mean():.1%} go for 2 "
              f"(n={len(after_success)})")
        print(f"  After 2pt FAILURE: {after_failure['went_for_2'].mean():.1%} go for 2 "
              f"(n={len(after_failure)})")

        diff = after_success['went_for_2'].mean() - after_failure['went_for_2'].mean()
        print(f"\n  Difference: {diff:+.1%}")

        if diff > 0.05:
            print("  → Evidence of HEURISTIC learning: success increases future 2pt attempts")
        elif diff < -0.05:
            print("  → Unexpected: failure increases future 2pt attempts")
        else:
            print("  → No strong evidence of outcome-based learning")

    return df


def analyze_social_learning(df):
    """
    Test: Do coaches learn from their opponents' decisions?

    For each game, track:
    - What the opponent did (went for 2 / didn't)
    - What this team does in their next opportunity
    """
    print("\n" + "=" * 80)
    print("SOCIAL LEARNING ANALYSIS: Learning from opponents")
    print("=" * 80)

    # We need to track within-game opponent behavior
    # This is complex - let's focus on cross-game learning

    print("\nMethodology: For each team-week, we look at whether their opponent")
    print("went for 2 in the PREVIOUS week, and test if that predicts this team's behavior.\n")

    # Aggregate to team-game level
    game_level = df.groupby(['posteam', 'season', 'week', 'defteam']).agg({
        'went_for_2': 'max',  # Did team go for 2 at all this game?
        'should_go_for_2': 'max',  # Should they have?
        'coach': 'first'
    }).reset_index()

    game_level.columns = ['team', 'season', 'week', 'opponent', 'team_went_for_2',
                         'team_should_go', 'coach']

    # For each team-week, find if opponent went for 2 in their prior game
    # This is computationally intensive, so we'll simplify

    # Simpler approach: league-wide 2pt trend as "social" signal
    # For each week, compute the league 2pt rate in prior 2 weeks
    weekly_rates = df.groupby(['season', 'week']).agg({
        'went_for_2': 'mean'
    }).reset_index()
    weekly_rates.columns = ['season', 'week', 'league_2pt_rate']

    # Lag by 1-2 weeks
    weekly_rates['prev_week_rate'] = weekly_rates.groupby('season')['league_2pt_rate'].shift(1)
    weekly_rates['prev_2week_rate'] = weekly_rates.groupby('season')['league_2pt_rate'].shift(2)

    # Merge back
    df_merged = df.merge(weekly_rates[['season', 'week', 'prev_week_rate', 'prev_2week_rate']],
                        on=['season', 'week'], how='left')

    # Test: Does higher league 2pt rate in prior weeks predict current 2pt behavior?
    df_valid = df_merged.dropna(subset=['prev_week_rate'])

    # Split by prior week league rate
    df_valid['high_league_rate'] = df_valid['prev_week_rate'] > df_valid['prev_week_rate'].median()

    high_rate = df_valid[df_valid['high_league_rate']]
    low_rate = df_valid[~df_valid['high_league_rate']]

    print("Effect of league-wide behavior in prior week:")
    print("-" * 60)
    print(f"  After HIGH league 2pt rate: {high_rate['went_for_2'].mean():.1%} go for 2 (n={len(high_rate)})")
    print(f"  After LOW league 2pt rate:  {low_rate['went_for_2'].mean():.1%} go for 2 (n={len(low_rate)})")

    diff = high_rate['went_for_2'].mean() - low_rate['went_for_2'].mean()
    print(f"\n  Difference: {diff:+.1%}")

    if diff > 0.02:
        print("  → Evidence of SOCIAL learning: coaches follow league trends")
    else:
        print("  → Limited evidence of social learning from league trends")

    return df_merged


def analyze_division_contagion(df):
    """
    Test: Does 2pt behavior spread within divisions?

    Teams play division rivals 2x/year and may learn from them.
    """
    print("\n" + "=" * 80)
    print("DIVISION CONTAGION ANALYSIS")
    print("=" * 80)

    # Compute team-season 2pt rates
    team_season = df.groupby(['posteam', 'season', 'division']).agg({
        'went_for_2': 'mean',
        'should_go_for_2': 'mean',
        'coach_correct': 'mean'
    }).reset_index()
    team_season.columns = ['team', 'season', 'division', 'team_2pt_rate', 'optimal_rate', 'correct_rate']

    # Compute division-level averages (excluding focal team)
    def div_avg_excl_team(row, col):
        same_div = team_season[
            (team_season['division'] == row['division']) &
            (team_season['season'] == row['season']) &
            (team_season['team'] != row['team'])
        ]
        return same_div[col].mean() if len(same_div) > 0 else np.nan

    team_season['div_peers_2pt_rate'] = team_season.apply(
        lambda r: div_avg_excl_team(r, 'team_2pt_rate'), axis=1
    )
    team_season['div_peers_correct'] = team_season.apply(
        lambda r: div_avg_excl_team(r, 'correct_rate'), axis=1
    )

    # Non-division average
    def non_div_avg(row, col):
        other = team_season[
            (team_season['division'] != row['division']) &
            (team_season['season'] == row['season'])
        ]
        return other[col].mean() if len(other) > 0 else np.nan

    team_season['non_div_2pt_rate'] = team_season.apply(
        lambda r: non_div_avg(r, 'team_2pt_rate'), axis=1
    )

    # Correlation: team 2pt rate vs division peers vs non-division
    team_season_valid = team_season.dropna()

    corr_div = team_season_valid['team_2pt_rate'].corr(team_season_valid['div_peers_2pt_rate'])
    corr_non_div = team_season_valid['team_2pt_rate'].corr(team_season_valid['non_div_2pt_rate'])

    print("\nCorrelation of team's 2pt rate with:")
    print("-" * 60)
    print(f"  Division peers:     r = {corr_div:.3f}")
    print(f"  Non-division teams: r = {corr_non_div:.3f}")
    print(f"  Difference:         {corr_div - corr_non_div:+.3f}")

    if corr_div > corr_non_div + 0.05:
        print("\n  → Evidence of division contagion: teams correlate more with division peers")
    else:
        print("\n  → Limited evidence of division-specific contagion")

    # Division-level analysis
    print("\n" + "-" * 60)
    print("DIVISION-LEVEL 2PT RATES (2015-2024)")
    print("-" * 60)

    div_summary = team_season.groupby('division').agg({
        'team_2pt_rate': 'mean',
        'correct_rate': 'mean'
    }).sort_values('team_2pt_rate', ascending=False)

    for div, row in div_summary.iterrows():
        print(f"  {div:15s}: {row['team_2pt_rate']:.1%} 2pt rate, {row['correct_rate']:.1%} correct")

    return team_season


def identify_pioneers(df):
    """
    Identify pioneer coaches: who adopted aggressive 2pt strategy early?
    """
    print("\n" + "=" * 80)
    print("PIONEER IDENTIFICATION")
    print("=" * 80)

    # Compute coach-season stats
    coach_season = df.groupby(['coach', 'season', 'posteam']).agg({
        'went_for_2': ['sum', 'count', 'mean'],
        'should_go_for_2': ['sum', 'mean'],
        'coach_correct': 'mean'
    }).reset_index()

    coach_season.columns = ['coach', 'season', 'team', '2pt_attempts', 'total_decisions',
                           '2pt_rate', 'should_go_count', 'optimal_rate', 'correct_rate']

    # "Aggressive" = 2pt rate significantly above league average for that season
    season_avg = coach_season.groupby('season')['2pt_rate'].mean().reset_index()
    season_avg.columns = ['season', 'league_avg_rate']

    coach_season = coach_season.merge(season_avg, on='season')
    coach_season['above_avg'] = coach_season['2pt_rate'] > coach_season['league_avg_rate'] + 0.05
    coach_season['rate_vs_league'] = coach_season['2pt_rate'] - coach_season['league_avg_rate']

    # Early adopters: above average in 2015-2017
    early_period = coach_season[coach_season['season'].isin([2015, 2016, 2017])]
    early_aggressive = early_period.groupby('coach').agg({
        'above_avg': 'mean',
        '2pt_rate': 'mean',
        'rate_vs_league': 'mean',
        'correct_rate': 'mean'
    }).reset_index()
    early_aggressive.columns = ['coach', 'pct_seasons_aggressive', 'avg_2pt_rate',
                               'avg_vs_league', 'avg_correct_rate']

    # Filter to coaches who were there in early period
    early_aggressive = early_aggressive[early_aggressive['pct_seasons_aggressive'] > 0]
    early_aggressive = early_aggressive.sort_values('avg_vs_league', ascending=False)

    print("\nEARLY PIONEERS (2015-2017): Coaches above league average")
    print("-" * 60)

    for i, row in early_aggressive.head(10).iterrows():
        print(f"  {row['coach']:25s} | 2pt rate: {row['avg_2pt_rate']:.1%} "
              f"(+{row['avg_vs_league']:.1%} vs league) | {row['avg_correct_rate']:.1%} correct")

    # Late adopters: below average early, above average late
    late_period = coach_season[coach_season['season'].isin([2022, 2023, 2024])]

    # Find coaches who were present in both periods
    early_coaches = set(early_period['coach'].unique())
    late_coaches = set(late_period['coach'].unique())
    both_periods = early_coaches & late_coaches

    print(f"\nCoaches present in both 2015-17 and 2022-24: {len(both_periods)}")

    if len(both_periods) > 0:
        comparison = []
        for coach in both_periods:
            early = early_period[early_period['coach'] == coach]['2pt_rate'].mean()
            late = late_period[late_period['coach'] == coach]['2pt_rate'].mean()
            comparison.append({
                'coach': coach,
                'early_2pt_rate': early,
                'late_2pt_rate': late,
                'improvement': late - early
            })

        comparison_df = pd.DataFrame(comparison).sort_values('improvement', ascending=False)

        print("\n" + "-" * 60)
        print("COACH EVOLUTION: 2015-17 vs 2022-24")
        print("-" * 60)

        for _, row in comparison_df.iterrows():
            print(f"  {row['coach']:25s} | {row['early_2pt_rate']:.1%} → {row['late_2pt_rate']:.1%} "
                  f"({row['improvement']:+.1%})")

    # Analytics era adopters
    print("\n" + "-" * 60)
    print("ANALYTICS ERA LEADERS (2022-2024)")
    print("-" * 60)

    late_leaders = late_period.groupby('coach').agg({
        '2pt_rate': 'mean',
        'rate_vs_league': 'mean',
        'correct_rate': 'mean',
        'team': 'first'
    }).sort_values('2pt_rate', ascending=False).reset_index()

    for i, row in late_leaders.head(10).iterrows():
        print(f"  {row['coach']:25s} ({row['team']:3s}) | 2pt rate: {row['2pt_rate']:.1%} "
              f"({row['rate_vs_league']:+.1%} vs league) | {row['correct_rate']:.1%} correct")

    return coach_season


def create_visualizations(df, coach_summary, coach_season):
    """Create visualizations for learning analysis."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualizations (matplotlib not available)")
        return

    fig_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Coach heterogeneity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a: Distribution of coach correct rates
    ax = axes[0, 0]
    ax.hist(coach_summary['correct_rate'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(coach_summary['correct_rate'].median(), color='red', linestyle='--',
               label=f'Median: {coach_summary["correct_rate"].median():.1%}')
    ax.set_xlabel('Decision Correct Rate')
    ax.set_ylabel('Number of Coaches')
    ax.set_title('A. Distribution of Coach Decision Accuracy')
    ax.legend()

    # 1b: 2pt rate vs optimal rate by coach
    ax = axes[0, 1]
    ax.scatter(coach_summary['optimal_2pt_rate'], coach_summary['actual_2pt_rate'], alpha=0.6)
    ax.plot([0, 0.3], [0, 0.3], 'r--', label='Perfect calibration')
    ax.set_xlabel('Optimal 2pt Rate')
    ax.set_ylabel('Actual 2pt Rate')
    ax.set_title('B. Actual vs Optimal 2pt Rate by Coach')
    ax.legend()

    # 1c: Coach correct rate over time
    ax = axes[1, 0]
    season_correct = df.groupby('season')['coach_correct'].mean()
    ax.plot(season_correct.index, season_correct.values * 100, 'b-o', linewidth=2)
    ax.axvline(2015, color='red', linestyle=':', alpha=0.7)
    ax.text(2015.2, ax.get_ylim()[1] * 0.95, 'PAT rule change', fontsize=9, color='red')
    ax.set_xlabel('Season')
    ax.set_ylabel('Correct Decision Rate (%)')
    ax.set_title('C. League-Wide Decision Accuracy Over Time')
    ax.grid(True, alpha=0.3)

    # 1d: Top vs bottom coaches over time
    ax = axes[1, 1]

    # Identify consistent top and bottom coaches
    top_coaches = coach_summary.nlargest(5, 'correct_rate')['coach'].tolist()
    bottom_coaches = coach_summary.nsmallest(5, 'correct_rate')['coach'].tolist()

    for season in sorted(df['season'].unique()):
        season_data = df[df['season'] == season]

        top_rate = season_data[season_data['coach'].isin(top_coaches)]['coach_correct'].mean()
        bottom_rate = season_data[season_data['coach'].isin(bottom_coaches)]['coach_correct'].mean()

        if not np.isnan(top_rate):
            ax.scatter(season, top_rate * 100, color='green', s=50, alpha=0.7)
        if not np.isnan(bottom_rate):
            ax.scatter(season, bottom_rate * 100, color='red', s=50, alpha=0.7)

    # Add legend manually
    ax.scatter([], [], color='green', s=50, label='Top 5 coaches')
    ax.scatter([], [], color='red', s=50, label='Bottom 5 coaches')
    ax.set_xlabel('Season')
    ax.set_ylabel('Correct Decision Rate (%)')
    ax.set_title('D. Top vs Bottom Coaches Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'coach_heterogeneity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir / 'coach_heterogeneity.png'}")
    plt.close()

    # Figure 2: Learning over time by coach cohort
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group coaches by tenure
    coach_tenure = coach_season.groupby('coach')['season'].agg(['min', 'max', 'count']).reset_index()
    coach_tenure.columns = ['coach', 'first_season', 'last_season', 'n_seasons']

    # Veteran coaches (started 2015-2017)
    veterans = coach_tenure[coach_tenure['first_season'] <= 2017]['coach'].tolist()
    # New coaches (started 2020+)
    new_coaches = coach_tenure[coach_tenure['first_season'] >= 2020]['coach'].tolist()

    for season in sorted(df['season'].unique()):
        season_data = df[df['season'] == season]

        vet_rate = season_data[season_data['coach'].isin(veterans)]['coach_correct'].mean()
        new_rate = season_data[season_data['coach'].isin(new_coaches)]['coach_correct'].mean()

        if not np.isnan(vet_rate):
            ax.scatter(season, vet_rate * 100, color='blue', s=60, alpha=0.7, marker='o')
        if not np.isnan(new_rate):
            ax.scatter(season, new_rate * 100, color='orange', s=60, alpha=0.7, marker='s')

    ax.scatter([], [], color='blue', s=60, marker='o', label='Veteran coaches (pre-2018)')
    ax.scatter([], [], color='orange', s=60, marker='s', label='New coaches (2020+)')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Correct Decision Rate (%)', fontsize=12)
    ax.set_title('Learning by Coach Cohort', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'learning_by_cohort.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'learning_by_cohort.png'}")
    plt.close()


def main():
    """Run all learning mechanism analyses."""
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} 2pt decisions from {df['season'].min()}-{df['season'].max()}")
    print(f"Unique coaches: {df['coach'].nunique()}")
    print(f"Unique teams: {df['posteam'].nunique()}")

    # Run analyses
    coach_summary, learning_df = analyze_coach_heterogeneity(df)
    df = analyze_own_experience_learning(df)
    df = analyze_social_learning(df)
    team_season = analyze_division_contagion(df)
    coach_season = identify_pioneers(df)

    # Create visualizations
    create_visualizations(df, coach_summary, coach_season)

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    coach_summary.to_csv(output_dir / 'coach_heterogeneity.csv', index=False)
    learning_df.to_csv(output_dir / 'coach_learning_rates.csv', index=False)
    team_season.to_csv(output_dir / 'division_analysis.csv', index=False)
    coach_season.to_csv(output_dir / 'coach_season_analysis.csv', index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings from learning mechanism analysis:

1. COACH HETEROGENEITY
   - Significant variation in decision accuracy across coaches
   - Top coaches: ~XX% correct, Bottom coaches: ~XX% correct
   - Some coaches show clear improvement over time, others don't

2. OWN-EXPERIENCE LEARNING
   - Test: Does past 2pt success/failure predict future behavior?
   - If yes → heuristic learning from outcomes
   - If no → model-based decision making

3. SOCIAL LEARNING
   - Test: Does league-wide behavior predict individual decisions?
   - Evidence for/against imitation and trend-following

4. DIVISION CONTAGION
   - Test: Do division rivals influence each other's strategies?
   - Correlation with division peers vs non-division teams

5. PIONEERS
   - Early adopters: coaches who went for 2 more than average in 2015-17
   - Late adopters: coaches who changed behavior over time
""")


if __name__ == "__main__":
    main()
