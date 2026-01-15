"""
Analyze specific fourth down decisions through the model.

This script allows us to run specific plays through our decision framework
to understand what the model recommends and why.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from analysis.decision_framework import (
    BayesianDecisionAnalyzer, GameState, load_models_with_off_def
)


def analyze_play(state: GameState, models_dir: Path = None) -> dict:
    """
    Analyze a specific fourth down situation.

    Returns detailed breakdown of expected WP for each action.
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / 'models'

    # Load models
    models = load_models_with_off_def(models_dir)
    analyzer = BayesianDecisionAnalyzer(models)

    # Run analysis
    result = analyzer.analyze(state)

    return {
        'state': state,
        'result': result,
        'analyzer': analyzer
    }


def print_analysis(analysis: dict):
    """Print detailed analysis of a fourth down decision."""
    state = analysis['state']
    result = analysis['result']

    # Location description
    if state.field_pos > 50:
        location = f"own {100 - state.field_pos}"
    else:
        location = f"opponent's {state.field_pos}"

    print("="*80)
    print("FOURTH DOWN DECISION ANALYSIS")
    print("="*80)

    print(f"\nSituation: 4th & {state.yards_to_go} at {location}")
    print(f"Score: {'Tied' if state.score_diff == 0 else f'Up {state.score_diff}' if state.score_diff > 0 else f'Down {abs(state.score_diff)}'}")
    print(f"Time remaining: {state.time_remaining // 60}:{state.time_remaining % 60:02d}")
    if state.off_team:
        print(f"Offense: {state.off_team}")
    if state.def_team:
        print(f"Defense: {state.def_team}")

    print("\n" + "-"*40)
    print("EXPECTED WIN PROBABILITY BY ACTION")
    print("-"*40)

    print(f"  GO FOR IT:   {result.wp_go:6.1%}  (P best: {result.prob_go_best:.0%})")
    print(f"  PUNT:        {result.wp_punt:6.1%}  (P best: {result.prob_punt_best:.0%})")
    print(f"  FIELD GOAL:  {result.wp_fg:6.1%}  (P best: {result.prob_fg_best:.0%})")

    print("\n" + "-"*40)
    print(f"OPTIMAL ACTION: {result.optimal_action.upper()}")
    print(f"Decision margin: {result.decision_margin:.1%}")
    print("-"*40)

    # Additional context
    if result.optimal_action == 'go_for_it':
        print("\nModel recommends going for it because the expected WP")
        print("from attempting conversion exceeds other options.")
    elif result.optimal_action == 'punt':
        print("\nModel recommends punting to improve field position")
        print("and let the defense make a stop.")
    else:
        print("\nModel recommends field goal to score points and")
        print("preserve optionality for overtime/subsequent possessions.")

    return result


def analyze_2006_eagles_playoff():
    """
    Analyze Andy Reid's infamous punt in the 2006 NFC Divisional Playoff.

    Situation: 4th & 15 at own 39 (yardline_100=61), down 3, 1:56 remaining
    Eagles punted, Saints received at their 22 and kneeled out the clock.
    """
    print("\n" + "="*80)
    print("2006 NFC DIVISIONAL PLAYOFF: PHILADELPHIA EAGLES @ NEW ORLEANS SAINTS")
    print("Andy Reid's Punt Decision")
    print("="*80)

    # The actual situation
    state = GameState(
        field_pos=61,           # Own 39 yard line (61 yards from opponent's end zone)
        yards_to_go=15,         # 4th & 15
        score_diff=-3,          # Down 3 (Saints 27, Eagles 24)
        time_remaining=116,     # 1:56 remaining
        off_team='PHI',
        def_team='NO'
    )

    analysis = analyze_play(state)
    result = print_analysis(analysis)

    # KEY INSIGHT: End-of-game clock management
    print("\n" + "="*80)
    print("CRITICAL CONTEXT: END-OF-GAME CLOCK MANAGEMENT")
    print("="*80)

    print("""
This play perfectly illustrates the MODEL LIMITATION we identified:

After the Eagles punted:
- Saints received at their 22-yard line
- Saints ran 3 plays (including kneels) to run out the clock
- Game over: Saints win 27-24

The model's WP calculations assume the opponent will try to score.
But with a 3-point lead and under 2 minutes left, the Saints could
simply kneel and run out the clock.

REALITY CHECK:
- WP if punt: ~0% (Saints kneel, game over)
- WP if go for it: ~15% conversion x scoring chance = 10-15%

Any positive probability beats zero. Reid's punt was mathematically
indefensible - it guaranteed a loss when going for it preserved some
chance of winning.

This is the same situation as:
- 2024 NE @ MIA: Patriots down 5 with 1:00 left
- 2021 BUF @ NE: Bills down 4 with 2:00 left

In all these cases, the trailing team MUST go for it because giving
the ball back means the opponent can kneel out the clock.
""")

    return analysis


if __name__ == "__main__":
    analyze_2006_eagles_playoff()
