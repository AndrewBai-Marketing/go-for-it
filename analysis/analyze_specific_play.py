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


def analyze_belichick_4th_and_2():
    """
    Analyze Bill Belichick's famous 4th & 2 decision (November 15, 2009).

    Situation: 4th & 2 at own 28 (yardline_100=72), up 6, 2:08 remaining
    Patriots went for it, failed to convert.
    Colts scored touchdown to win 35-34.

    This is the canonical case of:
    - Analytically correct decision
    - Bad outcome
    - Wrong learning signal to other coaches

    Most analyses showed going for it was correct because:
    1. Conversion probability ~60% on 4th & 2
    2. If convert: game over (can kneel out clock)
    3. If fail: Colts need TD from own 28 with 2:00 left vs Colts getting ball
       at own ~30 after punt with 2:00 left anyway
    4. Peyton Manning was highly likely to score either way
    """
    print("\n" + "="*80)
    print("NOVEMBER 15, 2009: NEW ENGLAND PATRIOTS @ INDIANAPOLIS COLTS")
    print("Bill Belichick's 4th & 2 Decision")
    print("="*80)

    # The actual situation
    state = GameState(
        field_pos=72,           # Own 28 yard line (72 yards from opponent's end zone)
        yards_to_go=2,          # 4th & 2
        score_diff=6,           # Up 6 (Patriots 34, Colts 28)
        time_remaining=128,     # 2:08 remaining
        off_team='NE',
        def_team='IND',
        is_home=0.0             # Patriots were away
    )

    analysis = analyze_play(state)
    result = print_analysis(analysis)

    print("\n" + "="*80)
    print("HEURISTIC LEARNING IMPLICATIONS (Strulov-Shlain Framework)")
    print("="*80)

    print("""
This play is a perfect example of OUTCOME-BASED LEARNING vs MODEL-BASED OPTIMIZATION:

WHAT HAPPENED:
- Belichick went for it (analytically correct)
- Pass to Kevin Faulk was ruled short (bad outcome)
- Colts scored TD, won 35-34
- Media CRUCIFIED Belichick for "arrogance"

CORRECT BAYESIAN ANALYSIS:
- P(convert 4th & 2) ≈ 60%
- If convert: P(win) ≈ 100% (kneel out clock)
- If fail: P(win) ≈ 30% (Colts need TD from 28, 2:00 left)
- If punt: P(win) ≈ 35-40% (Colts get ball at ~30, 2:00 left, Manning)

E[WP|Go] = 0.60 × 1.00 + 0.40 × 0.30 = 0.72
E[WP|Punt] ≈ 0.35-0.40

Going for it was ~30+ percentage points better!

THE LEARNING PROBLEM:
1. Other coaches observed: "Belichick went for it and LOST"
2. Outcome-based heuristic update: "Going for it is risky, I should punt"
3. This is EXACTLY WRONG - the decision was correct, outcome was unlucky

PREDICTION FOR YOUR ANALYSIS:
- Look at league-wide go-for-it rates in similar situations BEFORE and AFTER Nov 2009
- If coaches learned from the wrong signal, rates should DROP after this game
- This would be evidence of heuristic (outcome-based) learning, not model-based optimization
""")

    # Additional analysis: What if they had punted?
    print("\n" + "-"*40)
    print("COUNTERFACTUAL: What if Patriots had punted?")
    print("-"*40)

    # Approximate punt situation
    punt_state = GameState(
        field_pos=70,           # Colts get ball around own 30 after punt
        yards_to_go=10,         # 1st & 10
        score_diff=-6,          # Colts down 6
        time_remaining=120,     # ~2:00 after punt
        off_team='IND',
        def_team='NE',
        is_home=1.0             # Colts at home
    )

    print(f"\nColts would have ball at ~own 30, down 6, 2:00 left")
    print(f"With Peyton Manning, their P(TD) was extremely high")
    print(f"Punt only 'saves' ~8 yards of field position")

    return analysis


def analyze_learning_around_belichick():
    """
    Study whether the Belichick 4th & 2 caused a league-wide chilling effect.

    This is an empirical test: did coaches become MORE conservative after
    the Belichick play was criticized, despite it being analytically correct?
    """
    print("\n" + "="*80)
    print("EMPIRICAL TEST: Did Belichick 4th & 2 cause a chilling effect?")
    print("="*80)

    print("""
METHODOLOGY:

1. Define "Belichick-like situations":
   - 4th & short (1-3 yards)
   - Own territory (own 20-40)
   - Leading by 3-10 points
   - 1-4 minutes remaining

2. Compute go-for-it rate in these situations:
   - Pre-Belichick: 2006-2009 (up to Nov 15, 2009)
   - Post-Belichick: Nov 16, 2009 - 2012
   - Analytics era: 2018-2024

3. If heuristic learning from wrong signal:
   - Rate should DROP after Nov 2009
   - Rate should INCREASE again in analytics era

4. If model-based optimization:
   - Rate should be UNCHANGED by Belichick outcome
   - Coaches would recognize decision was correct despite outcome

This is a clean test because:
- The Belichick play was highly salient (massive media coverage)
- The outcome was negative (loss)
- The decision was analytically correct
- Other coaches could observe and potentially "learn" the wrong lesson
""")

    return None


if __name__ == "__main__":
    # Run both analyses
    analyze_2006_eagles_playoff()
    print("\n" * 3)
    analyze_belichick_4th_and_2()
    print("\n" * 3)
    analyze_learning_around_belichick()
