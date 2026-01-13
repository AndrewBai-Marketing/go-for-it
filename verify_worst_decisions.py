"""
Verify the 'worst decisions' from the paper using the neural WP model.
"""
from pathlib import Path
import sys
sys.path.insert(0, 'd:/VSCode/fourth_down')
from analysis.decision_framework import BayesianDecisionAnalyzer, GameState
from models.bayesian_models import load_all_models

models_dir = Path('d:/VSCode/fourth_down/models')
models = load_all_models(models_dir, use_neural_wp=True)
analyzer = BayesianDecisionAnalyzer(models)

# Verify we're using neural WP
print(f'WP Model type: {type(models["wp"]).__name__}')
print()

# Test each situation from the paper
# field_pos is distance from opponent's end zone (yardline_100)
# So 'own 49' means we are at our 49, which is 100-49 = 51 yards from opponent's end zone
# 'own 47' means at our 47 = 53 yards from opponent's end zone
# 'own 40' means at our 40 = 60 yards from opponent's end zone
# 'own 50' (midfield) = 50 yards from opponent's end zone
# 'own 46' = 54 yards from opponent's end zone

situations = [
    {'desc': '2016 BAL @ JAX', 'field_pos': 51, 'yards_to_go': 2, 'score_diff': -1, 'time_remaining': 134},
    {'desc': '2020 DAL @ SEA', 'field_pos': 53, 'yards_to_go': 3, 'score_diff': -1, 'time_remaining': 157},
    {'desc': '2012 DAL @ CAR', 'field_pos': 60, 'yards_to_go': 1, 'score_diff': -2, 'time_remaining': 131},
    {'desc': '2020 DEN @ NYJ', 'field_pos': 50, 'yards_to_go': 3, 'score_diff': -2, 'time_remaining': 125},
    {'desc': '2019 CLE @ PIT', 'field_pos': 54, 'yards_to_go': 2, 'score_diff': -1, 'time_remaining': 122},
]

print('='*80)
print('NEURAL WP MODEL - WORST DECISIONS VERIFICATION')
print('='*80)

correct_count = 0
mistake_count = 0

for sit in situations:
    state = GameState(
        field_pos=sit['field_pos'],
        yards_to_go=sit['yards_to_go'],
        score_diff=sit['score_diff'],
        time_remaining=sit['time_remaining']
    )
    result = analyzer.analyze(state)

    # What the coach chose vs what model recommends
    coach_action = 'GO'  # All were Go decisions
    model_action = result.optimal_action.upper()
    if model_action == 'GO_FOR_IT':
        model_action = 'GO'
    elif model_action == 'FIELD_GOAL':
        model_action = 'FG'

    was_mistake = (model_action != coach_action)

    if was_mistake:
        mistake_count += 1
    else:
        correct_count += 1

    print(f"\n{sit['desc']}: 4th & {sit['yards_to_go']}, {sit['field_pos']} yds from endzone, down {-sit['score_diff']}, {sit['time_remaining']}s left")
    print(f"  E[WP | GO]:    {result.wp_go:.1%}")
    print(f"  E[WP | PUNT]:  {result.wp_punt:.1%}")
    print(f"  E[WP | FG]:    {result.wp_fg:.1%}")
    print()
    print(f"  Model Optimal: {result.optimal_action.upper()} (margin: {result.decision_margin:.1%})")
    print(f"  Coach Choice:  GO")
    print()
    print(f"  P(GO best):    {result.prob_go_best:.0%}")
    print(f"  P(PUNT best):  {result.prob_punt_best:.0%}")
    print(f"  P(FG best):    {result.prob_fg_best:.0%}")

    if was_mistake:
        # Calculate WP cost
        wp_diff = max(result.wp_go, result.wp_punt, result.wp_fg) - result.wp_go
        print()
        print(f"  Verdict: COACH WAS WRONG - Model says {model_action}")
        print(f"  WP Cost: {wp_diff:.1%}")
    else:
        print()
        print(f"  Verdict: COACH WAS CORRECT - GO was optimal")

print()
print('='*80)
print('SUMMARY')
print('='*80)
print(f"Coaches were CORRECT in {correct_count} of 5 decisions (neural WP model agrees GO was right)")
print(f"Coaches were WRONG in {mistake_count} of 5 decisions (neural WP model recommends different action)")
