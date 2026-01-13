"""
Specific Game Analysis Tool

Allows analysis of any 4th down situation with full Bayesian decision theory output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.bayesian_models import load_all_models
from analysis.decision_framework import (
    BayesianDecisionAnalyzer,
    GameState, DecisionResult
)


class FourthDownAnalyzer:
    """
    Complete 4th down analysis tool with team/kicker-specific estimates.

    Usage:
        analyzer = FourthDownAnalyzer()

        # Basic analysis (league average)
        result = analyzer.analyze(
            field_pos=35,
            yards_to_go=3,
            score_diff=-3,
            time_remaining_min=5,
            timeouts=3,
            opp_timeouts=2
        )

        # Team/kicker-specific analysis
        result = analyzer.analyze(
            field_pos=35,
            yards_to_go=3,
            score_diff=-3,
            time_remaining_min=5,
            team='PHI',
            kicker='J.Elliott'  # or kicker_id for exact match
        )

        analyzer.print_report(result)
        analyzer.plot_analysis(result)
    """

    def __init__(self, models_dir: Path = None, hierarchical: bool = True):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / 'models'

        print("Loading models...")
        self.models = load_all_models(models_dir, hierarchical=hierarchical)
        self.bayesian = BayesianDecisionAnalyzer(self.models)

        # Build kicker name -> ID mapping if hierarchical
        self.kicker_name_to_id = {}
        if hasattr(self.models['fg'], 'kicker_list'):
            # Load FG data to get name mapping
            try:
                fgs = pd.read_parquet(models_dir.parent / 'data' / 'field_goals.parquet')
                self.kicker_name_to_id = fgs.groupby('kicker_player_name')['kicker_player_id'].first().to_dict()
            except:
                pass

        print("Ready for analysis!")
        if self.bayesian.has_team_effects:
            print("  Team effects: enabled")
        if self.bayesian.has_kicker_effects:
            print("  Kicker effects: enabled")

    def _resolve_kicker_id(self, kicker: str) -> Optional[str]:
        """Convert kicker name to ID, or return as-is if already an ID."""
        if kicker is None:
            return None
        # Check if it's a name we can map
        if kicker in self.kicker_name_to_id:
            return self.kicker_name_to_id[kicker]
        # Check if it's already an ID in the model
        if hasattr(self.models['fg'], 'kicker_effects'):
            if kicker in self.models['fg'].kicker_effects:
                return kicker
        return None

    def analyze(
        self,
        field_pos: int,
        yards_to_go: int,
        score_diff: int,
        time_remaining_min: float,
        timeouts: int = 3,
        opp_timeouts: int = 3,
        team: str = None,
        kicker: str = None
    ) -> Dict:
        """
        Analyze a 4th down situation.

        Args:
            field_pos: Yards from opponent's end zone (e.g., 35 = at their 35)
            yards_to_go: Yards needed for first down
            score_diff: Your score minus their score (positive = winning)
            time_remaining_min: Minutes remaining in game
            timeouts: Your timeouts remaining (0-3)
            opp_timeouts: Opponent's timeouts remaining (0-3)
            team: Team abbreviation (e.g., 'PHI', 'DET') for team-specific conversion rate
            kicker: Kicker name (e.g., 'J.Tucker') or player ID for kicker-specific FG rate

        Returns:
            Dict with complete analysis results
        """
        kicker_id = self._resolve_kicker_id(kicker)

        state = GameState(
            field_pos=field_pos,
            yards_to_go=yards_to_go,
            score_diff=score_diff,
            time_remaining=int(time_remaining_min * 60),
            timeout_diff=timeouts - opp_timeouts,
            team=team,
            kicker_id=kicker_id
        )

        # Bayesian analysis
        bayesian_result = self.bayesian.analyze(state)

        # Get probabilities (team/kicker-specific if available)
        if hasattr(self.models['conversion'], 'get_conversion_prob'):
            if team and hasattr(self.models['conversion'], 'team_effects'):
                conv_prob = self.models['conversion'].get_conversion_prob(yards_to_go, team=team)
            else:
                conv_prob = self.models['conversion'].get_conversion_prob(yards_to_go)
        else:
            conv_prob = self.models['conversion'].get_conversion_prob(yards_to_go)

        if hasattr(self.models['fg'], 'get_make_prob'):
            if kicker_id and hasattr(self.models['fg'], 'kicker_effects'):
                fg_prob = self.models['fg'].get_make_prob(field_pos + 17, kicker_id=kicker_id)
            else:
                fg_prob = self.models['fg'].get_make_prob(field_pos + 17)
        else:
            fg_prob = self.models['fg'].get_make_prob(field_pos + 17)

        result = {
            'state': state,
            'bayesian': bayesian_result,
            'fg_distance': field_pos + 17,
            'conversion_prob': conv_prob,
            'fg_prob': fg_prob,
            'punt_yards': self.models['punt'].get_expected_net_yards(field_pos),
            'team': team,
            'kicker': kicker,
            'kicker_id': kicker_id,
        }

        # Add kicker effect info if available
        if kicker_id and hasattr(self.models['fg'], 'get_kicker_effect'):
            result['kicker_effect'] = self.models['fg'].get_kicker_effect(kicker_id)

        return result

    def print_report(self, result: Dict):
        """
        Print a detailed analysis report.
        """
        state = result['state']
        bayesian = result['bayesian']

        print("\n" + "="*70)
        print("4TH DOWN DECISION ANALYSIS")
        print("="*70)

        # Game situation
        print(f"\n{'SITUATION':^70}")
        print("-"*70)
        print(f"4th & {state.yards_to_go} at opponent's {state.field_pos} yard line")

        score_str = "TIED" if state.score_diff == 0 else \
                   f"WINNING by {state.score_diff}" if state.score_diff > 0 else \
                   f"LOSING by {abs(state.score_diff)}"
        print(f"Score: {score_str}")

        minutes = state.time_remaining // 60
        seconds = state.time_remaining % 60
        print(f"Time remaining: {minutes}:{seconds:02d}")

        if state.timeout_diff != 0:
            print(f"Timeout advantage: {'+' if state.timeout_diff > 0 else ''}{state.timeout_diff}")

        # Team/kicker info if available
        if result.get('team') or result.get('kicker'):
            print()
            if result.get('team'):
                print(f"Team: {result['team']}")
            if result.get('kicker'):
                print(f"Kicker: {result['kicker']}")
                if result.get('kicker_effect'):
                    effect = result['kicker_effect']
                    if effect['known']:
                        print(f"  Kicker effect: {effect['effect']:+.3f} log-odds ({effect['n']} attempts)")

        # Key probabilities
        print(f"\n{'KEY PROBABILITIES':^70}")
        print("-"*70)
        team_note = f" ({result['team']})" if result.get('team') else ""
        print(f"4th & {state.yards_to_go} conversion probability: {result['conversion_prob']:.1%}{team_note}")

        kicker_note = f" ({result['kicker']})" if result.get('kicker') else ""
        print(f"{result['fg_distance']}-yard FG make probability: {result['fg_prob']:.1%}{kicker_note}")
        print(f"Expected net punt distance: {result['punt_yards']:.1f} yards")

        # Win probability analysis
        print(f"\n{'WIN PROBABILITY ANALYSIS':^70}")
        print("-"*70)
        print(f"{'Action':<20} {'Expected WP':>15} {'95% CI':>20} {'P(Best)':>12}")
        print("-"*70)

        for action, wp, samples, prob_best in [
            ('Go for it', bayesian.wp_go, bayesian.wp_go_samples, bayesian.prob_go_best),
            ('Punt', bayesian.wp_punt, bayesian.wp_punt_samples, bayesian.prob_punt_best),
            ('Field Goal', bayesian.wp_fg, bayesian.wp_fg_samples, bayesian.prob_fg_best)
        ]:
            ci_low = np.percentile(samples, 2.5)
            ci_high = np.percentile(samples, 97.5)
            ci_str = f"[{ci_low:.1%}, {ci_high:.1%}]"

            marker = " **" if action.lower().replace(' ', '_') == bayesian.optimal_action else ""
            print(f"{action:<20} {wp:>14.1%} {ci_str:>20} {prob_best:>11.0%}{marker}")

        # Recommendation
        print(f"\n{'RECOMMENDATION':^70}")
        print("-"*70)

        action_display = {
            'go_for_it': 'GO FOR IT',
            'punt': 'PUNT',
            'field_goal': 'KICK FIELD GOAL'
        }

        print(f"\n>>> {action_display[bayesian.optimal_action]} <<<\n")

        if bayesian.decision_margin > 0.05:
            print(f"This is a clear decision (margin: {bayesian.decision_margin:.1%})")
        elif bayesian.decision_margin > 0.02:
            print(f"This is a moderate decision (margin: {bayesian.decision_margin:.1%})")
        else:
            print(f"This is a close call (margin: {bayesian.decision_margin:.1%})")
            print("The decision could reasonably go either way.")

        print("\n" + "="*70)

    def plot_analysis(self, result: Dict, save_path: Path = None):
        """
        Create visualization of the analysis.
        """
        state = result['state']
        bayesian = result['bayesian']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: WP distributions
        ax = axes[0]
        for samples, color, label in [
            (bayesian.wp_go_samples, 'green', 'Go for it'),
            (bayesian.wp_punt_samples, 'blue', 'Punt'),
            (bayesian.wp_fg_samples, 'red', 'Field Goal')
        ]:
            ax.hist(samples * 100, bins=40, alpha=0.6, color=color, label=label, density=True)

        ax.set_xlabel('Win Probability (%)')
        ax.set_ylabel('Density')
        ax.set_title('Win Probability Distributions')
        ax.legend()

        # Plot 2: WP comparison bar chart
        ax = axes[1]
        actions = ['Go for it', 'Punt', 'Field Goal']
        wps = [bayesian.wp_go, bayesian.wp_punt, bayesian.wp_fg]
        colors = ['green', 'blue', 'red']

        bars = ax.bar(actions, [w * 100 for w in wps], color=colors, alpha=0.7)

        # Highlight optimal
        optimal_idx = wps.index(max(wps))
        bars[optimal_idx].set_alpha(1.0)
        bars[optimal_idx].set_edgecolor('black')
        bars[optimal_idx].set_linewidth(3)

        ax.set_ylabel('Expected Win Probability (%)')
        ax.set_title('Expected Value Comparison')

        # Add value labels
        for bar, wp in zip(bars, wps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{wp:.1%}', ha='center', va='bottom', fontsize=11)

        # Plot 3: Probability each is best
        ax = axes[2]
        probs = [bayesian.prob_go_best, bayesian.prob_punt_best, bayesian.prob_fg_best]

        bars = ax.bar(actions, [p * 100 for p in probs], color=colors, alpha=0.7)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Probability Each Option is Best')
        ax.set_ylim(0, 100)

        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{prob:.0%}', ha='center', va='bottom', fontsize=11)

        # Main title
        fig.suptitle(f"4th & {state.yards_to_go} at opponent's {state.field_pos}\n"
                    f"Score: {'+' if state.score_diff > 0 else ''}{state.score_diff}, "
                    f"Time: {state.time_remaining // 60}:{state.time_remaining % 60:02d}",
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def quick_analysis(
        self,
        field_pos: int,
        yards_to_go: int,
        score_diff: int = 0,
        time_remaining_min: float = 15
    ) -> str:
        """
        Quick one-liner analysis.

        Returns formatted string with recommendation.
        """
        result = self.analyze(field_pos, yards_to_go, score_diff, time_remaining_min)
        bayesian = result['bayesian']

        action_display = {
            'go_for_it': 'Go for it',
            'punt': 'Punt',
            'field_goal': 'FG'
        }

        return (f"4th&{yards_to_go} at {field_pos}: "
                f"{action_display[bayesian.optimal_action]} "
                f"(WP: {max(bayesian.wp_go, bayesian.wp_punt, bayesian.wp_fg):.1%}, "
                f"P(best): {max(bayesian.prob_go_best, bayesian.prob_punt_best, bayesian.prob_fg_best):.0%})")


def interactive_mode():
    """
    Run interactive analysis mode.
    """
    print("\n" + "="*70)
    print("4TH DOWN DECISION ANALYZER - INTERACTIVE MODE")
    print("="*70)
    print("\nThis tool analyzes any 4th down situation using Bayesian decision theory.")
    print("Type 'quit' to exit.\n")

    analyzer = FourthDownAnalyzer()

    while True:
        print("\n" + "-"*70)
        try:
            field_pos_str = input("Yards from opponent's end zone (e.g., 35 for 'at their 35'): ")
            if field_pos_str.lower() == 'quit':
                break
            field_pos = int(field_pos_str)

            yards_to_go = int(input("Yards to go (e.g., 3 for '4th and 3'): "))

            score_diff = int(input("Score differential (your score - their score): "))

            time_input = input("Time remaining (format: MM:SS or just minutes): ")
            if ':' in time_input:
                parts = time_input.split(':')
                time_remaining_min = int(parts[0]) + int(parts[1]) / 60
            else:
                time_remaining_min = float(time_input)

            timeouts = int(input("Your timeouts remaining (0-3) [default: 3]: ") or "3")
            opp_timeouts = int(input("Opponent's timeouts remaining (0-3) [default: 3]: ") or "3")

            result = analyzer.analyze(
                field_pos=field_pos,
                yards_to_go=yards_to_go,
                score_diff=score_diff,
                time_remaining_min=time_remaining_min,
                timeouts=timeouts,
                opp_timeouts=opp_timeouts
            )

            analyzer.print_report(result)

            show_plot = input("\nShow visualization? (y/n) [default: n]: ").lower()
            if show_plot == 'y':
                analyzer.plot_analysis(result)
                plt.show()

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="4th Down Decision Analyzer")
    parser.add_argument('--interactive', '-i', action='store_true',
                       help="Run in interactive mode")
    parser.add_argument('--field-pos', '-f', type=int,
                       help="Yards from opponent's end zone")
    parser.add_argument('--yards', '-y', type=int,
                       help="Yards to go")
    parser.add_argument('--score', '-s', type=int, default=0,
                       help="Score differential (default: 0)")
    parser.add_argument('--time', '-t', type=float, default=15,
                       help="Minutes remaining (default: 15)")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.field_pos and args.yards:
        analyzer = FourthDownAnalyzer()
        result = analyzer.analyze(
            field_pos=args.field_pos,
            yards_to_go=args.yards,
            score_diff=args.score,
            time_remaining_min=args.time
        )
        analyzer.print_report(result)
    else:
        # Demo mode
        print("Running demo analysis...")
        analyzer = FourthDownAnalyzer()

        # Example scenarios
        scenarios = [
            # (field_pos, yards_to_go, score_diff, time_min, description)
            (35, 1, 0, 15, "Classic 4th and 1 at midfield"),
            (2, 2, 0, 2, "Goal line, 2 minutes left, tied"),
            (45, 4, -7, 5, "Need touchdown, 5 minutes left"),
            (28, 3, 3, 10, "Up 3, protect lead"),
        ]

        for field_pos, ytg, score, time_min, desc in scenarios:
            print(f"\n\n{'='*70}")
            print(f"SCENARIO: {desc}")
            result = analyzer.analyze(field_pos, ytg, score, time_min)
            analyzer.print_report(result)
