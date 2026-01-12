"""
Visualizations for 4th down decision analysis.

Creates:
1. Decision boundary heat maps (like Romer's figures)
2. Uncertainty visualizations
3. Time series of conservatism
4. Team-level analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.bayesian_models import load_all_models
from analysis.decision_framework import BayesianDecisionAnalyzer, GameState

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


def create_decision_heatmap(
    analyzer: BayesianDecisionAnalyzer,
    score_diff: int = 0,
    time_remaining: int = 1800,
    output_path: Path = None
) -> plt.Figure:
    """
    Create Romer-style heat map showing optimal decision by field position and yards to go.

    Args:
        analyzer: BayesianDecisionAnalyzer instance
        score_diff: Score differential for the scenario
        time_remaining: Time remaining in seconds
        output_path: Path to save figure
    """
    # Create grid
    field_positions = np.arange(1, 100, 2)  # yards from opponent's end zone
    yards_to_go = np.arange(1, 16)

    # Decision matrix (0=go, 1=punt, 2=fg)
    decision_matrix = np.zeros((len(yards_to_go), len(field_positions)))
    wp_advantage = np.zeros((len(yards_to_go), len(field_positions)))  # WP_go - max(WP_punt, WP_fg)

    for i, ytg in enumerate(yards_to_go):
        for j, fp in enumerate(field_positions):
            state = GameState(
                field_pos=int(fp),
                yards_to_go=int(ytg),
                score_diff=score_diff,
                time_remaining=time_remaining
            )
            result = analyzer.analyze(state)

            if result.optimal_action == 'go_for_it':
                decision_matrix[i, j] = 0
            elif result.optimal_action == 'punt':
                decision_matrix[i, j] = 1
            else:
                decision_matrix[i, j] = 2

            # WP advantage of going for it
            wp_advantage[i, j] = result.wp_go - max(result.wp_punt, result.wp_fg)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Decision boundaries
    cmap_decision = plt.cm.colors.ListedColormap(['green', 'blue', 'red'])
    im1 = axes[0].imshow(decision_matrix, aspect='auto', cmap=cmap_decision,
                         extent=[field_positions[0], field_positions[-1],
                                yards_to_go[-1], yards_to_go[0]])

    axes[0].set_xlabel('Yards from Opponent\'s End Zone', fontsize=12)
    axes[0].set_ylabel('Yards to Go', fontsize=12)
    axes[0].set_title(f'Optimal 4th Down Decision\n(Score: {"Tied" if score_diff == 0 else f"+{score_diff}" if score_diff > 0 else score_diff}, '
                     f'{time_remaining//60} min remaining)', fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Go for it'),
                      Patch(facecolor='blue', label='Punt'),
                      Patch(facecolor='red', label='Field Goal')]
    axes[0].legend(handles=legend_elements, loc='upper right')

    # Right: WP advantage of going for it
    im2 = axes[1].imshow(wp_advantage * 100, aspect='auto', cmap='RdYlGn',
                         extent=[field_positions[0], field_positions[-1],
                                yards_to_go[-1], yards_to_go[0]],
                         vmin=-10, vmax=10)

    axes[1].set_xlabel('Yards from Opponent\'s End Zone', fontsize=12)
    axes[1].set_ylabel('Yards to Go', fontsize=12)
    axes[1].set_title('WP Advantage of Going For It\n(Green = Go for it better, Red = Kick better)',
                     fontsize=14)

    plt.colorbar(im2, ax=axes[1], label='WP Difference (percentage points)')

    # Add contour line at WP_advantage = 0
    contour = axes[1].contour(field_positions, yards_to_go, wp_advantage * 100,
                              levels=[0], colors='black', linewidths=2)
    axes[1].clabel(contour, inline=True, fontsize=10, fmt='Break-even')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved decision heatmap to {output_path}")

    return fig


def create_uncertainty_plot(
    analyzer: BayesianDecisionAnalyzer,
    state: GameState,
    output_path: Path = None
) -> plt.Figure:
    """
    Show posterior distributions of WP for each action.
    """
    result = analyzer.analyze(state)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histograms of WP for each action
    ax = axes[0]

    ax.hist(result.wp_go_samples * 100, bins=50, alpha=0.6, label='Go for it',
            color='green', density=True)
    ax.hist(result.wp_punt_samples * 100, bins=50, alpha=0.6, label='Punt',
            color='blue', density=True)
    ax.hist(result.wp_fg_samples * 100, bins=50, alpha=0.6, label='Field Goal',
            color='red', density=True)

    ax.axvline(result.wp_go * 100, color='green', linestyle='--', linewidth=2)
    ax.axvline(result.wp_punt * 100, color='blue', linestyle='--', linewidth=2)
    ax.axvline(result.wp_fg * 100, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Win Probability (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Posterior Win Probability Distributions\n'
                f'4th & {state.yards_to_go} at opponent\'s {state.field_pos}',
                fontsize=14)
    ax.legend()

    # Right: WP difference distribution (go - kick)
    ax = axes[1]

    wp_kick = np.maximum(result.wp_punt_samples, result.wp_fg_samples)
    wp_diff = (result.wp_go_samples - wp_kick) * 100

    ax.hist(wp_diff, bins=50, alpha=0.7, color='purple', density=True)
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Break-even')
    ax.axvline(np.mean(wp_diff), color='purple', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(wp_diff):.1f}pp')

    # Mark 95% CI
    ci_low, ci_high = np.percentile(wp_diff, [2.5, 97.5])
    ax.axvline(ci_low, color='gray', linestyle=':', linewidth=1)
    ax.axvline(ci_high, color='gray', linestyle=':', linewidth=1)
    ax.axvspan(ci_low, ci_high, alpha=0.2, color='gray', label=f'95% CI: [{ci_low:.1f}, {ci_high:.1f}]')

    prob_go_better = (wp_diff > 0).mean() * 100
    ax.text(0.95, 0.95, f'P(Go is better) = {prob_go_better:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('WP(Go for it) - WP(Best Kick Option) (pp)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Should You Go For It?\n(Posterior distribution of WP advantage)', fontsize=14)
    ax.legend(loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved uncertainty plot to {output_path}")

    return fig


def create_time_trend_plot(
    analysis_df: pd.DataFrame,
    output_path: Path = None
) -> plt.Figure:
    """
    Show how NFL conservatism has changed over time.
    """
    if 'season' not in analysis_df.columns:
        print("No season column in data")
        return None

    # Aggregate by season
    seasonal = analysis_df.groupby('season').agg({
        'actual_decision': [
            lambda x: (x == 'go_for_it').mean(),
            'count'
        ],
        'optimal_action': lambda x: (x == 'go_for_it').mean(),
        'is_optimal': 'mean',
        'wp_lost': ['mean', 'sum']
    })

    seasonal.columns = ['Actual Go Rate', 'N Plays', 'Optimal Go Rate',
                       'Pct Optimal', 'Avg WP Lost', 'Total WP Lost']
    seasonal = seasonal.reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Go-for-it rates
    ax = axes[0, 0]
    ax.plot(seasonal['season'], seasonal['Actual Go Rate'] * 100, 'o-',
            label='Actual', color='blue', linewidth=2, markersize=8)
    ax.plot(seasonal['season'], seasonal['Optimal Go Rate'] * 100, 's--',
            label='Optimal', color='green', linewidth=2, markersize=8)
    ax.fill_between(seasonal['season'],
                    seasonal['Actual Go Rate'] * 100,
                    seasonal['Optimal Go Rate'] * 100,
                    alpha=0.3, color='red', label='Conservatism Gap')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Go-For-It Rate (%)', fontsize=12)
    ax.set_title('NFL 4th Down Aggressiveness Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Optimal decision rate
    ax = axes[0, 1]
    ax.bar(seasonal['season'], seasonal['Pct Optimal'] * 100, color='steelblue')
    ax.axhline(y=seasonal['Pct Optimal'].mean() * 100, color='red',
               linestyle='--', label=f'Average: {seasonal["Pct Optimal"].mean()*100:.1f}%')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('% Optimal Decisions', fontsize=12)
    ax.set_title('Decision Quality Over Time', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 100)

    # Bottom left: WP cost
    ax = axes[1, 0]
    ax.bar(seasonal['season'], seasonal['Avg WP Lost'] * 100, color='coral')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Average WP Lost (percentage points)', fontsize=12)
    ax.set_title('Cost of Suboptimal Decisions', fontsize=14)

    # Bottom right: Number of plays analyzed
    ax = axes[1, 1]
    ax.bar(seasonal['season'], seasonal['N Plays'], color='gray')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Number of 4th Down Plays', fontsize=12)
    ax.set_title('Sample Size by Season', fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved time trend plot to {output_path}")

    return fig


def create_team_analysis_plot(
    analysis_df: pd.DataFrame,
    output_path: Path = None
) -> plt.Figure:
    """
    Show which teams are closest to optimal.
    """
    if 'posteam' not in analysis_df.columns:
        print("No team column in data")
        return None

    # Aggregate by team
    team_stats = analysis_df.groupby('posteam').agg({
        'actual_decision': [
            lambda x: (x == 'go_for_it').mean(),
            'count'
        ],
        'optimal_action': lambda x: (x == 'go_for_it').mean(),
        'is_optimal': 'mean',
        'wp_lost': 'mean'
    })

    team_stats.columns = ['Actual Go Rate', 'N Plays', 'Optimal Go Rate',
                         'Pct Optimal', 'Avg WP Lost']
    team_stats = team_stats.reset_index()

    # Calculate "aggressiveness gap" (how much more they should go for it)
    team_stats['Aggressiveness Gap'] = team_stats['Optimal Go Rate'] - team_stats['Actual Go Rate']

    # Sort by optimal rate
    team_stats = team_stats.sort_values('Pct Optimal', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Left: Bar chart of optimal decision %
    ax = axes[0]
    colors = plt.cm.RdYlGn(team_stats['Pct Optimal'].values)
    ax.barh(team_stats['posteam'], team_stats['Pct Optimal'] * 100, color=colors)
    ax.axvline(x=team_stats['Pct Optimal'].mean() * 100, color='black',
               linestyle='--', label='League Average')
    ax.set_xlabel('% Optimal Decisions', fontsize=12)
    ax.set_ylabel('Team', fontsize=12)
    ax.set_title('Decision Quality by Team\n(Higher = Better)', fontsize=14)
    ax.legend()

    # Right: Aggressiveness gap
    ax = axes[1]
    team_stats_sorted = team_stats.sort_values('Aggressiveness Gap', ascending=False)
    colors = ['red' if x > 0 else 'green' for x in team_stats_sorted['Aggressiveness Gap']]
    ax.barh(team_stats_sorted['posteam'], team_stats_sorted['Aggressiveness Gap'] * 100,
            color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Aggressiveness Gap (Optimal - Actual Go Rate, pp)', fontsize=12)
    ax.set_ylabel('Team', fontsize=12)
    ax.set_title('How Much More Should Each Team Go For It?\n(Red = Too Conservative)',
                fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved team analysis plot to {output_path}")

    return fig


def create_all_visualizations(output_dir: Path):
    """
    Generate all visualizations for the analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'
    tables_dir = Path(__file__).parent.parent / 'outputs' / 'tables'

    # Load models and create analyzer
    print("Loading models...")
    models = load_all_models(models_dir)
    analyzer = BayesianDecisionAnalyzer(models)

    # 1. Decision heatmaps for different game situations
    print("\nCreating decision heatmaps...")

    scenarios = [
        (0, 1800, 'tied_30min'),      # Tied, 30 min left
        (0, 300, 'tied_5min'),        # Tied, 5 min left
        (-7, 600, 'down7_10min'),     # Down 7, 10 min left
        (3, 900, 'up3_15min'),        # Up 3, 15 min left
    ]

    for score_diff, time_remaining, name in scenarios:
        fig = create_decision_heatmap(
            analyzer, score_diff, time_remaining,
            output_path=output_dir / f'decision_heatmap_{name}.png'
        )
        plt.close(fig)

    # 2. Uncertainty plots for example situations
    print("\nCreating uncertainty plots...")

    example_states = [
        GameState(field_pos=35, yards_to_go=1, score_diff=0, time_remaining=300),
        GameState(field_pos=45, yards_to_go=4, score_diff=-3, time_remaining=180),
        GameState(field_pos=10, yards_to_go=2, score_diff=0, time_remaining=120),
    ]

    for i, state in enumerate(example_states):
        fig = create_uncertainty_plot(
            analyzer, state,
            output_path=output_dir / f'uncertainty_example_{i+1}.png'
        )
        plt.close(fig)

    # 3. Time trends and team analysis (requires analysis results)
    analysis_path = tables_dir / 'decision_analysis_full.parquet'
    if analysis_path.exists():
        print("\nCreating time trend and team plots...")
        analysis_df = pd.read_parquet(analysis_path)

        fig = create_time_trend_plot(
            analysis_df,
            output_path=output_dir / 'time_trends.png'
        )
        if fig:
            plt.close(fig)

        fig = create_team_analysis_plot(
            analysis_df,
            output_path=output_dir / 'team_analysis.png'
        )
        if fig:
            plt.close(fig)
    else:
        print(f"\nAnalysis results not found at {analysis_path}")
        print("Run analyze_all_decisions.py first")

    print(f"\nAll visualizations saved to {output_dir}")


def create_off_def_quality_heatmap(
    analyzer: BayesianDecisionAnalyzer,
    field_pos: int = 40,
    yards_to_go: int = 1,
    score_diff: int = 0,
    time_remaining: int = 480,
    output_path: Path = None
) -> plt.Figure:
    """
    Create heat map showing how offense/defense quality shifts optimal decisions.

    Fixed situation: e.g., 4th & 1 at opponent's 40, tied, 8:00 left in Q4.
    X-axis: Offense quality (standard deviations from average)
    Y-axis: Defense quality (opponent's stopping ability, SDs from average)
    Color: Optimal decision (go=green, FG=yellow, punt=blue)

    This visualization shows the "it depends" has precise mathematical structure.
    """
    from models.hierarchical_off_def_model import HierarchicalOffDefConversionModel

    # Check if we have the off/def conversion model
    if not isinstance(analyzer.conversion, HierarchicalOffDefConversionModel):
        print("Warning: Analyzer does not have offense/defense model. Using synthetic effects.")
        has_real_effects = False
    else:
        has_real_effects = True
        # Get the SDs for scaling
        tau_off = np.sqrt(analyzer.conversion.tau_sq_off)
        tau_def = np.sqrt(analyzer.conversion.tau_sq_def)

    # Create grid of offense/defense quality
    # Range: -2 to +2 standard deviations
    off_range = np.linspace(-2, 2, 25)  # Offense quality (higher = better)
    def_range = np.linspace(-2, 2, 25)  # Defense quality (higher = worse at stopping)

    # Decision matrix and WP matrices
    decision_matrix = np.zeros((len(def_range), len(off_range)))
    wp_go_matrix = np.zeros((len(def_range), len(off_range)))
    wp_fg_matrix = np.zeros((len(def_range), len(off_range)))
    wp_punt_matrix = np.zeros((len(def_range), len(off_range)))

    # Compute optimal decision for each cell
    for i, def_sd in enumerate(def_range):
        for j, off_sd in enumerate(off_range):
            # Create synthetic team effects by modifying the conversion probability
            # We'll manually compute conversion prob with these effects
            if has_real_effects:
                # Use the model's tau values to scale
                off_effect = off_sd * tau_off
                def_effect = def_sd * tau_def
            else:
                # Use reasonable defaults (~0.15 log-odds SD)
                off_effect = off_sd * 0.15
                def_effect = def_sd * 0.15

            # Get base conversion probability at this distance
            base_conv = analyzer.conversion.get_posterior_samples(yards_to_go).mean()

            # Adjust for team effects (in log-odds space)
            from scipy.special import logit, expit
            base_logit = logit(np.clip(base_conv, 0.01, 0.99))
            adjusted_logit = base_logit + off_effect + def_effect
            adjusted_conv = expit(adjusted_logit)

            # Compute WP for each action using adjusted conversion prob
            # Go for it
            state = GameState(
                field_pos=int(field_pos),
                yards_to_go=int(yards_to_go),
                score_diff=score_diff,
                time_remaining=time_remaining
            )

            # State after conversion
            new_pos = max(field_pos - yards_to_go - 2, 1)
            wp_if_convert = analyzer.wp.get_win_prob(
                score_diff, time_remaining - 6, new_pos
            )

            # State after failed conversion
            opp_pos = 100 - field_pos
            wp_opponent_if_fail = analyzer.wp.get_win_prob(
                -score_diff, time_remaining - 6, opp_pos
            )
            wp_if_fail = 1 - wp_opponent_if_fail

            wp_go = adjusted_conv * wp_if_convert + (1 - adjusted_conv) * wp_if_fail

            # Punt (doesn't depend on conversion ability)
            punt_yards = analyzer.punt.get_posterior_samples(field_pos).mean()
            punt_landing = field_pos - punt_yards
            if punt_landing <= 0:
                opp_punt_pos = 75
            else:
                opp_punt_pos = 100 - punt_landing
            wp_opp_punt = analyzer.wp.get_win_prob(
                -score_diff, time_remaining - 5, opp_punt_pos
            )
            wp_punt = 1 - wp_opp_punt

            # Field goal (doesn't depend on conversion ability)
            fg_distance = field_pos + 17
            if fg_distance > 63:
                wp_fg = 0.0  # Not realistic
            else:
                p_make = analyzer.fg.get_posterior_samples(fg_distance).mean()
                # If make
                wp_opp_fg_make = analyzer.wp.get_win_prob(
                    -(score_diff + 3), time_remaining - 5, 75
                )
                wp_if_make = 1 - wp_opp_fg_make
                # If miss
                opp_fg_miss_pos = max(100 - field_pos, 80)
                wp_opp_fg_miss = analyzer.wp.get_win_prob(
                    -score_diff, time_remaining - 5, opp_fg_miss_pos
                )
                wp_if_miss = 1 - wp_opp_fg_miss
                wp_fg = p_make * wp_if_make + (1 - p_make) * wp_if_miss

            # Store WPs
            wp_go_matrix[i, j] = wp_go
            wp_punt_matrix[i, j] = wp_punt
            wp_fg_matrix[i, j] = wp_fg

            # Optimal decision
            wps = {'go_for_it': wp_go, 'punt': wp_punt, 'field_goal': wp_fg}
            optimal = max(wps, key=wps.get)

            if optimal == 'go_for_it':
                decision_matrix[i, j] = 0
            elif optimal == 'punt':
                decision_matrix[i, j] = 1
            else:
                decision_matrix[i, j] = 2

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Decision regions
    cmap_decision = plt.cm.colors.ListedColormap(['#2ecc71', '#3498db', '#e74c3c'])  # green, blue, red
    im1 = axes[0].imshow(decision_matrix, aspect='auto', cmap=cmap_decision,
                         extent=[off_range[0], off_range[-1],
                                def_range[-1], def_range[0]],
                         origin='upper')

    axes[0].set_xlabel('Offense Quality (σ from average)', fontsize=12)
    axes[0].set_ylabel('Opponent Defense Quality (σ from average)\n← Better at stopping | Worse at stopping →', fontsize=11)

    # Title with situation
    if field_pos > 50:
        location = f"own {100 - field_pos}"
    else:
        location = f"opponent's {field_pos}"
    score_str = "Tied" if score_diff == 0 else f"+{score_diff}" if score_diff > 0 else str(score_diff)
    mins = time_remaining // 60
    secs = time_remaining % 60

    axes[0].set_title(f'Optimal Decision by Team Quality\n'
                     f'4th & {yards_to_go} at {location}, {score_str}, {mins}:{secs:02d} left',
                     fontsize=13)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Go for it'),
                      Patch(facecolor='#3498db', label='Punt'),
                      Patch(facecolor='#e74c3c', label='Field Goal')]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Add contour lines for decision boundaries
    axes[0].contour(off_range, def_range, decision_matrix,
                   levels=[0.5, 1.5], colors='black', linewidths=2)

    # Mark average team point
    axes[0].plot(0, 0, 'ko', markersize=10, markerfacecolor='white', markeredgewidth=2)
    axes[0].annotate('League\nAverage', xy=(0, 0), xytext=(0.3, 0.5),
                    fontsize=9, ha='left')

    # Right: WP advantage of going for it
    wp_advantage = (wp_go_matrix - np.maximum(wp_punt_matrix, wp_fg_matrix)) * 100

    im2 = axes[1].imshow(wp_advantage, aspect='auto', cmap='RdYlGn',
                         extent=[off_range[0], off_range[-1],
                                def_range[-1], def_range[0]],
                         origin='upper', vmin=-5, vmax=5)

    axes[1].set_xlabel('Offense Quality (σ from average)', fontsize=12)
    axes[1].set_ylabel('Opponent Defense Quality (σ from average)\n← Better at stopping | Worse at stopping →', fontsize=11)
    axes[1].set_title('WP Advantage of Going For It\n(Green = Go preferred, Red = Kick preferred)',
                     fontsize=13)

    cbar = plt.colorbar(im2, ax=axes[1], label='WP Difference (percentage points)')

    # Add contour at break-even
    contour = axes[1].contour(off_range, def_range, wp_advantage,
                              levels=[0], colors='black', linewidths=2)
    axes[1].clabel(contour, inline=True, fontsize=9, fmt='Break-even')

    # Mark average team
    axes[1].plot(0, 0, 'ko', markersize=10, markerfacecolor='white', markeredgewidth=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved offense/defense quality heatmap to {output_path}")

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    create_all_visualizations(output_dir)
