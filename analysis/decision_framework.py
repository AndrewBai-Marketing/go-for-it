"""
Bayesian Decision Theory Framework for 4th Down Analysis

Implements:
1. Standard Bayesian expected WP maximization
2. Hierarchical models with team/kicker effects
3. CHMM (2025) misspecification-robust decision making with multiple structured models

Reference: Cerreia-Vioglio, Hansen, Maccheroni, Marinacci (2025).
"Making Decisions Under Model Misspecification." Review of Economic Studies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.bayesian_models import (
    ConversionModel, PuntModel, FieldGoalModel, WinProbabilityModel,
    HierarchicalConversionModel, HierarchicalFieldGoalModel,
    load_all_models
)
# Optional enhanced models (may not exist)
try:
    from models.bayesian_models import WeatherAwareFieldGoalModel
except ImportError:
    WeatherAwareFieldGoalModel = None
try:
    from models.bayesian_models import ContextAwareConversionModel
except ImportError:
    ContextAwareConversionModel = None
try:
    from models.bayesian_models import ContextAwarePuntModel
except ImportError:
    ContextAwarePuntModel = None
# Try importing off/def model (may not exist yet in all environments)
try:
    from models.hierarchical_off_def_model import HierarchicalOffDefConversionModel
    HAS_OFF_DEF_MODEL = True
except ImportError:
    HAS_OFF_DEF_MODEL = False


@dataclass
class GameState:
    """Represents a 4th down game state."""
    field_pos: int          # yardline_100: yards from opponent's end zone
    yards_to_go: int        # yards needed for first down
    score_diff: int         # positive = winning
    time_remaining: int     # seconds remaining in game
    timeout_diff: int = 0   # timeout differential
    off_team: str = None    # offensive team abbreviation (e.g., 'PHI')
    def_team: str = None    # defensive team abbreviation (e.g., 'NYG')
    kicker_id: str = None   # kicker player ID for FG model
    punter_id: str = None   # punter player ID for punt model

    # Contextual factors for enhanced models
    is_home: bool = False   # True if offense is home team
    is_dome: bool = False   # True if playing in dome/closed stadium
    temp: float = 65.0      # Temperature in Fahrenheit
    wind: float = 5.0       # Wind speed in mph

    # Backward compatibility: alias 'team' to 'off_team'
    @property
    def team(self):
        return self.off_team

    @team.setter
    def team(self, value):
        self.off_team = value


# Clock consumption constants (fallback when Bayesian model not available)
# These represent mean time until next change of possession for each action-outcome
CLOCK_CONSUMPTION = {
    'go_convert': 151,    # Go for it + convert: retain possession, run more plays
    'go_fail': 48,        # Go for it + fail: opponent gets ball, runs their drive
    'punt': 69,           # Punt: opponent gets ball at worse field position
    'fg_make': 99,        # FG make: kickoff + opponent drive
    'fg_miss': 48,        # FG miss: opponent gets ball at LOS, similar to failed conversion
}

# Try to import ClockConsumptionModel
try:
    from models.bayesian_models import ClockConsumptionModel
    HAS_CLOCK_MODEL = True
except ImportError:
    HAS_CLOCK_MODEL = False

# Threshold for "end of game" - when time remaining is below this,
# we use immediate play time rather than full drive time.
# This is because with <2 min left, teams are in hurry-up mode and
# the drive time estimates don't apply.
END_OF_GAME_THRESHOLD = 120  # 2 minutes

# Immediate play time (used for end-of-game scenarios)
IMMEDIATE_PLAY_TIME = {
    'go': 6,        # Conversion attempt takes ~6 seconds
    'punt': 5,      # Punt play takes ~5 seconds
    'fg': 5,        # Field goal attempt takes ~5 seconds
}


@dataclass
class DecisionResult:
    """Result of analyzing a 4th down decision."""
    state: GameState

    # Expected WP for each action
    wp_go: float
    wp_punt: float
    wp_fg: float

    # Posterior distributions
    wp_go_samples: np.ndarray
    wp_punt_samples: np.ndarray
    wp_fg_samples: np.ndarray

    # Optimal decision
    optimal_action: str

    # Probability each action is best
    prob_go_best: float
    prob_punt_best: float
    prob_fg_best: float

    # Value of optimal vs second best
    decision_margin: float


class BayesianDecisionAnalyzer:
    """
    Standard Bayesian decision analysis.
    Maximizes expected win probability integrating over parameter uncertainty.

    Supports hierarchical models with team/kicker effects when available.
    Now also supports offense/defense fixed effects for conversion model.

    Clock Model:
    ------------
    When use_clock_model=True (default), the analyzer uses a hybrid approach:

    **Mid-game (>2 minutes remaining):**
    Accounts for asymmetric clock consumption between action-outcome pairs:
    - Go + Convert: ~151s until next change of possession (retain ball, run plays)
    - Go + Fail: ~48s (opponent gets ball, runs their drive)
    - Punt: ~69s (opponent gets ball at worse position)
    - FG Make: ~99s (kickoff + opponent drive)
    - FG Miss: ~48s (opponent gets ball at LOS)

    This is critical for situations where clock management matters.
    When winning, burning more clock (converting) helps protect the lead.
    When losing, burning clock hurts because less time remains to catch up.

    **End-of-game (<2 minutes remaining):**
    Uses immediate play time (~5-6 seconds per play) rather than full drive time.
    This is because in end-of-game scenarios:
    1. Teams are in hurry-up mode, not running full drives
    2. Scoring happens immediately (TD/FG) or game ends
    3. Full drive time estimates don't apply

    This hybrid approach allows proper handling of both mid-game clock
    dynamics AND end-of-game desperation scenarios.
    """

    def __init__(self, models: Dict, use_clock_model: bool = True):
        """
        Initialize the decision analyzer.

        Args:
            models: Dict with 'conversion', 'punt', 'fg', 'wp' model objects
                   Optionally includes 'clock' for Bayesian clock model
            use_clock_model: If True, use clock consumption model for time dynamics.
                            This properly accounts for time burned by subsequent drives
                            and is essential for late-game decision accuracy.
        """
        self.conversion = models['conversion']
        self.punt = models['punt']
        self.fg = models['fg']
        self.wp = models['wp']
        self.n_samples = self.wp.n_samples
        self.use_clock_model = use_clock_model

        # Check for Bayesian clock model
        if 'clock' in models and models['clock'] is not None:
            self.clock = models['clock']
            self.has_bayesian_clock = True
        else:
            self.clock = None
            self.has_bayesian_clock = False

        # Check if we have hierarchical models
        self.has_team_effects = isinstance(self.conversion, HierarchicalConversionModel)
        self.has_kicker_effects = isinstance(self.fg, HierarchicalFieldGoalModel)

        # Check if we have enhanced context-aware models
        self.has_context_aware_conversion = (ContextAwareConversionModel is not None and
                                              isinstance(self.conversion, ContextAwareConversionModel))
        self.has_weather_aware_fg = (WeatherAwareFieldGoalModel is not None and
                                      isinstance(self.fg, WeatherAwareFieldGoalModel))
        self.has_context_aware_punt = (ContextAwarePuntModel is not None and
                                        isinstance(self.punt, ContextAwarePuntModel))

        # Check if we have offense/defense model
        self.has_off_def_effects = (HAS_OFF_DEF_MODEL and
                                     isinstance(self.conversion, HierarchicalOffDefConversionModel))

    def _get_clock_consumption(self, category: str, posterior_idx: int = None) -> float:
        """
        Get clock consumption for a given action-outcome category.

        If Bayesian clock model is available, draws from posterior.
        Otherwise uses fixed constants.

        Args:
            category: One of 'go_convert', 'go_fail', 'punt', 'fg_make', 'fg_miss'
            posterior_idx: If specified, return specific posterior sample
        """
        if self.has_bayesian_clock and self.clock is not None:
            return self.clock.get_time_consumed(category, posterior_idx)
        else:
            return CLOCK_CONSUMPTION[category]

    def _get_clock_consumption_samples(self, category: str) -> np.ndarray:
        """
        Get all posterior samples for clock consumption in given category.

        If Bayesian clock model is available, returns posterior samples.
        Otherwise returns array of constant values.
        """
        if self.has_bayesian_clock and self.clock is not None:
            return self.clock.get_posterior_samples(category)
        else:
            return np.full(self.n_samples, CLOCK_CONSUMPTION[category])

    def _state_after_conversion(self, state: GameState, yards_gained: int = None,
                                  use_clock_model: bool = True,
                                  posterior_idx: int = None) -> Tuple[int, int]:
        """
        Return (new_field_pos, time_remaining) after successful conversion.
        Assumes average gain of yards_to_go + 2 if not specified.

        If use_clock_model=True, uses the Bayesian clock consumption model
        which accounts for the full time until next change of possession.

        For end-of-game scenarios (under 2 minutes), uses immediate play time
        instead of full drive time, as teams are in hurry-up mode.
        """
        if yards_gained is None:
            yards_gained = state.yards_to_go + 2  # Assume slightly more than needed

        new_field_pos = max(state.field_pos - yards_gained, 1)  # Don't go past goal line

        if use_clock_model and state.time_remaining > END_OF_GAME_THRESHOLD:
            # Mid-game: account for full drive time after conversion
            time_consumed = self._get_clock_consumption('go_convert', posterior_idx)
        else:
            # End-of-game or legacy mode: just the immediate play time
            time_consumed = IMMEDIATE_PLAY_TIME['go']

        new_time = max(0, state.time_remaining - time_consumed)
        return new_field_pos, new_time

    def _state_after_failed_conversion(self, state: GameState,
                                        use_clock_model: bool = True,
                                        posterior_idx: int = None) -> Tuple[int, int]:
        """
        Return (opponent's field_pos, time_remaining) after failed conversion.
        Opponent gets ball at spot of failed attempt.

        If use_clock_model=True, uses the Bayesian clock consumption model
        which accounts for the opponent's subsequent drive time.

        For end-of-game scenarios (under 2 minutes), uses immediate play time.
        """
        # Opponent's yardline_100 is 100 - current field_pos
        opp_field_pos = 100 - state.field_pos

        if use_clock_model and state.time_remaining > END_OF_GAME_THRESHOLD:
            # Mid-game: account for opponent's drive time
            time_consumed = self._get_clock_consumption('go_fail', posterior_idx)
        else:
            # End-of-game or legacy mode: just the immediate play time
            time_consumed = IMMEDIATE_PLAY_TIME['go']

        new_time = max(0, state.time_remaining - time_consumed)
        return opp_field_pos, new_time

    def _state_after_punt(self, state: GameState, net_yards: float,
                          use_clock_model: bool = True,
                          posterior_idx: int = None) -> Tuple[int, int]:
        """
        Return (opponent's field_pos, time_remaining) after punt.

        If use_clock_model=True, uses the Bayesian clock consumption model
        which accounts for the opponent's subsequent drive time.

        For end-of-game scenarios (under 2 minutes), uses immediate play time.
        """
        punt_landing = state.field_pos - net_yards

        # Touchback if punt goes into end zone
        if punt_landing <= 0:
            opp_field_pos = 75  # Ball at opponent's 25 (75 yards from their end zone)
        else:
            opp_field_pos = 100 - punt_landing

        opp_field_pos = max(min(opp_field_pos, 99), 1)

        if use_clock_model and state.time_remaining > END_OF_GAME_THRESHOLD:
            # Mid-game: account for opponent's drive time
            time_consumed = self._get_clock_consumption('punt', posterior_idx)
        else:
            # End-of-game or legacy mode: just the punt play time
            time_consumed = IMMEDIATE_PLAY_TIME['punt']

        new_time = max(0, state.time_remaining - time_consumed)
        return opp_field_pos, new_time

    def _state_after_fg_make(self, state: GameState,
                              use_clock_model: bool = True,
                              posterior_idx: int = None) -> Tuple[int, int, int]:
        """
        Return (opponent's field_pos, time_remaining, new_score_diff) after made FG.
        Opponent gets ball at ~25 after kickoff.

        If use_clock_model=True, uses the Bayesian clock consumption model
        which accounts for the kickoff + opponent's subsequent drive time.

        For end-of-game scenarios (under 2 minutes), uses immediate play time.
        """
        opp_field_pos = 75  # Opponent at their own 25
        new_score_diff = state.score_diff + 3

        if use_clock_model and state.time_remaining > END_OF_GAME_THRESHOLD:
            # Mid-game: FG attempt + kickoff + opponent drive
            time_consumed = self._get_clock_consumption('fg_make', posterior_idx)
        else:
            # End-of-game or legacy mode: just the FG play time
            time_consumed = IMMEDIATE_PLAY_TIME['fg']

        new_time = max(0, state.time_remaining - time_consumed)
        return opp_field_pos, new_time, new_score_diff

    def _state_after_fg_miss(self, state: GameState,
                              use_clock_model: bool = True,
                              posterior_idx: int = None) -> Tuple[int, int]:
        """
        Return (opponent's field_pos, time_remaining) after missed FG.
        Opponent gets ball at LOS or their 20, whichever is better for them.

        If use_clock_model=True, uses the Bayesian clock consumption model
        which accounts for the opponent's subsequent drive time.

        For end-of-game scenarios (under 2 minutes), uses immediate play time.
        """
        opp_field_pos_at_los = 100 - state.field_pos
        opp_field_pos = max(opp_field_pos_at_los, 80)  # At least at their 20

        if use_clock_model and state.time_remaining > END_OF_GAME_THRESHOLD:
            # Mid-game: opponent drive time (similar to failed conversion)
            time_consumed = self._get_clock_consumption('fg_miss', posterior_idx)
        else:
            # End-of-game or legacy mode: just the FG play time
            time_consumed = IMMEDIATE_PLAY_TIME['fg']

        new_time = max(0, state.time_remaining - time_consumed)
        return opp_field_pos, new_time

    def compute_wp_go_for_it(self, state: GameState) -> np.ndarray:
        """
        Compute posterior distribution of WP if going for it.

        WP_go = P(convert) * WP(state_if_convert) + P(fail) * WP(state_if_fail)

        With clock model enabled, this properly accounts for the asymmetric
        clock consumption: converting burns ~151s (retain possession),
        while failing burns only ~48s (opponent gets ball).
        """
        # Get conversion probabilities (all posterior samples)
        # Use context-aware model if available, else fall back to hierarchical
        if self.has_context_aware_conversion:
            p_convert = self.conversion.get_posterior_samples(
                state.yards_to_go, off_team=state.off_team, def_team=state.def_team,
                is_home=state.is_home, is_dome=state.is_dome
            )
        elif self.has_off_def_effects and (state.off_team is not None or state.def_team is not None):
            p_convert = self.conversion.get_posterior_samples(
                state.yards_to_go, off_team=state.off_team, def_team=state.def_team
            )
        elif self.has_team_effects and state.off_team is not None:
            p_convert = self.conversion.get_posterior_samples(state.yards_to_go, team=state.off_team)
        else:
            p_convert = self.conversion.get_posterior_samples(state.yards_to_go)

        # State after conversion (uses clock model if enabled)
        new_pos, new_time = self._state_after_conversion(state, use_clock_model=self.use_clock_model)
        wp_if_convert = self.wp.get_posterior_samples(
            state.score_diff, new_time, new_pos, state.timeout_diff
        )

        # State after failed conversion (opponent has ball)
        # WP for us = 1 - WP for opponent
        opp_pos, opp_time = self._state_after_failed_conversion(state, use_clock_model=self.use_clock_model)
        wp_opponent = self.wp.get_posterior_samples(
            -state.score_diff, opp_time, opp_pos, -state.timeout_diff
        )
        wp_if_fail = 1 - wp_opponent

        # Expected WP
        wp_go = p_convert * wp_if_convert + (1 - p_convert) * wp_if_fail
        return wp_go

    def compute_wp_punt(self, state: GameState) -> np.ndarray:
        """
        Compute posterior distribution of WP if punting.

        WP_punt = 1 - WP_opponent(state_after_punt)

        With clock model enabled, accounts for the ~69s typically burned
        by the punt play + opponent's subsequent drive.
        """
        # Get expected punt distance - use context-aware model if available
        if self.has_context_aware_punt:
            punt_yards = self.punt.get_posterior_samples(
                state.field_pos, wind=state.wind, is_dome=state.is_dome,
                punter_id=state.punter_id
            )
        else:
            punt_yards = self.punt.get_posterior_samples(state.field_pos)

        # Use mean punt distance for state transition
        mean_punt = punt_yards.mean()
        opp_pos, new_time = self._state_after_punt(state, mean_punt, use_clock_model=self.use_clock_model)

        # Opponent's WP
        wp_opponent = self.wp.get_posterior_samples(
            -state.score_diff, new_time, opp_pos, -state.timeout_diff
        )
        wp_punt = 1 - wp_opponent
        return wp_punt

    def compute_wp_field_goal(self, state: GameState) -> np.ndarray:
        """
        Compute posterior distribution of WP if attempting field goal.

        WP_fg = P(make) * WP(state_if_make) + P(miss) * WP(state_if_miss)

        With clock model enabled, accounts for:
        - FG make: ~99s (kick + kickoff + opponent drive)
        - FG miss: ~48s (opponent gets ball, runs their drive)
        """
        fg_distance = state.field_pos + 17  # Add 17 for snap/hold

        # Use weather-aware model if available (handles extreme distances internally)
        if self.has_weather_aware_fg:
            # Weather-aware model handles weather effects, but still cap at realistic distances
            # NFL record is 66 yards, but beyond 63 is extremely rare and risky
            if fg_distance > 63:
                return np.full(self.n_samples, -np.inf)
            p_make = self.fg.get_posterior_samples(
                fg_distance, kicker_id=state.kicker_id,
                temp=state.temp, wind=state.wind
            )
        else:
            # Fall back to original logic with hard cutoffs
            # FG is not a realistic option beyond ~63 yards (NFL record is 66)
            if fg_distance > 63:
                return np.full(self.n_samples, -np.inf)

            # For long FGs (58-63 yards), cap probability at realistic levels
            if fg_distance > 60:
                p_make = np.full(self.n_samples, 0.10)
            elif fg_distance > 58:
                p_make = np.full(self.n_samples, 0.20)
            elif self.has_kicker_effects and state.kicker_id is not None:
                p_make = self.fg.get_posterior_samples(fg_distance, kicker_id=state.kicker_id)
            else:
                p_make = self.fg.get_posterior_samples(fg_distance)

        # State after make (uses clock model if enabled)
        opp_pos, new_time, new_score = self._state_after_fg_make(state, use_clock_model=self.use_clock_model)
        wp_opponent_if_make = self.wp.get_posterior_samples(
            -new_score, new_time, opp_pos, -state.timeout_diff
        )
        wp_if_make = 1 - wp_opponent_if_make

        # State after miss (uses clock model if enabled)
        opp_pos, new_time = self._state_after_fg_miss(state, use_clock_model=self.use_clock_model)
        wp_opponent_if_miss = self.wp.get_posterior_samples(
            -state.score_diff, new_time, opp_pos, -state.timeout_diff
        )
        wp_if_miss = 1 - wp_opponent_if_miss

        wp_fg = p_make * wp_if_make + (1 - p_make) * wp_if_miss
        return wp_fg

    def analyze(self, state: GameState) -> DecisionResult:
        """
        Full Bayesian decision analysis for a 4th down situation.
        """
        # Compute WP distributions for each action
        wp_go_samples = self.compute_wp_go_for_it(state)
        wp_punt_samples = self.compute_wp_punt(state)
        wp_fg_samples = self.compute_wp_field_goal(state)

        # Handle infeasible actions (e.g., FG too long)
        # Replace -inf with 0 for mean calculation but track infeasibility
        fg_infeasible = np.isinf(wp_fg_samples).all()
        if fg_infeasible:
            wp_fg_samples_clean = np.zeros_like(wp_fg_samples)
        else:
            wp_fg_samples_clean = wp_fg_samples

        # Expected WP (posterior mean)
        wp_go = wp_go_samples.mean()
        wp_punt = wp_punt_samples.mean()
        wp_fg = 0.0 if fg_infeasible else wp_fg_samples.mean()

        # Determine optimal action (only consider feasible actions)
        if fg_infeasible:
            wps = {'go_for_it': wp_go, 'punt': wp_punt}
        else:
            wps = {'go_for_it': wp_go, 'punt': wp_punt, 'field_goal': wp_fg}
        optimal_action = max(wps, key=wps.get)

        # Probability each action is best (across posterior samples)
        if fg_infeasible:
            go_best = wp_go_samples > wp_punt_samples
            punt_best = wp_punt_samples > wp_go_samples
            fg_best = np.zeros_like(go_best, dtype=bool)
        else:
            go_best = (wp_go_samples > wp_punt_samples) & (wp_go_samples > wp_fg_samples)
            punt_best = (wp_punt_samples > wp_go_samples) & (wp_punt_samples > wp_fg_samples)
            fg_best = (wp_fg_samples > wp_go_samples) & (wp_fg_samples > wp_punt_samples)

        prob_go_best = go_best.mean()
        prob_punt_best = punt_best.mean()
        prob_fg_best = fg_best.mean()

        # Decision margin (difference between best and second-best)
        if fg_infeasible:
            sorted_wps = sorted([wp_go, wp_punt], reverse=True)
        else:
            sorted_wps = sorted([wp_go, wp_punt, wp_fg], reverse=True)
        decision_margin = sorted_wps[0] - sorted_wps[1] if len(sorted_wps) > 1 else 0.0

        return DecisionResult(
            state=state,
            wp_go=wp_go,
            wp_punt=wp_punt,
            wp_fg=wp_fg,
            wp_go_samples=wp_go_samples,
            wp_punt_samples=wp_punt_samples,
            wp_fg_samples=wp_fg_samples_clean,
            optimal_action=optimal_action,
            prob_go_best=prob_go_best,
            prob_punt_best=prob_punt_best,
            prob_fg_best=prob_fg_best,
            decision_margin=decision_margin
        )


class CHMMDecisionAnalyzer:
    """
    CHMM (2025) Misspecification-Robust Decision Analysis.

    Implements the decision criterion from Cerreia-Vioglio, Hansen, Maccheroni, Marinacci:
    V(action) = min_{q ∈ Q} [ -λ × log E_q[exp(-U/λ)] ]

    Where:
    - Q is the set of structured models
    - λ is the misspecification index (lower = more fear of misspecification)
    - U is win probability (our utility)
    """

    def __init__(self, structured_models: Dict[str, Dict]):
        """
        Args:
            structured_models: Dict of model name -> dict with 'conversion', 'punt', 'fg', 'wp' models
        """
        self.structured_models = structured_models
        self.model_names = list(structured_models.keys())

        # Create Bayesian analyzers for each structured model
        self.analyzers = {
            name: BayesianDecisionAnalyzer(models)
            for name, models in structured_models.items()
        }

    def risk_sensitive_value(self, action: str, state: GameState,
                              model_name: str, lambda_param: float) -> float:
        """
        Compute risk-sensitive value for an action under a specific model.

        V_q(action) = -λ × log E_q[exp(-U/λ)]

        For λ → ∞, this approaches E_q[U] (risk-neutral)
        For λ → 0, this approaches min support of U (extreme risk aversion)
        """
        analyzer = self.analyzers[model_name]

        if action == 'go_for_it':
            wp_samples = analyzer.compute_wp_go_for_it(state)
        elif action == 'punt':
            wp_samples = analyzer.compute_wp_punt(state)
        elif action == 'field_goal':
            wp_samples = analyzer.compute_wp_field_goal(state)
        else:
            raise ValueError(f"Unknown action: {action}")

        if lambda_param >= 100:  # Treat as risk-neutral
            return wp_samples.mean()

        # Risk-sensitive expectation: -λ × log E[exp(-U/λ)]
        # Use log-sum-exp trick for numerical stability
        scaled_wp = -wp_samples / lambda_param
        max_val = scaled_wp.max()
        log_expectation = max_val + np.log(np.mean(np.exp(scaled_wp - max_val)))

        return -lambda_param * log_expectation

    def misspecification_robust_value(self, action: str, state: GameState,
                                       lambda_param: float) -> Tuple[float, str, Dict[str, float]]:
        """
        Compute worst-case risk-sensitive value across all structured models.

        V(action) = min_{q ∈ Q} V_q(action)

        Returns:
            - worst_case_value: Minimum value across models
            - worst_case_model: Name of the model giving minimum
            - all_values: Dict of model_name -> value
        """
        values = {}
        for model_name in self.model_names:
            values[model_name] = self.risk_sensitive_value(
                action, state, model_name, lambda_param
            )

        worst_case_model = min(values, key=values.get)
        worst_case_value = values[worst_case_model]

        return worst_case_value, worst_case_model, values

    def optimal_decision(self, state: GameState, lambda_param: float) -> Dict:
        """
        Find optimal action under CHMM criterion.

        a* = argmax_a min_{q ∈ Q} V_q(a)

        Returns dict with:
            - optimal_action
            - action_values: Dict of action -> worst-case value
            - worst_models: Dict of action -> which model was worst case
            - all_model_values: Dict of action -> {model -> value}
        """
        actions = ['go_for_it', 'punt', 'field_goal']

        action_values = {}
        worst_models = {}
        all_model_values = {}

        for action in actions:
            value, worst_model, values = self.misspecification_robust_value(
                action, state, lambda_param
            )
            action_values[action] = value
            worst_models[action] = worst_model
            all_model_values[action] = values

        optimal_action = max(action_values, key=action_values.get)

        return {
            'optimal_action': optimal_action,
            'action_values': action_values,
            'worst_models': worst_models,
            'all_model_values': all_model_values
        }

    def compare_lambda_values(self, state: GameState,
                               lambda_values: List[float] = None) -> pd.DataFrame:
        """
        Show how optimal decision varies with λ.
        """
        if lambda_values is None:
            lambda_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 100.0]

        results = []
        for lam in lambda_values:
            decision = self.optimal_decision(state, lam)
            results.append({
                'lambda': lam,
                'optimal_action': decision['optimal_action'],
                'wp_go': decision['action_values']['go_for_it'],
                'wp_punt': decision['action_values']['punt'],
                'wp_fg': decision['action_values']['field_goal'],
                'worst_model_go': decision['worst_models']['go_for_it'],
                'worst_model_punt': decision['worst_models']['punt'],
                'worst_model_fg': decision['worst_models']['field_goal'],
            })

        return pd.DataFrame(results)


def create_structured_models(data_dir: Path, models_dir: Path,
                              n_bootstrap: int = 1000) -> Dict[str, Dict]:
    """
    Create multiple structured models for CHMM analysis.

    Creates variations in:
    1. Different feature specifications for WP model
    2. Different subsets of data
    3. Different regularization
    """
    from sklearn.linear_model import LogisticRegression

    print("Creating structured model set for CHMM analysis...")

    # Load base data
    all_plays = pd.read_parquet(data_dir / 'cleaned_pbp.parquet')
    attempts = pd.read_parquet(data_dir / 'fourth_down_attempts.parquet')
    punts = pd.read_parquet(data_dir / 'punts.parquet')
    fgs = pd.read_parquet(data_dir / 'field_goals.parquet')

    # Prepare clean data
    df_clean = all_plays.dropna(subset=['score_diff', 'game_seconds_remaining',
                                         'yardline_100', 'timeout_diff', 'team_won']).copy()

    structured_models = {}

    # Model 1: Baseline (same as single model)
    print("  Fitting baseline model...")
    baseline_conversion = ConversionModel()
    baseline_conversion.fit(attempts, n_bootstrap=n_bootstrap)

    baseline_punt = PuntModel()
    baseline_punt.fit(punts, n_bootstrap=n_bootstrap)

    baseline_fg = FieldGoalModel()
    baseline_fg.fit(fgs, n_bootstrap=n_bootstrap)

    baseline_wp = WinProbabilityModel()
    baseline_wp.fit(all_plays, n_bootstrap=n_bootstrap)

    structured_models['baseline'] = {
        'conversion': baseline_conversion,
        'punt': baseline_punt,
        'fg': baseline_fg,
        'wp': baseline_wp
    }

    # Model 2: Recent years only (2022-2024) - captures recent trends
    print("  Fitting recent-years model...")
    recent_plays = all_plays[all_plays['season'] >= 2022].copy()
    recent_attempts = attempts[attempts['season'] >= 2022].copy() if 'season' in attempts.columns else attempts

    recent_wp = WinProbabilityModel()
    recent_wp.fit(recent_plays, n_bootstrap=n_bootstrap)

    structured_models['recent_years'] = {
        'conversion': baseline_conversion,  # Reuse
        'punt': baseline_punt,
        'fg': baseline_fg,
        'wp': recent_wp
    }

    # Model 3: Conservative conversion estimates (lower bound)
    print("  Fitting conservative conversion model...")
    conservative_conversion = ConversionModel()
    conservative_conversion.fit(attempts, n_bootstrap=n_bootstrap)
    # Shift conversion probabilities down by adjusting intercept
    conservative_conversion.samples[:, 0] -= 0.2  # Lower baseline

    structured_models['conservative_conversion'] = {
        'conversion': conservative_conversion,
        'punt': baseline_punt,
        'fg': baseline_fg,
        'wp': baseline_wp
    }

    # Model 4: Aggressive conversion estimates (upper bound)
    print("  Fitting aggressive conversion model...")
    aggressive_conversion = ConversionModel()
    aggressive_conversion.fit(attempts, n_bootstrap=n_bootstrap)
    aggressive_conversion.samples[:, 0] += 0.2  # Higher baseline

    structured_models['aggressive_conversion'] = {
        'conversion': aggressive_conversion,
        'punt': baseline_punt,
        'fg': baseline_fg,
        'wp': baseline_wp
    }

    # Model 5: Different WP specification (no time interaction)
    print("  Fitting alternative WP specification...")

    alt_wp = WinProbabilityModel()
    # Fit with simpler specification by zeroing out interaction term
    alt_wp.fit(all_plays, n_bootstrap=n_bootstrap)
    alt_wp.samples[:, 3] = 0  # Zero out interaction term

    structured_models['no_interaction'] = {
        'conversion': baseline_conversion,
        'punt': baseline_punt,
        'fg': baseline_fg,
        'wp': alt_wp
    }

    print(f"Created {len(structured_models)} structured models")

    # Save structured models
    with open(models_dir / 'structured_models.pkl', 'wb') as f:
        pickle.dump(structured_models, f)

    return structured_models


def load_structured_models(models_dir: Path) -> Dict[str, Dict]:
    """Load saved structured models."""
    with open(models_dir / 'structured_models.pkl', 'rb') as f:
        return pickle.load(f)


def load_models_with_off_def(models_dir: Path) -> Dict:
    """
    Load models including the offense/defense hierarchical model if available.

    Returns dict with keys: 'conversion', 'punt', 'fg', 'wp'
    """
    from models.bayesian_models import PuntModel, WinProbabilityModel, HierarchicalFieldGoalModel

    models = {}

    # Try to load off/def conversion model first
    off_def_path = models_dir / 'hierarchical_off_def_conversion_model.pkl'
    if off_def_path.exists() and HAS_OFF_DEF_MODEL:
        print("Loading offense/defense conversion model...")
        models['conversion'] = HierarchicalOffDefConversionModel().load(off_def_path)
    else:
        # Fall back to standard hierarchical model
        hier_path = models_dir / 'hierarchical_conversion_model.pkl'
        if hier_path.exists():
            print("Loading hierarchical conversion model...")
            from models.bayesian_models import HierarchicalConversionModel
            models['conversion'] = HierarchicalConversionModel().load(hier_path)
        else:
            print("Loading basic conversion model...")
            from models.bayesian_models import ConversionModel
            models['conversion'] = ConversionModel().load(models_dir / 'conversion_model.pkl')

    # Load other models
    models['punt'] = PuntModel().load(models_dir / 'punt_model.pkl')

    hier_fg_path = models_dir / 'hierarchical_fg_model.pkl'
    if hier_fg_path.exists():
        models['fg'] = HierarchicalFieldGoalModel().load(hier_fg_path)
    else:
        from models.bayesian_models import FieldGoalModel
        models['fg'] = FieldGoalModel().load(models_dir / 'fg_model.pkl')

    models['wp'] = WinProbabilityModel().load(models_dir / 'wp_model.pkl')

    return models


if __name__ == "__main__":
    # Demo analysis
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'

    # Load models (with off/def effects if available)
    print("Loading models...")
    models = load_models_with_off_def(models_dir)

    # Create Bayesian analyzer
    analyzer = BayesianDecisionAnalyzer(models)
    print(f"Has off/def effects: {analyzer.has_off_def_effects}")

    # Example 4th down situations - now with matchups
    print("\n" + "="*80)
    print("BAYESIAN DECISION ANALYSIS - EXAMPLE SCENARIOS")
    print("="*80)

    # Generic scenarios (no team info)
    scenarios = [
        GameState(field_pos=40, yards_to_go=1, score_diff=0, time_remaining=1800),  # 4th&1 at opp 40, tied, 30min
        GameState(field_pos=35, yards_to_go=3, score_diff=-3, time_remaining=300),  # 4th&3 at opp 35, down 3, 5min
    ]

    for state in scenarios:
        result = analyzer.analyze(state)

        print(f"\n4th & {state.yards_to_go} at opponent's {state.field_pos}")
        print(f"Score: {'Tied' if state.score_diff == 0 else f'+{state.score_diff}' if state.score_diff > 0 else state.score_diff}")
        print(f"Time remaining: {state.time_remaining // 60} min")
        print("-" * 40)
        print(f"Expected WP if go for it:   {result.wp_go:.1%}")
        print(f"Expected WP if punt:        {result.wp_punt:.1%}")
        print(f"Expected WP if field goal:  {result.wp_fg:.1%}")
        print("-" * 40)
        print(f"OPTIMAL: {result.optimal_action.upper()}")
        print(f"P(go is best): {result.prob_go_best:.0%}")

    # Matchup scenarios (with team info)
    if analyzer.has_off_def_effects:
        print("\n" + "="*80)
        print("MATCHUP ANALYSIS - SAME SITUATION, DIFFERENT TEAMS")
        print("="*80)

        # 4th & 3 at opponent's 40, tied, 10 min left
        base_state = dict(field_pos=40, yards_to_go=3, score_diff=0, time_remaining=600)

        matchups = [
            ('PHI', 'CLE', 'Good offense vs bad defense'),
            ('CHI', 'BAL', 'Bad offense vs good defense'),
            ('KC', 'NE', 'Good offense vs good defense'),
            (None, None, 'League average'),
        ]

        for off, def_, desc in matchups:
            state = GameState(**base_state, off_team=off, def_team=def_)
            result = analyzer.analyze(state)

            print(f"\n{desc} ({off or 'AVG'} vs {def_ or 'AVG'})")
            print(f"  WP(go): {result.wp_go:.1%} | WP(punt): {result.wp_punt:.1%}")
            print(f"  Optimal: {result.optimal_action.upper()} (margin: {result.decision_margin:.1%})")
