"""
Two-Point Conversion Decision Analysis

Evaluates whether teams should go for 2 or kick the extra point (PAT).
This analysis is for the post-2015 rule change era where PATs are kicked
from the 15-yard line (33-yard kick).

Key insight: The decision is NOT about expected points. It's about win probability.
Going for 2 introduces variance, which can be valuable or harmful depending on
the game situation (score differential, time remaining).

- When trailing significantly, variance helps (go for 2)
- When leading comfortably, variance hurts (kick PAT)
- The break-even points create interesting strategic thresholds (e.g., up 14 vs 13)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats
import pickle


@dataclass
class TwoPointState:
    """Game state after a touchdown, before PAT decision."""
    score_diff_pre_td: int    # Score diff BEFORE the touchdown (positive = leading)
    time_remaining: int       # Seconds remaining in game
    posteam: str = None       # Team that scored the TD (offensive team for 2pt)
    defteam: str = None       # Opponent (defensive team for 2pt)
    kicker_id: str = None     # Kicker player ID (for PAT)


class TwoPointConversionModel:
    """
    Model for 2-point conversion success probability.

    Uses Bayesian estimation with bootstrap samples for uncertainty quantification.
    """

    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.success_rate_samples = None
        self.overall_rate = None

    def fit(self, two_pt_data: pd.DataFrame):
        """
        Fit the model using historical 2-point conversion data.

        Args:
            two_pt_data: DataFrame with 'two_point_conv_result' column
        """
        successes = (two_pt_data['two_point_conv_result'] == 'success').sum()
        attempts = len(two_pt_data)

        self.overall_rate = successes / attempts

        # Bayesian posterior: Beta(successes + 1, failures + 1)
        # Using conjugate prior Beta(1, 1) = Uniform
        alpha = successes + 1
        beta = (attempts - successes) + 1

        self.success_rate_samples = np.random.beta(alpha, beta, self.n_samples)

        print(f"2-point conversion model fit:")
        print(f"  Data: {successes}/{attempts} = {self.overall_rate:.1%}")
        print(f"  Posterior: Beta({alpha}, {beta})")
        print(f"  95% CI: [{np.percentile(self.success_rate_samples, 2.5):.1%}, {np.percentile(self.success_rate_samples, 97.5):.1%}]")

    def get_posterior_samples(self, team: str = None) -> np.ndarray:
        """Return posterior samples of success probability."""
        return self.success_rate_samples

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class HierarchicalTwoPointModel:
    """
    Hierarchical Bayesian model for 2-point conversion success (offense only).

    DEPRECATED: Use HierarchicalOffDefTwoPointModel instead, which captures
    both offensive conversion ability AND defensive stopping ability.

    This simpler model only uses posteam (offensive team) effects.

    Model:
        logit(p_team) = mu + alpha_team
        alpha_team ~ N(0, tau^2)

    Where:
        - mu is the league-average log-odds of conversion
        - alpha_team is the offensive team's conversion ability
        - tau^2 is the between-team variance (estimated via empirical Bayes)
    """

    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.mu = None  # League average (log-odds)
        self.tau_sq = None  # Between-team variance
        self.team_effects = {}  # team -> array of posterior samples
        self.teams = []
        self.overall_rate = None

    def fit(self, two_pt_data: pd.DataFrame):
        """
        Fit hierarchical model using historical 2-point conversion data.

        Args:
            two_pt_data: DataFrame with 'two_point_conv_result' and 'posteam' columns
        """
        from scipy.special import logit, expit
        from scipy.optimize import minimize_scalar

        # Compute team-level statistics
        two_pt_data = two_pt_data.copy()
        two_pt_data['success'] = (two_pt_data['two_point_conv_result'] == 'success').astype(int)

        team_stats = two_pt_data.groupby('posteam').agg(
            successes=('success', 'sum'),
            attempts=('success', 'count')
        ).reset_index()

        # Filter to teams with at least 5 attempts for stable estimates
        team_stats = team_stats[team_stats['attempts'] >= 5].copy()
        team_stats['rate'] = team_stats['successes'] / team_stats['attempts']

        self.teams = team_stats['posteam'].tolist()

        # Overall statistics
        total_successes = team_stats['successes'].sum()
        total_attempts = team_stats['attempts'].sum()
        self.overall_rate = total_successes / total_attempts
        self.mu = logit(self.overall_rate)

        print(f"Hierarchical 2-point conversion model:")
        print(f"  Total: {total_successes}/{total_attempts} = {self.overall_rate:.1%}")
        print(f"  Teams with 5+ attempts: {len(self.teams)}")

        # Estimate tau^2 using method of moments
        # Var(observed rates) = Var(true rates) + sampling variance
        # Var(true rates) approx tau^2 * p(1-p) for logistic model
        observed_var = team_stats['rate'].var()
        avg_sampling_var = np.mean(
            team_stats['rate'] * (1 - team_stats['rate']) / team_stats['attempts']
        )
        tau_sq_estimate = max(0.001, (observed_var - avg_sampling_var) /
                              (self.overall_rate * (1 - self.overall_rate)))
        self.tau_sq = tau_sq_estimate

        print(f"  Between-team variance (tau^2): {self.tau_sq:.4f}")
        print(f"  Team effect SD: {np.sqrt(self.tau_sq):.3f} log-odds")

        # Compute shrinkage estimates for each team
        for _, row in team_stats.iterrows():
            team = row['posteam']
            n = row['attempts']
            y = row['successes']

            # Shrinkage factor (empirical Bayes)
            # Higher n -> less shrinkage, lower tau^2 -> more shrinkage
            sampling_var = self.overall_rate * (1 - self.overall_rate) / n
            shrinkage = self.tau_sq / (self.tau_sq + sampling_var)

            # Team's observed log-odds
            team_rate = np.clip(y / n, 0.01, 0.99)
            team_logit = logit(team_rate)

            # Shrunk estimate
            alpha_hat = shrinkage * (team_logit - self.mu)

            # Posterior variance of alpha
            posterior_var = shrinkage * sampling_var

            # Generate posterior samples
            alpha_samples = np.random.normal(alpha_hat, np.sqrt(posterior_var), self.n_samples)
            self.team_effects[team] = alpha_samples

        # Show top/bottom teams
        team_means = {t: self.team_effects[t].mean() for t in self.teams}
        sorted_teams = sorted(team_means.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Top 5 special teams (2pt):")
        for team, effect in sorted_teams[:5]:
            prob = expit(self.mu + effect)
            print(f"    {team}: {prob:.1%} ({effect:+.3f} log-odds)")

        print(f"\n  Bottom 5 special teams (2pt):")
        for team, effect in sorted_teams[-5:]:
            prob = expit(self.mu + effect)
            print(f"    {team}: {prob:.1%} ({effect:+.3f} log-odds)")

    def get_posterior_samples(self, team: str = None) -> np.ndarray:
        """
        Return posterior samples of success probability.

        Args:
            team: Team abbreviation. If None, returns league average.
        """
        from scipy.special import expit

        if team is None or team not in self.team_effects:
            # League average
            return expit(self.mu + np.random.normal(0, np.sqrt(self.tau_sq), self.n_samples))

        # Team-specific
        alpha = self.team_effects[team]
        return expit(self.mu + alpha)

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class HierarchicalOffDefTwoPointModel:
    """
    Hierarchical Bayesian model for 2-point conversion with BOTH
    offensive team effects and defensive team effects.

    Model:
        logit(p_ij) = μ + α_i^off + α_j^def

    Where:
        - μ is the league-average log-odds of conversion
        - α_i^off ~ N(0, τ²_off) is offensive team i's conversion ability
        - α_j^def ~ N(0, τ²_def) is defensive team j's stopping ability

    This properly captures that 2pt is offense vs defense, not special teams.
    Uses empirical Bayes for partial pooling of both effect types.
    """

    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.mu = None  # League average (log-odds)
        self.overall_rate = None

        # Offensive effects
        self.tau_sq_off = None
        self.off_effects = {}  # team -> shrunk effect estimate
        self.off_samples = {}  # team -> posterior samples
        self.off_teams = []

        # Defensive effects
        self.tau_sq_def = None
        self.def_effects = {}  # team -> shrunk effect estimate
        self.def_samples = {}  # team -> posterior samples
        self.def_teams = []

    def fit(self, two_pt_data: pd.DataFrame, min_attempts: int = 5):
        """
        Fit hierarchical model with offense and defense effects.

        Args:
            two_pt_data: DataFrame with 'two_point_conv_result', 'posteam', 'defteam'
            min_attempts: Minimum attempts for team to get its own effect
        """
        from scipy.special import logit, expit

        df = two_pt_data.copy()
        df['success'] = (df['two_point_conv_result'] == 'success').astype(int)
        df = df.dropna(subset=['posteam', 'defteam', 'success'])

        # Overall statistics
        total_successes = df['success'].sum()
        total_attempts = len(df)
        self.overall_rate = total_successes / total_attempts
        self.mu = logit(self.overall_rate)

        print(f"Hierarchical Off/Def 2-point conversion model:")
        print(f"  Total: {total_successes}/{total_attempts} = {self.overall_rate:.1%}")
        print(f"  Unique matchups: {df.groupby(['posteam', 'defteam']).ngroups}")

        # Step 1: Compute offensive team statistics
        off_stats = df.groupby('posteam').agg(
            successes=('success', 'sum'),
            attempts=('success', 'count')
        ).reset_index()
        off_stats = off_stats[off_stats['attempts'] >= min_attempts].copy()
        off_stats['rate'] = off_stats['successes'] / off_stats['attempts']
        self.off_teams = off_stats['posteam'].tolist()

        # Estimate tau_sq_off using method of moments
        observed_var_off = off_stats['rate'].var()
        avg_sampling_var_off = np.mean(
            off_stats['rate'] * (1 - off_stats['rate']) / off_stats['attempts']
        )
        self.tau_sq_off = max(0.001, (observed_var_off - avg_sampling_var_off) /
                              (self.overall_rate * (1 - self.overall_rate)))

        print(f"  Offensive teams with {min_attempts}+ attempts: {len(self.off_teams)}")
        print(f"  Offensive effect SD (tau_off): {np.sqrt(self.tau_sq_off):.3f} log-odds")

        # Step 2: Compute defensive team statistics
        def_stats = df.groupby('defteam').agg(
            successes=('success', 'sum'),
            attempts=('success', 'count')
        ).reset_index()
        def_stats = def_stats[def_stats['attempts'] >= min_attempts].copy()
        def_stats['rate'] = def_stats['successes'] / def_stats['attempts']
        self.def_teams = def_stats['defteam'].tolist()

        # Estimate tau_sq_def
        observed_var_def = def_stats['rate'].var()
        avg_sampling_var_def = np.mean(
            def_stats['rate'] * (1 - def_stats['rate']) / def_stats['attempts']
        )
        self.tau_sq_def = max(0.001, (observed_var_def - avg_sampling_var_def) /
                              (self.overall_rate * (1 - self.overall_rate)))

        print(f"  Defensive teams with {min_attempts}+ attempts: {len(self.def_teams)}")
        print(f"  Defensive effect SD (tau_def): {np.sqrt(self.tau_sq_def):.3f} log-odds")

        # Step 3: Compute shrinkage estimates for offensive teams
        for _, row in off_stats.iterrows():
            team = row['posteam']
            n = row['attempts']
            y = row['successes']

            sampling_var = self.overall_rate * (1 - self.overall_rate) / n
            shrinkage = self.tau_sq_off / (self.tau_sq_off + sampling_var)

            team_rate = np.clip(y / n, 0.01, 0.99)
            team_logit = logit(team_rate)

            alpha_hat = shrinkage * (team_logit - self.mu)
            posterior_var = shrinkage * sampling_var

            self.off_effects[team] = alpha_hat
            self.off_samples[team] = np.random.normal(alpha_hat, np.sqrt(posterior_var), self.n_samples)

        # Step 4: Compute shrinkage estimates for defensive teams
        # Note: Higher rate against = WORSE defense, so we flip the sign
        for _, row in def_stats.iterrows():
            team = row['defteam']
            n = row['attempts']
            y = row['successes']

            sampling_var = self.overall_rate * (1 - self.overall_rate) / n
            shrinkage = self.tau_sq_def / (self.tau_sq_def + sampling_var)

            team_rate = np.clip(y / n, 0.01, 0.99)
            team_logit = logit(team_rate)

            # Defensive effect: positive = easier to convert against (bad defense)
            delta_hat = shrinkage * (team_logit - self.mu)
            posterior_var = shrinkage * sampling_var

            self.def_effects[team] = delta_hat
            self.def_samples[team] = np.random.normal(delta_hat, np.sqrt(posterior_var), self.n_samples)

        # Show top/bottom offenses and defenses
        off_means = {t: self.off_effects[t] for t in self.off_teams}
        sorted_off = sorted(off_means.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Top 5 2pt offenses:")
        for team, effect in sorted_off[:5]:
            prob = expit(self.mu + effect)
            print(f"    {team}: {prob:.1%} ({effect:+.3f} log-odds)")

        print(f"\n  Bottom 5 2pt offenses:")
        for team, effect in sorted_off[-5:]:
            prob = expit(self.mu + effect)
            print(f"    {team}: {prob:.1%} ({effect:+.3f} log-odds)")

        def_means = {t: self.def_effects[t] for t in self.def_teams}
        sorted_def = sorted(def_means.items(), key=lambda x: x[1])  # Lower = better defense

        print(f"\n  Top 5 2pt defenses (hardest to convert against):")
        for team, effect in sorted_def[:5]:
            prob = expit(self.mu + effect)
            print(f"    {team}: {prob:.1%} opponent rate ({effect:+.3f} log-odds)")

        print(f"\n  Bottom 5 2pt defenses (easiest to convert against):")
        for team, effect in sorted_def[-5:]:
            prob = expit(self.mu + effect)
            print(f"    {team}: {prob:.1%} opponent rate ({effect:+.3f} log-odds)")

    def get_posterior_samples(self, off_team: str = None, def_team: str = None) -> np.ndarray:
        """
        Return posterior samples of conversion probability.

        Args:
            off_team: Offensive team. If None, uses league average.
            def_team: Defensive team. If None, uses league average.
        """
        from scipy.special import expit

        # Get offensive effect samples
        if off_team is not None and off_team in self.off_samples:
            off_effect = self.off_samples[off_team]
        else:
            off_effect = np.random.normal(0, np.sqrt(self.tau_sq_off), self.n_samples)

        # Get defensive effect samples
        if def_team is not None and def_team in self.def_samples:
            def_effect = self.def_samples[def_team]
        else:
            def_effect = np.random.normal(0, np.sqrt(self.tau_sq_def), self.n_samples)

        return expit(self.mu + off_effect + def_effect)

    def get_matchup_advantage(self, off_team: str, def_team: str) -> dict:
        """Get the expected conversion rate for a specific matchup."""
        from scipy.special import expit

        off_eff = self.off_effects.get(off_team, 0)
        def_eff = self.def_effects.get(def_team, 0)

        matchup_prob = expit(self.mu + off_eff + def_eff)
        baseline_prob = expit(self.mu)

        return {
            'matchup_prob': matchup_prob,
            'baseline_prob': baseline_prob,
            'off_effect': off_eff,
            'def_effect': def_eff,
            'total_effect': off_eff + def_eff,
        }

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class PATModel:
    """
    Model for PAT (extra point) success probability.
    Post-2015: 33-yard kicks (snap from 15-yard line).
    """

    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.success_rate_samples = None
        self.overall_rate = None

    def fit(self, pat_data: pd.DataFrame):
        """
        Fit the model using historical PAT data.

        Args:
            pat_data: DataFrame with 'extra_point_result' column
        """
        successes = (pat_data['extra_point_result'] == 'good').sum()
        attempts = len(pat_data)

        self.overall_rate = successes / attempts

        # Bayesian posterior
        alpha = successes + 1
        beta = (attempts - successes) + 1

        self.success_rate_samples = np.random.beta(alpha, beta, self.n_samples)

        print(f"PAT model fit:")
        print(f"  Data: {successes}/{attempts} = {self.overall_rate:.1%}")
        print(f"  Posterior: Beta({alpha}, {beta})")
        print(f"  95% CI: [{np.percentile(self.success_rate_samples, 2.5):.1%}, {np.percentile(self.success_rate_samples, 97.5):.1%}]")

    def get_posterior_samples(self, kicker_id: str = None) -> np.ndarray:
        """Return posterior samples of success probability."""
        return self.success_rate_samples

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class HierarchicalPATModel:
    """
    Hierarchical Bayesian model for PAT (extra point) success.

    Models kicker-specific accuracy with shrinkage toward league average.
    Uses empirical Bayes to estimate the variance of kicker effects.

    Model:
        logit(p_kicker) = mu + alpha_kicker
        alpha_kicker ~ N(0, tau^2)

    Where:
        - mu is the league-average log-odds of PAT success
        - alpha_kicker is the kicker-specific effect
        - tau^2 is the between-kicker variance (estimated via empirical Bayes)

    Note: With 94% league-wide success, kicker effects are relatively small
    but still meaningful (range roughly 90-97%).
    """

    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.mu = None  # League average (log-odds)
        self.tau_sq = None  # Between-kicker variance
        self.kicker_effects = {}  # kicker_id -> array of posterior samples
        self.kickers = []
        self.overall_rate = None

    def fit(self, pat_data: pd.DataFrame):
        """
        Fit hierarchical model using historical PAT data.

        Args:
            pat_data: DataFrame with 'extra_point_result' and 'kicker_player_id' columns
        """
        from scipy.special import logit, expit

        # Compute kicker-level statistics
        pat_data = pat_data.copy()
        pat_data['success'] = (pat_data['extra_point_result'] == 'good').astype(int)

        # Filter to rows with kicker info
        pat_data = pat_data[pat_data['kicker_player_id'].notna()].copy()

        kicker_stats = pat_data.groupby('kicker_player_id').agg(
            successes=('success', 'sum'),
            attempts=('success', 'count'),
            kicker_name=('kicker_player_name', 'first')
        ).reset_index()

        # Filter to kickers with at least 20 PAT attempts for stable estimates
        kicker_stats = kicker_stats[kicker_stats['attempts'] >= 20].copy()
        kicker_stats['rate'] = kicker_stats['successes'] / kicker_stats['attempts']

        self.kickers = kicker_stats['kicker_player_id'].tolist()

        # Overall statistics
        total_successes = kicker_stats['successes'].sum()
        total_attempts = kicker_stats['attempts'].sum()
        self.overall_rate = total_successes / total_attempts
        self.mu = logit(self.overall_rate)

        print(f"Hierarchical PAT model:")
        print(f"  Total: {total_successes}/{total_attempts} = {self.overall_rate:.1%}")
        print(f"  Kickers with 20+ attempts: {len(self.kickers)}")

        # Estimate tau^2 using method of moments
        observed_var = kicker_stats['rate'].var()
        avg_sampling_var = np.mean(
            kicker_stats['rate'] * (1 - kicker_stats['rate']) / kicker_stats['attempts']
        )
        # With high success rates, need to be careful about variance estimation
        tau_sq_estimate = max(0.001, (observed_var - avg_sampling_var) /
                              (self.overall_rate * (1 - self.overall_rate)))
        self.tau_sq = tau_sq_estimate

        print(f"  Between-kicker variance (tau^2): {self.tau_sq:.4f}")
        print(f"  Kicker effect SD: {np.sqrt(self.tau_sq):.3f} log-odds")

        # Compute shrinkage estimates for each kicker
        for _, row in kicker_stats.iterrows():
            kicker_id = row['kicker_player_id']
            n = row['attempts']
            y = row['successes']

            # Shrinkage factor
            sampling_var = self.overall_rate * (1 - self.overall_rate) / n
            shrinkage = self.tau_sq / (self.tau_sq + sampling_var)

            # Kicker's observed log-odds
            kicker_rate = np.clip(y / n, 0.01, 0.99)
            kicker_logit = logit(kicker_rate)

            # Shrunk estimate
            alpha_hat = shrinkage * (kicker_logit - self.mu)

            # Posterior variance
            posterior_var = shrinkage * sampling_var

            # Generate posterior samples
            alpha_samples = np.random.normal(alpha_hat, np.sqrt(posterior_var), self.n_samples)
            self.kicker_effects[kicker_id] = alpha_samples

        # Show top/bottom kickers
        kicker_means = {k: self.kicker_effects[k].mean() for k in self.kickers}
        sorted_kickers = sorted(kicker_means.items(), key=lambda x: x[1], reverse=True)

        # Get names for display
        kicker_names = dict(zip(kicker_stats['kicker_player_id'], kicker_stats['kicker_name']))

        print(f"\n  Top 5 kickers (PAT):")
        for kicker_id, effect in sorted_kickers[:5]:
            prob = expit(self.mu + effect)
            name = kicker_names.get(kicker_id, kicker_id)
            print(f"    {name}: {prob:.1%} ({effect:+.3f} log-odds)")

        print(f"\n  Bottom 5 kickers (PAT):")
        for kicker_id, effect in sorted_kickers[-5:]:
            prob = expit(self.mu + effect)
            name = kicker_names.get(kicker_id, kicker_id)
            print(f"    {name}: {prob:.1%} ({effect:+.3f} log-odds)")

    def get_posterior_samples(self, kicker_id: str = None) -> np.ndarray:
        """
        Return posterior samples of PAT success probability.

        Args:
            kicker_id: Kicker player ID. If None, returns league average.
        """
        from scipy.special import expit

        if kicker_id is None or kicker_id not in self.kicker_effects:
            # League average
            return expit(self.mu + np.random.normal(0, np.sqrt(self.tau_sq), self.n_samples))

        # Kicker-specific
        alpha = self.kicker_effects[kicker_id]
        return expit(self.mu + alpha)

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class TwoPointDecisionAnalyzer:
    """
    Analyzes whether to go for 2 or kick PAT based on win probability.

    The key insight: Expected points are roughly equal (PAT: 0.94, 2pt: 0.96),
    but the variance differs. The optimal choice depends on game state.

    Supports hierarchical models:
    - Offense/defense-specific 2-point conversion rates (HierarchicalOffDefTwoPointModel)
    - Kicker-specific PAT success rates (HierarchicalPATModel)

    This uses the same win probability model as the fourth down analysis.
    """

    def __init__(self, two_pt_model, pat_model, wp_model):
        """
        Args:
            two_pt_model: Model for 2-point conversion success
                          (HierarchicalOffDefTwoPointModel, HierarchicalTwoPointModel, or TwoPointConversionModel)
            pat_model: Model for PAT success
                       (PATModel or HierarchicalPATModel)
            wp_model: Win probability model (same as fourth down analysis)
        """
        self.two_pt = two_pt_model
        self.pat = pat_model
        self.wp = wp_model
        self.n_samples = two_pt_model.n_samples

        # Check if we have off/def model (preferred)
        self.has_off_def_effects = isinstance(two_pt_model, HierarchicalOffDefTwoPointModel)
        # Fallback to simple team effects
        self.has_team_effects = isinstance(two_pt_model, HierarchicalTwoPointModel)
        self.has_kicker_effects = isinstance(pat_model, HierarchicalPATModel)

    def _score_after_pat(self, state: TwoPointState) -> Tuple[np.ndarray, int, int, int]:
        """
        Compute score differential after TD + PAT attempt.

        Returns:
            (p_pat samples, score_after_td, score_if_make, score_if_miss)
        """
        score_after_td = state.score_diff_pre_td + 6

        # PAT success probability - use kicker-specific if available
        p_pat = self.pat.get_posterior_samples(kicker_id=state.kicker_id)

        score_if_make = score_after_td + 1  # +7 total from TD
        score_if_miss = score_after_td      # +6 total from TD

        return p_pat, score_after_td, score_if_make, score_if_miss

    def _score_after_2pt(self, state: TwoPointState) -> Tuple[np.ndarray, int, int, int]:
        """
        Compute score differential after TD + 2-point attempt.

        Returns:
            (p_2pt samples, score_after_td, score_if_success, score_if_fail)
        """
        score_after_td = state.score_diff_pre_td + 6

        # 2pt success probability - use off/def model if available
        if self.has_off_def_effects:
            p_2pt = self.two_pt.get_posterior_samples(
                off_team=state.posteam,
                def_team=getattr(state, 'defteam', None)
            )
        elif self.has_team_effects:
            p_2pt = self.two_pt.get_posterior_samples(team=state.posteam)
        else:
            p_2pt = self.two_pt.get_posterior_samples()

        score_if_success = score_after_td + 2  # +8 total from TD
        score_if_fail = score_after_td         # +6 total from TD

        return p_2pt, score_after_td, score_if_success, score_if_fail

    def compute_wp_pat(self, state: TwoPointState) -> np.ndarray:
        """
        Compute win probability distribution if kicking PAT.

        WP_pat = P(make) * WP(+7) + P(miss) * WP(+6)

        where +7 and +6 are the score differentials after the TD.
        """
        p_pat, _, score_if_make, score_if_miss = self._score_after_pat(state)

        # Win probability at each score differential
        # After kickoff, opponent has ball at ~25 yard line (75 yards from their endzone)
        opp_field_pos = 75

        # Opponent's WP (they have ball after kickoff)
        wp_opp_if_make = self.wp.get_posterior_samples(
            -score_if_make, state.time_remaining, opp_field_pos, 0
        )
        wp_opp_if_miss = self.wp.get_posterior_samples(
            -score_if_miss, state.time_remaining, opp_field_pos, 0
        )

        # Our WP is 1 - opponent's WP
        wp_if_make = 1 - wp_opp_if_make
        wp_if_miss = 1 - wp_opp_if_miss

        # Expected WP
        wp_pat = p_pat * wp_if_make + (1 - p_pat) * wp_if_miss

        return wp_pat

    def compute_wp_2pt(self, state: TwoPointState) -> np.ndarray:
        """
        Compute win probability distribution if going for 2.

        WP_2pt = P(success) * WP(+8) + P(fail) * WP(+6)
        """
        p_2pt, _, score_if_success, score_if_fail = self._score_after_2pt(state)

        opp_field_pos = 75

        wp_opp_if_success = self.wp.get_posterior_samples(
            -score_if_success, state.time_remaining, opp_field_pos, 0
        )
        wp_opp_if_fail = self.wp.get_posterior_samples(
            -score_if_fail, state.time_remaining, opp_field_pos, 0
        )

        wp_if_success = 1 - wp_opp_if_success
        wp_if_fail = 1 - wp_opp_if_fail

        wp_2pt = p_2pt * wp_if_success + (1 - p_2pt) * wp_if_fail

        return wp_2pt

    def analyze(self, state: TwoPointState) -> Dict:
        """
        Full analysis of PAT vs 2-point conversion decision.

        Returns dict with:
            - wp_pat: Expected WP if kicking PAT
            - wp_2pt: Expected WP if going for 2
            - optimal_action: 'pat' or 'two_point'
            - wp_margin: WP difference (positive = 2pt better)
            - prob_2pt_better: Probability 2pt is better across posterior
        """
        wp_pat_samples = self.compute_wp_pat(state)
        wp_2pt_samples = self.compute_wp_2pt(state)

        wp_pat = wp_pat_samples.mean()
        wp_2pt = wp_2pt_samples.mean()

        optimal_action = 'two_point' if wp_2pt > wp_pat else 'pat'
        wp_margin = wp_2pt - wp_pat

        prob_2pt_better = (wp_2pt_samples > wp_pat_samples).mean()

        # Also compute the score scenarios (team/kicker-specific if available)
        p_pat = self.pat.get_posterior_samples(kicker_id=state.kicker_id).mean()
        if self.has_off_def_effects:
            p_2pt = self.two_pt.get_posterior_samples(
                off_team=state.posteam,
                def_team=getattr(state, 'defteam', None)
            ).mean()
        elif self.has_team_effects:
            p_2pt = self.two_pt.get_posterior_samples(team=state.posteam).mean()
        else:
            p_2pt = self.two_pt.get_posterior_samples().mean()
        score_after_td = state.score_diff_pre_td + 6

        return {
            'state': state,
            'wp_pat': wp_pat,
            'wp_2pt': wp_2pt,
            'wp_pat_samples': wp_pat_samples,
            'wp_2pt_samples': wp_2pt_samples,
            'optimal_action': optimal_action,
            'wp_margin': wp_margin,
            'prob_2pt_better': prob_2pt_better,
            'p_pat': p_pat,
            'p_2pt': p_2pt,
            'score_if_pat_make': score_after_td + 1,
            'score_if_pat_miss': score_after_td,
            'score_if_2pt_success': score_after_td + 2,
            'score_if_2pt_fail': score_after_td,
        }

    def analyze_score_differential_grid(self, time_remaining: int = 600,
                                         score_range: range = range(-21, 22)) -> pd.DataFrame:
        """
        Analyze optimal decision across different pre-TD score differentials.

        Returns DataFrame showing when 2pt is better vs PAT.
        """
        results = []
        for score_diff_pre_td in score_range:
            state = TwoPointState(score_diff_pre_td=score_diff_pre_td,
                                  time_remaining=time_remaining)
            analysis = self.analyze(state)
            results.append({
                'score_diff_pre_td': score_diff_pre_td,
                'score_after_td': score_diff_pre_td + 6,
                'time_remaining': time_remaining,
                'wp_pat': analysis['wp_pat'],
                'wp_2pt': analysis['wp_2pt'],
                'wp_margin': analysis['wp_margin'],
                'optimal': analysis['optimal_action'],
                'prob_2pt_better': analysis['prob_2pt_better'],
            })
        return pd.DataFrame(results)

    def analyze_time_remaining_grid(self, score_diff_pre_td: int,
                                     time_points: List[int] = None) -> pd.DataFrame:
        """
        Analyze how optimal decision changes with time remaining.
        """
        if time_points is None:
            # Every 2 minutes from 0 to 60 minutes
            time_points = list(range(0, 3601, 120))

        results = []
        for time_remaining in time_points:
            state = TwoPointState(score_diff_pre_td=score_diff_pre_td,
                                  time_remaining=time_remaining)
            analysis = self.analyze(state)
            results.append({
                'score_diff_pre_td': score_diff_pre_td,
                'time_remaining': time_remaining,
                'time_remaining_min': time_remaining / 60,
                'wp_pat': analysis['wp_pat'],
                'wp_2pt': analysis['wp_2pt'],
                'wp_margin': analysis['wp_margin'],
                'optimal': analysis['optimal_action'],
                'prob_2pt_better': analysis['prob_2pt_better'],
            })
        return pd.DataFrame(results)


def prepare_two_point_data(data_path: Path, start_year: int = 2015) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for two-point conversion analysis.

    Args:
        data_path: Path to all_pbp_1999_2024.parquet
        start_year: First year to include (2015 = post-PAT rule change)

    Returns:
        (two_pt_data, pat_data)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Filter to post-rule-change era
    df = df[df['season'] >= start_year].copy()

    # Two-point conversions
    two_pt = df[df['two_point_attempt'] == 1].copy()
    two_pt = two_pt[two_pt['two_point_conv_result'].notna()].copy()

    # PAT attempts
    pat = df[df['extra_point_attempt'] == 1].copy()
    pat = pat[pat['extra_point_result'].notna()].copy()

    print(f"Post-{start_year} data:")
    print(f"  Two-point attempts: {len(two_pt)}")
    print(f"  PAT attempts: {len(pat)}")

    return two_pt, pat


def train_and_save_models(data_dir: Path, models_dir: Path, start_year: int = 2015,
                          hierarchical: bool = True):
    """
    Train and save two-point conversion models.

    Args:
        data_dir: Directory containing play-by-play data
        models_dir: Directory to save trained models
        start_year: First year to include (2015 = post-PAT rule change)
        hierarchical: If True, train hierarchical models with team/kicker effects
    """
    two_pt_data, pat_data = prepare_two_point_data(
        data_dir / 'all_pbp_1999_2024.parquet',
        start_year=start_year
    )

    if hierarchical:
        # Hierarchical models with team/kicker effects
        print("\n--- Training Hierarchical Two-Point Model (team effects) ---")
        two_pt_model = HierarchicalTwoPointModel()
        two_pt_model.fit(two_pt_data)
        two_pt_model.save(models_dir / 'hierarchical_two_point_model.pkl')

        print("\n--- Training Hierarchical PAT Model (kicker effects) ---")
        pat_model = HierarchicalPATModel()
        pat_model.fit(pat_data)
        pat_model.save(models_dir / 'hierarchical_pat_model.pkl')
    else:
        # Simple pooled models
        two_pt_model = TwoPointConversionModel()
        two_pt_model.fit(two_pt_data)
        two_pt_model.save(models_dir / 'two_point_model.pkl')

        pat_model = PATModel()
        pat_model.fit(pat_data)
        pat_model.save(models_dir / 'pat_model.pkl')

    return two_pt_model, pat_model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Demo analysis
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'

    # Train models
    print("Training two-point conversion models...")
    two_pt_model, pat_model = train_and_save_models(data_dir, models_dir)

    # Load win probability model
    from models.bayesian_models import WinProbabilityModel
    wp_model = WinProbabilityModel().load(models_dir / 'wp_model.pkl')

    # Create analyzer
    analyzer = TwoPointDecisionAnalyzer(two_pt_model, pat_model, wp_model)

    print("\n" + "="*80)
    print("TWO-POINT CONVERSION DECISION ANALYSIS")
    print("="*80)

    # Example: Patriots vs Chargers scenario
    # Score was 9-3 before the TD. After TD: 15-3 (up 12 before decision)
    # Time remaining: approximately 10 minutes (let's say 600 seconds)

    print("\n--- Patriots vs Chargers Wild Card (Example) ---")
    print("Situation: Pats up 9-3, score TD, now deciding PAT vs 2pt")
    print("Pre-TD score diff: +6, After TD: +12, ~10 min remaining")

    state = TwoPointState(score_diff_pre_td=6, time_remaining=600)
    result = analyzer.analyze(state)

    print(f"\nAnalysis:")
    print(f"  If kick PAT:  {result['p_pat']:.1%} make -> {result['score_if_pat_make']:+d} or {result['score_if_pat_miss']:+d}")
    print(f"  If go for 2:  {result['p_2pt']:.1%} success -> {result['score_if_2pt_success']:+d} or {result['score_if_2pt_fail']:+d}")
    print(f"\n  WP if PAT:    {result['wp_pat']:.1%}")
    print(f"  WP if 2pt:    {result['wp_2pt']:.1%}")
    print(f"  WP margin:    {result['wp_margin']:+.2%} ({'2pt better' if result['wp_margin'] > 0 else 'PAT better'})")
    print(f"  P(2pt better): {result['prob_2pt_better']:.1%}")
    print(f"\n  Optimal: {result['optimal_action'].upper()}")

    # Grid analysis
    print("\n" + "="*80)
    print("GRID ANALYSIS: When is 2pt better than PAT?")
    print("="*80)

    print("\n--- By Pre-TD Score Differential (10 min remaining) ---")
    grid = analyzer.analyze_score_differential_grid(time_remaining=600)

    # Show key thresholds
    two_pt_better = grid[grid['optimal'] == 'two_point']
    pat_better = grid[grid['optimal'] == 'pat']

    print("\n2pt is better when pre-TD score diff is:")
    if len(two_pt_better) > 0:
        print(f"  {two_pt_better['score_diff_pre_td'].min()} to {two_pt_better['score_diff_pre_td'].max()}")
    else:
        print("  Never (PAT always better)")

    print("\nDetailed results for interesting scenarios:")
    interesting = grid[grid['score_diff_pre_td'].isin([-7, -1, 0, 1, 6, 7, 8, 13, 14])]
    print(interesting[['score_diff_pre_td', 'score_after_td', 'wp_pat', 'wp_2pt', 'wp_margin', 'optimal']].to_string(index=False))
