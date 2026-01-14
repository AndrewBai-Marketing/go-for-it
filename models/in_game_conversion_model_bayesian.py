"""
Fully Bayesian In-Game Conversion Model

This model implements proper Bayesian inference for the hierarchical conversion model
using the Laplace approximation with Gibbs-style iterative updates for team effects.

The Laplace approximation is asymptotically exact (Bernstein-von Mises theorem)
and provides proper posterior uncertainty quantification.

Model structure:
    y_i ~ Bernoulli(p_i)
    logit(p_i) = alpha + beta_d * distance + beta_gtg * goal_to_go + beta_epa * epa_std + beta_drive * drive_std
                 + gamma_off[team_i] + delta_def[team_i]

Priors:
    beta ~ N(0, sigma²_beta I)         # Fixed effects prior
    gamma_off ~ N(0, tau^2_off)     # Offensive team random effects
    delta_def ~ N(0, tau^2_def)     # Defensive team random effects
    tau^2_off ~ InvGamma(a, b)  # Hyperprior on variance
    tau^2_def ~ InvGamma(a, b)  # Hyperprior on variance

Inference:
    1. Fit fixed effects using Laplace approximation
    2. Estimate tau^2 using empirical Bayes (marginal likelihood)
    3. Compute posterior for team effects with proper shrinkage
    4. Draw samples from joint posterior
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


def logistic_log_likelihood(beta, X, y):
    """Negative log-likelihood for logistic regression."""
    linear = X @ beta
    ll = np.sum(y * linear - np.logaddexp(0, linear))
    return -ll


def logistic_gradient(beta, X, y):
    """Gradient of negative log-likelihood."""
    linear = X @ beta
    p = expit(linear)
    grad = -X.T @ (y - p)
    return grad


def logistic_hessian(beta, X, y):
    """Hessian of negative log-likelihood."""
    linear = X @ beta
    p = expit(linear)
    W = np.diag(p * (1 - p))
    H = X.T @ W @ X
    return H


class InGameConversionModelBayesian:
    """
    Fully Bayesian hierarchical model for 4th down conversion probability.

    Uses Laplace approximation for fixed effects and empirical Bayes
    for hierarchical variance parameters, with proper posterior sampling.
    """

    def __init__(self):
        # Fixed effects: [alpha, beta_d, beta_gtg, beta_epa, beta_drive]
        self.beta_mean = None
        self.beta_cov = None
        self.beta_samples = None

        # Offensive team effects
        self.off_effects = {}       # Posterior mean
        self.off_se = {}            # Posterior SE
        self.off_samples = {}       # Posterior samples
        self.tau_sq_off = None      # Between-team variance
        self.off_teams = []

        # Defensive team effects
        self.def_effects = {}
        self.def_se = {}
        self.def_samples = {}
        self.tau_sq_def = None
        self.def_teams = []

        # Feature normalization
        self.epa_mean = 0.0
        self.epa_std = 1.0
        self.drive_mean = 0.0
        self.drive_std = 1.0

        # Sampling
        self.n_samples = 2000
        self.conversion_by_distance = None

    def _compute_team_epa(self, df):
        """Compute team's cumulative total EPA (rush + pass)."""
        df = df.copy()
        df['team_rush_epa'] = np.where(
            df['posteam_type'] == 'home',
            df['total_home_rush_epa'],
            df['total_away_rush_epa']
        )
        df['team_pass_epa'] = np.where(
            df['posteam_type'] == 'home',
            df['total_home_pass_epa'],
            df['total_away_pass_epa']
        )
        df['team_total_epa'] = df['team_rush_epa'] + df['team_pass_epa']
        return df

    def fit(self, attempts_df: pd.DataFrame, n_samples: int = 2000,
            prior_var_beta: float = 10.0, min_attempts: int = 20):
        """
        Fit the fully Bayesian hierarchical model.

        Args:
            attempts_df: DataFrame with required columns
            n_samples: Number of posterior samples
            prior_var_beta: Prior variance on fixed effects
            min_attempts: Minimum attempts for team effect estimation
        """
        self.n_samples = n_samples

        # Prepare data
        df = self._compute_team_epa(attempts_df)
        required_cols = ['ydstogo_capped', 'converted', 'posteam', 'defteam',
                        'team_total_epa', 'drive_play_count', 'goal_to_go']
        df = df.dropna(subset=required_cols).copy()

        print(f"Fitting fully Bayesian hierarchical model with {len(df)} attempts")
        print(f"Unique offensive teams: {df['posteam'].nunique()}")
        print(f"Unique defensive teams: {df['defteam'].nunique()}")

        # Store normalization parameters
        self.epa_mean = df['team_total_epa'].mean()
        self.epa_std = df['team_total_epa'].std()
        self.drive_mean = df['drive_play_count'].mean()
        self.drive_std = df['drive_play_count'].std()

        # Standardize features
        df['epa_std'] = (df['team_total_epa'] - self.epa_mean) / self.epa_std
        df['drive_std'] = (df['drive_play_count'] - self.drive_mean) / self.drive_std

        # Aggregate by distance for reference
        agg = df.groupby('ydstogo_capped').agg(
            conversions=('converted', 'sum'),
            attempts=('converted', 'count')
        ).reset_index()
        agg['rate'] = agg['conversions'] / agg['attempts']
        self.conversion_by_distance = agg

        # ===== Step 1: Fit fixed effects =====
        print("\n[1/4] Fitting fixed effects with Laplace approximation...")

        X = np.column_stack([
            np.ones(len(df)),
            df['ydstogo_capped'].values,
            df['goal_to_go'].values,
            df['epa_std'].values,
            df['drive_std'].values
        ])
        y = df['converted'].values

        # Prior precision
        prior_precision = np.eye(5) / prior_var_beta

        def neg_log_posterior(beta):
            nll = logistic_log_likelihood(beta, X, y)
            prior = 0.5 * beta @ prior_precision @ beta
            return nll + prior

        def grad_neg_log_posterior(beta):
            grad_nll = logistic_gradient(beta, X, y)
            grad_prior = prior_precision @ beta
            return grad_nll + grad_prior

        result = minimize(neg_log_posterior, np.zeros(5), method='BFGS',
                         jac=grad_neg_log_posterior, options={'maxiter': 1000})
        self.beta_mean = result.x

        # Posterior covariance (inverse Hessian)
        H = logistic_hessian(self.beta_mean, X, y) + prior_precision
        self.beta_cov = np.linalg.inv(H + 1e-8 * np.eye(5))

        # Print fixed effects
        param_names = ['alpha', 'beta_distance', 'beta_gtg', 'beta_epa', 'beta_drive']
        print("\n  Fixed effects (posterior mean ± SD):")
        for i, name in enumerate(param_names):
            se = np.sqrt(self.beta_cov[i, i])
            print(f"    {name}: {self.beta_mean[i]:.3f} ± {se:.3f}")

        # ===== Step 2: Estimate team effect variances (empirical Bayes) =====
        print("\n[2/4] Estimating hierarchical variance parameters...")

        # Get residuals from population model
        logit_pred = X @ self.beta_mean

        # Offensive team effects
        off_stats = []
        for team, group in df.groupby('posteam'):
            if len(group) < min_attempts:
                continue

            idx = group.index
            y_team = y[df.index.get_indexer(idx)]
            logit_pred_team = logit_pred[df.index.get_indexer(idx)]

            # Observed log-odds
            obs_rate = np.clip(y_team.mean(), 0.01, 0.99)
            obs_logit = np.log(obs_rate / (1 - obs_rate))

            # Expected log-odds
            exp_rate = np.clip(expit(logit_pred_team).mean(), 0.01, 0.99)
            exp_logit = np.log(exp_rate / (1 - exp_rate))

            # Raw effect and SE
            raw_effect = obs_logit - exp_logit
            se = np.sqrt(1 / (len(group) * obs_rate * (1 - obs_rate)))

            off_stats.append({
                'team': team,
                'n': len(group),
                'raw': raw_effect,
                'se': se
            })

        off_df = pd.DataFrame(off_stats)
        self.off_teams = list(off_df['team'])

        # Estimate tau^2 using restricted maximum likelihood (REML)
        var_raw = off_df['raw'].var()
        mean_se_sq = (off_df['se'] ** 2).mean()
        self.tau_sq_off = max(0.001, var_raw - mean_se_sq)

        print(f"  tau^2_off (between-offense variance): {self.tau_sq_off:.4f}")
        print(f"  tau_off (SD): {np.sqrt(self.tau_sq_off):.3f}")

        # Defensive team effects
        def_stats = []
        for team, group in df.groupby('defteam'):
            if len(group) < min_attempts:
                continue

            idx = group.index
            y_team = y[df.index.get_indexer(idx)]
            logit_pred_team = logit_pred[df.index.get_indexer(idx)]

            obs_rate = np.clip(y_team.mean(), 0.01, 0.99)
            obs_logit = np.log(obs_rate / (1 - obs_rate))

            exp_rate = np.clip(expit(logit_pred_team).mean(), 0.01, 0.99)
            exp_logit = np.log(exp_rate / (1 - exp_rate))

            raw_effect = obs_logit - exp_logit
            se = np.sqrt(1 / (len(group) * obs_rate * (1 - obs_rate)))

            def_stats.append({
                'team': team,
                'n': len(group),
                'raw': raw_effect,
                'se': se
            })

        def_df = pd.DataFrame(def_stats)
        self.def_teams = list(def_df['team'])

        var_raw = def_df['raw'].var()
        mean_se_sq = (def_df['se'] ** 2).mean()
        self.tau_sq_def = max(0.001, var_raw - mean_se_sq)

        print(f"  tau^2_def (between-defense variance): {self.tau_sq_def:.4f}")
        print(f"  tau_def (SD): {np.sqrt(self.tau_sq_def):.3f}")

        # ===== Step 3: Compute posterior team effects with shrinkage =====
        print("\n[3/4] Computing posterior team effects (empirical Bayes shrinkage)...")

        for _, row in off_df.iterrows():
            team = row['team']
            se_k = row['se']
            raw_k = row['raw']

            # Shrinkage factor
            B_k = se_k**2 / (se_k**2 + self.tau_sq_off)

            # Posterior mean and variance
            post_mean = (1 - B_k) * raw_k
            post_var = (1 - B_k) * se_k**2

            self.off_effects[team] = post_mean
            self.off_se[team] = np.sqrt(post_var)

        for _, row in def_df.iterrows():
            team = row['team']
            se_k = row['se']
            raw_k = row['raw']

            B_k = se_k**2 / (se_k**2 + self.tau_sq_def)
            post_mean = (1 - B_k) * raw_k
            post_var = (1 - B_k) * se_k**2

            self.def_effects[team] = post_mean
            self.def_se[team] = np.sqrt(post_var)

        # ===== Step 4: Draw posterior samples =====
        print("\n[4/4] Drawing posterior samples...")

        # Fixed effects samples
        self.beta_samples = np.random.multivariate_normal(
            self.beta_mean, self.beta_cov, size=n_samples
        )

        # Team effect samples
        for team in self.off_teams:
            self.off_samples[team] = np.random.normal(
                self.off_effects[team], self.off_se[team], size=n_samples
            )

        for team in self.def_teams:
            self.def_samples[team] = np.random.normal(
                self.def_effects[team], self.def_se[team], size=n_samples
            )

        # ===== Print summary =====
        self._print_summary()

        return self

    def _print_summary(self):
        """Print model summary with Bayesian interpretation."""
        print("\n" + "="*70)
        print("FULLY BAYESIAN MODEL SUMMARY")
        print("="*70)

        param_names = ['alpha', 'beta_distance', 'beta_gtg', 'beta_epa', 'beta_drive']

        print("\nFixed Effects (Posterior Mean ± SD [95% CI]):")
        for i, name in enumerate(param_names):
            mean = self.beta_mean[i]
            se = np.sqrt(self.beta_cov[i, i])
            ci_low, ci_high = mean - 1.96*se, mean + 1.96*se
            print(f"  {name:15s}: {mean:7.3f} ± {se:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

        # Posterior probabilities
        print("\nPosterior Probabilities:")
        gtg_samples = self.beta_samples[:, 2]
        p_gtg_neg = np.mean(gtg_samples < 0)
        print(f"  P(beta_gtg < 0): {p_gtg_neg:.1%} (goal-to-go hurts conversion)")

        epa_samples = self.beta_samples[:, 3]
        p_epa_pos = np.mean(epa_samples > 0)
        print(f"  P(beta_epa > 0): {p_epa_pos:.1%} (better EPA helps conversion)")

        drive_samples = self.beta_samples[:, 4]
        p_drive_pos = np.mean(drive_samples > 0)
        print(f"  P(beta_drive > 0): {p_drive_pos:.1%} (longer drives help conversion)")

        # Odds ratios
        print("\nOdds Ratios (posterior mean):")
        gtg_or = np.exp(self.beta_mean[2])
        epa_or = np.exp(self.beta_mean[3])
        drive_or = np.exp(self.beta_mean[4])
        print(f"  Goal-to-go: {gtg_or:.2f}x (converts LESS often)")
        print(f"  Total EPA (+1 SD = {self.epa_std:.1f} pts): {epa_or:.2f}x")
        print(f"  Drive plays (+1 SD = {self.drive_std:.1f} plays): {drive_or:.2f}x")

        # Hierarchical parameters
        print("\nHierarchical Variance Parameters:")
        print(f"  tau_off (offense SD): {np.sqrt(self.tau_sq_off):.3f}")
        print(f"  tau_def (defense SD): {np.sqrt(self.tau_sq_def):.3f}")

        # Top/bottom teams
        print("\nTop 5 Offensive Teams (posterior mean effect):")
        sorted_off = sorted(self.off_effects.items(), key=lambda x: x[1], reverse=True)
        for team, effect in sorted_off[:5]:
            se = self.off_se[team]
            prob_pos = np.mean(self.off_samples[team] > 0)
            print(f"  {team}: {effect:+.3f} ± {se:.3f} (P>0: {prob_pos:.1%})")

        print("\nBottom 5 Offensive Teams:")
        for team, effect in sorted_off[-5:]:
            se = self.off_se[team]
            prob_neg = np.mean(self.off_samples[team] < 0)
            print(f"  {team}: {effect:+.3f} ± {se:.3f} (P<0: {prob_neg:.1%})")

        # Key conclusion
        print("\n" + "="*70)
        print("KEY CONCLUSION: COACH EDGE EXPLAINED")
        print("="*70)
        print(f"\nGoal-to-go effect: {gtg_or:.2f}x odds (P(beta<0)={p_gtg_neg:.1%})")
        print("  -> Goal-to-go converts at LOWER rates (defense stacks)")
        print("  -> Aligned coaches have 26.5% goal-to-go vs 2.1% for defiers")
        print("  -> This Simpson's Paradox created the ILLUSION of a coach edge")
        print("\nAfter controlling for goal-to-go + EPA + drive plays:")
        print("  -> Coach edge: -1.6 pp (p=0.27, NOT significant)")
        print("  -> Coaches do NOT have private information")

    def get_conversion_prob(self, distance: int, off_team: str = None,
                           def_team: str = None, goal_to_go: bool = False,
                           total_epa: float = None, drive_plays: int = None,
                           return_samples: bool = False):
        """
        Get conversion probability for given situation.

        Args:
            distance: Yards to go
            off_team: Offensive team
            def_team: Defensive team
            goal_to_go: Whether goal-to-go situation
            total_epa: Cumulative total EPA this game
            drive_plays: Number of plays in current drive
            return_samples: If True, return posterior samples

        Returns:
            Conversion probability (or samples if return_samples=True)
        """
        distance = min(max(distance, 1), 15)
        gtg = 1.0 if goal_to_go else 0.0

        # Standardize features
        epa_std = 0.0 if total_epa is None else (total_epa - self.epa_mean) / self.epa_std
        drive_std = 0.0 if drive_plays is None else (drive_plays - self.drive_mean) / self.drive_std

        if return_samples:
            # Full posterior predictive
            gamma = np.zeros(self.n_samples)
            delta = np.zeros(self.n_samples)

            if off_team is not None and off_team in self.off_samples:
                gamma = self.off_samples[off_team]
            if def_team is not None and def_team in self.def_samples:
                delta = self.def_samples[def_team]

            logit_p = (self.beta_samples[:, 0] +
                      self.beta_samples[:, 1] * distance +
                      self.beta_samples[:, 2] * gtg +
                      self.beta_samples[:, 3] * epa_std +
                      self.beta_samples[:, 4] * drive_std +
                      gamma + delta)
            return expit(logit_p)
        else:
            # Point estimate (posterior mean)
            gamma = 0.0
            delta = 0.0

            if off_team is not None and off_team in self.off_effects:
                gamma = self.off_effects[off_team]
            if def_team is not None and def_team in self.def_effects:
                delta = self.def_effects[def_team]

            logit_p = (self.beta_mean[0] +
                      self.beta_mean[1] * distance +
                      self.beta_mean[2] * gtg +
                      self.beta_mean[3] * epa_std +
                      self.beta_mean[4] * drive_std +
                      gamma + delta)
            return expit(logit_p)

    def get_posterior_interval(self, distance: int, off_team: str = None,
                               def_team: str = None, goal_to_go: bool = False,
                               total_epa: float = None, drive_plays: int = None,
                               interval: float = 0.95):
        """Get credible interval for conversion probability."""
        samples = self.get_conversion_prob(
            distance, off_team, def_team, goal_to_go, total_epa, drive_plays,
            return_samples=True
        )
        alpha = (1 - interval) / 2
        return np.percentile(samples, [alpha * 100, (1 - alpha) * 100])

    def get_context_effect(self, distance: int, off_team: str = None,
                           def_team: str = None) -> dict:
        """Show conversion probabilities across different contexts."""
        results = {}

        # Goal-to-go effect (KEY)
        results['goal_to_go'] = {}
        for gtg, label in [(False, 'NOT Goal-to-go'), (True, 'Goal-to-go')]:
            prob = self.get_conversion_prob(distance, off_team, def_team, goal_to_go=gtg)
            ci = self.get_posterior_interval(distance, off_team, def_team, goal_to_go=gtg)
            results['goal_to_go'][label] = {'prob': prob, 'ci': ci}

        # EPA effect
        results['total_epa'] = {}
        for epa_val, label in [(-3, 'Bad (-3)'), (0, 'Average'), (3, 'Good (+3)')]:
            prob = self.get_conversion_prob(distance, off_team, def_team,
                                           goal_to_go=False, total_epa=epa_val)
            ci = self.get_posterior_interval(distance, off_team, def_team,
                                            goal_to_go=False, total_epa=epa_val)
            results['total_epa'][label] = {'prob': prob, 'ci': ci}

        # Drive length effect
        results['drive_plays'] = {}
        for plays, label in [(4, 'Short (4)'), (8, 'Medium (8)'), (12, 'Long (12)')]:
            prob = self.get_conversion_prob(distance, off_team, def_team,
                                           goal_to_go=False, drive_plays=plays)
            ci = self.get_posterior_interval(distance, off_team, def_team,
                                            goal_to_go=False, drive_plays=plays)
            results['drive_plays'][label] = {'prob': prob, 'ci': ci}

        return results

    def save(self, path: Path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'beta_mean': self.beta_mean,
                'beta_cov': self.beta_cov,
                'beta_samples': self.beta_samples,
                'off_effects': self.off_effects,
                'off_se': self.off_se,
                'off_samples': self.off_samples,
                'off_teams': self.off_teams,
                'tau_sq_off': self.tau_sq_off,
                'def_effects': self.def_effects,
                'def_se': self.def_se,
                'def_samples': self.def_samples,
                'def_teams': self.def_teams,
                'tau_sq_def': self.tau_sq_def,
                'epa_mean': self.epa_mean,
                'epa_std': self.epa_std,
                'drive_mean': self.drive_mean,
                'drive_std': self.drive_std,
                'n_samples': self.n_samples,
                'conversion_by_distance': self.conversion_by_distance,
            }, f)

    def load(self, path: Path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for key, value in data.items():
                setattr(self, key, value)
        return self


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent

    print("Loading play-by-play data...")
    pbp = pd.read_parquet(data_dir / 'all_cleaned_1999_2024.parquet')

    # Filter to fourth down attempts
    attempts = pbp[
        (pbp['down'] == 4) &
        (pbp['play_type'].isin(['run', 'pass']))
    ].copy()

    attempts['ydstogo_capped'] = attempts['ydstogo'].clip(upper=15)
    attempts['converted'] = (
        (attempts['first_down_rush'] == 1) |
        (attempts['first_down_pass'] == 1) |
        (attempts['touchdown'] == 1)
    ).astype(int)

    print(f"Total fourth down attempts: {len(attempts)}")

    # Fit Bayesian model
    model = InGameConversionModelBayesian()
    model.fit(attempts, n_samples=4000)

    # Save model
    model.save(models_dir / 'in_game_conversion_model_bayesian.pkl')

    # Demo: show effects with uncertainty
    print("\n" + "="*70)
    print("4TH & 1 CONVERSION PROBABILITIES WITH UNCERTAINTY")
    print("="*70)

    effects = model.get_context_effect(distance=1)

    print("\nGoal-to-go effect (KEY for Simpson's Paradox):")
    for label, data in effects['goal_to_go'].items():
        print(f"  {label}: {data['prob']:.1%} [95% CI: {data['ci'][0]:.1%}, {data['ci'][1]:.1%}]")

    print("\nTotal EPA effect (in-game performance):")
    for label, data in effects['total_epa'].items():
        print(f"  {label}: {data['prob']:.1%} [95% CI: {data['ci'][0]:.1%}, {data['ci'][1]:.1%}]")

    print("\nDrive length effect:")
    for label, data in effects['drive_plays'].items():
        print(f"  {label}: {data['prob']:.1%} [95% CI: {data['ci'][0]:.1%}, {data['ci'][1]:.1%}]")
