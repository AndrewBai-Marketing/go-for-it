"""
Hierarchical Conversion Model with Offense and Defense Fixed Effects

This model captures:
1. Offensive team's ability to convert fourth downs
2. Defensive team's ability to stop fourth downs
3. Both effects are shrunk via empirical Bayes (partial pooling)

This supports the selection-on-observables argument:
- At decision time, the coach knows their offense quality and the opponent's defense quality
- By controlling for both, we capture the coach's information set
- Remaining variation should be noise, not private information
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
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


def fit_bayesian_logistic(X, y, prior_var=100.0):
    """Fit Bayesian logistic regression using Laplace approximation."""
    n, p = X.shape
    prior_precision = np.eye(p) / prior_var

    def neg_log_posterior(beta):
        nll = logistic_log_likelihood(beta, X, y)
        prior_term = 0.5 * beta @ prior_precision @ beta
        return nll + prior_term

    def grad_neg_log_posterior(beta):
        grad_nll = logistic_gradient(beta, X, y)
        grad_prior = prior_precision @ beta
        return grad_nll + grad_prior

    beta_init = np.zeros(p)
    result = minimize(neg_log_posterior, beta_init, method='BFGS',
                     jac=grad_neg_log_posterior, options={'maxiter': 1000})
    beta_map = result.x

    H = logistic_hessian(beta_map, X, y) + prior_precision
    try:
        beta_cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        beta_cov = np.linalg.inv(H + 1e-6 * np.eye(p))

    return beta_map, beta_cov


class HierarchicalOffDefConversionModel:
    """
    Hierarchical Bayesian model for 4th down conversion with BOTH
    offensive team effects and defensive team effects.

    P(convert | distance, off_team, def_team) = σ(α + β*d + γ_off + δ_def)

    Where:
    - γ_off ~ N(0, tau^2_off): offensive team's conversion ability
    - δ_def ~ N(0, tau^2_def): defensive team's ability to stop conversions

    Uses empirical Bayes for partial pooling of both effect types.

    This model supports the selection-on-observables argument:
    coaches observe both their offensive quality and the opponent's
    defensive quality when making decisions.
    """

    def __init__(self):
        self.beta_mean = None       # Population [alpha, beta]
        self.beta_cov = None

        # Offensive effects
        self.off_effects = {}       # {team: γ} shrunk effects
        self.off_se = {}
        self.off_raw = {}
        self.off_n = {}
        self.tau_sq_off = None
        self.off_samples = {}

        # Defensive effects
        self.def_effects = {}       # {team: δ} shrunk effects
        self.def_se = {}
        self.def_raw = {}
        self.def_n = {}
        self.tau_sq_def = None
        self.def_samples = {}

        self.samples = None
        self.n_samples = 2000
        self.off_teams = []
        self.def_teams = []
        self.conversion_by_distance = None

    def fit(self, attempts_df: pd.DataFrame, n_samples: int = 2000,
            prior_var: float = 100.0, min_attempts: int = 20):
        """
        Fit hierarchical conversion model with offense and defense effects.

        Args:
            attempts_df: DataFrame with 'ydstogo_capped', 'converted', 'posteam', 'defteam'
        """
        self.n_samples = n_samples

        df = attempts_df.dropna(subset=['ydstogo_capped', 'converted', 'posteam', 'defteam']).copy()

        # Aggregate by distance for reference
        agg = df.groupby('ydstogo_capped').agg(
            conversions=('converted', 'sum'),
            attempts=('converted', 'count')
        ).reset_index()
        agg['rate'] = agg['conversions'] / agg['attempts']
        self.conversion_by_distance = agg

        print(f"Fitting hierarchical off/def conversion model with {len(df)} attempts")
        print(f"Unique offensive teams: {df['posteam'].nunique()}")
        print(f"Unique defensive teams: {df['defteam'].nunique()}")

        # Step 1: Population model (no team effects)
        X_pop = np.column_stack([np.ones(len(df)), df['ydstogo_capped'].values])
        y = df['converted'].values

        self.beta_mean, self.beta_cov = fit_bayesian_logistic(X_pop, y, prior_var)
        print(f"Population: alpha = {self.beta_mean[0]:.3f} (SE: {np.sqrt(self.beta_cov[0,0]):.3f})")
        print(f"           beta = {self.beta_mean[1]:.3f} (SE: {np.sqrt(self.beta_cov[1,1]):.3f})")

        # Step 2a: Estimate OFFENSIVE team effects
        print("\nEstimating offensive team effects...")
        off_stats = []

        for team, group in df.groupby('posteam'):
            n_k = len(group)
            if n_k < min_attempts:
                continue

            X_k = np.column_stack([np.ones(n_k), group['ydstogo_capped'].values])
            logit_expected = X_k @ self.beta_mean

            y_k = group['converted'].values
            observed_rate = np.clip(y_k.mean(), 0.01, 0.99)
            logit_observed = np.log(observed_rate / (1 - observed_rate))

            expected_rate = np.clip(expit(logit_expected).mean(), 0.01, 0.99)
            logit_expected_avg = np.log(expected_rate / (1 - expected_rate))

            raw_gamma = logit_observed - logit_expected_avg
            se_gamma = np.sqrt(1 / (n_k * observed_rate) + 1 / (n_k * (1 - observed_rate)))

            off_stats.append({
                'team': team,
                'n': n_k,
                'raw_gamma': raw_gamma,
                'se': se_gamma,
                'rate': observed_rate
            })

            self.off_raw[team] = raw_gamma
            self.off_n[team] = n_k

        off_df = pd.DataFrame(off_stats)
        self.off_teams = list(off_df['team'])

        # Estimate tau^2_off
        var_raw = off_df['raw_gamma'].var()
        mean_se_sq = (off_df['se'] ** 2).mean()
        self.tau_sq_off = max(0.001, var_raw - mean_se_sq)

        print(f"  Teams with {min_attempts}+ attempts (offense): {len(off_df)}")
        print(f"  Between-offense variance (tau^2_off): {self.tau_sq_off:.4f}")
        print(f"  Implied offense SD: {np.sqrt(self.tau_sq_off):.3f} log-odds")

        # Shrink offensive effects
        for _, row in off_df.iterrows():
            team = row['team']
            se_k = row['se']
            raw_gamma = row['raw_gamma']

            B_k = se_k**2 / (se_k**2 + self.tau_sq_off)
            shrunk_gamma = (1 - B_k) * raw_gamma
            posterior_var = (1 - B_k) * se_k**2

            self.off_effects[team] = shrunk_gamma
            self.off_se[team] = np.sqrt(posterior_var)

        # Step 2b: Estimate DEFENSIVE team effects
        print("\nEstimating defensive team effects...")
        def_stats = []

        for team, group in df.groupby('defteam'):
            n_k = len(group)
            if n_k < min_attempts:
                continue

            X_k = np.column_stack([np.ones(n_k), group['ydstogo_capped'].values])
            logit_expected = X_k @ self.beta_mean

            y_k = group['converted'].values
            observed_rate = np.clip(y_k.mean(), 0.01, 0.99)
            logit_observed = np.log(observed_rate / (1 - observed_rate))

            expected_rate = np.clip(expit(logit_expected).mean(), 0.01, 0.99)
            logit_expected_avg = np.log(expected_rate / (1 - expected_rate))

            # Note: for defense, NEGATIVE effect means good at stopping
            raw_delta = logit_observed - logit_expected_avg
            se_delta = np.sqrt(1 / (n_k * observed_rate) + 1 / (n_k * (1 - observed_rate)))

            def_stats.append({
                'team': team,
                'n': n_k,
                'raw_delta': raw_delta,
                'se': se_delta,
                'rate': observed_rate  # conversion rate ALLOWED
            })

            self.def_raw[team] = raw_delta
            self.def_n[team] = n_k

        def_df = pd.DataFrame(def_stats)
        self.def_teams = list(def_df['team'])

        # Estimate tau^2_def
        var_raw = def_df['raw_delta'].var()
        mean_se_sq = (def_df['se'] ** 2).mean()
        self.tau_sq_def = max(0.001, var_raw - mean_se_sq)

        print(f"  Teams with {min_attempts}+ attempts (defense): {len(def_df)}")
        print(f"  Between-defense variance (tau^2_def): {self.tau_sq_def:.4f}")
        print(f"  Implied defense SD: {np.sqrt(self.tau_sq_def):.3f} log-odds")

        # Shrink defensive effects
        for _, row in def_df.iterrows():
            team = row['team']
            se_k = row['se']
            raw_delta = row['raw_delta']

            B_k = se_k**2 / (se_k**2 + self.tau_sq_def)
            shrunk_delta = (1 - B_k) * raw_delta
            posterior_var = (1 - B_k) * se_k**2

            self.def_effects[team] = shrunk_delta
            self.def_se[team] = np.sqrt(posterior_var)

        # Step 3: Draw posterior samples
        print("\nDrawing posterior samples...")
        self.samples = np.random.multivariate_normal(
            self.beta_mean, self.beta_cov, size=n_samples
        )

        for team in self.off_teams:
            mean_k = self.off_effects[team]
            se_k = self.off_se[team]
            self.off_samples[team] = np.random.normal(mean_k, se_k, size=n_samples)

        for team in self.def_teams:
            mean_k = self.def_effects[team]
            se_k = self.def_se[team]
            self.def_samples[team] = np.random.normal(mean_k, se_k, size=n_samples)

        # Print summary
        print("\n" + "="*60)
        print("OFFENSIVE TEAM EFFECTS (positive = better at converting)")
        print("="*60)
        sorted_off = sorted(self.off_effects.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 offenses:")
        for team, effect in sorted_off[:5]:
            raw = self.off_raw[team]
            n = self.off_n[team]
            print(f"  {team}: +{effect:.3f} (raw: +{raw:.3f}, n={n})")

        print("\nBottom 5 offenses:")
        for team, effect in sorted_off[-5:]:
            raw = self.off_raw[team]
            n = self.off_n[team]
            print(f"  {team}: {effect:.3f} (raw: {raw:.3f}, n={n})")

        print("\n" + "="*60)
        print("DEFENSIVE TEAM EFFECTS (negative = better at stopping)")
        print("="*60)
        sorted_def = sorted(self.def_effects.items(), key=lambda x: x[1])  # Lower is better
        print("\nTop 5 defenses (best at stopping):")
        for team, effect in sorted_def[:5]:
            raw = self.def_raw[team]
            n = self.def_n[team]
            print(f"  {team}: {effect:.3f} (raw: {raw:.3f}, n={n})")

        print("\nBottom 5 defenses (worst at stopping):")
        for team, effect in sorted_def[-5:]:
            raw = self.def_raw[team]
            n = self.def_n[team]
            print(f"  {team}: +{effect:.3f} (raw: +{raw:.3f}, n={n})")

        return self

    def get_conversion_prob(self, distance: int, off_team: str = None,
                           def_team: str = None, posterior_idx: int = None) -> float:
        """
        Get conversion probability for given distance and team matchup.

        Args:
            distance: Yards to go
            off_team: Offensive team (None for league average)
            def_team: Defensive team (None for league average)
            posterior_idx: Specific posterior sample index
        """
        distance = min(max(distance, 1), 15)

        # Get offensive effect
        if off_team is not None and off_team in self.off_effects:
            if posterior_idx is not None:
                gamma = self.off_samples[off_team][posterior_idx % self.n_samples]
            else:
                gamma = self.off_effects[off_team]
        else:
            gamma = 0.0

        # Get defensive effect
        if def_team is not None and def_team in self.def_effects:
            if posterior_idx is not None:
                delta = self.def_samples[def_team][posterior_idx % self.n_samples]
            else:
                delta = self.def_effects[def_team]
        else:
            delta = 0.0

        if posterior_idx is not None:
            beta = self.samples[posterior_idx % self.n_samples]
            logit_p = beta[0] + beta[1] * distance + gamma + delta
        else:
            logit_p = self.beta_mean[0] + self.beta_mean[1] * distance + gamma + delta

        return expit(logit_p)

    def get_posterior_samples(self, distance: int, off_team: str = None,
                             def_team: str = None) -> np.ndarray:
        """Get all posterior samples for conversion prob."""
        distance = min(max(distance, 1), 15)

        if off_team is not None and off_team in self.off_samples:
            gamma_samples = self.off_samples[off_team]
        else:
            gamma_samples = np.zeros(self.n_samples)

        if def_team is not None and def_team in self.def_samples:
            delta_samples = self.def_samples[def_team]
        else:
            delta_samples = np.zeros(self.n_samples)

        logit_p = (self.samples[:, 0] + self.samples[:, 1] * distance +
                   gamma_samples + delta_samples)
        return expit(logit_p)

    def get_credible_interval(self, distance: int, off_team: str = None,
                             def_team: str = None, alpha: float = 0.05) -> tuple:
        """Get credible interval for conversion probability."""
        samples = self.get_posterior_samples(distance, off_team, def_team)
        return np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])

    def get_matchup_advantage(self, off_team: str, def_team: str) -> dict:
        """
        Get the matchup advantage for a specific offense vs defense.

        Returns the combined effect (γ_off + δ_def) and its interpretation.
        """
        off_effect = self.off_effects.get(off_team, 0.0)
        def_effect = self.def_effects.get(def_team, 0.0)
        combined = off_effect + def_effect

        # Convert to probability difference at 4th & 3
        base_prob = expit(self.beta_mean[0] + self.beta_mean[1] * 3)
        matchup_prob = expit(self.beta_mean[0] + self.beta_mean[1] * 3 + combined)

        return {
            'off_effect': off_effect,
            'def_effect': def_effect,
            'combined_effect': combined,
            'base_prob_4th_and_3': base_prob,
            'matchup_prob_4th_and_3': matchup_prob,
            'prob_difference': matchup_prob - base_prob
        }

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({
                'beta_mean': self.beta_mean,
                'beta_cov': self.beta_cov,
                'off_effects': self.off_effects,
                'off_se': self.off_se,
                'off_raw': self.off_raw,
                'off_n': self.off_n,
                'tau_sq_off': self.tau_sq_off,
                'off_samples': self.off_samples,
                'off_teams': self.off_teams,
                'def_effects': self.def_effects,
                'def_se': self.def_se,
                'def_raw': self.def_raw,
                'def_n': self.def_n,
                'tau_sq_def': self.tau_sq_def,
                'def_samples': self.def_samples,
                'def_teams': self.def_teams,
                'samples': self.samples,
                'conversion_by_distance': self.conversion_by_distance
            }, f)

    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.beta_mean = data['beta_mean']
            self.beta_cov = data['beta_cov']
            self.off_effects = data['off_effects']
            self.off_se = data['off_se']
            self.off_raw = data['off_raw']
            self.off_n = data['off_n']
            self.tau_sq_off = data['tau_sq_off']
            self.off_samples = data['off_samples']
            self.off_teams = data['off_teams']
            self.def_effects = data['def_effects']
            self.def_se = data['def_se']
            self.def_raw = data['def_raw']
            self.def_n = data['def_n']
            self.tau_sq_def = data['tau_sq_def']
            self.def_samples = data['def_samples']
            self.def_teams = data['def_teams']
            self.samples = data['samples']
            self.conversion_by_distance = data.get('conversion_by_distance')
            self.n_samples = len(self.samples)
        return self


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent

    # Load data
    print("Loading fourth down attempts data...")
    attempts = pd.read_parquet(data_dir / 'fourth_down_attempts.parquet')

    # Fit model
    model = HierarchicalOffDefConversionModel()
    model.fit(attempts, n_samples=2000)

    # Save model
    model.save(models_dir / 'hierarchical_off_def_conversion_model.pkl')

    # Demo: show some matchup comparisons
    print("\n" + "="*60)
    print("MATCHUP EXAMPLES")
    print("="*60)

    matchups = [
        ('PHI', 'NYG'),  # Good offense vs bad defense
        ('CHI', 'BAL'),  # Bad offense vs good defense
        ('DET', 'GB'),   # Divisional matchup
        ('KC', 'BUF'),   # AFC matchup
    ]

    for off_team, def_team in matchups:
        if off_team in model.off_effects and def_team in model.def_effects:
            matchup = model.get_matchup_advantage(off_team, def_team)
            print(f"\n{off_team} offense vs {def_team} defense:")
            print(f"  Offense effect: {matchup['off_effect']:+.3f}")
            print(f"  Defense effect: {matchup['def_effect']:+.3f}")
            print(f"  Combined: {matchup['combined_effect']:+.3f}")
            print(f"  4th & 3 conversion prob: {matchup['matchup_prob_4th_and_3']:.1%} "
                  f"(vs {matchup['base_prob_4th_and_3']:.1%} league avg)")
