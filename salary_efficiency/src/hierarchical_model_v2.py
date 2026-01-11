"""
Bayesian Hierarchical Model v2: Improved Specification

Key improvements from feedback:
1. Latent talent model - WAR is noisy proxy for true value
2. Random walk prior for year effects (smooth market inflation)
3. Piecewise WAR slope for star premium (knot at 3 WAR)
4. Heteroskedastic residuals (variance scales with WAR)
5. Uses REAL salary data from Lahman

Model:
    theta_i ~ latent true talent (partially pooled by player)
    WAR_i | theta_i ~ N(theta_i, sigma_war^2)  [observation model]
    log(salary_i) = alpha + beta1*theta_i + beta2*(theta_i-3)_+ + gamma_team + gamma_year + epsilon_i
    gamma_year ~ RandomWalk(tau)  [smooth market trends]
    gamma_team ~ N(0, sigma_team^2)  [partial pooling]
    epsilon_i ~ N(0, sigma_i^2) where log(sigma_i) = a + b*theta_i  [heteroskedasticity]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import pickle
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ImprovedEfficiencyModel:
    """
    Bayesian hierarchical model with:
    - Latent talent (WAR measurement error)
    - Random walk year effects
    - Piecewise linear star premium
    - Heteroskedastic residuals
    """

    def __init__(self,
                 sigma_war: float = 0.5,  # WAR measurement error SD
                 star_knot: float = 3.0,  # WAR threshold for star premium
                 prior_sigma_team: float = 0.2,
                 prior_tau_year: float = 0.1):  # Random walk innovation SD
        """
        Initialize model with priors.

        Parameters
        ----------
        sigma_war : float
            Assumed measurement error SD for WAR
        star_knot : float
            WAR value where star premium kicks in
        prior_sigma_team : float
            Prior scale for team effect SD
        prior_tau_year : float
            Prior scale for year random walk innovation
        """
        self.sigma_war = sigma_war
        self.star_knot = star_knot
        self.prior_sigma_team = prior_sigma_team
        self.prior_tau_year = prior_tau_year

        self.teams = None
        self.years = None
        self.team_to_idx = None
        self.year_to_idx = None

        self.posterior_samples = None

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """Prepare data for modeling."""
        df = df.copy()

        # Log salary as outcome
        y = np.log(df['salary'].values)

        # Observed WAR
        war_obs = df['war'].values

        # Team indices
        self.teams = sorted(df['team'].unique())
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        team_idx = df['team'].map(self.team_to_idx).values.astype(int)

        # Year indices (sorted for random walk)
        self.years = sorted(df['season'].unique())
        self.year_to_idx = {y: i for i, y in enumerate(self.years)}
        year_idx = df['season'].map(self.year_to_idx).values.astype(int)

        return y, war_obs, team_idx, year_idx

    def _star_premium_feature(self, war: np.ndarray) -> np.ndarray:
        """Compute (WAR - knot)_+ for star premium."""
        return np.maximum(war - self.star_knot, 0)

    def _log_likelihood(self, params: Dict, y: np.ndarray, war_obs: np.ndarray,
                        team_idx: np.ndarray, year_idx: np.ndarray) -> float:
        """
        Compute log likelihood with latent talent and heteroskedasticity.
        """
        # Unpack parameters
        alpha = params['alpha']
        beta1 = params['beta1']  # Base WAR coefficient
        beta2 = params['beta2']  # Star premium coefficient
        theta = params['theta']  # Latent talent (per observation)
        gamma_team = params['gamma_team']
        gamma_year = params['gamma_year']
        sigma_base = params['sigma_base']
        sigma_slope = params['sigma_slope']

        n = len(y)

        # WAR observation model: WAR_obs ~ N(theta, sigma_war^2)
        ll_war = np.sum(norm.logpdf(war_obs, loc=theta, scale=self.sigma_war))

        # Star premium feature
        star_feat = self._star_premium_feature(theta)

        # Predicted log salary
        mu = (alpha + beta1 * theta + beta2 * star_feat +
              gamma_team[team_idx] + gamma_year[year_idx])

        # Heteroskedastic residual SD: log(sigma_i) = sigma_base + sigma_slope * theta
        log_sigma = sigma_base + sigma_slope * theta
        sigma = np.exp(np.clip(log_sigma, -2, 2))  # Clip for stability

        # Salary log likelihood
        ll_salary = np.sum(norm.logpdf(y, loc=mu, scale=sigma))

        return ll_war + ll_salary

    def _log_prior(self, params: Dict) -> float:
        """Compute log prior with random walk for year effects."""
        lp = 0.0

        # Fixed effects priors (weakly informative)
        lp += norm.logpdf(params['alpha'], loc=14.0, scale=2.0)
        lp += norm.logpdf(params['beta1'], loc=0.3, scale=0.2)  # ~exp(0.3)=1.35x per WAR
        lp += norm.logpdf(params['beta2'], loc=0.1, scale=0.1)  # Star premium

        # Latent talent prior: theta ~ N(war_obs, sigma_talent^2)
        # This provides regularization toward observed WAR
        sigma_talent = params.get('sigma_talent', 1.0)
        # Prior on theta is implicit through the observation model

        # Team effects: gamma_team ~ N(0, sigma_team^2)
        sigma_team = params['sigma_team']
        lp += norm.logpdf(sigma_team, loc=0, scale=self.prior_sigma_team) + np.log(2)  # Half-normal
        if sigma_team > 0:
            lp += np.sum(norm.logpdf(params['gamma_team'], loc=0, scale=sigma_team))

        # Year effects: Random walk prior
        # gamma_year[0] ~ N(0, 0.5)
        # gamma_year[t] | gamma_year[t-1] ~ N(gamma_year[t-1], tau^2)
        tau = params['tau_year']
        lp += norm.logpdf(tau, loc=0, scale=self.prior_tau_year) + np.log(2)  # Half-normal

        gamma_year = params['gamma_year']
        lp += norm.logpdf(gamma_year[0], loc=0, scale=0.5)
        if tau > 0:
            for t in range(1, len(gamma_year)):
                lp += norm.logpdf(gamma_year[t], loc=gamma_year[t-1], scale=tau)

        # Heteroskedasticity parameters
        lp += norm.logpdf(params['sigma_base'], loc=-0.5, scale=0.5)
        lp += norm.logpdf(params['sigma_slope'], loc=0, scale=0.1)

        return lp

    def _log_posterior(self, params: Dict, y: np.ndarray, war_obs: np.ndarray,
                       team_idx: np.ndarray, year_idx: np.ndarray) -> float:
        """Compute log posterior."""
        # Check constraints
        if params['sigma_team'] <= 0 or params['tau_year'] <= 0:
            return -np.inf

        ll = self._log_likelihood(params, y, war_obs, team_idx, year_idx)
        lp = self._log_prior(params)

        return ll + lp

    def _initialize_params(self, y: np.ndarray, war_obs: np.ndarray) -> Dict:
        """Initialize parameters from data."""
        n = len(y)
        n_teams = len(self.teams)
        n_years = len(self.years)

        return {
            'alpha': 14.0,
            'beta1': 0.3,
            'beta2': 0.1,
            'theta': war_obs.copy(),  # Initialize latent at observed
            'gamma_team': np.zeros(n_teams),
            'gamma_year': np.zeros(n_years),
            'sigma_team': 0.1,
            'tau_year': 0.05,
            'sigma_base': -0.5,
            'sigma_slope': 0.05,
        }

    def _mcmc_sampler(self, y: np.ndarray, war_obs: np.ndarray,
                      team_idx: np.ndarray, year_idx: np.ndarray,
                      n_samples: int = 3000, n_warmup: int = 1000,
                      seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Sample from posterior using Metropolis-within-Gibbs.
        """
        np.random.seed(seed)

        n = len(y)
        n_teams = len(self.teams)
        n_years = len(self.years)

        # Initialize
        params = self._initialize_params(y, war_obs)

        # Proposal SDs
        prop_sd = {
            'alpha': 0.05,
            'beta1': 0.02,
            'beta2': 0.02,
            'theta': np.ones(n) * 0.1,
            'gamma_team': np.ones(n_teams) * 0.05,
            'gamma_year': np.ones(n_years) * 0.05,
            'sigma_team': 0.02,
            'tau_year': 0.01,
            'sigma_base': 0.05,
            'sigma_slope': 0.02,
        }

        # Storage
        n_total = n_samples + n_warmup
        samples = {
            'alpha': np.zeros(n_total),
            'beta1': np.zeros(n_total),
            'beta2': np.zeros(n_total),
            'gamma_team': np.zeros((n_total, n_teams)),
            'gamma_year': np.zeros((n_total, n_years)),
            'sigma_team': np.zeros(n_total),
            'tau_year': np.zeros(n_total),
            'sigma_base': np.zeros(n_total),
            'sigma_slope': np.zeros(n_total),
        }

        # We don't store all theta samples (too large), just summaries
        theta_sum = np.zeros(n)
        theta_sumsq = np.zeros(n)

        log_post_current = self._log_posterior(params, y, war_obs, team_idx, year_idx)

        print(f"  Running MCMC: {n_warmup} warmup + {n_samples} samples...")

        for i in range(n_total):
            # Update scalar parameters with MH
            for key in ['alpha', 'beta1', 'beta2', 'sigma_base', 'sigma_slope']:
                params_prop = params.copy()
                params_prop[key] = params[key] + np.random.normal(0, prop_sd[key])
                log_post_prop = self._log_posterior(params_prop, y, war_obs, team_idx, year_idx)
                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop

            # Update variance parameters (positive constraint)
            for key in ['sigma_team', 'tau_year']:
                params_prop = params.copy()
                params_prop[key] = abs(params[key] + np.random.normal(0, prop_sd[key]))
                log_post_prop = self._log_posterior(params_prop, y, war_obs, team_idx, year_idx)
                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop

            # Update latent theta (block of 100 at a time for efficiency)
            theta_new = params['theta'].copy()
            for block_start in range(0, n, 100):
                block_end = min(block_start + 100, n)
                block_idx = slice(block_start, block_end)

                theta_new[block_idx] = params['theta'][block_idx] + \
                    np.random.normal(0, prop_sd['theta'][block_idx])

                params_prop = params.copy()
                params_prop['theta'] = theta_new.copy()
                log_post_prop = self._log_posterior(params_prop, y, war_obs, team_idx, year_idx)

                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop
                else:
                    theta_new[block_idx] = params['theta'][block_idx]

            # Update team effects
            for t in range(n_teams):
                params_prop = params.copy()
                params_prop['gamma_team'] = params['gamma_team'].copy()
                params_prop['gamma_team'][t] += np.random.normal(0, prop_sd['gamma_team'][t])
                log_post_prop = self._log_posterior(params_prop, y, war_obs, team_idx, year_idx)
                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop

            # Update year effects
            for yr in range(n_years):
                params_prop = params.copy()
                params_prop['gamma_year'] = params['gamma_year'].copy()
                params_prop['gamma_year'][yr] += np.random.normal(0, prop_sd['gamma_year'][yr])
                log_post_prop = self._log_posterior(params_prop, y, war_obs, team_idx, year_idx)
                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop

            # Store samples
            samples['alpha'][i] = params['alpha']
            samples['beta1'][i] = params['beta1']
            samples['beta2'][i] = params['beta2']
            samples['gamma_team'][i] = params['gamma_team']
            samples['gamma_year'][i] = params['gamma_year']
            samples['sigma_team'][i] = params['sigma_team']
            samples['tau_year'][i] = params['tau_year']
            samples['sigma_base'][i] = params['sigma_base']
            samples['sigma_slope'][i] = params['sigma_slope']

            # Accumulate theta statistics (after warmup)
            if i >= n_warmup:
                theta_sum += params['theta']
                theta_sumsq += params['theta']**2

            if (i + 1) % 500 == 0:
                phase = "warmup" if i < n_warmup else "sampling"
                print(f"    Iteration {i+1}/{n_total} ({phase}), log_post = {log_post_current:.1f}")

        # Discard warmup
        posterior = {k: v[n_warmup:] for k, v in samples.items()}

        # Add theta summary statistics
        posterior['theta_mean'] = theta_sum / n_samples
        posterior['theta_sd'] = np.sqrt(theta_sumsq / n_samples - (theta_sum / n_samples)**2)

        return posterior

    def fit(self, df: pd.DataFrame, n_samples: int = 3000, n_warmup: int = 1000) -> 'ImprovedEfficiencyModel':
        """Fit the improved hierarchical model."""
        print(f"Fitting improved hierarchical model on {len(df):,} player-seasons...")
        print(f"  Teams: {df['team'].nunique()}")
        print(f"  Years: {df['season'].min()}-{df['season'].max()}")
        print(f"  WAR measurement error assumed: {self.sigma_war}")
        print(f"  Star premium knot: {self.star_knot} WAR")

        # Prepare data
        y, war_obs, team_idx, year_idx = self._prepare_data(df)

        print(f"  Log salary range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  WAR range: [{war_obs.min():.2f}, {war_obs.max():.2f}]")

        # Run MCMC
        self.posterior_samples = self._mcmc_sampler(
            y, war_obs, team_idx, year_idx,
            n_samples=n_samples, n_warmup=n_warmup
        )

        # Print summary
        self._print_summary()

        return self

    def _print_summary(self):
        """Print posterior summary."""
        print("\n" + "="*60)
        print("POSTERIOR SUMMARY")
        print("="*60)

        ps = self.posterior_samples

        print(f"\nFixed Effects:")
        print(f"  alpha (intercept):     {ps['alpha'].mean():.3f} +/- {ps['alpha'].std():.3f}")
        print(f"  beta1 (base WAR):      {ps['beta1'].mean():.3f} +/- {ps['beta1'].std():.3f}")
        print(f"  beta2 (star premium):  {ps['beta2'].mean():.3f} +/- {ps['beta2'].std():.3f}")

        # Interpret coefficients
        base_mult = np.exp(ps['beta1'].mean())
        star_mult = np.exp(ps['beta1'].mean() + ps['beta2'].mean())
        print(f"\n  Interpretation:")
        print(f"    Each WAR below {self.star_knot}: {base_mult:.2f}x salary multiplier")
        print(f"    Each WAR above {self.star_knot}: {star_mult:.2f}x salary multiplier")

        print(f"\nVariance Components:")
        print(f"  sigma_team:  {ps['sigma_team'].mean():.3f}")
        print(f"  tau_year (RW innovation): {ps['tau_year'].mean():.3f}")

        print(f"\nHeteroskedasticity:")
        print(f"  sigma_base:  {ps['sigma_base'].mean():.3f}")
        print(f"  sigma_slope: {ps['sigma_slope'].mean():.3f}")

    def get_team_effects(self) -> pd.DataFrame:
        """Get posterior summaries for team effects."""
        gamma_samples = self.posterior_samples['gamma_team']

        results = []
        for i, team in enumerate(self.teams):
            samples = gamma_samples[:, i]
            efficiency = np.exp(-samples)

            results.append({
                'team': team,
                'gamma_mean': samples.mean(),
                'gamma_sd': samples.std(),
                'gamma_ci_lower': np.percentile(samples, 2.5),
                'gamma_ci_upper': np.percentile(samples, 97.5),
                'efficiency_mean': efficiency.mean(),
                'efficiency_ci_lower': np.percentile(efficiency, 2.5),
                'efficiency_ci_upper': np.percentile(efficiency, 97.5),
                'prob_above_avg': np.mean(samples < 0),  # Negative gamma = more efficient
            })

        return pd.DataFrame(results).sort_values('efficiency_mean', ascending=False)

    def get_year_effects(self) -> pd.DataFrame:
        """Get posterior summaries for year effects (random walk)."""
        gamma_samples = self.posterior_samples['gamma_year']

        results = []
        for i, year in enumerate(self.years):
            samples = gamma_samples[:, i]

            results.append({
                'year': year,
                'gamma_mean': samples.mean(),
                'gamma_sd': samples.std(),
                'gamma_ci_lower': np.percentile(samples, 2.5),
                'gamma_ci_upper': np.percentile(samples, 97.5),
            })

        return pd.DataFrame(results).sort_values('year')

    def save(self, filepath: Path):
        """Save model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> 'ImprovedEfficiencyModel':
        """Load model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def main():
    """Fit improved model on real salary data."""
    data_dir = Path('salary_efficiency/data/processed')
    model_dir = Path('salary_efficiency/models')
    output_dir = Path('salary_efficiency/outputs/tables')

    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load REAL salary data
    print("Loading real salary data...")
    df = pd.read_parquet(data_dir / 'player_seasons_real_salary.parquet')
    print(f"Loaded {len(df):,} player-seasons with real salaries")
    print(f"Year range: {df['season'].min()}-{df['season'].max()}")

    # Fit improved model
    model = ImprovedEfficiencyModel(
        sigma_war=0.5,  # Assume ~0.5 WAR measurement error
        star_knot=3.0,  # Star premium kicks in at 3 WAR
        prior_sigma_team=0.2,
        prior_tau_year=0.1
    )
    model.fit(df, n_samples=3000, n_warmup=1000)

    # Save
    model.save(model_dir / 'improved_efficiency_model.pkl')

    # Get team effects
    team_effects = model.get_team_effects()
    team_effects.to_csv(output_dir / 'team_efficiency_rankings_v2.csv', index=False)

    print("\n" + "="*60)
    print("TEAM EFFICIENCY RANKINGS (Improved Model)")
    print("="*60)
    print("\nTop 10 Most Efficient:")
    print(team_effects.head(10)[['team', 'gamma_mean', 'efficiency_mean',
                                  'efficiency_ci_lower', 'efficiency_ci_upper',
                                  'prob_above_avg']].to_string(index=False))

    print("\nBottom 10:")
    print(team_effects.tail(10)[['team', 'gamma_mean', 'efficiency_mean',
                                  'efficiency_ci_lower', 'efficiency_ci_upper',
                                  'prob_above_avg']].to_string(index=False))

    # Get year effects
    year_effects = model.get_year_effects()
    year_effects.to_csv(output_dir / 'year_effects_v2.csv', index=False)

    print("\n" + "="*60)
    print("YEAR EFFECTS (Random Walk)")
    print("="*60)
    print(year_effects.to_string(index=False))

    return model, team_effects, year_effects


if __name__ == "__main__":
    model, team_effects, year_effects = main()
