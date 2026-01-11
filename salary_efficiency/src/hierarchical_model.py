"""
Bayesian Hierarchical Model for Team Salary Efficiency

Models team-level efficiency with:
- Team random effects (some teams consistently better at valuation)
- Position random effects
- Year random effects (to capture market trends)
- Partial pooling via hierarchical priors

Uses MCMC for posterior sampling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.special import logsumexp
import pickle
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class TeamEfficiencyModel:
    """
    Bayesian hierarchical model for team salary efficiency.

    Model:
        log(salary_i) = alpha + beta * WAR_i + gamma_team[t_i] + gamma_year[y_i] + epsilon_i

    Where:
        - alpha: intercept (baseline salary)
        - beta: $/WAR coefficient
        - gamma_team: team random effects ~ N(0, sigma_team^2)
        - gamma_year: year random effects ~ N(0, sigma_year^2)
        - epsilon: residual ~ N(0, sigma^2)

    Efficiency = exp(-gamma_team): teams with negative gamma pay less for same WAR
    """

    def __init__(self,
                 prior_beta_mean: float = 16.0,  # log($8M) ~ 16
                 prior_beta_sd: float = 1.0,
                 prior_sigma_team: float = 0.5,
                 prior_sigma_year: float = 0.5):
        """
        Initialize model with priors.

        Parameters
        ----------
        prior_beta_mean : float
            Prior mean for log($/WAR)
        prior_beta_sd : float
            Prior SD for coefficients
        prior_sigma_team : float
            Prior scale for team random effect SD
        prior_sigma_year : float
            Prior scale for year random effect SD
        """
        self.prior_beta_mean = prior_beta_mean
        self.prior_beta_sd = prior_beta_sd
        self.prior_sigma_team = prior_sigma_team
        self.prior_sigma_year = prior_sigma_year

        self.teams = None
        self.years = None
        self.team_to_idx = None
        self.year_to_idx = None

        self.posterior_samples = None
        self.param_names = None

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for modeling."""
        df = df.copy()

        # Log salary as outcome
        y = np.log(df['salary'].values)

        # WAR as main predictor
        X_war = df['war'].values

        # Team indices
        self.teams = sorted(df['team'].unique())
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        team_idx = df['team'].map(self.team_to_idx).values

        # Year indices
        self.years = sorted(df['season'].unique())
        self.year_to_idx = {y: i for i, y in enumerate(self.years)}
        year_idx = df['season'].map(self.year_to_idx).values

        return y, X_war, team_idx.astype(int), year_idx.astype(int)

    def _log_likelihood(self, params: Dict[str, np.ndarray],
                        y: np.ndarray, X_war: np.ndarray,
                        team_idx: np.ndarray, year_idx: np.ndarray) -> float:
        """Compute log likelihood."""
        alpha = params['alpha']
        beta = params['beta']
        gamma_team = params['gamma_team']
        gamma_year = params['gamma_year']
        sigma = params['sigma']

        # Predicted log salary
        mu = alpha + beta * X_war + gamma_team[team_idx] + gamma_year[year_idx]

        # Normal log likelihood
        ll = np.sum(norm.logpdf(y, loc=mu, scale=sigma))
        return ll

    def _log_prior(self, params: Dict[str, np.ndarray]) -> float:
        """Compute log prior."""
        lp = 0.0

        # Fixed effects priors
        lp += norm.logpdf(params['alpha'], loc=14.0, scale=2.0)  # log(~$1M)
        lp += norm.logpdf(params['beta'], loc=self.prior_beta_mean, scale=self.prior_beta_sd)

        # Hierarchical priors for random effects
        sigma_team = params['sigma_team']
        sigma_year = params['sigma_year']

        # Half-normal priors on variance components
        lp += norm.logpdf(sigma_team, loc=0, scale=self.prior_sigma_team) + np.log(2)
        lp += norm.logpdf(sigma_year, loc=0, scale=self.prior_sigma_year) + np.log(2)

        # Random effects centered at zero
        lp += np.sum(norm.logpdf(params['gamma_team'], loc=0, scale=sigma_team))
        lp += np.sum(norm.logpdf(params['gamma_year'], loc=0, scale=sigma_year))

        # Residual SD prior (half-normal)
        lp += norm.logpdf(params['sigma'], loc=0, scale=1.0) + np.log(2)

        return lp

    def _log_posterior(self, params: Dict[str, np.ndarray],
                       y: np.ndarray, X_war: np.ndarray,
                       team_idx: np.ndarray, year_idx: np.ndarray) -> float:
        """Compute log posterior (unnormalized)."""
        # Check constraints
        if params['sigma'] <= 0 or params['sigma_team'] <= 0 or params['sigma_year'] <= 0:
            return -np.inf

        return self._log_likelihood(params, y, X_war, team_idx, year_idx) + self._log_prior(params)

    def _initialize_params(self, y: np.ndarray, X_war: np.ndarray) -> Dict[str, np.ndarray]:
        """Initialize parameters from data."""
        n_teams = len(self.teams)
        n_years = len(self.years)

        # Simple OLS for initialization
        mean_y = y.mean()
        mean_war = X_war.mean()

        return {
            'alpha': mean_y - 0.5 * mean_war,  # Rough intercept
            'beta': 0.5,  # Rough slope in log scale
            'gamma_team': np.zeros(n_teams),
            'gamma_year': np.zeros(n_years),
            'sigma_team': 0.3,
            'sigma_year': 0.3,
            'sigma': 1.0
        }

    def _gibbs_sampler(self, y: np.ndarray, X_war: np.ndarray,
                       team_idx: np.ndarray, year_idx: np.ndarray,
                       n_samples: int = 5000, n_warmup: int = 1000,
                       seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Sample from posterior using Gibbs sampling with Metropolis steps.
        """
        np.random.seed(seed)

        n = len(y)
        n_teams = len(self.teams)
        n_years = len(self.years)

        # Initialize
        params = self._initialize_params(y, X_war)

        # Proposal SDs (will be adapted)
        proposal_sd = {
            'alpha': 0.1,
            'beta': 0.05,
            'gamma_team': np.ones(n_teams) * 0.1,
            'gamma_year': np.ones(n_years) * 0.1,
            'sigma': 0.05,
            'sigma_team': 0.05,
            'sigma_year': 0.05
        }

        # Storage
        n_total = n_samples + n_warmup
        samples = {
            'alpha': np.zeros(n_total),
            'beta': np.zeros(n_total),
            'gamma_team': np.zeros((n_total, n_teams)),
            'gamma_year': np.zeros((n_total, n_years)),
            'sigma': np.zeros(n_total),
            'sigma_team': np.zeros(n_total),
            'sigma_year': np.zeros(n_total)
        }

        log_post_current = self._log_posterior(params, y, X_war, team_idx, year_idx)

        print(f"  Running Gibbs/MH sampler: {n_warmup} warmup + {n_samples} samples...")

        accept_counts = {k: 0 for k in proposal_sd.keys()}

        for i in range(n_total):
            # Update each parameter with MH step

            # Alpha
            params_prop = params.copy()
            params_prop['alpha'] = params['alpha'] + np.random.normal(0, proposal_sd['alpha'])
            log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
            if np.log(np.random.random()) < log_post_prop - log_post_current:
                params = params_prop
                log_post_current = log_post_prop
                if i < n_warmup:
                    accept_counts['alpha'] += 1

            # Beta
            params_prop = params.copy()
            params_prop['beta'] = params['beta'] + np.random.normal(0, proposal_sd['beta'])
            log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
            if np.log(np.random.random()) < log_post_prop - log_post_current:
                params = params_prop
                log_post_current = log_post_prop
                if i < n_warmup:
                    accept_counts['beta'] += 1

            # Team random effects (block update)
            for t in range(n_teams):
                params_prop = params.copy()
                params_prop['gamma_team'] = params['gamma_team'].copy()
                params_prop['gamma_team'][t] += np.random.normal(0, proposal_sd['gamma_team'][t])
                log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop

            # Year random effects
            for yr in range(n_years):
                params_prop = params.copy()
                params_prop['gamma_year'] = params['gamma_year'].copy()
                params_prop['gamma_year'][yr] += np.random.normal(0, proposal_sd['gamma_year'][yr])
                log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
                if np.log(np.random.random()) < log_post_prop - log_post_current:
                    params = params_prop
                    log_post_current = log_post_prop

            # Sigma
            params_prop = params.copy()
            params_prop['sigma'] = abs(params['sigma'] + np.random.normal(0, proposal_sd['sigma']))
            log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
            if np.log(np.random.random()) < log_post_prop - log_post_current:
                params = params_prop
                log_post_current = log_post_prop
                if i < n_warmup:
                    accept_counts['sigma'] += 1

            # Sigma_team
            params_prop = params.copy()
            params_prop['sigma_team'] = abs(params['sigma_team'] + np.random.normal(0, proposal_sd['sigma_team']))
            log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
            if np.log(np.random.random()) < log_post_prop - log_post_current:
                params = params_prop
                log_post_current = log_post_prop
                if i < n_warmup:
                    accept_counts['sigma_team'] += 1

            # Sigma_year
            params_prop = params.copy()
            params_prop['sigma_year'] = abs(params['sigma_year'] + np.random.normal(0, proposal_sd['sigma_year']))
            log_post_prop = self._log_posterior(params_prop, y, X_war, team_idx, year_idx)
            if np.log(np.random.random()) < log_post_prop - log_post_current:
                params = params_prop
                log_post_current = log_post_prop
                if i < n_warmup:
                    accept_counts['sigma_year'] += 1

            # Store samples
            samples['alpha'][i] = params['alpha']
            samples['beta'][i] = params['beta']
            samples['gamma_team'][i] = params['gamma_team']
            samples['gamma_year'][i] = params['gamma_year']
            samples['sigma'][i] = params['sigma']
            samples['sigma_team'][i] = params['sigma_team']
            samples['sigma_year'][i] = params['sigma_year']

            # Adapt proposals during warmup
            if i < n_warmup and (i + 1) % 100 == 0:
                for key in ['alpha', 'beta', 'sigma', 'sigma_team', 'sigma_year']:
                    rate = accept_counts[key] / 100
                    if rate < 0.3:
                        proposal_sd[key] *= 0.8
                    elif rate > 0.5:
                        proposal_sd[key] *= 1.2
                    accept_counts[key] = 0

            if (i + 1) % 1000 == 0:
                phase = "warmup" if i < n_warmup else "sampling"
                print(f"    Iteration {i+1}/{n_total} ({phase})")

        # Discard warmup
        posterior = {k: v[n_warmup:] for k, v in samples.items()}

        return posterior

    def fit(self, df: pd.DataFrame, n_samples: int = 5000, n_warmup: int = 1000) -> 'TeamEfficiencyModel':
        """
        Fit the hierarchical model.

        Parameters
        ----------
        df : DataFrame with columns: salary, war, team, season
        n_samples : Number of posterior samples
        n_warmup : Number of warmup samples
        """
        print(f"Fitting hierarchical model on {len(df):,} player-seasons...")
        print(f"  Teams: {df['team'].nunique()}")
        print(f"  Years: {df['season'].min()}-{df['season'].max()}")

        # Prepare data
        y, X_war, team_idx, year_idx = self._prepare_data(df)

        print(f"  Log salary range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  WAR range: [{X_war.min():.2f}, {X_war.max():.2f}]")

        # Run MCMC
        self.posterior_samples = self._gibbs_sampler(
            y, X_war, team_idx, year_idx,
            n_samples=n_samples, n_warmup=n_warmup
        )

        # Print summary
        print("\nPosterior Summary:")
        print(f"  alpha (intercept): {self.posterior_samples['alpha'].mean():.3f} +/- {self.posterior_samples['alpha'].std():.3f}")
        print(f"  beta ($/WAR in log): {self.posterior_samples['beta'].mean():.3f} +/- {self.posterior_samples['beta'].std():.3f}")
        print(f"  sigma (residual): {self.posterior_samples['sigma'].mean():.3f}")
        print(f"  sigma_team: {self.posterior_samples['sigma_team'].mean():.3f}")
        print(f"  sigma_year: {self.posterior_samples['sigma_year'].mean():.3f}")

        return self

    def get_team_effects(self) -> pd.DataFrame:
        """Get posterior summaries for team random effects."""
        gamma_samples = self.posterior_samples['gamma_team']

        results = []
        for i, team in enumerate(self.teams):
            samples = gamma_samples[:, i]
            # Efficiency = exp(-gamma): negative gamma = pay less = more efficient
            efficiency_samples = np.exp(-samples)

            results.append({
                'team': team,
                'gamma_mean': samples.mean(),
                'gamma_sd': samples.std(),
                'gamma_ci_lower': np.percentile(samples, 2.5),
                'gamma_ci_upper': np.percentile(samples, 97.5),
                'efficiency_mean': efficiency_samples.mean(),
                'efficiency_ci_lower': np.percentile(efficiency_samples, 2.5),
                'efficiency_ci_upper': np.percentile(efficiency_samples, 97.5),
            })

        return pd.DataFrame(results).sort_values('efficiency_mean', ascending=False)

    def get_year_effects(self) -> pd.DataFrame:
        """Get posterior summaries for year random effects."""
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
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> 'TeamEfficiencyModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def main():
    """Fit hierarchical model and analyze team efficiency."""
    data_dir = Path('salary_efficiency/data/processed')
    model_dir = Path('salary_efficiency/models')
    output_dir = Path('salary_efficiency/outputs/tables')

    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_dir / 'player_seasons.parquet')
    print(f"Loaded {len(df):,} player-seasons")

    # Filter to free agents for market efficiency analysis
    # (pre-arb and arb salaries are constrained by rules)
    fa_df = df[df['service_time_class'] == 'free_agent'].copy()
    print(f"Free agents: {len(fa_df):,}")

    # Focus on recent years with more data
    recent_df = fa_df[fa_df['season'] >= 2010].copy()
    print(f"Recent (2010+): {len(recent_df):,}")

    # Fit model
    model = TeamEfficiencyModel(
        prior_beta_mean=0.5,  # log scale: ~1.6x per WAR
        prior_beta_sd=0.5,
        prior_sigma_team=0.3,
        prior_sigma_year=0.3
    )
    model.fit(recent_df, n_samples=5000, n_warmup=1000)

    # Save model
    model.save(model_dir / 'team_efficiency_model.pkl')

    # Get team effects
    team_effects = model.get_team_effects()
    team_effects.to_csv(output_dir / 'team_efficiency_rankings.csv', index=False)

    print("\n" + "="*60)
    print("TEAM EFFICIENCY RANKINGS")
    print("="*60)
    print("(Efficiency > 1 means team pays less than average for same WAR)")
    print()

    print("Top 10 Most Efficient Teams:")
    print(team_effects.head(10)[['team', 'gamma_mean', 'efficiency_mean', 'efficiency_ci_lower', 'efficiency_ci_upper']].to_string(index=False))

    print("\nBottom 10 (Least Efficient):")
    print(team_effects.tail(10)[['team', 'gamma_mean', 'efficiency_mean', 'efficiency_ci_lower', 'efficiency_ci_upper']].to_string(index=False))

    # Get year effects
    year_effects = model.get_year_effects()
    year_effects.to_csv(output_dir / 'year_effects.csv', index=False)

    print("\n" + "="*60)
    print("YEAR EFFECTS (market inflation)")
    print("="*60)
    print(year_effects.to_string(index=False))

    return model, team_effects, year_effects


if __name__ == "__main__":
    model, team_effects, year_effects = main()
