"""
Bayesian Success Probability Model for Stolen Bases

Models P(success | attempt, game state) using Bayesian logistic regression
with proper MCMC posterior sampling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import pickle
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class StealSuccessModel:
    """
    Bayesian logistic regression for stolen base success probability.

    P(success | x) = σ(x'β)

    Prior: β ~ N(0, σ²_prior * I)

    Posterior sampled via Metropolis-Hastings MCMC.

    Features:
    - outs (0, 1)
    - inning
    - score differential
    - steal base (2B vs 3B)
    - pitcher handedness (LHP harder to steal against)
    """

    def __init__(self, prior_mean: float = 0.0, prior_sd: float = 2.5):
        """
        Initialize model with prior specification.

        Parameters
        ----------
        prior_mean : float
            Prior mean for coefficients (default 0)
        prior_sd : float
            Prior standard deviation for coefficients (default 2.5, weakly informative)
        """
        self.prior_mean = prior_mean
        self.prior_sd = prior_sd
        self.feature_names = None
        self.posterior_samples = None
        self.beta_mean = None
        self.beta_sd = None

    def _create_design_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create design matrix for regression."""
        df = df.copy()
        features = pd.DataFrame(index=df.index)

        # Intercept
        features['intercept'] = 1.0

        # Game state features - convert to float explicitly
        features['outs'] = df['outs_when_up'].astype(float).fillna(0)
        features['inning'] = (df['inning'].astype(float).fillna(5) - 5) / 4  # Standardize around inning 5
        features['score_diff'] = df['score_diff'].astype(float).fillna(0) / 5  # Standardize

        # Steal base (3B is harder)
        features['stealing_3rd'] = (df['steal_base'] == '3B').astype(float)

        # Pitcher handedness (LHP harder to steal against)
        if 'p_throws' in df.columns:
            features['lhp'] = (df['p_throws'] == 'L').astype(float)
        else:
            features['lhp'] = 0.0

        self.feature_names = list(features.columns)
        X = features.values.astype(np.float64)

        # Outcome
        y = df['steal_success'].astype(np.float64).values

        return X, y

    def _log_likelihood(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log likelihood for logistic regression."""
        eta = X @ beta
        # Use numerically stable formula
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        return ll

    def _log_prior(self, beta: np.ndarray) -> float:
        """Compute log prior: β ~ N(prior_mean, prior_sd²)."""
        return np.sum(norm.logpdf(beta, loc=self.prior_mean, scale=self.prior_sd))

    def _log_posterior(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log posterior (unnormalized)."""
        return self._log_likelihood(beta, X, y) + self._log_prior(beta)

    def _metropolis_hastings(self, X: np.ndarray, y: np.ndarray,
                              n_samples: int = 10000,
                              n_warmup: int = 2000,
                              proposal_sd: Optional[np.ndarray] = None,
                              seed: int = 42) -> np.ndarray:
        """
        Sample from posterior using Metropolis-Hastings algorithm.

        Parameters
        ----------
        X : design matrix
        y : outcomes
        n_samples : number of posterior samples to retain
        n_warmup : number of warmup samples to discard
        proposal_sd : proposal standard deviations (tuned during warmup if None)
        seed : random seed

        Returns
        -------
        samples : (n_samples, n_params) array of posterior samples
        """
        np.random.seed(seed)
        n_params = X.shape[1]

        # Initialize at prior mean
        beta_current = np.zeros(n_params)
        # Initialize intercept at observed log-odds
        success_rate = y.mean()
        beta_current[0] = np.log(success_rate / (1 - success_rate))

        log_post_current = self._log_posterior(beta_current, X, y)

        # Initial proposal SDs (will be tuned)
        if proposal_sd is None:
            proposal_sd = np.ones(n_params) * 0.1

        # Storage
        samples = np.zeros((n_samples + n_warmup, n_params))
        accept_count = np.zeros(n_params)

        # Adaptive tuning during warmup
        adapt_interval = 100
        target_accept = 0.44  # Optimal for 1D proposals

        print(f"  Running MCMC: {n_warmup} warmup + {n_samples} samples...")

        for i in range(n_samples + n_warmup):
            # Update each parameter with Gibbs-like single-site updates
            for j in range(n_params):
                # Propose new value for parameter j
                beta_proposal = beta_current.copy()
                beta_proposal[j] += np.random.normal(0, proposal_sd[j])

                # Compute acceptance probability
                log_post_proposal = self._log_posterior(beta_proposal, X, y)
                log_accept_ratio = log_post_proposal - log_post_current

                # Accept or reject
                if np.log(np.random.random()) < log_accept_ratio:
                    beta_current = beta_proposal
                    log_post_current = log_post_proposal
                    if i < n_warmup:
                        accept_count[j] += 1

            samples[i] = beta_current

            # Adapt proposal SDs during warmup
            if i < n_warmup and (i + 1) % adapt_interval == 0:
                accept_rates = accept_count / adapt_interval
                for j in range(n_params):
                    if accept_rates[j] < target_accept - 0.1:
                        proposal_sd[j] *= 0.8
                    elif accept_rates[j] > target_accept + 0.1:
                        proposal_sd[j] *= 1.2
                accept_count = np.zeros(n_params)

            # Progress update
            if (i + 1) % 2000 == 0:
                phase = "warmup" if i < n_warmup else "sampling"
                print(f"    Iteration {i+1}/{n_samples + n_warmup} ({phase})")

        # Discard warmup
        posterior_samples = samples[n_warmup:]

        # Compute acceptance rate for post-warmup
        print(f"  MCMC complete. Final proposal SDs: {proposal_sd}")

        return posterior_samples

    def fit(self, df: pd.DataFrame, n_samples: int = 5000, n_warmup: int = 1000) -> 'StealSuccessModel':
        """
        Fit the model on steal attempts using MCMC.

        Parameters
        ----------
        df : DataFrame with steal attempts (must have steal_success column)
        n_samples : Number of posterior samples to retain
        n_warmup : Number of warmup samples to discard
        """
        # Only fit on steal attempts
        attempts = df[df['steal_attempt'] == True].copy()

        if len(attempts) < 50:
            raise ValueError(f"Need at least 50 steal attempts for estimation, got {len(attempts)}")

        print(f"Fitting Bayesian model on {len(attempts):,} steal attempts...")
        success_rate = attempts['steal_success'].mean()
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Prior: beta ~ N({self.prior_mean}, {self.prior_sd}^2)")

        # Create design matrix
        X, y = self._create_design_matrix(attempts)

        print(f"  Features: {self.feature_names}")

        # Run MCMC
        self.posterior_samples = self._metropolis_hastings(
            X, y, n_samples=n_samples, n_warmup=n_warmup
        )

        # Compute posterior summaries
        self.beta_mean = self.posterior_samples.mean(axis=0)
        self.beta_sd = self.posterior_samples.std(axis=0)

        # Compute 95% credible intervals
        ci_lower = np.percentile(self.posterior_samples, 2.5, axis=0)
        ci_upper = np.percentile(self.posterior_samples, 97.5, axis=0)

        print(f"\nPosterior estimates (mean ± SD [95% CI]):")
        for i, name in enumerate(self.feature_names):
            sig = '*' if (ci_lower[i] > 0 or ci_upper[i] < 0) else ''
            print(f"    {name:15s}: {self.beta_mean[i]:+.3f} ± {self.beta_sd[i]:.3f} "
                  f"[{ci_lower[i]:+.3f}, {ci_upper[i]:+.3f}] {sig}")

        # Effective sample size (simple approximation)
        self._compute_diagnostics()

        return self

    def _compute_diagnostics(self):
        """Compute MCMC diagnostics."""
        n_samples, n_params = self.posterior_samples.shape

        print(f"\nMCMC Diagnostics:")

        for i, name in enumerate(self.feature_names):
            chain = self.posterior_samples[:, i]

            # Effective sample size (using autocorrelation)
            acf = np.correlate(chain - chain.mean(), chain - chain.mean(), mode='full')
            acf = acf[len(acf)//2:] / acf[len(acf)//2]

            # Find first negative autocorrelation
            first_neg = np.where(acf < 0)[0]
            if len(first_neg) > 0:
                tau = 1 + 2 * np.sum(acf[1:first_neg[0]])
            else:
                tau = 1 + 2 * np.sum(acf[1:min(100, len(acf))])

            n_eff = n_samples / max(tau, 1)
            print(f"    {name:15s}: n_eff = {n_eff:.0f}")

    def predict_proba(self, df: pd.DataFrame, use_mean: bool = True) -> np.ndarray:
        """
        Predict success probability for each row.

        Parameters
        ----------
        df : DataFrame with features
        use_mean : If True, use posterior mean. If False, return samples.
        """
        X, _ = self._create_design_matrix(df)

        if use_mean:
            eta = X @ self.beta_mean
            return 1 / (1 + np.exp(-eta))
        else:
            # Return predictions for each posterior sample
            n_samples = self.posterior_samples.shape[0]
            probs = np.zeros((len(df), n_samples))
            for s in range(n_samples):
                eta = X @ self.posterior_samples[s]
                probs[:, s] = 1 / (1 + np.exp(-eta))
            return probs

    def predict_proba_with_uncertainty(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict success probability with uncertainty quantification.

        Returns
        -------
        mean : posterior mean of probability
        lower : 2.5th percentile
        upper : 97.5th percentile
        """
        probs = self.predict_proba(df, use_mean=False)
        mean = probs.mean(axis=1)
        lower = np.percentile(probs, 2.5, axis=1)
        upper = np.percentile(probs, 97.5, axis=1)
        return mean, lower, upper

    def get_baseline_success_prob(self) -> float:
        """Get baseline success probability (intercept only)."""
        return 1 / (1 + np.exp(-self.beta_mean[0]))

    def save(self, filepath: Path):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> 'StealSuccessModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def main():
    """Fit the Bayesian success probability model."""
    data_dir = Path('stolen_base_analysis/data/processed')
    model_dir = Path('stolen_base_analysis/models')
    output_dir = Path('stolen_base_analysis/outputs/tables')

    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading steal opportunities...")
    df = pd.read_parquet(data_dir / 'steal_opportunities.parquet')
    print(f"Loaded {len(df):,} steal opportunities")
    print(f"Steal attempts: {df['steal_attempt'].sum():,}")

    # Fit Bayesian model
    model = StealSuccessModel(prior_mean=0.0, prior_sd=2.5)
    model.fit(df, n_samples=5000, n_warmup=1000)

    # Save model
    model.save(model_dir / 'success_model.pkl')

    # Save posterior summaries
    ci_lower = np.percentile(model.posterior_samples, 2.5, axis=0)
    ci_upper = np.percentile(model.posterior_samples, 97.5, axis=0)

    estimates = pd.DataFrame({
        'feature': model.feature_names,
        'posterior_mean': model.beta_mean,
        'posterior_sd': model.beta_sd,
        'ci_lower_2.5': ci_lower,
        'ci_upper_97.5': ci_upper
    })
    estimates['significant'] = (estimates['ci_lower_2.5'] > 0) | (estimates['ci_upper_97.5'] < 0)
    estimates.to_csv(output_dir / 'success_model_estimates.csv', index=False)

    print("\nPosterior estimates saved to success_model_estimates.csv")

    # Predicted probabilities by situation
    print("\n" + "="*60)
    print("PREDICTED SUCCESS PROBABILITIES BY SITUATION")
    print("="*60)

    baseline = model.get_baseline_success_prob()
    print(f"\nBaseline (0 outs, avg inning/score, stealing 2B, RHP): {baseline:.1%}")

    # Create scenarios
    scenarios = pd.DataFrame({
        'outs_when_up': [0, 1, 0, 0, 0, 0],
        'inning': [5, 5, 9, 5, 5, 5],
        'score_diff': [0, 0, 0, -3, 0, 0],
        'steal_base': ['2B', '2B', '2B', '2B', '3B', '2B'],
        'p_throws': ['R', 'R', 'R', 'R', 'R', 'L'],
        'steal_attempt': [True]*6,
        'steal_success': [True]*6
    })

    means, lowers, uppers = model.predict_proba_with_uncertainty(scenarios)

    scenario_names = [
        'Baseline (0 out, 5th, tie, 2B, RHP)',
        '1 out',
        '9th inning',
        'Down by 3',
        'Stealing 3rd',
        'vs LHP'
    ]

    for name, mean, lower, upper in zip(scenario_names, means, lowers, uppers):
        print(f"  {name:40s}: {mean:.1%} [{lower:.1%}, {upper:.1%}]")

    return model


if __name__ == "__main__":
    model = main()
