# Neural Bayesian Estimation: Future Work

## Overview

This document outlines a proposed enhancement to use neural networks for Bayesian posterior estimation, replacing the current Laplace approximation approach. The goal is to leverage modern deep learning while maintaining principled Bayesian uncertainty quantification.

## Current Approach: Laplace Approximation

Our current models use Laplace approximation for Bayesian inference:

1. Find the MAP estimate (mode of posterior)
2. Approximate posterior as Gaussian centered at MAP
3. Use inverse Hessian as covariance matrix

**Advantages:**
- Computationally efficient
- Well-understood theoretical properties
- Works well for logistic regression with reasonable sample sizes

**Limitations:**
- Assumes unimodal, approximately Gaussian posterior
- May be poor approximation for complex models
- Limited expressiveness for non-linear relationships
- Cannot easily incorporate complex feature interactions

## Proposed Approach: Neural Bayesian Estimation

### Architecture Options

#### Option 1: Bayesian Neural Networks (BNNs)

Replace point estimates with distributions over weights:

```
P(y|x) = integral P(y|x,w) P(w|D) dw
```

**Implementation approaches:**
- **Variational Inference**: Approximate P(w|D) with a variational distribution q(w)
- **Monte Carlo Dropout**: Use dropout at test time as approximate Bayesian inference
- **Stochastic Weight Averaging Gaussian (SWAG)**: Fit Gaussian to SGD trajectory

**Libraries:**
- PyTorch + Pyro
- TensorFlow Probability
- Blitz (PyTorch BNN library)

#### Option 2: Normalizing Flows for Posterior Estimation

Use normalizing flows to learn flexible posterior distributions:

```
z ~ N(0, I)
w = f_theta(z)  # invertible neural network
log P(w|D) = log P(z) - log |det(df/dz)|
```

**Advantages:**
- Can represent arbitrary posterior shapes (multimodal, skewed)
- Exact density evaluation
- Efficient sampling

**Implementation:**
- Use Neural Spline Flows or Real-NVP architectures
- Train to maximize log P(D|w) + log P(w) for sampled weights

#### Option 3: Neural Posterior Estimation (NPE)

Train a neural network to directly output posterior parameters:

```
Input: game state features x
Output: posterior distribution parameters (mu, sigma for each model parameter)
```

This is essentially "amortized inference" - once trained, posterior estimation is a single forward pass.

### Proposed Model Architecture

For fourth down decisions, we propose a hierarchical neural architecture:

```
                    ┌─────────────────┐
                    │   Game State    │
                    │  (features x)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Shared Encoder │
                    │   (MLP/Trans)   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
   ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
   │  Conversion   │ │   FG Make     │ │     Punt      │
   │  Probability  │ │  Probability  │ │   Distance    │
   │    Head       │ │    Head       │ │    Head       │
   └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
           │                 │                 │
           ▼                 ▼                 ▼
       P(convert)        P(FG make)      E[punt_yards]
       + uncertainty     + uncertainty   + uncertainty
```

### Features to Include

**Current features:**
- yards_to_go
- field_position
- score_differential
- time_remaining
- timeout_differential

**Enhanced features:**
- Temperature (continuous)
- Wind speed and direction
- Precipitation (rain/snow indicator)
- Altitude (Denver effect)
- Kicker/punter quality (learned embedding)
- Team offense/defense quality (learned embedding)
- Historical matchup data
- Pre-game spread line (team quality proxy)

**Sequence features (optional):**
- Previous plays in drive
- Game momentum (recent scoring)
- Fatigue indicators (time since last possession)

### Training Procedure

1. **Data preparation:**
   - Split by season for temporal cross-validation
   - Ensure no leakage between train/test

2. **Loss function:**
   ```
   L = -log P(y|x) + KL(q(w) || P(w))  # ELBO for variational inference
   ```

   Or for NPE:
   ```
   L = -log P(outcome | predicted_distribution_params)
   ```

3. **Uncertainty calibration:**
   - Use calibration plots to verify predicted uncertainties
   - Adjust temperature scaling if needed

4. **Hyperparameter tuning:**
   - Network depth/width
   - Dropout rate
   - Prior variance
   - Learning rate schedule

### Validation Metrics

1. **Predictive accuracy:**
   - Brier score for probability predictions
   - MAE for punt distance

2. **Calibration:**
   - Reliability diagrams
   - Expected calibration error (ECE)

3. **Uncertainty quality:**
   - Negative log-likelihood on held-out data
   - Coverage of credible intervals
   - Proper scoring rules

4. **Decision quality:**
   - Backtest against historical decisions
   - WP cost reduction vs current model

### Implementation Plan

**Phase 1: Foundation (2-4 weeks)**
- Set up PyTorch infrastructure
- Implement base neural network for P(convert|x)
- Compare to Laplace approximation baseline

**Phase 2: Bayesian Extension (2-4 weeks)**
- Add MC Dropout for uncertainty
- Implement proper scoring and calibration
- Validate on held-out seasons

**Phase 3: Full Model (4-6 weeks)**
- Extend to all model components (FG, punt, WP)
- Add team/kicker embeddings
- Incorporate weather and context features

**Phase 4: Decision Integration (2-4 weeks)**
- Update decision framework to use neural posteriors
- Validate end-to-end decision quality
- Backtest against historical decisions

### Expected Benefits

1. **Better uncertainty quantification:**
   - More accurate credible intervals
   - Better handling of rare situations

2. **Richer feature interactions:**
   - Learn non-linear relationships
   - Capture interactions (e.g., wind * distance for FG)

3. **Adaptive team/kicker effects:**
   - Learn embeddings that capture quality
   - Can incorporate recent performance

4. **Scalability:**
   - Easy to add new features
   - Can handle larger datasets

### Risks and Mitigations

1. **Overfitting:**
   - Use strong regularization
   - Temporal cross-validation
   - Keep architecture simple initially

2. **Calibration degradation:**
   - Monitor calibration during training
   - Use temperature scaling post-hoc
   - Ensemble multiple models

3. **Interpretability loss:**
   - Keep parallel simple models for comparison
   - Use attention/SHAP for feature importance
   - Validate against domain knowledge

### References

1. Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
2. Maddox et al. (2019). "A Simple Baseline for Bayesian Inference in Deep Learning" (SWAG)
3. Papamakarios et al. (2021). "Normalizing Flows for Probabilistic Modeling and Inference"
4. Greenberg et al. (2019). "Automatic Posterior Transformation for Likelihood-Free Inference"
