# Go For It: Bayesian Decision Theory for NFL Fourth Down and Two-Point Decisions

A fully Bayesian decision-theoretic framework for analyzing fourth down and two-point conversion decisions in the NFL.

**Author:** Andrew Bai (University of Chicago)

---

## Overview

---

## Mathematical Framework

### 1. The State Space

The state of the game at any moment is represented by the tuple:

$$s = (\Delta, \tau, x, d)$$

where:
- $\Delta \in \mathbb{Z}$ is the **score differential** (positive if the possession team is winning)
- $\tau \in [0, T]$ is the **time remaining** in seconds
- $x \in \lbrace 1, \ldots, 99 \rbrace$ is the **field position** measured in yards from the opponent's end zone
- $d \in \lbrace 1, \ldots, 99 \rbrace$ is the **yards to go** for a first down

The full state space also includes timeouts and half indicators, but the reduced state captures the most decision-relevant variation.

### 2. The Action Space

On fourth down, the coach chooses an action $a$ from:

$$\mathcal{A} = \lbrace\texttt{go}, \texttt{punt}, \texttt{fg}\rbrace$$

where:
- $\texttt{go}$ denotes attempting to convert the fourth down
- $\texttt{punt}$ denotes punting the ball
- $\texttt{fg}$ denotes attempting a field goal

The field goal action is infeasible when $x > 60$ (requiring a kick longer than 77 yards).

### 3. Transition Dynamics

Each action induces a probability distribution over successor states. Let $P(s' \mid s, a; \theta)$ denote the transition probability parameterized by $\theta$.

**Going for it.** Let $\pi(d; \theta)$ denote the probability of converting with $d$ yards to go. The transition is:

$$P(s' \mid s, \texttt{go}; \theta) = \pi(d; \theta) \cdot \mathbb{1}\lbrace s' = s_{\text{convert}}\rbrace + (1 - \pi(d; \theta)) \cdot \mathbb{1}\lbrace s' = s_{\text{fail}}\rbrace$$

where $s_{\text{convert}}$ retains possession with updated field position, and $s_{\text{fail}}$ gives the opponent the ball at the current spot.

**Punting.** Let $Y(x; \theta)$ denote net punt yards from field position $x$. The opponent receives the ball at:

$$x' = \min(\max(100 - (x - Y), 1), 80)$$

where the bounds reflect touchbacks and downing inside the 20.

**Field goal.** Let $\phi(x; \theta)$ denote the probability of making a field goal from $x$ yards (kick distance = $x + 17$):

$$P(s' \mid s, \texttt{fg}; \theta) = \phi(x; \theta) \cdot \mathbb{1}\lbrace +3, \text{kickoff}\rbrace + (1 - \phi(x; \theta)) \cdot \mathbb{1}\lbrace \text{opp. at } \max(x, 20)\rbrace$$

### 4. The Objective: Maximizing Win Probability

The coach's objective is to maximize the probability of winning the game. The expected win probability for action $a$ in state $s$ is:

$$\mathbb{E}[W \mid a, s] = \sum_{s'} W(s') \cdot P(s' \mid s, a)$$

where $W(s')$ is the probability of winning from successor state $s'$.

The optimal action is:

$$a^* = \arg\max_{a \in \mathcal{A}} \mathbb{E}[W \mid a, s]$$

### 5. Fully Bayesian Framework

The key innovation is treating the transition parameters $\theta$ as **uncertain** rather than known. We place a prior $p(\theta)$ on the parameters and update to the posterior $p(\theta \mid \mathcal{D})$ given observed data $\mathcal{D}$.

The Bayesian expected win probability integrates over parameter uncertainty:

$$\mathbb{E}[W \mid a, s] = \int W(s' \mid a, s, \theta) \cdot p(\theta \mid \mathcal{D}) \, d\theta$$

This integral is computed via Monte Carlo. For $M$ posterior draws $\theta^{(1)}, \ldots, \theta^{(M)} \sim p(\theta \mid \mathcal{D})$:

$$\mathbb{E}[W \mid a, s] \approx \frac{1}{M} \sum_{m=1}^{M} W(s' \mid a, s, \theta^{(m)})$$

For the action $\texttt{go}$, this expands to:

$$\mathbb{E}[W \mid \texttt{go}, s] = \int \left[\pi(d; \theta) \cdot W(s_{\text{convert}}; \theta) + (1 - \pi(d; \theta)) \cdot W(s_{\text{fail}}; \theta)\right] \cdot p(\theta \mid \mathcal{D}) \, d\theta$$

### 6. The Bayes-Optimal Decision

**Definition (Bayes-Optimal Decision).** The Bayes-optimal action is:

$$a^* = \arg\max_{a \in \mathcal{A}} \int W(s' \mid a, s, \theta) \cdot p(\theta \mid \mathcal{D}) \, d\theta$$

This decision criterion accounts for both:
1. **Transition uncertainty**: Given parameters $\theta$, outcomes are stochastic
2. **Parameter uncertainty**: The parameters $\theta$ themselves are uncertain

### 7. Decision Confidence (Posterior Probability of Optimality)

A key advantage of the Bayesian framework is quantifying **uncertainty about which action is optimal**.

**Definition (Decision Confidence).** The posterior probability that action $a$ is optimal is:

$$\mathbb{P}(a \text{ is optimal} \mid s, \mathcal{D}) = \mathbb{P}_{\theta \mid \mathcal{D}}\left(W_a(s; \theta) > \max_{a' \neq a} W_{a'}(s; \theta)\right)$$

This is estimated by Monte Carlo:
1. Draw $\theta^{(m)} \sim p(\theta \mid \mathcal{D})$ for $m = 1, \ldots, M$
2. Compute $W_a(s; \theta^{(m)})$ for each action $a \in \mathcal{A}$
3. Calculate the fraction of draws for which action $a$ has the highest WP

Interpretation:
- $\mathbb{P}(\texttt{go} \text{ is optimal}) \approx 1$ → **obvious** go-for-it decision
- $\mathbb{P}(\texttt{go} \text{ is optimal}) \approx 0.5$ → **close call** where data does not clearly favor one action

This allows us to distinguish between:
- **Clear mistakes**: Coach chose suboptimally when the data strongly favored another action
- **Close calls**: Coach's choice was reasonable given decision uncertainty

---

## Component Models

### Hierarchical Conversion Model

Conversion probability is modeled as logistic in yards to go with **both** offensive and defensive team random effects:

$$\mathbb{P}(\text{convert} \mid d, \text{off} = j, \text{def} = k) = \sigma(\alpha + \beta d + \gamma_j^{\text{off}} + \delta_k^{\text{def}})$$

where $\sigma(\cdot)$ is the logistic function, and:
- $\gamma_j^{\text{off}} \sim \mathcal{N}(0, \tau_{\text{off}}^2)$ captures offensive team conversion ability
- $\delta_k^{\text{def}} \sim \mathcal{N}(0, \tau_{\text{def}}^2)$ captures defensive team stopping ability

Both effects are shrunk toward zero via **empirical Bayes**, with shrinkage factor:

$$B_k = \frac{\text{SE}_k^2}{\text{SE}_k^2 + \tau^2}$$

This ensures stable estimates even for teams with few observations.

**Population-level estimates** (1999-2024, N = 13,884 attempts):
- $\hat{\alpha} = 0.660$ (SE: 0.026)
- $\hat{\beta} = -0.160$ (SE: 0.005)

| Yards to Go | Conversion % | 95% CI |
|-------------|--------------|--------|
| 1 | 64.8% | [62.9%, 66.4%] |
| 2 | 60.7% | [58.9%, 62.2%] |
| 3 | 56.4% | [54.8%, 57.9%] |
| 5 | 47.6% | [46.0%, 49.3%] |
| 10 | 27.4% | [24.9%, 30.4%] |

### Hierarchical Field Goal Model

Make probability is logistic in kick distance (centered at 35 yards) with kicker-specific effects:

$$\mathbb{P}(\text{make} \mid d, \text{kicker} = j) = \sigma(\alpha + \beta (d - 35) + \gamma_j)$$

where $\gamma_j \sim \mathcal{N}(0, \tau^2)$ captures kicker ability relative to league average.

**Population-level estimates:**
- $\hat{\alpha} = 2.383$ (SE: 0.056)
- $\hat{\beta} = -0.105$ (SE: 0.004)
- Between-kicker variance: $\hat{\tau}^2 = 0.031$

The best-to-worst kicker spread at 50 yards is approximately 9 percentage points (73.1% vs 63.9%).

### Punt Model

Net punt yards are modeled as linear in field position with Gaussian errors:

$$Y \mid x \sim \mathcal{N}(\alpha + \beta x, \sigma^2)$$

**Estimates:**
- $\hat{\alpha} = 32.8$ (SE: 0.41)
- $\hat{\beta} = 0.154$ (SE: 0.006)
- $\hat{\sigma} = 9.3$ yards

Punts from deeper in own territory travel further (positive $\beta$), reflecting punter adjustment.

### Win Probability Model

Win probability is modeled using a **neural network** trained on 710,664 plays from 2006-2024. The architecture is a 3-layer multilayer perceptron (128-64-32 hidden units) with ReLU activations and 20% dropout, trained with early stopping to minimize binary cross-entropy loss.

**Input features:**
- Score differential ($\Delta$)
- Time remaining ($\tau$)
- Field position ($x$ yards from opponent's end zone)
- Down and yards to go
- Timeout differential ($k$)
- Half indicator
- Goal-to-go indicator
- Interaction terms: score×time, field position×time
- Binary indicators for late-game ($\tau < 300$s), red zone ($x \leq 20$), goal-line ($x \leq 5$)

**Model validation:**
- 5-fold cross-validated **Brier score: 0.164** (±0.0004)
- Expected calibration error (ECE): **0.0049** (nearly perfect calibration)
- Across deciles of predicted WP, actual win rates match predictions closely

**Why neural network over logistic regression?**

The neural network substantially outperforms logistic regression for edge cases such as goal-line situations with minimal time remaining. For example, a team trailing by 3 with 16 seconds remaining at the opponent's 2-yard line (1st & goal) has a win probability of approximately **71%** under the neural model—reflecting the ~67% historical touchdown rate from this position—compared to only ~14% under a logistic specification that cannot capture the nonlinear interaction between field position and time in such extreme scenarios.

---

## End-of-Game Filtering

Following standard practice in sports analytics (Baldwin 2021, nfl4th), we exclude plays with fewer than 60 seconds remaining from our analysis. In these situations, win probability models become unreliable because outcomes depend heavily on factors not captured in our model:

- **Timeout availability** and clock management strategy
- **Kneel-out scenarios** where the opponent can simply run out the clock
- **Hail Mary situations** requiring different probability calculations

This filtering removes approximately 1.4% of plays. The excluded plays are disproportionately high-leverage situations where our model's assumptions are least valid.

---

## Key Findings

### Fourth Down Decisions (2006-2024)

- **72.3% match rate** with model recommendations overall
- **79% of deviations are close calls** (decision margin < 2 percentage points)
- Only **1.3% are clear mistakes** (margin ≥ 5pp) where the optimal action was obvious
- Go-for-it rates increased from 12.6% (2006-2014) → 19.2% (2019-2024)
- Model recommends go-for-it **30%** of the time vs coaches' actual **14%**

### Two-Point Conversions

- **58.6% optimal** overall
- Virtually all deviations are close calls
- Optimality improving at **+1.0 pp/year** ($p = 0.0003$)

### Takeaway

Coaches are learning at the simple decision (two-point conversions) but the complex decision (fourth downs) shows no improvement over time. The analytics revolution changed behavior (more aggression) but not accuracy.

---

## Real-Time Knowability

A natural objection: perhaps coaches couldn't have known the optimal decision at the time.

We implement an **expanding window analysis** with a 7-year minimum training window. For each test year $Y$:
1. Train models on data from 1999 through $Y-1$ only (the "ex ante" model)
2. Compute optimal decisions under the ex ante model
3. Compare to ex post (full sample) recommendations

**Result:** 96.5% of optimal decisions were knowable in real-time.

---

## Project Structure

```
├── analysis/
│   ├── decision_framework.py      # Core Bayesian decision analysis
│   ├── decision_categorization.py # Mistake classification
│   ├── two_point_analysis.py      # Two-point conversion analysis
│   ├── expanding_window_analysis.py # Real-time knowability tests
│   └── era_comparison.py          # Temporal trends analysis
├── models/
│   ├── bayesian_models.py         # Conversion, punt, FG, WP models
│   └── hierarchical_off_def_model.py # Offense/defense effects
├── data/
│   └── acquire_data.py            # Data acquisition via nflfastR
├── slides/
│   └── slides.tex                 # Beamer presentation
└── outputs/
    ├── figures/                   # Generated visualizations
    └── tables/                    # Generated tables
```

## Data

Play-by-play data from 1999-2024 NFL seasons via [nflfastR](https://www.nflfastr.com/).

- **71,786 fourth-down situations** (2006-2024 evaluation sample)
- 7-year minimum training window for expanding window analysis

## Requirements

```
python >= 3.8
numpy
pandas
scipy
scikit-learn
pyarrow
tqdm
```

## Usage

```python
from analysis.decision_framework import BayesianDecisionAnalyzer, GameState
from models.bayesian_models import load_all_models

# Load trained models
models = load_all_models('models/')
analyzer = BayesianDecisionAnalyzer(models)

# Analyze a 4th down situation
state = GameState(
    field_pos=40,       # 40 yards from opponent's end zone
    yards_to_go=3,      # 4th and 3
    score_diff=-3,      # Down by 3
    time_remaining=300, # 5 minutes left
    off_team='PHI',     # Eagles offense
    def_team='DAL'      # Cowboys defense
)

result = analyzer.analyze(state)
print(f"Optimal: {result.optimal_action}")
print(f"WP(go): {result.wp_go:.1%}")
print(f"WP(punt): {result.wp_punt:.1%}")
print(f"WP(fg): {result.wp_fg:.1%}")
print(f"P(go is best): {result.prob_go_best:.0%}")
```

## Citation

```bibtex
@misc{bai2026goforit,
  author = {Bai, Andrew},
  title = {Is Management Learning? Evidence from NFL Fourth Down and Two-Point Decisions},
  year = {2026},
  institution = {University of Chicago}
}
```

## References

- Romer, D. (2006). Do Firms Maximize? Evidence from Professional Football. *Journal of Political Economy*, 114(2), 340-365.
- Baldwin, B. & Eager, E. (2021). nflfastR: Functions to Efficiently Access NFL Play by Play Data. R package.

## License

MIT License - see [LICENSE](LICENSE) for details.
