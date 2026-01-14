# Go For It: Bayesian Decision Theory for NFL Fourth Down and Two-Point Decisions

A fully Bayesian decision-theoretic framework for analyzing fourth down and two-point conversion decisions in the NFL.

**Author:** Andrew Bai (University of Chicago)

---

## Overview

This project develops a rigorous framework for evaluating NFL coaching decisions. Unlike existing approaches that rely on point estimates, this framework:

1. Optimizes **win probability** directly (not expected points)
2. Propagates **parameter uncertainty** through to decision uncertainty via fully Bayesian inference
3. Uses **hierarchical Bayes** to capture team-specific and kicker-specific effects with empirical Bayes shrinkage
4. Incorporates **in-game context features** (goal-to-go, EPA, drive momentum) that explain apparent "coach intuition"
5. Tests **real-time knowability** via expanding window estimation

---

## Key Findings

### Fourth Down Decisions (2006-2024)

- **82.2% optimal** overall
- **82% of mistakes are close calls** (decision margin < 2 percentage points)
- Only **0.3% are clear mistakes** where the optimal action was obvious
- Go-for-it rates increased from 12% to 20% (2006-2024)
- Decision **quality has not improved**—the analytics revolution changed *behavior* but not *accuracy*

### Two-Point Conversions

- **The Down 8 vs Down 9 Paradox**: When down 8, coaches go for 2 at 79% (model says 85%). When down 9, coaches go for 2 at 1% (model says 91%). Down 9 has a *higher* optimal rate, yet coaches almost never do it.
- **Behavioral explanation**: Present bias—going for 2 when down 8 ties the game *now*; going for 2 when down 9 sets up a future tying field goal

### The Coach Override Test

When coaches override the model to go for it, do they convert at higher rates? **No.**
- Short yardage: Raw +6.5pp advantage disappears after controlling for in-game context
- Long yardage: Coaches who override are *overconfident*—converting 31% vs 42% when model agrees
- **Conclusion**: No evidence of private information; in-game context model captures what coaches "know"

---

## Mathematical Framework

### State Space

The game state is represented as:

$$s = (\Delta, \tau, x, d, h, k_1, k_2)$$

where $\Delta$ is score differential, $\tau$ is time remaining, $x$ is field position (yards from opponent's end zone), $d$ is yards to go, $h$ is the half, and $k_1, k_2$ are timeouts remaining.

### Action Space

On fourth down, the coach chooses from:

$$\mathcal{A} = \lbrace\texttt{go}, \texttt{punt}, \texttt{fg}\rbrace$$

### Transition Dynamics

Each action induces transitions to successor states:

**Going for it:**
$$P(s' \mid s, \texttt{go}; \theta) = \pi(d; \theta) \cdot P(s_{\text{convert}}) + (1 - \pi(d; \theta)) \cdot P(s_{\text{fail}})$$

**Field goal:**
$$P(s' \mid s, \texttt{fg}; \theta) = \phi(x; \theta) \cdot P(s_{\text{make}}) + (1 - \phi(x; \theta)) \cdot P(s_{\text{miss}})$$

### Bayesian Expected Win Probability

For action $a$ in state $s$, expected win probability integrating over parameter uncertainty:

$$\mathbb{E}[W \mid a, s] = \int W(s' \mid a, s, \theta) \cdot p(\theta \mid \mathcal{D}) \, d\theta$$

The Bayes-optimal decision is:

$$a^* = \arg\max_{a \in \mathcal{A}} \mathbb{E}[W \mid a, s]$$

---

## Component Models

### Hierarchical Conversion Model with In-Game Context

Conversion probability is modeled as logistic with **in-game context features** and team random effects:

$$\mathbb{P}(\text{convert} \mid d, g, e, p, \text{off} = j, \text{def} = k) = \sigma(\alpha + \beta_d d + \beta_g g + \beta_e e + \beta_p p + \gamma_j^{\text{off}} + \delta_k^{\text{def}})$$

where:
- $d$ = yards to go
- $g$ = goal-to-go indicator
- $e$ = standardized in-game EPA (team's cumulative rush + pass EPA in that game)
- $p$ = standardized drive play count
- $\gamma_j^{\text{off}}, \delta_k^{\text{def}}$ = team random effects (shrunk via empirical Bayes)

**Key finding**: Goal-to-go situations convert at *lower* rates ($\hat{\beta}_g = -1.129$). At 4th & 1, conversion probability drops from 64.3% (non-goal-to-go) to 36.8% (goal-to-go)—a 27.5pp penalty.

| Yards to Go | Conversion % | 95% CI |
|-------------|--------------|--------|
| 1 | 64.3% | [63.0%, 65.6%] |
| 2 | 61.2% | [60.0%, 62.4%] |
| 3 | 58.0% | [56.9%, 59.1%] |
| 5 | 51.4% | [50.3%, 52.4%] |
| 10 | 35.2% | [33.4%, 36.8%] |

### Hierarchical Field Goal Model

Make probability is logistic in kick distance (centered at 35 yards) with kicker-specific effects:

$$\mathbb{P}(\text{make} \mid d, \text{kicker} = j) = \sigma(\alpha + \beta (d - 35) + \gamma_j)$$

**Population-level estimates:**
- $\hat{\alpha} = 2.383$ (SE: 0.056)
- $\hat{\beta} = -0.105$ (SE: 0.004)
- Between-kicker variance: $\hat{\tau}^2 = 0.031$

### Win Probability Model

Win probability is estimated using a neural network (3-layer MLP with 128-64-32 hidden units) trained on 710,664 plays. Features include score differential, time remaining, field position, down, yards to go, timeout differential, and interaction terms (score×time).

---

## Real-Time Knowability

A natural objection: perhaps coaches couldn't have known the optimal decision at the time.

We implement an **expanding window analysis** with a 7-year minimum training window. For each test year $Y$:
1. Train models on data from 1999 through $Y-1$ only (the "ex ante" model)
2. Compute optimal decisions under the ex ante model
3. Compare to ex post (full sample) recommendations

**Result:** 96.1% agreement between ex ante and ex post recommendations across 70,006 plays.

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

- **70,006 fourth-down situations** (2006-2024 evaluation sample)
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
