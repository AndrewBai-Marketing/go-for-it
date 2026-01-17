# Research Notes: Extending the Fourth Down Analysis

## Current State of Findings

Using nflfastR's vegas_wp model:
- **Optimal GO rate: 46.6%** vs actual 14.8% (32 pp gap)
- **4th & 1 optimal GO: 70.8%** vs actual ~25%
- **Match rate: 52.5%** - coaches wrong roughly half the time
- **Minimal learning: +0.22 pp/year** (p=0.0005) on fourth down
- **Stronger learning: +1.03 pp/year** (p=0.06) on two-point conversions

---

## Research Direction 1: Learning from Others' Mistakes

### Core Question
Do coaches learn from *other teams'* high-profile fourth down decisions (successes and failures)?

### Testable Hypotheses

**H1a: League-wide response to high-profile failures**
- After a nationally televised fourth down failure (e.g., team loses game on failed 4th down attempt), does the league become MORE conservative?
- Prediction: Yes - availability heuristic makes failures salient

**H1b: League-wide response to high-profile successes**
- After a high-profile fourth down success (e.g., team wins playoff game on gutsy 4th down call), does the league become MORE aggressive?
- Prediction: Weaker effect than failures (negativity bias)

**H1c: Asymmetric learning**
- Is the magnitude of conservative shift after failures > aggressive shift after successes?
- Prediction: Yes - loss aversion and blame avoidance

### Key Events to Analyze

| Date | Event | Expected Effect |
|------|-------|-----------------|
| 2015 Super Bowl | Seahawks pass on goal line (INT) | League-wide conservatism spike? |
| 2017 Week 17 | Eagles "Philly Special" (success) | Aggression increase? |
| 2018 AFC Championship | Dee Ford offsides, Chiefs conservative | ? |
| 2021 Playoffs | Kevin Stefanski aggressive calls | Browns influence? |
| Various | High-profile punts that cost games | Delayed aggression? |

### Methodology

1. **Identify "treatment" events**
   - National TV games (SNF, MNF, playoffs)
   - Fourth down in high-leverage situation (WP swing > 5%)
   - Clear outcome attribution (won/lost game on the play)

2. **Measure league-wide response**
   - Compare GO rates in similar situations before/after event
   - Use difference-in-differences with team fixed effects
   - Control for game state, score, time remaining

3. **Event study design**
   - Window: 4 weeks pre, 8 weeks post
   - Outcome: GO rate in comparable situations
   - Heterogeneity: By team's geographic/divisional proximity to event team

### Preliminary Finding: The Tied/Close Game Paradox

**BIGGEST GAP IS WHEN TIED OR BARELY LOSING**

| Score State | Actual GO | Optimal GO | Gap |
|-------------|-----------|------------|-----|
| Tied | 8.0% | 67.7% | **+59.7pp** |
| Down 1-3 | 12.4% | 41.0% | +28.6pp |
| Up 1-3 | 8.8% | 35.0% | +26.2pp |
| Losing big (<-14) | 36.5% | 48.4% | +11.9pp |

**Key insight**: Coaches are MOST conservative exactly when they should be MOST aggressive (tied games). This is consistent with:
- **Blame aversion**: "Don't want to be the one who lost a close game"
- **Status quo bias**: Punting is the "safe" default
- **Regret aversion**: Going for it and failing is viscerally painful when the game is close

This is **NOT** standard loss aversion (which predicts risk-seeking when losing). Instead, it's more like **blame aversion** - coaches are most conservative when the stakes are highest and attribution is clearest.

**Late game (Q4) confirms this**:
- Losing big: 54.5% actual vs 59.5% optimal (small gap - coaches appropriately desperate)
- Losing small: 24.4% actual vs 61.8% optimal (HUGE gap - blame aversion kicks in)

### Data Needed
- [ ] Flag nationally televised games
- [ ] Identify high-leverage fourth down plays
- [ ] Create "treatment" indicator for post-event periods
- [ ] Calculate situation-adjusted GO rates by week

---

## Research Direction 2: Loss Aversion

### Core Question
Do coaches exhibit loss aversion on fourth down decisions?

### Theoretical Framework

Standard expected utility: Choose action maximizing E[WP]

Loss aversion predicts: Coaches weight potential losses more than equivalent gains
- "Going for it and failing" feels worse than "punting and losing later"
- The immediate, attributable loss looms larger than the diffuse, probabilistic loss

### Testable Predictions

**H2a: Asymmetric response to score differential**
- When trailing (in "loss domain"), coaches should be MORE aggressive (seeking risk)
- When leading (in "gain domain"), coaches should be MORE conservative (avoiding risk)
- Reference point = tied game

**H2b: End-of-half effects**
- Before halftime: Losses not yet "realized" → more aggressive
- End of game: Losses imminent → behavior depends on score

**H2c: Comparison to model**
- Model is risk-neutral (maximizes E[WP])
- If coaches are loss-averse, they should:
  - Under-go when winning (protect lead)
  - Over-go when losing big (nothing to lose)
  - Under-go when losing small (loss aversion dominates)

### Analysis Plan

1. **Score differential asymmetry**
   ```
   GO_rate = f(score_diff) + controls

   Test: Is slope steeper for score_diff < 0 vs > 0?
   ```

2. **Deviation from model by domain**
   ```
   For each play:
   - Calculate model_optimal (GO/PUNT/FG)
   - Calculate deviation = (actual == GO) - (optimal == GO)

   Compare deviation when:
   - Winning by 1-7 (small lead)
   - Winning by 8-14 (comfortable lead)
   - Losing by 1-7 (small deficit)
   - Losing by 8-14 (large deficit)
   ```

3. **Reference point analysis**
   - Test if "tied" is the reference point
   - Or is it "expected score based on pre-game line"?
   - Vegas spread as alternative reference point

### Current Data Check
```python
# Quick analysis of GO rate by score differential
# Already have: score_diff in nflfastr_wp_proper_results.csv
# Need to compute: actual_go_rate and optimal_go_rate by score bin
```

---

## Research Direction 3: When Do Teams Learn?

### Dimensions of Learning

**A. By decision complexity**
- Simple (4th & 1 at midfield) vs complex (4th & 3 at own 35, down 4, 8 min left)
- Hypothesis: Learning faster on simple decisions

**B. By feedback clarity**
- Immediate feedback (conversion/failure) vs delayed (game outcome)
- Hypothesis: Learning faster when feedback is immediate and attributable

**C. By stakes**
- Regular season vs playoffs
- Early season vs late season / playoff race
- Hypothesis: Stakes increase attention but may increase conservatism

**D. By coach tenure**
- New coaches vs established coaches
- Hypothesis: New coaches more willing to deviate from norms

**E. By team analytics investment**
- Teams with known analytics departments vs traditional
- Hypothesis: Analytics teams closer to optimal

### Analysis Framework

```
Learning_rate = f(complexity, feedback_clarity, stakes, coach_tenure, analytics_investment)
```

### Specific Tests

1. **Complexity interaction**
   ```
   For each play, compute decision_margin = |WP_go - max(WP_punt, WP_fg)|

   Test: Learning trend steeper when margin > 5% (clear decisions)?
   ```

2. **Feedback attribution**
   ```
   Compare learning on:
   - Fourth downs where team won/lost by < 3 (clear attribution)
   - Fourth downs where team won/lost by > 14 (weak attribution)
   ```

3. **Coach fixed effects**
   ```
   Model: GO_rate = coach_FE + year_FE + situation_controls

   Identify which coaches are:
   - Consistently aggressive (positive FE)
   - Consistently conservative (negative FE)
   - Improving over tenure
   ```

4. **Team analytics proxies**
   ```
   Use public info on team analytics hires:
   - Eagles (early adopters)
   - Ravens (known analytics focus)
   - Browns (Haslam era analytics push)

   Compare to traditional teams
   ```

---

## Research Direction 4: The Down 8 vs Down 9 Puzzle (Two-Point)

### Current Finding
- Down 8: 79% go for 2 (model says 85%) ✓ Good compliance
- Down 9: 1% go for 2 (model says 91%) ✗ Terrible compliance

### Why This Matters
Perfect natural experiment for testing behavioral theories:
- Same decision (go for 2 or not)
- Similar optimal rates (85% vs 91%)
- Dramatically different compliance (79% vs 1%)

### Behavioral Explanations to Test

**Explanation 1: Present bias**
- Down 8: Go for 2 ties game NOW
- Down 9: Go for 2 sets up future FG to tie
- Test: Is compliance related to temporal proximity of payoff?

**Explanation 2: Salience**
- "Tie the game" is a salient, concrete goal
- "Set up a future tying FG" is abstract
- Test: Compliance by whether outcome is immediately visible

**Explanation 3: Regret aversion**
- Failing to tie immediately → clear regret
- Failing to set up future FG → diffuse regret
- Test: Does compliance correlate with potential for blame?

### Extended Analysis
- Down 7 (go for 2 avoids potential future OT): 3% compliance, 67% optimal
- Down 14 (go for 2 to get to down 8): 8% compliance, 94% optimal
- Down 15 (go for 2 to get to down 9): 23% compliance, 99% optimal

Pattern: Compliance drops when the benefit is deferred or probabilistic.

---

## Data Requirements Summary

### Already Have
- [x] nflfastR play-by-play 1999-2024
- [x] vegas_wp for each play
- [x] Fourth down optimal decisions (nflfastr_wp_proper_results.csv)
- [x] Two-point conversion analysis

### Need to Add
- [ ] National TV game indicator
- [ ] Coach ID and tenure for each game
- [ ] Team analytics investment proxy (manual coding)
- [ ] High-profile event dates (manual identification)
- [ ] Playoff/elimination game indicator

### Analyses to Run
1. [ ] Event study around high-profile fourth down plays
2. [ ] Loss aversion test (GO rate asymmetry by score domain)
3. [ ] Learning decomposition by complexity/feedback/stakes
4. [ ] Coach fixed effects model
5. [ ] Extended two-point analysis (all score differentials)

---

## Next Steps

1. **Short term**: Run loss aversion analysis (data already available)
2. **Medium term**: Build event study framework for learning from others
3. **Long term**: Comprehensive coach-level analysis with tenure effects

## Notes for Paper Structure

Potential paper title: "Do Managers Learn? Evidence from NFL Fourth Down Decisions"

Structure:
1. **Intro**: Romer (2006) showed suboptimality; 20 years later, still suboptimal
2. **Framework**: nflfastR WP model, decision theory
3. **Main finding**: Coaches too conservative, minimal learning
4. **Mechanism 1**: Loss aversion explains conservatism
5. **Mechanism 2**: Learning from others (or lack thereof)
6. **Heterogeneity**: When do teams learn?
7. **Two-point comparison**: Why learning differs by complexity
8. **Conclusion**: Implications for organizational learning
