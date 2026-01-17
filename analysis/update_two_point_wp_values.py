"""
Update the Down 9 decision tree WP values using nflfastR WP lookup.

The Down 9 paradox shows:
- After TD from down 9, you're down 3
- PAT path: 94% → Down 2; 6% → Down 3
- 2pt path: 48% → Down 1; 52% → Down 3

We need WP values for:
- WP(Down 1) - any score wins
- WP(Down 2) - need TD to be comfortable
- WP(Down 3) - FG ties

These are WP values after kickoff (opponent has ball at ~25 yard line).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_wp_for_score_diff(pbp: pd.DataFrame, score_diff: int, time_remaining: int = 600) -> float:
    """
    Get average WP for a given score differential using nflfastR data.

    Uses plays where:
    - Team has the score_diff (positive = winning)
    - Opponent has ball after kickoff (around 25 yard line = 75 yardline_100 for them)
    - Similar time remaining
    """
    wp_col = 'vegas_wp' if 'vegas_wp' in pbp.columns else 'wp'

    # Filter to situations after kickoffs where opponent has ball
    # Score diff from kicking team's perspective
    # After kickoff, posteam is the receiving team

    # Find plays where:
    # - 1st and 10
    # - Around 25 yard line (opponent's perspective after kickoff)
    # - Score differential matches

    quarter = max(1, 4 - int(time_remaining // 900))

    relevant = pbp[
        (pbp['down'] == 1) &
        (pbp['ydstogo'] >= 8) & (pbp['ydstogo'] <= 12) &
        (pbp['yardline_100'] >= 70) & (pbp['yardline_100'] <= 80) &  # ~25 yard line for receiving team
        (pbp['score_differential'] == -score_diff) &  # Opponent has ball, so we flip perspective
        (pbp['qtr'] == quarter) &
        (pbp[wp_col].notna())
    ]

    if len(relevant) > 10:
        # Opponent's WP when they have ball at their 25
        opp_wp = relevant[wp_col].mean()
        # Our WP is 1 - their WP
        return 1 - opp_wp

    # Fallback: use broader time window
    relevant = pbp[
        (pbp['down'] == 1) &
        (pbp['ydstogo'] >= 8) & (pbp['ydstogo'] <= 12) &
        (pbp['yardline_100'] >= 70) & (pbp['yardline_100'] <= 80) &
        (pbp['score_differential'] == -score_diff) &
        (pbp[wp_col].notna())
    ]

    if len(relevant) > 10:
        opp_wp = relevant[wp_col].mean()
        return 1 - opp_wp

    # Last resort: simple estimate
    print(f"Warning: Using fallback for score_diff={score_diff}")
    return 0.5 + score_diff * 0.03


def main():
    print("=" * 60)
    print("COMPUTING NFLFASTR WP VALUES FOR DOWN 9 DECISION TREE")
    print("=" * 60)

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'all_pbp_1999_2024.parquet'
    print(f"\nLoading data from {data_path}...")
    pbp = pd.read_parquet(data_path)

    # Use recent years for most accurate WP estimates
    pbp = pbp[pbp['season'] >= 2015].copy()
    print(f"Using {len(pbp):,} plays from 2015-2024")

    # Scenario: Down 9, score TD, now down 3
    # After PAT/2pt decision, kickoff, opponent has ball
    # Time remaining: ~10 minutes (Q4, close game)

    print("\n" + "-" * 60)
    print("DOWN 9 SCENARIO: After TD, down 3, making PAT/2pt decision")
    print("-" * 60)

    # Get WP values for each possible outcome
    # These are from the perspective of the scoring team after kickoff

    # Down 1 (2pt success): Any score wins
    wp_down_1 = get_wp_for_score_diff(pbp, -1, time_remaining=600)
    print(f"\nWP(Down 1) = {wp_down_1:.1%}")
    print("  -> After successful 2pt, down by 1, any score wins")

    # Down 2 (PAT make): Need TD to be comfortable
    wp_down_2 = get_wp_for_score_diff(pbp, -2, time_remaining=600)
    print(f"\nWP(Down 2) = {wp_down_2:.1%}")
    print("  -> After PAT, down by 2, need TD (FG only gets you to -2 still)")

    # Down 3 (PAT miss or 2pt fail): FG ties
    wp_down_3 = get_wp_for_score_diff(pbp, -3, time_remaining=600)
    print(f"\nWP(Down 3) = {wp_down_3:.1%}")
    print("  -> After PAT miss or 2pt fail, down by 3, FG ties")

    # Calculate expected WP for each decision
    p_pat = 0.94  # PAT success rate
    p_2pt = 0.48  # 2pt conversion rate

    print("\n" + "-" * 60)
    print("EXPECTED WIN PROBABILITY CALCULATION")
    print("-" * 60)

    ewp_pat = p_pat * wp_down_2 + (1 - p_pat) * wp_down_3
    print(f"\nE[WP|PAT] = {p_pat:.0%} × {wp_down_2:.1%} + {1-p_pat:.0%} × {wp_down_3:.1%}")
    print(f"         = {ewp_pat:.1%}")

    ewp_2pt = p_2pt * wp_down_1 + (1 - p_2pt) * wp_down_3
    print(f"\nE[WP|2pt] = {p_2pt:.0%} × {wp_down_1:.1%} + {1-p_2pt:.0%} × {wp_down_3:.1%}")
    print(f"         = {ewp_2pt:.1%}")

    diff = ewp_2pt - ewp_pat
    print(f"\nDIFFERENCE: {diff:+.1%}")
    print(f"RECOMMENDATION: {'GO FOR 2' if diff > 0 else 'KICK PAT'}")

    # Output values for README
    print("\n" + "=" * 60)
    print("VALUES FOR README.md DECISION TREE:")
    print("=" * 60)
    print(f"""
```
                    DOWN 3
                (after TD from down 9)
                 /            \\
              PAT           GO FOR 2
               |                 |
         ┌─────┴─────┐      ┌────┴────┐
        {p_pat:.0%}         {1-p_pat:.0%}    {p_2pt:.0%}       {1-p_2pt:.0%}
         |           |      |         |
      DOWN 2     DOWN 3  DOWN 1   DOWN 3
     WP = {wp_down_2:.0%}   WP = {wp_down_3:.0%} WP = {wp_down_1:.0%} WP = {wp_down_3:.0%}

E[WP|PAT] = {p_pat:.2f} × {wp_down_2:.2f} + {1-p_pat:.2f} × {wp_down_3:.2f} = {ewp_pat:.1%}
E[WP|2pt] = {p_2pt:.2f} × {wp_down_1:.2f} + {1-p_2pt:.2f} × {wp_down_3:.2f} = {ewp_2pt:.1%}

DIFFERENCE: {diff:+.1%} → {'GO FOR 2 IS BETTER' if diff > 0 else 'PAT IS BETTER'}
```
""")

    return {
        'wp_down_1': wp_down_1,
        'wp_down_2': wp_down_2,
        'wp_down_3': wp_down_3,
        'ewp_pat': ewp_pat,
        'ewp_2pt': ewp_2pt,
        'diff': diff,
    }


if __name__ == "__main__":
    main()
