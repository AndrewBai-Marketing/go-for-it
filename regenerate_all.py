"""
Master script to regenerate all analysis results.

Run this after making changes to the data pipeline or models.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path: str, description: str):
    """Run a Python script and report status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=Path(__file__).parent,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"WARNING: {script_path} returned non-zero exit code: {result.returncode}")
    else:
        print(f"SUCCESS: {description}")

    return result.returncode == 0


def main():
    base_dir = Path(__file__).parent

    print("="*60)
    print("REGENERATING ALL ANALYSIS RESULTS")
    print("="*60)

    # Step 1: Re-download and clean data
    print("\n\n" + "="*60)
    print("STEP 1: DATA PIPELINE")
    print("="*60)
    run_script("data/acquire_data.py", "Download and clean play-by-play data")

    # Step 2: Retrain models
    print("\n\n" + "="*60)
    print("STEP 2: RETRAIN BAYESIAN MODELS")
    print("="*60)
    run_script("models/bayesian_models.py", "Train all Bayesian models (conversion, punt, FG, WP)")

    # Step 3: Fourth down expanding window analysis
    print("\n\n" + "="*60)
    print("STEP 3: FOURTH DOWN EXPANDING WINDOW ANALYSIS")
    print("="*60)
    run_script("analysis/expanding_window_analysis.py", "Ex ante vs ex post fourth down analysis")

    # Step 4: Fourth down decision categorization
    print("\n\n" + "="*60)
    print("STEP 4: FOURTH DOWN DECISION CATEGORIZATION")
    print("="*60)
    run_script("analysis/decision_categorization.py", "Categorize mistakes and compute WP costs")

    # Step 5: Learning by margin analysis
    print("\n\n" + "="*60)
    print("STEP 5: LEARNING BY MARGIN ANALYSIS")
    print("="*60)
    run_script("analysis/learning_by_margin.py", "Analyze learning trends by decision margin")

    # Step 6: Era comparison
    print("\n\n" + "="*60)
    print("STEP 6: ERA COMPARISON")
    print("="*60)
    run_script("analysis/era_comparison.py", "Compare early vs late era decision quality")

    # Step 7: Team WP loss analysis
    print("\n\n" + "="*60)
    print("STEP 7: TEAM WP LOSS ANALYSIS")
    print("="*60)
    run_script("analysis/team_wp_loss_analysis.py", "Compute expected wins lost per team/season")

    # Step 8: Two-point rule change analysis
    print("\n\n" + "="*60)
    print("STEP 8: TWO-POINT RULE CHANGE ANALYSIS")
    print("="*60)
    run_script("analysis/two_point_rule_change_analysis.py", "Analyze two-point decisions with expanding window")

    # Step 9: Create defensible figure (two-point learning curve)
    print("\n\n" + "="*60)
    print("STEP 9: TWO-POINT DEFENSIBLE ANALYSIS")
    print("="*60)
    run_script("analysis/create_defensible_figure.py", "Create two-point defensible learning figure")

    # Step 10: Two-point analysis (down 8 vs 9 paradox)
    print("\n\n" + "="*60)
    print("STEP 10: TWO-POINT PARADOX ANALYSIS")
    print("="*60)
    run_script("analysis/two_point_analysis.py", "Analyze down 8 vs down 9 paradox")

    # Step 11: High stakes analysis
    print("\n\n" + "="*60)
    print("STEP 11: HIGH STAKES ANALYSIS")
    print("="*60)
    run_script("analysis/high_stakes_analysis.py", "Compare playoff vs regular season decisions")

    # Step 12: Generate all visualizations
    print("\n\n" + "="*60)
    print("STEP 12: GENERATE VISUALIZATIONS")
    print("="*60)
    run_script("analysis/visualizations.py", "Generate main figures")
    run_script("analysis/update_visualizations.py", "Generate update figures")

    print("\n\n" + "="*60)
    print("REGENERATION COMPLETE")
    print("="*60)
    print("\nCheck outputs/tables/ for CSV files")
    print("Check outputs/figures/ for PNG files")
    print("\nRemember to update the numbers in:")
    print("  - paper_summary.tex")
    print("  - update_margin_analysis.tex")
    print("  - README.md")


if __name__ == "__main__":
    main()
