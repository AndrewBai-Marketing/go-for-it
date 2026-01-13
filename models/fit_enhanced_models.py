"""
Fit enhanced Bayesian models with contextual factors.

This script fits:
1. WeatherAwareFieldGoalModel - FG probability with kicker effects, weather, and long-distance handling
2. ContextAwareConversionModel - Conversion probability with home/dome/team effects
3. ContextAwarePuntModel - Punt distance with wind and punter effects

These models extend the base hierarchical models with additional contextual information
that can improve decision accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.bayesian_models import (
    WeatherAwareFieldGoalModel,
    ContextAwareConversionModel,
    ContextAwarePuntModel
)


def load_data(data_dir: Path):
    """Load all necessary data files."""
    print("Loading data...")

    # Main play-by-play data
    pbp = pd.read_parquet(data_dir / 'all_pbp_2006_2024.parquet')
    print(f"  Play-by-play: {len(pbp):,} plays")

    # Fourth down attempts
    attempts = pd.read_parquet(data_dir / 'all_fourth_downs_1999_2024.parquet')
    attempts = attempts[(attempts['rush_attempt']==1) | (attempts['pass_attempt']==1)]
    attempts['converted'] = attempts['fourth_down_converted'].fillna(0).astype(int)
    attempts['ydstogo_capped'] = attempts['ydstogo'].clip(1, 15)
    print(f"  Fourth down attempts: {len(attempts):,}")

    # Field goals
    fgs = pbp[pbp['field_goal_attempt'] == 1].copy()
    fgs['fg_made'] = (fgs['field_goal_result'] == 'made').astype(int)
    fgs['fg_distance'] = fgs['kick_distance']
    print(f"  Field goal attempts: {len(fgs):,}")

    # Punts
    punts = pbp[pbp['punt_attempt'] == 1].copy()
    print(f"  Punt attempts: {len(punts):,}")

    return pbp, attempts, fgs, punts


def fit_fg_model(fgs: pd.DataFrame, models_dir: Path, n_samples: int = 2000):
    """Fit weather-aware field goal model."""
    print("\n" + "="*60)
    print("WEATHER-AWARE FIELD GOAL MODEL")
    print("="*60)

    model = WeatherAwareFieldGoalModel()
    model.fit(fgs, n_samples=n_samples)

    # Validation: compare predictions to empirical rates
    print("\n--- Model Validation ---")
    print("Distance | Model | Actual | Diff")
    for dist in [25, 35, 45, 50, 55, 60]:
        model_p = model.get_make_prob(dist)
        actual = fgs[fgs['fg_distance'] == dist]['fg_made'].mean()
        n = len(fgs[fgs['fg_distance'] == dist])
        diff = model_p - actual
        if n > 0:
            print(f"  {dist:2d} yd   | {model_p:.1%}  | {actual:.1%} (n={n:4d}) | {diff:+.1%}")

    # Weather effect validation
    print("\n--- Weather Effect Validation ---")
    long_fgs = fgs[fgs['fg_distance'] >= 50]

    cold_fgs = long_fgs[long_fgs['temp'] < 40]
    warm_fgs = long_fgs[long_fgs['temp'] >= 40]

    if len(cold_fgs) > 0 and len(warm_fgs) > 0:
        print(f"Long FGs (50+) in cold (<40°F): {cold_fgs['fg_made'].mean():.1%} (n={len(cold_fgs)})")
        print(f"Long FGs (50+) in warm (40°F+): {warm_fgs['fg_made'].mean():.1%} (n={len(warm_fgs)})")

        # Model predictions for same conditions
        model_cold = model.get_make_prob(53, temp=30, wind=10)
        model_warm = model.get_make_prob(53, temp=70, wind=5)
        print(f"Model 53yd cold: {model_cold:.1%}")
        print(f"Model 53yd warm: {model_warm:.1%}")

    # Save model
    model.save(models_dir / 'weather_aware_fg_model.pkl')
    print(f"\nModel saved to {models_dir / 'weather_aware_fg_model.pkl'}")

    return model


def fit_conversion_model(attempts: pd.DataFrame, models_dir: Path, n_samples: int = 2000):
    """Fit context-aware conversion model."""
    print("\n" + "="*60)
    print("CONTEXT-AWARE CONVERSION MODEL")
    print("="*60)

    model = ContextAwareConversionModel()
    model.fit(attempts, n_samples=n_samples)

    # Validation
    print("\n--- Model Validation ---")
    print("Distance | Model | Actual | Diff")
    for dist in [1, 2, 3, 5, 10]:
        model_p = model.get_conversion_prob(dist)
        actual = attempts[attempts['ydstogo_capped'] == dist]['converted'].mean()
        n = len(attempts[attempts['ydstogo_capped'] == dist])
        diff = model_p - actual
        print(f"  {dist:2d} yd   | {model_p:.1%}  | {actual:.1%} (n={n:4d}) | {diff:+.1%}")

    # Context effect validation
    print("\n--- Context Effect Validation ---")
    home = attempts[attempts['posteam_type'] == 'home']
    away = attempts[attempts['posteam_type'] == 'away']
    print(f"Home conversion rate: {home['converted'].mean():.1%} (n={len(home)})")
    print(f"Away conversion rate: {away['converted'].mean():.1%} (n={len(away)})")

    dome = attempts[attempts['roof'].isin(['dome', 'closed'])]
    outdoor = attempts[attempts['roof'] == 'outdoors']
    print(f"Dome conversion rate: {dome['converted'].mean():.1%} (n={len(dome)})")
    print(f"Outdoor conversion rate: {outdoor['converted'].mean():.1%} (n={len(outdoor)})")

    # Model predictions
    print(f"\nModel (4th & 3, home, dome): {model.get_conversion_prob(3, is_home=True, is_dome=True):.1%}")
    print(f"Model (4th & 3, away, outdoor): {model.get_conversion_prob(3, is_home=False, is_dome=False):.1%}")

    # Save model
    model.save(models_dir / 'context_aware_conversion_model.pkl')
    print(f"\nModel saved to {models_dir / 'context_aware_conversion_model.pkl'}")

    return model


def fit_punt_model(punts: pd.DataFrame, models_dir: Path, n_samples: int = 2000):
    """Fit context-aware punt model."""
    print("\n" + "="*60)
    print("CONTEXT-AWARE PUNT MODEL")
    print("="*60)

    model = ContextAwarePuntModel()
    model.fit(punts, n_samples=n_samples)

    # Validation
    print("\n--- Model Validation ---")

    # Wind effect
    punts_wind = punts.dropna(subset=['wind', 'kick_distance'])
    light_wind = punts_wind[punts_wind['wind'] <= 5]['kick_distance'].mean()
    heavy_wind = punts_wind[punts_wind['wind'] >= 15]['kick_distance'].mean()
    print(f"Actual punt distance (light wind): {light_wind:.1f} yards")
    print(f"Actual punt distance (heavy wind): {heavy_wind:.1f} yards")
    print(f"Wind effect (actual): {heavy_wind - light_wind:.1f} yards")

    # Model predictions
    print(f"\nModel punt (field_pos=50, wind=5):  {model.get_expected_net(50, wind=5):.1f} yards")
    print(f"Model punt (field_pos=50, wind=20): {model.get_expected_net(50, wind=20):.1f} yards")

    # Save model
    model.save(models_dir / 'context_aware_punt_model.pkl')
    print(f"\nModel saved to {models_dir / 'context_aware_punt_model.pkl'}")

    return model


def main():
    """Fit all enhanced models."""
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent

    # Load data
    pbp, attempts, fgs, punts = load_data(data_dir)

    # Fit models
    fg_model = fit_fg_model(fgs, models_dir)
    conv_model = fit_conversion_model(attempts, models_dir)
    punt_model = fit_punt_model(punts, models_dir)

    print("\n" + "="*60)
    print("ALL ENHANCED MODELS FITTED SUCCESSFULLY")
    print("="*60)
    print("\nNew model files:")
    print("  - weather_aware_fg_model.pkl")
    print("  - context_aware_conversion_model.pkl")
    print("  - context_aware_punt_model.pkl")
    print("\nTo use these models, update load_all_models() or create a new")
    print("loading function that uses these enhanced versions.")


if __name__ == "__main__":
    main()
