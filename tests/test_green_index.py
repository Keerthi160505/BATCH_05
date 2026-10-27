"""
Unit Tests for Green AI Index Calculator & Pareto Analysis
Validates:
- Linear & Harmonic Green AI Index formulations
- Alpha boundary values (alpha = 0.0, 0.5, 1.0)
- Input validation (alpha in [0, 1], non-negative weights, positive sum)
- Pareto frontier dominance rules (dominated, non-dominated, and identical ties)
- Constant-valued metric handling (zero variance)
- Deterministic sorting behavior
"""

import pytest
import numpy as np
import pandas as pd
from src.green_index import GreenAIIndexCalculator


def test_gai_linear_and_harmonic_calculations():
    sample_data = pd.DataFrame([
        {
            "model_name": "Model_Fast_Green",
            "accuracy": 0.82, "precision": 0.81, "recall": 0.80, "f1_score": 0.80,
            "energy_joules": 2.5, "training_time_s": 0.15, "prediction_time_s": 0.01,
            "cpu_avg_pct": 15.0, "ram_peak_mb": 120.0
        },
        {
            "model_name": "Model_Heavy_Accurate",
            "accuracy": 0.86, "precision": 0.85, "recall": 0.85, "f1_score": 0.85,
            "energy_joules": 180.0, "training_time_s": 15.0, "prediction_time_s": 0.25,
            "cpu_avg_pct": 85.0, "ram_peak_mb": 500.0
        }
    ])

    calc = GreenAIIndexCalculator(alpha=0.5)
    results = calc.calculate_gai(sample_data)

    assert "gai_linear" in results.columns
    assert "gai_harmonic" in results.columns
    assert "pareto_optimal" in results.columns
    assert "rank_gai" in results.columns

    # Verify linear and harmonic bounds [0, 1]
    assert (results["gai_linear"] >= 0.0).all() and (results["gai_linear"] <= 1.0).all()
    assert (results["gai_harmonic"] >= 0.0).all() and (results["gai_harmonic"] <= 1.0).all()

    # Model_Fast_Green has significantly lower resource cost, so under alpha=0.5 it should have higher resource efficiency
    fg_eff = results.loc[results["model_name"] == "Model_Fast_Green", "resource_efficiency"].values[0]
    ha_eff = results.loc[results["model_name"] == "Model_Heavy_Accurate", "resource_efficiency"].values[0]
    assert fg_eff > ha_eff


def test_alpha_boundaries():
    sample_data = pd.DataFrame([
        {
            "model_name": "Model_A",
            "accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1_score": 0.95,
            "energy_joules": 500.0, "training_time_s": 20.0, "prediction_time_s": 1.0,
            "cpu_avg_pct": 90.0, "ram_peak_mb": 600.0
        },
        {
            "model_name": "Model_B",
            "accuracy": 0.70, "precision": 0.70, "recall": 0.70, "f1_score": 0.70,
            "energy_joules": 1.0, "training_time_s": 0.05, "prediction_time_s": 0.005,
            "cpu_avg_pct": 10.0, "ram_peak_mb": 100.0
        }
    ])

    # alpha = 1.0: Pure predictive performance
    calc_perf = GreenAIIndexCalculator(alpha=1.0)
    res_perf = calc_perf.calculate_gai(sample_data)
    assert res_perf.iloc[0]["model_name"] == "Model_A", "Under alpha=1.0, highest accuracy model must rank #1"

    # alpha = 0.0: Pure sustainability
    calc_sust = GreenAIIndexCalculator(alpha=0.0)
    res_sust = calc_sust.calculate_gai(sample_data)
    assert res_sust.iloc[0]["model_name"] == "Model_B", "Under alpha=0.0, lowest energy/latency model must rank #1"


def test_weight_and_alpha_validation():
    # Invalid alpha (<0 or >1)
    with pytest.raises(ValueError, match="alpha must be a float in"):
        GreenAIIndexCalculator(alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be a float in"):
        GreenAIIndexCalculator(alpha=1.5)

    # Negative weight
    with pytest.raises(ValueError, match="must be non-negative"):
        GreenAIIndexCalculator(cost_weights={"energy_joules": -0.5, "cpu_avg_pct": 1.0})

    # Zero total weights
    with pytest.raises(ValueError, match="must be strictly positive"):
        GreenAIIndexCalculator(perf_weights={"accuracy": 0.0, "f1_score": 0.0})


def test_hand_crafted_pareto_frontier():
    """
    Test 3 distinct scenarios:
    1. Model A strictly dominates Model B (higher F1, lower energy).
    2. Model A and Model C are trade-off non-dominated (A has higher F1, C has lower energy).
    3. Model D and Model E have identical objectives (ties remain non-dominated).
    """
    data = pd.DataFrame([
        {"model_name": "Model_A", "f1_score": 0.90, "energy_joules": 10.0},
        {"model_name": "Model_B", "f1_score": 0.80, "energy_joules": 20.0}, # Dominated by Model_A
        {"model_name": "Model_C", "f1_score": 0.70, "energy_joules": 2.0},  # Non-dominated (lowest energy)
        {"model_name": "Model_D", "f1_score": 0.85, "energy_joules": 5.0},  # Non-dominated
        {"model_name": "Model_E", "f1_score": 0.85, "energy_joules": 5.0},  # Identical to D -> both non-dominated
    ])

    calc = GreenAIIndexCalculator(alpha=0.5, pareto_perf_col="f1_score")
    results = calc.calculate_gai(data)

    pareto_map = dict(zip(results["model_name"], results["pareto_optimal"]))

    assert pareto_map["Model_A"] is True, "Model_A should be Pareto-optimal (high F1)"
    assert pareto_map["Model_B"] is False, "Model_B is strictly dominated by Model_A"
    assert pareto_map["Model_C"] is True, "Model_C should be Pareto-optimal (lowest energy)"
    assert pareto_map["Model_D"] is True, "Model_D should be Pareto-optimal"
    assert pareto_map["Model_E"] is True, "Model_E should be Pareto-optimal alongside tied Model_D"


def test_constant_valued_metrics():
    """Verify that constant-valued columns (zero variance) normalize safely without crashing or returning NaN."""
    data = pd.DataFrame([
        {"model_name": "M1", "accuracy": 0.80, "f1_score": 0.80, "energy_joules": 10.0},
        {"model_name": "M2", "accuracy": 0.85, "f1_score": 0.85, "energy_joules": 10.0}, # identical energy
    ])
    calc = GreenAIIndexCalculator(alpha=0.5)
    res = calc.calculate_gai(data)
    assert not res["gai_linear"].isna().any()
    assert not res["resource_cost"].isna().any()
