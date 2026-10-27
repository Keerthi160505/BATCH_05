"""
Green AI Index (GAI) Mathematical Formulation & Ranking Engine
Defines:
1. Performance Composite Score (P)
2. Normalized Resource Cost (C) and Efficiency (R = 1 - C)
3. Linear Green AI Index (GAI_linear)
4. Harmonic Green AI Index (GAI_harmonic / Green F-Measure)
5. Eco-Efficiency Ratio (F1 per Joule)
6. Pareto Frontier Dominance Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class GreenAIIndexCalculator:
    """
    Computes the Green AI Index (GAI) across evaluated models.
    Provides customizable weights, sensitivity sweeps, and Pareto dominance analysis.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        perf_weights: Optional[Dict[str, float]] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        pareto_perf_col: str = "f1_score"
    ):
        """
        Parameters:
        -----------
        alpha : float
            Tradeoff parameter between prediction performance (alpha) and
            resource efficiency (1 - alpha). Range: [0.0, 1.0]. Default 0.50 (balanced).
        perf_weights : dict, optional
            Weights for performance metrics: accuracy, precision, recall, f1_score.
        cost_weights : dict, optional
            Weights for resource costs: energy_joules, training_time_s, prediction_time_s, cpu_avg_pct, ram_peak_mb.
        pareto_perf_col : str
            Column used as performance objective for Pareto frontier extraction (default 'f1_score').
        """
        # Validate alpha
        if alpha is None or np.isnan(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Trade-off parameter alpha must be a float in [0.0, 1.0]. Got: {alpha}")
        self.alpha = float(alpha)
        self.pareto_perf_col = pareto_perf_col

        # Default performance weights: balanced, F1-prioritized
        default_perf_weights = {
            "accuracy": 0.25,
            "precision": 0.20,
            "recall": 0.20,
            "f1_score": 0.35
        }
        self.perf_weights = dict(perf_weights) if perf_weights is not None else default_perf_weights
        self._validate_weights(self.perf_weights, "perf_weights")
        self._normalize_dict(self.perf_weights)

        # Default resource cost weights: energy and time prioritized
        default_cost_weights = {
            "energy_joules": 0.35,
            "training_time_s": 0.20,
            "prediction_time_s": 0.15,
            "cpu_avg_pct": 0.15,
            "ram_peak_mb": 0.15
        }
        self.cost_weights = dict(cost_weights) if cost_weights is not None else default_cost_weights
        self._validate_weights(self.cost_weights, "cost_weights")
        self._normalize_dict(self.cost_weights)

    def _validate_weights(self, weights: Dict[str, float], name: str):
        if not weights or not isinstance(weights, dict):
            raise ValueError(f"{name} must be a non-empty dictionary of float weights.")
        for k, v in weights.items():
            if v is None or np.isnan(v) or v < 0.0:
                raise ValueError(f"Weight for '{k}' in {name} must be non-negative. Got: {v}")
        if sum(weights.values()) <= 0.0:
            raise ValueError(f"Sum of weights in {name} must be strictly positive. Got sum: {sum(weights.values())}")

    def _normalize_dict(self, d: Dict[str, float]):
        total = sum(d.values())
        if total > 0:
            for k in d:
                d[k] /= total

    def _min_max_scale(self, series: pd.Series, eps: float = 1e-9) -> pd.Series:
        """Min-Max normalization to [0, 1]. Safe against constant and missing values."""
        clean = series.fillna(0.0).astype(float)
        min_val = clean.min()
        max_val = clean.max()
        if max_val - min_val < eps:
            return pd.Series(0.0, index=clean.index)
        return (clean - min_val) / (max_val - min_val + eps)

    def calculate_gai(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw model evaluation DataFrame and adds:
        - performance_score (P)
        - resource_cost (C)
        - resource_efficiency (R)
        - gai_linear
        - gai_harmonic
        - eco_efficiency (F1 per Joule * 100)
        - pareto_optimal (bool)
        - rank_gai (int)
        """
        if results_df is None or results_df.empty:
            raise ValueError("Input DataFrame for GAI calculation is empty.")

        df = results_df.copy()

        # Fill any missing metric columns safely
        for col in ["accuracy", "precision", "recall", "f1_score", "energy_joules",
                    "training_time_s", "prediction_time_s", "cpu_avg_pct", "ram_peak_mb"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # 1. Performance Composite Score (P in [0, 1])
        perf_score = (
            self.perf_weights["accuracy"] * df["accuracy"] +
            self.perf_weights["precision"] * df["precision"] +
            self.perf_weights["recall"] * df["recall"] +
            self.perf_weights["f1_score"] * df["f1_score"]
        )
        df["performance_score"] = perf_score.round(4)

        # 2. Normalized Resource Costs (each in [0, 1])
        norm_energy = self._min_max_scale(df["energy_joules"])
        norm_t_train = self._min_max_scale(df["training_time_s"])
        norm_t_pred = self._min_max_scale(df["prediction_time_s"])
        norm_cpu = self._min_max_scale(df["cpu_avg_pct"])
        norm_ram = self._min_max_scale(df["ram_peak_mb"])

        resource_cost = (
            self.cost_weights["energy_joules"] * norm_energy +
            self.cost_weights["training_time_s"] * norm_t_train +
            self.cost_weights["prediction_time_s"] * norm_t_pred +
            self.cost_weights["cpu_avg_pct"] * norm_cpu +
            self.cost_weights["ram_peak_mb"] * norm_ram
        )
        df["resource_cost"] = resource_cost.round(4)

        # 3. Resource Efficiency (R in [0, 1], higher is greener)
        resource_efficiency = 1.0 - resource_cost
        df["resource_efficiency"] = resource_efficiency.round(4)

        # 4. GAI Formulations
        # Linear: alpha * P + (1 - alpha) * R
        gai_linear = self.alpha * df["performance_score"] + (1.0 - self.alpha) * df["resource_efficiency"]
        df["gai_linear"] = gai_linear.round(4)

        # Harmonic: 2 * (P * R) / (P + R + eps)
        eps = 1e-9
        gai_harmonic = (2.0 * df["performance_score"] * df["resource_efficiency"]) / (
            df["performance_score"] + df["resource_efficiency"] + eps
        )
        df["gai_harmonic"] = gai_harmonic.round(4)

        # 5. Eco-Efficiency Ratio (F1 per Joule * 100)
        df["eco_efficiency_f1_per_joule"] = (
            (df["f1_score"] / (df["energy_joules"] + eps)) * 100.0
        ).round(3)

        # 6. Pareto Optimality (Performance objective vs Energy Consumption)
        perf_objective = df[self.pareto_perf_col].values if self.pareto_perf_col in df.columns else df["f1_score"].values
        df["pareto_optimal"] = self._compute_pareto_frontier(
            perf_objective,
            df["energy_joules"].values
        )

        # 7. Model Rankings with deterministic tie-breaking
        # Sort key order: gai_linear (desc), f1_score (desc), energy_joules (asc), model_name (asc)
        df = df.sort_values(
            by=["gai_linear", "f1_score", "energy_joules", "model_name"],
            ascending=[False, False, True, True]
        ).reset_index(drop=True)

        df["rank_gai"] = df["gai_linear"].rank(ascending=False, method="min").astype(int)
        df["rank_f1"] = df["f1_score"].rank(ascending=False, method="min").astype(int)
        df["rank_energy"] = df["energy_joules"].rank(ascending=True, method="min").astype(int)

        return df

    def _compute_pareto_frontier(self, perf_scores: np.ndarray, energies: np.ndarray) -> List[bool]:
        """
        Determines Pareto optimality:
        A model is Pareto-optimal if no other model is at least as good in both objectives
        and strictly better in at least one objective.
        Objective 1: Maximize performance score
        Objective 2: Minimize energy consumption
        """
        n = len(perf_scores)
        is_pareto = [True] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Does model j dominate model i?
                    # Dominated if j has >= perf and <= energy, with at least one strict inequality
                    if (perf_scores[j] >= perf_scores[i] and energies[j] <= energies[i]) and (
                        perf_scores[j] > perf_scores[i] or energies[j] < energies[i]
                    ):
                        is_pareto[i] = False
                        break
        return is_pareto

    def sensitivity_analysis(
        self,
        results_df: pd.DataFrame,
        alphas: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Sweeps alpha from 0.0 (pure sustainability) to 1.0 (pure predictive accuracy)
        to evaluate stability of rankings under varying decision-maker preferences.
        """
        if alphas is None:
            alphas = np.linspace(0.0, 1.0, 21)

        sweep_records = []
        for a in alphas:
            calc = GreenAIIndexCalculator(
                alpha=float(a),
                perf_weights=self.perf_weights,
                cost_weights=self.cost_weights,
                pareto_perf_col=self.pareto_perf_col
            )
            evaluated = calc.calculate_gai(results_df)
            for _, row in evaluated.iterrows():
                sweep_records.append({
                    "alpha": round(float(a), 2),
                    "model": row["model_name"],
                    "gai": row["gai_linear"],
                    "rank": row["rank_gai"]
                })

        return pd.DataFrame(sweep_records)
