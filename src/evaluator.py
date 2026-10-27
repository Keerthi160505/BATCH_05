"""
Evaluation Orchestrator
Executes end-to-end benchmarking of the 6 machine learning models:
Measures:
1. Accuracy
2. Precision (weighted, zero_division=0)
3. Recall (weighted, zero_division=0)
4. F1-Score (weighted, zero_division=0)
5. CPU Usage (% average and peak)
6. Memory Usage (peak RAM and delta RAM in MB)
7. Training Time (s)
8. Prediction Time & Latency per 1k samples (s)
9. Energy Consumption (Joules & kWh)
10. Carbon Emissions (gCO2eq)
Supports multi-run statistical evaluation (mean +/- std) for research publication.
"""

import time
import json
import psutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .models import ModelFactory
from .profiler import ResourceEnergyProfiler
from .green_index import GreenAIIndexCalculator
from .data_pipeline import MemoryBudgetExceededError


class GreenAIEvaluator:
    """
    Coordinates model training, inference, resource profiling,
    performance metrics computation, and Green AI Index ranking.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        use_codecarbon: bool = True,
        perf_weights: Optional[Dict[str, float]] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        pareto_perf_col: str = "f1_score",
        max_memory_budget_mb: float = 2048.0
    ):
        self.alpha = float(alpha)
        self.use_codecarbon = bool(use_codecarbon)
        self.pareto_perf_col = pareto_perf_col
        self.max_memory_budget_mb = float(max_memory_budget_mb)
        self.gai_calculator = GreenAIIndexCalculator(
            alpha=self.alpha,
            perf_weights=perf_weights,
            cost_weights=cost_weights,
            pareto_perf_col=self.pareto_perf_col
        )

    def evaluate_single_model(
        self,
        model_name: str,
        X_train: Union[np.ndarray, sp.spmatrix],
        X_test: Union[np.ndarray, sp.spmatrix],
        y_train: np.ndarray,
        y_test: np.ndarray,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Trains and tests an individual model while actively profiling its resource consumption.
        Validates memory safety prior to execution.
        """
        # Memory safety check before model training
        if sp.issparse(X_train):
            data_bytes = X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes
        else:
            data_bytes = X_train.nbytes
        data_mb = data_bytes / (1024 * 1024)
        
        # Model training overhead estimation (internal buffers, gradient arrays, histograms)
        est_training_mb = data_mb * 3.0
        if est_training_mb > self.max_memory_budget_mb:
            raise MemoryBudgetExceededError(
                message=(
                    f"Memory budget exceeded for model training: estimated memory for model '{model_name}' "
                    f"({est_training_mb:.1f} MB) exceeds the memory budget of {self.max_memory_budget_mb:.1f} MB. "
                    f"Training aborted to avoid system exhaustion."
                ),
                estimated_mb=est_training_mb,
                budget_mb=self.max_memory_budget_mb,
                stage=f"model_training ({model_name})",
                recommendations=[
                    "Sample the training data in the sidebar or reduce sample_size.",
                    "Select lighter baseline models (e.g. Decision Tree or Logistic Regression).",
                    "Increase max_memory_budget_mb if additional host RAM is available."
                ]
            )

        # Check system physical memory headroom
        try:
            avail_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
            if avail_ram_mb < est_training_mb * 0.5:
                raise MemoryBudgetExceededError(
                    message=(
                        f"System available RAM ({avail_ram_mb:.1f} MB) is critically insufficient "
                        f"for model '{model_name}' estimated requirement ({est_training_mb:.1f} MB)."
                    ),
                    estimated_mb=est_training_mb,
                    budget_mb=avail_ram_mb,
                    stage=f"system_memory_check ({model_name})"
                )
        except Exception as e:
            if isinstance(e, MemoryBudgetExceededError):
                raise

        model = ModelFactory.get_model(model_name, random_state=random_state)

        # ---------------- TRAINING PROFILING ----------------
        train_profiler = ResourceEnergyProfiler(use_codecarbon=self.use_codecarbon)
        train_profiler.start()
        model.fit(X_train, y_train)
        train_stats = train_profiler.stop()

        # ---------------- PREDICTION PROFILING ----------------
        pred_profiler = ResourceEnergyProfiler(use_codecarbon=self.use_codecarbon)
        pred_profiler.start()
        y_pred = model.predict(X_test)
        pred_stats = pred_profiler.stop()

        # ---------------- PREDICTION PERFORMANCE ----------------
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Combined resource metrics
        total_time_s = train_stats["duration_s"] + pred_stats["duration_s"]
        total_energy_j = train_stats["energy_joules"] + pred_stats["energy_joules"]
        total_energy_kwh = train_stats["energy_kwh"] + pred_stats["energy_kwh"]
        total_co2_g = train_stats["co2_emissions_g"] + pred_stats["co2_emissions_g"]
        avg_cpu_pct = round((train_stats["avg_cpu_pct"] + pred_stats["avg_cpu_pct"]) / 2.0, 2)
        peak_cpu_pct = max(train_stats["peak_cpu_pct"], pred_stats["peak_cpu_pct"])
        peak_ram_mb = max(train_stats["peak_ram_mb"], pred_stats["peak_ram_mb"])
        delta_ram_mb = max(train_stats["delta_ram_mb"], pred_stats["delta_ram_mb"])

        return {
            "model_name": model_name,
            # Predictive Metrics
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            # Resource & Sustainability Metrics
            "cpu_avg_pct": avg_cpu_pct,
            "cpu_peak_pct": peak_cpu_pct,
            "ram_peak_mb": round(peak_ram_mb, 2),
            "ram_delta_mb": round(delta_ram_mb, 2),
            "training_time_s": train_stats["duration_s"],
            "prediction_time_s": pred_stats["duration_s"],
            "total_time_s": round(total_time_s, 5),
            "energy_joules": round(total_energy_j, 4),
            "energy_kwh": float(f"{total_energy_kwh:.8f}"),
            "co2_emissions_g": float(f"{total_co2_g:.6f}"),
            # Detailed breakdown
            "train_energy_joules": train_stats["energy_joules"],
            "pred_energy_joules": pred_stats["energy_joules"],
            "test_sample_count": int(X_test.shape[0]),
            "latency_per_1k_samples_s": round((pred_stats["duration_s"] / max(1, int(X_test.shape[0]))) * 1000.0, 4),
            "power_measurement_method": train_stats.get("power_measurement_method", "TDP empirical integration"),
            "is_energy_estimated": train_stats.get("is_energy_estimated", True),
            "tdp_watts": train_stats.get("tdp_watts", 65.0)
        }

    def evaluate_all(
        self,
        X_train: Union[np.ndarray, sp.spmatrix],
        X_test: Union[np.ndarray, sp.spmatrix],
        y_train: np.ndarray,
        y_test: np.ndarray,
        models: Optional[List[str]] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Evaluates all requested models sequentially, profiles them,
        and computes the Green AI Index rankings.
        """
        model_list = models or ModelFactory.SUPPORTED_MODELS
        results = []

        for m_name in model_list:
            res = self.evaluate_single_model(
                model_name=m_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                random_state=random_state
            )
            results.append(res)

        raw_df = pd.DataFrame(results)
        return self.gai_calculator.calculate_gai(raw_df)

    def evaluate_multi_run(
        self,
        X_train: Union[np.ndarray, sp.spmatrix],
        X_test: Union[np.ndarray, sp.spmatrix],
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_runs: int = 3,
        models: Optional[List[str]] = None,
        base_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes n_runs independent iterations across different seeds.
        Computes Mean +/- Std for all metrics, essential for research publication.
        Returns:
            (aggregated_df_with_std, raw_all_runs_df)
        """
        model_list = models or ModelFactory.SUPPORTED_MODELS
        all_runs = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            for m_name in model_list:
                res = self.evaluate_single_model(
                    model_name=m_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    random_state=seed
                )
                res["run_idx"] = run_idx + 1
                res["seed"] = seed
                all_runs.append(res)

        raw_all = pd.DataFrame(all_runs)

        # Aggregate numeric metrics
        numeric_cols = [
            "accuracy", "precision", "recall", "f1_score",
            "cpu_avg_pct", "cpu_peak_pct", "ram_peak_mb", "ram_delta_mb",
            "training_time_s", "prediction_time_s", "total_time_s",
            "energy_joules", "energy_kwh", "co2_emissions_g",
            "latency_per_1k_samples_s"
        ]

        grouped = raw_all.groupby("model_name")
        means = grouped[numeric_cols].mean()
        stds = grouped[numeric_cols].std().fillna(0.0)

        # Compute GAI on mean values
        mean_results = means.reset_index()
        mean_with_gai = self.gai_calculator.calculate_gai(mean_results)

        # Add formatted mean +/- std columns for paper presentation
        for col in ["accuracy", "f1_score", "training_time_s", "energy_joules", "ram_peak_mb"]:
            mean_with_gai[f"{col}_mean_std"] = mean_with_gai.apply(
                lambda r: f"{r[col]:.4f} +/- {stds.loc[r['model_name'], col]:.4f}"
                if (r["model_name"] in stds.index and col in stds.columns) else f"{r[col]:.4f}",
                axis=1
            )

        return mean_with_gai, raw_all
