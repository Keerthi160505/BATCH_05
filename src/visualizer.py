"""
Publication-Quality Visualization Suite
Generates 300 DPI figures for research papers:
1. Pareto Frontier Plot (Performance vs Energy)
2. Multi-Criteria Radar / Spider Chart
3. Green AI Index Leaderboard
4. Energy and Time Breakdown
5. Sensitivity Analysis Curves (Alpha parameter sweep)
6. Tradeoff Correlation Heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

# Set publication style aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

MODEL_COLORS = {
    "Decision Tree": "#1f77b4",
    "Random Forest": "#2ca02c",
    "XGBoost": "#ff7f0e",
    "LightGBM": "#9467bd",
    "CatBoost": "#d62728",
    "Logistic Regression": "#8c564b"
}


class GreenAIVisualizer:
    """
    Creates and saves publication-ready figures.
    """
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_pareto_frontier(self, df: pd.DataFrame, dataset_name: str = "Benchmark", save_name: str = "pareto_frontier.png") -> str:
        """
        Plots F1-Score vs Energy Consumption (Joules) with Pareto Frontier.
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        for _, row in df.iterrows():
            m_name = row["model_name"]
            color = MODEL_COLORS.get(m_name, "#333333")
            marker = "*" if row.get("pareto_optimal", False) else "o"
            size = 200 if row.get("pareto_optimal", False) else 120

            ax.scatter(
                row["energy_joules"], row["f1_score"],
                color=color, s=size, alpha=0.9, edgecolors="black", linewidth=1.2,
                label=f"{m_name} {'(Pareto Optimal)' if row.get('pareto_optimal', False) else ''}"
            )
            ax.annotate(
                f" {m_name}\n (GAI: {row['gai_linear']:.2f})",
                (row["energy_joules"], row["f1_score"]),
                fontsize=9, weight="bold" if row.get("pareto_optimal", False) else "normal"
            )

        # Draw Pareto frontier line
        pareto_df = df[df["pareto_optimal"] == True].sort_values("energy_joules")
        if len(pareto_df) > 1:
            ax.plot(
                pareto_df["energy_joules"], pareto_df["f1_score"],
                linestyle="--", color="#e74c3c", linewidth=2.0, alpha=0.8,
                label="Pareto Frontier"
            )

        ax.set_xlabel("Energy Consumption (Joules) [Lower is Greener]", fontweight="bold")
        ax.set_ylabel("F1-Score [Higher is Better]", fontweight="bold")
        ax.set_title(f"Pareto Trade-Off: Predictive Performance vs. Energy Usage\n[{dataset_name}]", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Place legend outside or below
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        return save_path

    def plot_radar_chart(self, df: pd.DataFrame, dataset_name: str = "Benchmark", save_name: str = "radar_chart.png") -> str:
        """
        Multi-criteria Radar / Spider Chart comparing all 6 models.
        """
        categories = [
            "Accuracy",
            "F1-Score",
            "CPU Eff.",
            "RAM Eff.",
            "Time Eff.",
            "Energy Eff."
        ]
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

        # Normalize metrics to [0.1, 1.0] for radar visibility
        def norm_inv(s):
            denom = (s.max() - s.min()) if (s.max() - s.min()) > 1e-9 else 1.0
            return 1.0 - ((s - s.min()) / denom)

        norm_acc = df["accuracy"]
        norm_f1 = df["f1_score"]
        eff_cpu = norm_inv(df["cpu_avg_pct"])
        eff_ram = norm_inv(df["ram_peak_mb"])
        eff_time = norm_inv(df["total_time_s"])
        eff_energy = norm_inv(df["energy_joules"])

        for idx, row in df.iterrows():
            m_name = row["model_name"]
            values = [
                norm_acc.iloc[idx],
                norm_f1.iloc[idx],
                eff_cpu.iloc[idx],
                eff_ram.iloc[idx],
                eff_time.iloc[idx],
                eff_energy.iloc[idx]
            ]
            values += values[:1]
            color = MODEL_COLORS.get(m_name, "#333333")

            ax.plot(angles, values, color=color, linewidth=2, label=m_name)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontweight="bold", fontsize=11)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Multi-Dimensional Green AI Profile\n[{dataset_name}]", fontweight="bold", pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1.1), loc='upper right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        return save_path

    def plot_gai_leaderboard(self, df: pd.DataFrame, dataset_name: str = "Benchmark", save_name: str = "gai_leaderboard.png") -> str:
        """
        Stacked horizontal bar chart showing Performance vs Resource Efficiency breakdown of GAI.
        """
        df_sorted = df.sort_values("gai_linear", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

        y_positions = np.arange(len(df_sorted))
        bar_height = 0.55

        # Show weighted performance and weighted sustainability portions
        perf_contrib = 0.5 * df_sorted["performance_score"]
        sust_contrib = 0.5 * df_sorted["resource_efficiency"]

        ax.barh(y_positions, perf_contrib, bar_height, label="Performance (50%)", color="#2ecc71", alpha=0.85)
        ax.barh(y_positions, sust_contrib, bar_height, left=perf_contrib, label="Sustainability / Efficiency (50%)", color="#3498db", alpha=0.85)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_sorted["model_name"], fontweight="bold", fontsize=11)
        ax.set_xlabel("Green AI Index (GAI)", fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_title(f"Green AI Index Leaderboard\n[{dataset_name}]", fontweight="bold")

        # Annotate total GAI and rank
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(
                row["gai_linear"] + 0.02, i,
                f"GAI: {row['gai_linear']:.3f} (Rank #{row['rank_gai']})",
                va="center", fontsize=9, fontweight="bold"
            )

        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        return save_path

    def plot_time_energy_breakdown(self, df: pd.DataFrame, dataset_name: str = "Benchmark", save_name: str = "time_energy_breakdown.png") -> str:
        """
        Grouped bar chart showing Training Time vs Prediction Time and Total Energy.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # Plot 1: Training vs Prediction Time (log scale if large range)
        x = np.arange(len(df))
        w = 0.35

        ax1.bar(x - w/2, df["training_time_s"], w, label="Training Time (s)", color="#e67e22")
        ax1.bar(x + w/2, df["prediction_time_s"], w, label="Prediction Time (s)", color="#9b59b6")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["model_name"], rotation=30, ha="right", fontweight="bold")
        ax1.set_ylabel("Execution Time (seconds)", fontweight="bold")
        ax1.set_title("Training vs. Prediction Latency", fontweight="bold")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Plot 2: Total Energy Consumption (Joules)
        colors = [MODEL_COLORS.get(m, "#34495e") for m in df["model_name"]]
        bars = ax2.bar(df["model_name"], df["energy_joules"], color=colors, alpha=0.85, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["model_name"], rotation=30, ha="right", fontweight="bold")
        ax2.set_ylabel("Energy Consumed (Joules)", fontweight="bold")
        ax2.set_title("Total Energy Consumption (Training + Inference)", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05 * h, f"{h:.2f} J", ha="center", va="bottom", fontsize=8)

        plt.suptitle(f"Resource Profiling Details [{dataset_name}]", fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        return save_path

    def plot_sensitivity_sweep(self, sweep_df: pd.DataFrame, dataset_name: str = "Benchmark", save_name: str = "sensitivity_analysis.png") -> str:
        """
        Plots GAI score as a function of Alpha (0.0 to 1.0) showing rank transitions.
        """
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        for m_name in sweep_df["model"].unique():
            subset = sweep_df[sweep_df["model"] == m_name]
            color = MODEL_COLORS.get(m_name, "#333333")
            ax.plot(subset["alpha"], subset["gai"], marker="o", markersize=3, label=m_name, color=color, linewidth=2)

        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Default Balanced (alpha = 0.5)")
        ax.set_xlabel(r"Trade-off Parameter $\alpha$ (0 = Pure Green, 1 = Pure Accuracy)", fontweight="bold")
        ax.set_ylabel("Green AI Index (GAI)", fontweight="bold")
        ax.set_title(f"Sensitivity Analysis: Ranking Robustness Across Preferences\n[{dataset_name}]", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        return save_path

    def plot_cross_dataset_comparison(self, cross_df: pd.DataFrame, save_name: str = "cross_dataset_comparison.png") -> str:
        """
        Grouped bar chart comparing GAI scores of all 6 algorithms across all benchmark datasets.
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Expected columns: dataset, model_name, gai_linear
        datasets = cross_df["dataset"].unique()
        models = [m for m in MODEL_COLORS.keys() if m in cross_df["model_name"].unique()]
        n_datasets = len(datasets)
        n_models = len(models)

        x = np.arange(n_datasets)
        total_width = 0.8
        bar_width = total_width / max(1, n_models)

        for i, m_name in enumerate(models):
            m_data = []
            for d in datasets:
                val = cross_df[(cross_df["dataset"] == d) & (cross_df["model_name"] == m_name)]["gai_linear"].values
                m_data.append(val[0] if len(val) > 0 else 0.0)
            offset = (i - n_models / 2.0 + 0.5) * bar_width
            color = MODEL_COLORS.get(m_name, "#333333")
            ax.bar(x + offset, m_data, width=bar_width, label=m_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontweight="bold", fontsize=11)
        ax.set_ylabel("Green AI Index (GAI, alpha=0.5)", fontweight="bold")
        ax.set_title("Cross-Dataset Model Sustainability & Performance Comparison", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        return save_path

    def generate_all_plots(self, df: pd.DataFrame, sweep_df: Optional[pd.DataFrame] = None, dataset_name: str = "Benchmark") -> Dict[str, str]:
        """Generates the full suite of figures."""
        paths = {
            "pareto": self.plot_pareto_frontier(df, dataset_name=dataset_name),
            "radar": self.plot_radar_chart(df, dataset_name=dataset_name),
            "leaderboard": self.plot_gai_leaderboard(df, dataset_name=dataset_name),
            "breakdown": self.plot_time_energy_breakdown(df, dataset_name=dataset_name)
        }
        if sweep_df is not None:
            paths["sensitivity"] = self.plot_sensitivity_sweep(sweep_df, dataset_name=dataset_name)
        return paths
