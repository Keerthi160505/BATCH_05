"""
Scientific Visualization Script: Experimental Results Charts
Reads actual experimental CSV summaries from results/tables/:
- results/tables/adult_census_summary.csv
- results/tables/telco_churn_summary.csv
- results/tables/breast_cancer_summary.csv
- results/tables/pareto_summary.csv
- results/tables/aggregated_summary.csv

Generates 10 publication-quality experimental charts:
1. Accuracy comparison across models
2. F1-score comparison across models
3. Energy consumption comparison (Training vs Inference Joules + TDP disclaimer)
4. CPU utilization comparison (Average vs Peak CPU %)
5. RAM consumption comparison (Peak RAM vs Delta RAM MB)
6. Training time and inference latency comparison
7. Green AI Index ranking breakdown (Performance vs Sustainability contributions)
8. Pareto frontier of F1-score versus energy consumption
9. Cross-dataset performance and sustainability comparison
10. Multi-criteria Radar chart of normalized model metrics

Outputs:
- 300 DPI publication PNGs in documentation/figures/results/
- Scalable Vector Graphics (SVG) in documentation/figures/results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "documentation", "figures", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication styling defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13.5
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# Palette definitions
MODEL_PALETTE = {
    "Decision Tree": "#059669",      # Emerald Green
    "LightGBM": "#0d9488",           # Teal
    "Logistic Regression": "#2563eb",# Royal Blue
    "XGBoost": "#d97706",            # Amber
    "Random Forest": "#7c3aed",      # Violet
    "CatBoost": "#dc2626"            # Red
}

DISCLAIMER_TEXT = "Note: Energy modeled via TDP-calibrated empirical numerical integration (CPU + DRAM); not direct hardware shunt measurement."


def load_datasets():
    """Loads actual benchmark data from CSV tables."""
    adult_df = pd.read_csv(os.path.join(RESULTS_TABLES, "adult_census_summary.csv"))
    telco_df = pd.read_csv(os.path.join(RESULTS_TABLES, "telco_churn_summary.csv"))
    cancer_df = pd.read_csv(os.path.join(RESULTS_TABLES, "breast_cancer_summary.csv"))
    pareto_df = pd.read_csv(os.path.join(RESULTS_TABLES, "pareto_summary.csv"))
    return adult_df, telco_df, cancer_df, pareto_df


# =========================================================================
# CHART 1: ACCURACY COMPARISON
# =========================================================================
def plot_accuracy_comparison(df):
    df_sorted = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    colors = [MODEL_PALETTE.get(m, "#475569") for m in df_sorted["model_name"]]
    
    bars = ax.bar(df_sorted["model_name"], df_sorted["accuracy"] * 100.0, color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.9)
    
    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_ylim(80, 92)
    ax.set_title("Predictive Accuracy Comparison Across Algorithms\n[UCI Adult Census Income Benchmark (N=25,000, Test=5,000)]", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color="#0f172a")
        
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["model_name"], rotation=18, ha="right", fontweight="bold")
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    svg_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: accuracy_comparison")


# =========================================================================
# CHART 2: F1-SCORE COMPARISON
# =========================================================================
def plot_f1_comparison(df):
    df_sorted = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    colors = [MODEL_PALETTE.get(m, "#475569") for m in df_sorted["model_name"]]
    
    bars = ax.bar(df_sorted["model_name"], df_sorted["f1_score"], color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.9)
    
    ax.set_ylabel("F1-Score (Weighted Macro-Average)", fontweight="bold")
    ax.set_ylim(0.80, 0.91)
    ax.set_title("F1-Score Comparison Across Algorithms\n[UCI Adult Census Income Benchmark (N=25,000)]", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        rank = df_sorted.loc[i, "rank_f1"]
        ax.annotate(f"{h:.4f}\n(Rank #{rank})",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#0f172a")
        
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["model_name"], rotation=18, ha="right", fontweight="bold")
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "f1_comparison.png")
    svg_path = os.path.join(OUTPUT_DIR, "f1_comparison.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: f1_comparison")


# =========================================================================
# CHART 3: ENERGY CONSUMPTION COMPARISON
# =========================================================================
def plot_energy_comparison(df):
    df_sorted = df.sort_values("energy_joules", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=300)
    
    x = np.arange(len(df_sorted))
    w = 0.55
    
    train_bars = ax.bar(x, df_sorted["train_energy_joules"], w, label="Training Energy (Joules)",
                        color="#059669", edgecolor="#064e3b", linewidth=0.8, alpha=0.85)
    pred_bars = ax.bar(x, df_sorted["pred_energy_joules"], w, bottom=df_sorted["train_energy_joules"],
                       label="Inference Energy (Joules)", color="#14b8a6", edgecolor="#0f766e", linewidth=0.8, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["model_name"], rotation=18, ha="right", fontweight="bold")
    ax.set_ylabel("Energy Consumption (Joules)", fontweight="bold")
    ax.set_title("Empirical Energy Consumption Breakdown (Training vs Inference)\n[UCI Adult Census Income Benchmark]", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        total_j = row["energy_joules"]
        co2_mg = row["co2_emissions_g"] * 1000.0
        ax.annotate(f"{total_j:.2f} J\n({co2_mg:.1f} mg CO₂)",
                    xy=(i, total_j),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.2, fontweight='bold', color="#0f172a")
        
    # Energy disclaimer footnote
    ax.text(0.5, -0.22, DISCLAIMER_TEXT, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.8, fontstyle="italic", color="#64748b")
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "energy_comparison.png")
    svg_path = os.path.join(OUTPUT_DIR, "energy_comparison.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: energy_comparison")


# =========================================================================
# CHART 4: CPU UTILIZATION COMPARISON
# =========================================================================
def plot_cpu_comparison(df):
    df_sorted = df.sort_values("cpu_avg_pct", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)
    
    x = np.arange(len(df_sorted))
    w = 0.35
    
    bars_avg = ax.bar(x - w/2, df_sorted["cpu_avg_pct"], w, label="Average CPU Utilization (%)",
                      color="#0d9488", edgecolor="#0f766e", linewidth=0.8, alpha=0.85)
    bars_peak = ax.bar(x + w/2, df_sorted["cpu_peak_pct"], w, label="Peak CPU Utilization (%)",
                       color="#3b82f6", edgecolor="#1d4ed8", linewidth=0.8, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["model_name"], rotation=18, ha="right", fontweight="bold")
    ax.set_ylabel("Processor Utilization (%)", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title("System & Process CPU Utilization (Average vs Peak)\n[High-Frequency Telemetry, 20ms Polling Interval]", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    for bar in bars_avg:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars_peak:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#1e40af")
        
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "cpu_utilization_comparison.png")
    svg_path = os.path.join(OUTPUT_DIR, "cpu_utilization_comparison.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: cpu_utilization_comparison")


# =========================================================================
# CHART 5: RAM CONSUMPTION COMPARISON
# =========================================================================
def plot_ram_comparison(df):
    df_sorted = df.sort_values("ram_peak_mb", ascending=True).reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300)
    
    # Subplot 1: Peak RAM
    colors = [MODEL_PALETTE.get(m, "#475569") for m in df_sorted["model_name"]]
    bars1 = ax1.bar(df_sorted["model_name"], df_sorted["ram_peak_mb"], color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.85)
    ax1.set_ylabel("Peak Process RAM Footprint (MB)", fontweight="bold")
    ax1.set_ylim(250, 315)
    ax1.set_title("Peak Process Memory (RSS)", fontweight="bold")
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted["model_name"], rotation=25, ha="right", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f} MB", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    # Subplot 2: Delta RAM
    df_delta = df.sort_values("ram_delta_mb", ascending=True).reset_index(drop=True)
    colors_delta = [MODEL_PALETTE.get(m, "#475569") for m in df_delta["model_name"]]
    bars2 = ax2.bar(df_delta["model_name"], df_delta["ram_delta_mb"], color=colors_delta, width=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.85)
    ax2.set_ylabel("Non-Negative Delta RAM Δ (MB)", fontweight="bold")
    ax2.set_ylim(0, 26)
    ax2.set_title("Differential Allocation Δ RAM (Peak - Baseline)", fontweight="bold")
    ax2.set_xticks(range(len(df_delta)))
    ax2.set_xticklabels(df_delta["model_name"], rotation=25, ha="right", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f} MB", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    plt.suptitle("Computational Memory Footprint Profiling Across Machine Learning Algorithms", fontweight="bold", y=1.01)
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "ram_consumption_comparison.png")
    svg_path = os.path.join(OUTPUT_DIR, "ram_consumption_comparison.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: ram_consumption_comparison")


# =========================================================================
# CHART 6: TRAINING TIME AND INFERENCE LATENCY COMPARISON
# =========================================================================
def plot_time_latency_comparison(df):
    df_sorted = df.sort_values("training_time_s", ascending=True).reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300)
    
    # Subplot 1: Training Time
    colors = [MODEL_PALETTE.get(m, "#475569") for m in df_sorted["model_name"]]
    bars1 = ax1.bar(df_sorted["model_name"], df_sorted["training_time_s"], color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.85)
    ax1.set_ylabel("Training Duration (seconds)", fontweight="bold")
    ax1.set_title("Training Execution Duration", fontweight="bold")
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted["model_name"], rotation=25, ha="right", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}s", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    # Subplot 2: Inference Latency per 1k samples
    df_lat = df.sort_values("latency_per_1k_samples_s", ascending=True).reset_index(drop=True)
    colors_lat = [MODEL_PALETTE.get(m, "#475569") for m in df_lat["model_name"]]
    bars2 = ax2.bar(df_lat["model_name"], df_lat["latency_per_1k_samples_s"] * 1000.0, color=colors_lat, width=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.85)
    ax2.set_ylabel("Inference Latency per 1,000 Samples (ms)", fontweight="bold")
    ax2.set_title("Inference Latency (Milliseconds per 1k)", fontweight="bold")
    ax2.set_xticks(range(len(df_lat)))
    ax2.set_xticklabels(df_lat["model_name"], rotation=25, ha="right", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f} ms", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    plt.suptitle("Computational Time & Latency Benchmarks (Adult Census, 20,000 Train / 5,000 Test)", fontweight="bold", y=1.01)
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "training_inference_latency.png")
    svg_path = os.path.join(OUTPUT_DIR, "training_inference_latency.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: training_inference_latency")


# =========================================================================
# CHART 7: GREEN AI INDEX RANKING
# =========================================================================
def plot_gai_ranking(df):
    df_sorted = df.sort_values("gai_linear", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=300)
    
    y = np.arange(len(df_sorted))
    h = 0.55
    
    perf_contrib = 0.5 * df_sorted["performance_score"]
    sust_contrib = 0.5 * df_sorted["resource_efficiency"]
    
    ax.barh(y, perf_contrib, h, label="Predictive Performance (w = 0.50)", color="#059669", edgecolor="#064e3b", linewidth=0.8, alpha=0.9)
    ax.barh(y, sust_contrib, h, left=perf_contrib, label="Resource Sustainability (w = 0.50)", color="#0d9488", edgecolor="#0f766e", linewidth=0.8, alpha=0.9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["model_name"], fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Green AI Index (GAI, alpha = 0.50) [0.0 = Red AI, 1.0 = Ideal Green AI]", fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_title("Green AI Index (GAI) Model Ranking & Component Synthesis\n[UCI Adult Census Income Benchmark]", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        gai_lin = row["gai_linear"]
        gai_harm = row["gai_harmonic"]
        rank = row["rank_gai"]
        ax.annotate(f"GAI: {gai_lin:.4f} (Rank #{rank})\nHarmonic: {gai_harm:.4f}",
                    xy=(gai_lin + 0.015, i),
                    va="center", fontsize=8.5, fontweight="bold", color="#0f172a")
        
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "gai_ranking.png")
    svg_path = os.path.join(OUTPUT_DIR, "gai_ranking.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: gai_ranking")


# =========================================================================
# CHART 8: PARETO FRONTIER (F1-SCORE VS ENERGY)
# =========================================================================
def plot_pareto_frontier(df, pareto_df):
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=300)
    
    # Merge pareto_optimal column
    merged = df.copy()
    if "pareto_optimal" not in merged.columns and "pareto_optimal" in pareto_df.columns:
        p_map = dict(zip(pareto_df["model_name"], pareto_df["pareto_optimal"]))
        merged["pareto_optimal"] = merged["model_name"].map(p_map).fillna(False)

    pareto_pts = merged[merged["pareto_optimal"] == True].sort_values("energy_joules")
    non_pareto_pts = merged[merged["pareto_optimal"] == False]
    
    # Draw convex Pareto frontier line
    if len(pareto_pts) > 1:
        ax.plot(pareto_pts["energy_joules"], pareto_pts["f1_score"],
                linestyle="--", color="#059669", linewidth=2.2, alpha=0.85,
                label="Empirical Pareto Frontier (Non-Dominated)", zorder=2)
    
    # Plot Dominated Models
    for _, row in non_pareto_pts.iterrows():
        m = row["model_name"]
        color = MODEL_PALETTE.get(m, "#64748b")
        ax.scatter(row["energy_joules"], row["f1_score"],
                   color=color, s=130, marker="o", edgecolors="#1e293b", linewidth=1.2, alpha=0.85, zorder=3)
        ax.annotate(f" {m}\n (GAI: {row['gai_linear']:.3f})",
                    xy=(row["energy_joules"], row["f1_score"]),
                    xytext=(6, -4), textcoords="offset points",
                    fontsize=8.5, fontweight="normal", color="#334155")
        
    # Plot Pareto-Optimal Models (Stars)
    for _, row in pareto_pts.iterrows():
        m = row["model_name"]
        color = MODEL_PALETTE.get(m, "#059669")
        ax.scatter(row["energy_joules"], row["f1_score"],
                   color=color, s=260, marker="*", edgecolors="#064e3b", linewidth=1.4, alpha=1.0, zorder=4,
                   label=f"{m} (Pareto Optimal *)")
        ax.annotate(f" {m} *\n F1: {row['f1_score']:.4f}\n Energy: {row['energy_joules']:.2f} J",
                    xy=(row["energy_joules"], row["f1_score"]),
                    xytext=(8, -14 if m == "XGBoost" else 8), textcoords="offset points",
                    fontsize=9.0, fontweight="bold", color="#064e3b")

    ax.set_xlabel("Total Energy Consumption (Joules) [Lower is Greener]", fontweight="bold")
    ax.set_ylabel("Predictive Performance (F1-Score) [Higher is Better]", fontweight="bold")
    ax.set_title("Pareto Frontier: Predictive Utility vs. Computational Energy Footprint\n[UCI Adult Census Income Benchmark]", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    
    # Footnote
    ax.text(0.5, -0.15, DISCLAIMER_TEXT, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.8, fontstyle="italic", color="#64748b")
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "pareto_frontier.png")
    svg_path = os.path.join(OUTPUT_DIR, "pareto_frontier.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: pareto_frontier")


# =========================================================================
# CHART 9: CROSS-DATASET PERFORMANCE AND SUSTAINABILITY COMPARISON
# =========================================================================
def plot_cross_dataset_comparison(adult_df, telco_df, cancer_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5), dpi=300)
    
    datasets = ["Adult Census\n(N=25k)", "Telco Churn\n(N=7k)", "Breast Cancer\n(N=569)"]
    dfs = [adult_df, telco_df, cancer_df]
    models = list(MODEL_PALETTE.keys())
    
    x = np.arange(len(datasets))
    w = 0.13
    
    # Subplot 1: F1-Score across datasets
    for i, m in enumerate(models):
        f1_vals = []
        for d in dfs:
            row = d[d["model_name"] == m]
            f1_vals.append(row["f1_score"].values[0] if not row.empty else 0.0)
        offset = (i - len(models)/2.0 + 0.5) * w
        ax1.bar(x + offset, f1_vals, width=w, label=m, color=MODEL_PALETTE[m], edgecolor="#1e293b", linewidth=0.6, alpha=0.9)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight="bold")
    ax1.set_ylabel("F1-Score (Predictive Utility)", fontweight="bold")
    ax1.set_ylim(0.65, 1.05)
    ax1.set_title("Cross-Dataset Predictive Performance (F1-Score)", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Subplot 2: Green AI Index across datasets
    for i, m in enumerate(models):
        gai_vals = []
        for d in dfs:
            row = d[d["model_name"] == m]
            gai_vals.append(row["gai_linear"].values[0] if not row.empty else 0.0)
        offset = (i - len(models)/2.0 + 0.5) * w
        ax2.bar(x + offset, gai_vals, width=w, label=m, color=MODEL_PALETTE[m], edgecolor="#1e293b", linewidth=0.6, alpha=0.9)
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontweight="bold")
    ax2.set_ylabel("Green AI Index (GAI, alpha = 0.50)", fontweight="bold")
    ax2.set_ylim(0.40, 1.05)
    ax2.set_title("Cross-Dataset Green AI Index (GAI Ranking)", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.suptitle("Generalization & Sustainability Comparison Across Three Benchmark Workloads", fontweight="bold", y=1.01)
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "cross_dataset_comparison.png")
    svg_path = os.path.join(OUTPUT_DIR, "cross_dataset_comparison.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: cross_dataset_comparison")


# =========================================================================
# CHART 10: MULTI-CRITERIA RADAR / SPIDER CHART
# =========================================================================
def plot_radar_chart(df):
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
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True), dpi=300)
    
    # Inverted min-max for efficiency (lower cost = higher efficiency in [0.2, 1.0])
    def norm_inv(s):
        denom = (s.max() - s.min()) if (s.max() - s.min()) > 1e-9 else 1.0
        return 0.2 + 0.8 * (1.0 - ((s - s.min()) / denom))

    def norm_direct(s):
        denom = (s.max() - s.min()) if (s.max() - s.min()) > 1e-9 else 1.0
        return 0.2 + 0.8 * ((s - s.min()) / denom)

    norm_acc = norm_direct(df["accuracy"])
    norm_f1 = norm_direct(df["f1_score"])
    eff_cpu = norm_inv(df["cpu_avg_pct"])
    eff_ram = norm_inv(df["ram_peak_mb"])
    eff_time = norm_inv(df["total_time_s"])
    eff_energy = norm_inv(df["energy_joules"])

    for idx, row in df.iterrows():
        m_name = row["model_name"]
        color = MODEL_PALETTE.get(m_name, "#333333")
        values = [
            norm_acc.iloc[idx],
            norm_f1.iloc[idx],
            eff_cpu.iloc[idx],
            eff_ram.iloc[idx],
            eff_time.iloc[idx],
            eff_energy.iloc[idx]
        ]
        values += values[:1]
        
        ax.plot(angles, values, color=color, linewidth=2.0, label=m_name)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight="bold", fontsize=10.5)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.08)
    ax.set_title("Multi-Criteria Normalized Green AI Profile\n[Predictive Utility vs Hardware Efficiency Dimensions]",
                 fontweight="bold", pad=24)
    ax.legend(bbox_to_anchor=(1.18, 1.08), loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "radar_chart.png")
    svg_path = os.path.join(OUTPUT_DIR, "radar_chart.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print("Generated: radar_chart")


if __name__ == "__main__":
    print("Loading actual benchmark result tables...")
    adult_df, telco_df, cancer_df, pareto_df = load_datasets()
    
    print("Generating 10 publication-quality experimental charts...")
    plot_accuracy_comparison(adult_df)
    plot_f1_comparison(adult_df)
    plot_energy_comparison(adult_df)
    plot_cpu_comparison(adult_df)
    plot_ram_comparison(adult_df)
    plot_time_latency_comparison(adult_df)
    plot_gai_ranking(adult_df)
    plot_pareto_frontier(adult_df, pareto_df)
    plot_cross_dataset_comparison(adult_df, telco_df, cancer_df)
    plot_radar_chart(adult_df)
    print("All 10 experimental charts generated successfully!")
