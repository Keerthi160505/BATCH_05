"""
Main Experiment Runner & CLI
Executes the empirical study on canonical benchmark datasets:
- Adult Census Income (UCI)
- Telco Customer Churn
- Breast Cancer Diagnostic (UCI)

Performs statistical multi-run evaluations, calculates Green AI Index (GAI),
and outputs publication-ready LaTeX tables, high-resolution figures,
machine-readable metadata, and Pareto summaries.
"""

import os
import sys
import json
import psutil
import platform
import datetime
import argparse
import pandas as pd
import numpy as np

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_pipeline import DynamicDataPipeline
from src.evaluator import GreenAIEvaluator
from src.visualizer import GreenAIVisualizer
from src.green_index import GreenAIIndexCalculator
from src.profiler import get_cpu_brand_name


DATASETS = {
    "adult_census": {
        "file": os.path.join(current_dir, "data", "adult_census.csv"),
        "target": "income",
        "description": "UCI Adult Census Income Benchmark"
    },
    "telco_churn": {
        "file": os.path.join(current_dir, "data", "telco_churn.csv"),
        "target": "Churn",
        "description": "Telco Customer Churn Benchmark"
    },
    "breast_cancer": {
        "file": os.path.join(current_dir, "data", "breast_cancer.csv"),
        "target": "target",
        "description": "UCI Breast Cancer Diagnostic Benchmark"
    }
}


def export_latex_table(df: pd.DataFrame, output_path: str, caption: str, label: str):
    """
    Exports a clean, formal IEEE/Springer style LaTeX table with booktabs.
    """
    cols_to_show = [
        "model_name", "accuracy", "precision", "recall", "f1_score",
        "cpu_avg_pct", "ram_peak_mb", "training_time_s", "energy_joules",
        "gai_linear", "rank_gai"
    ]
    sub_df = df[cols_to_show].copy()
    sub_df.columns = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "CPU (\\%)", "RAM (MB)", "Train Time (s)", "Energy (J)",
        "GAI", "Rank"
    ]

    latex_str = sub_df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        position="htbp"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"Exported LaTeX table to: {output_path}")


def run_single_dataset_benchmark(
    dataset_key: str,
    n_runs: int = 1,
    alpha: float = 0.5,
    output_dir: str = "results"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 75)
    print(f"RUNNING GREEN AI EVALUATION: {DATASETS[dataset_key]['description']}")
    print("=" * 75)

    csv_path = DATASETS[dataset_key]["file"]
    target_col = DATASETS[dataset_key]["target"]

    if not os.path.exists(csv_path):
        print(f"Dataset file missing: {csv_path}. Running downloader...")
        from data.download_datasets import prepare_adult_dataset, prepare_telco_churn_dataset, prepare_breast_cancer_dataset
        prepare_adult_dataset()
        prepare_telco_churn_dataset()
        prepare_breast_cancer_dataset()

    # 1. Pipeline: Clean, Encode, Scale, Split
    print("\n[Step 1/4] Running Dataset Preparation Pipeline...")
    pipeline = DynamicDataPipeline(target_column=target_col, scaler_type="standard", test_size=0.2, random_state=42)
    df_raw = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test, meta = pipeline.prepare(df_raw)

    print(f" -> Samples: Train={meta['train_samples']}, Test={meta['test_samples']}")
    print(f" -> Features: Original={meta['original_features']} -> Processed={meta['processed_features']}")
    print(f" -> Classes: {meta['classes']}, Scaler: {meta['scaler_applied']}")

    # 2. Evaluation & Resource Profiling
    print(f"\n[Step 2/4] Executing Resource Profiling & Model Evaluation ({n_runs} runs)...")
    evaluator = GreenAIEvaluator(alpha=alpha, use_codecarbon=True)

    if n_runs > 1:
        results_df, raw_runs = evaluator.evaluate_multi_run(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            n_runs=n_runs
        )
    else:
        results_df = evaluator.evaluate_all(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        raw_runs = results_df.copy()
        raw_runs["run_idx"] = 1
        raw_runs["seed"] = 42

    results_df["dataset"] = dataset_key
    raw_runs["dataset"] = dataset_key

    # 3. Sensitivity Analysis
    print("\n[Step 3/4] Performing Green AI Sensitivity Sweep (Alpha 0.0 -> 1.0)...")
    calc = GreenAIIndexCalculator(alpha=alpha)
    sweep_df = calc.sensitivity_analysis(results_df)

    # 4. Save Tables & Generate Plots
    print("\n[Step 4/4] Generating Publication Figures and Tables...")
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save CSVs
    summary_csv = os.path.join(tables_dir, f"{dataset_key}_summary.csv")
    results_df.to_csv(summary_csv, index=False)
    raw_csv = os.path.join(tables_dir, f"{dataset_key}_all_runs.csv")
    raw_runs.to_csv(raw_csv, index=False)

    # Save LaTeX table
    latex_path = os.path.join(tables_dir, f"{dataset_key}_table.tex")
    export_latex_table(
        results_df, latex_path,
        caption=f"Green AI Evaluation Results for {DATASETS[dataset_key]['description']}",
        label=f"tab:{dataset_key}_results"
    )

    # Generate Figures
    visualizer = GreenAIVisualizer(output_dir=figures_dir)
    fig_paths = visualizer.generate_all_plots(
        results_df, sweep_df=sweep_df,
        dataset_name=DATASETS[dataset_key]["description"]
    )

    # Print Formatted Terminal Summary
    print("\n" + "-" * 75)
    print(f"RESULTS SUMMARY: {dataset_key.upper()} (Alpha = {alpha})")
    print("-" * 75)
    summary_cols = ["rank_gai", "model_name", "accuracy", "f1_score", "energy_joules", "training_time_s", "gai_linear", "pareto_optimal"]
    display_df = results_df[summary_cols].copy()
    display_df.columns = ["Rank", "Model", "Accuracy", "F1", "Energy (J)", "Train (s)", "GAI", "Pareto"]
    print(display_df.to_string(index=False))
    print("-" * 75)

    return results_df, raw_runs


def save_experiment_metadata(output_dir: str, all_results: pd.DataFrame, alpha: float, n_runs: int):
    """Saves comprehensive machine-readable metadata and hardware specifications."""
    metadata = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": get_cpu_brand_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "python_version": sys.version
        },
        "experiment_configuration": {
            "alpha": alpha,
            "runs_per_model": n_runs,
            "datasets_evaluated": list(DATASETS.keys()),
            "models_evaluated": list(all_results["model_name"].unique()),
            "total_benchmark_runs": len(all_results),
            "profiling_methodology": "Asynchronous 20ms Riemann numerical integration of TDP-calibrated power model (CPU + DRAM)",
            "gpu_scope": "Unsupported and not claimed; CPU-only profiling"
        }
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved machine-readable experiment metadata to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic Green AI Evaluation Framework CLI")
    parser.add_argument("--dataset", type=str, default="all", choices=list(DATASETS.keys()) + ["all"],
                        help="Benchmark dataset to evaluate (default: all)")
    parser.add_argument("--runs", type=int, default=1, help="Number of experimental runs for statistical averaging (default: 1)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Green AI tradeoff weight (default: 0.5)")
    parser.add_argument("--output", type=str, default=os.path.join(current_dir, "results"), help="Results output folder")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    tables_dir = os.path.join(args.output, "tables")
    figures_dir = os.path.join(args.output, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    log_file = os.path.join(args.output, "execution.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Green AI Benchmark Execution: {datetime.datetime.now().isoformat()} ---\n")
        f.write(f"Args: dataset={args.dataset}, runs={args.runs}, alpha={args.alpha}, output={args.output}\n")

    datasets_to_run = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    all_summaries = []
    all_raw_runs = []

    for d in datasets_to_run:
        res_df, raw_df = run_single_dataset_benchmark(d, n_runs=args.runs, alpha=args.alpha, output_dir=args.output)
        all_summaries.append(res_df)
        all_raw_runs.append(raw_df)

    # Combine into aggregated summary across all datasets
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        agg_csv = os.path.join(tables_dir, "aggregated_summary.csv")
        combined_df.to_csv(agg_csv, index=False)
        print(f"\nSaved combined cross-dataset summary to: {agg_csv}")

        # Save Pareto optimality summary table
        pareto_df = combined_df[["dataset", "model_name", "f1_score", "energy_joules", "gai_linear", "pareto_optimal", "rank_gai"]].copy()
        pareto_csv = os.path.join(tables_dir, "pareto_summary.csv")
        pareto_df.to_csv(pareto_csv, index=False)
        print(f"Saved Pareto frontier summary to: {pareto_csv}")

        # Save machine-readable metadata
        save_experiment_metadata(args.output, combined_df, alpha=args.alpha, n_runs=args.runs)

        # Generate Cross-Dataset Comparison Plot if multiple datasets
        if len(datasets_to_run) > 1:
            viz = GreenAIVisualizer(output_dir=figures_dir)
            cross_fig_path = viz.plot_cross_dataset_comparison(combined_df)
            print(f"Generated cross-dataset visualization at: {cross_fig_path}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Execution completed successfully. Total models evaluated: {len(combined_df)}\n")


if __name__ == "__main__":
    main()
