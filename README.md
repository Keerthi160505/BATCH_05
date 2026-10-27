# Dynamic Green AI Evaluation Framework 🌿

An automated, dataset-independent empirical benchmarking framework that evaluates machine learning models on both **predictive performance** and **computational sustainability** (CPU utilization, RAM footprint, latency, energy draw in Joules, and carbon emissions in $\text{gCO}_2\text{eq}$).

Developed as a publishable final-year undergraduate research capstone project targeting peer-reviewed venues (IEEE / Springer / ACM).

---

## 📌 Executive Summary & Abstract
Machine learning models are traditionally evaluated exclusively using predictive metrics: **Accuracy, Precision, Recall, and F1-score**. However, these measures do not reveal how much CPU, memory, time, or energy a model consumes.

The **Dynamic Green AI Evaluation Framework** addresses this critical limitation by:
1. **Automating Data Preparation**: Cleans, imputes missing values, one-hot encodes categoricals, standardizes features, and creates stratified train-test splits without statistical data leakage.
2. **Evaluating Six Canonical Models**:
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
   - CatBoost
   - Logistic Regression
3. **Profiling Comprehensive Resource Footprints**:
   - Accuracy, Precision, Recall, F1-Score (weighted average, zero-division safe)
   - Process & System CPU Usage (%)
   - Baseline RAM, Peak RAM, and non-negative Differential $\Delta$ RAM (MB)
   - Training & Inference Latency (seconds and latency per 1k samples)
   - Energy Draw (Joules and kWh) via Thermal Design Power (TDP) integration & CodeCarbon
   - Operational Carbon Footprint ($\text{gCO}_2\text{eq}$)
4. **Formulating the Green AI Index (GAI)**:
   - Harmonizes predictive utility and computational efficiency into a unified, mathematically rigorous index.
   - Computes Pareto-optimal frontiers and performs $\alpha$-preference sensitivity sweeps.
   - Supports both Linear and Harmonic GAI formulations with configurable sub-weights.
5. **Interactive Dashboard & LaTeX Export**:
   - Streamlit interactive GUI with dynamic weight sliders.
   - One-click export of publication-ready IEEE/Springer LaTeX tables and 300 DPI vector figures.

---

## 📂 Repository Architecture

```text
dynamic-green-ai-framework/
├── .venv/                        # Local virtual environment
├── data/
│   ├── adult_census.csv          # UCI Adult Census Income benchmark (25k rows)
│   ├── telco_churn.csv           # Telco Customer Churn benchmark (7k rows)
│   ├── breast_cancer.csv         # Diagnostic benchmark (UCI/sklearn, 569 rows)
│   └── download_datasets.py      # Automated dataset downloader & offline generator
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Automated cleaning, encoding, scaling, splitting (zero leakage)
│   ├── profiler.py               # Asynchronous CPU, RAM, Time, Energy & CO2 profiler
│   ├── models.py                 # Registry for the 6 canonical algorithms
│   ├── green_index.py            # Mathematical Green AI Index & Pareto engine
│   ├── evaluator.py              # Statistical multi-run benchmarking orchestrator
│   └── visualizer.py             # 300 DPI publication plots (Pareto, Radar, Leaderboard, Cross-Dataset)
├── app/
│   └── dashboard.py              # Interactive Streamlit Web Application
├── paper/
│   ├── research_paper.tex        # Full IEEE conference/journal LaTeX paper
│   └── research_paper.md         # Full Markdown research paper draft
├── results/
│   ├── figures/                  # Publication-ready charts (300 DPI PNG)
│   ├── tables/                   # Experimental CSV summaries, LaTeX tables, Pareto summary
│   ├── metadata.json             # Machine-readable experiment & hardware metadata
│   └── execution.log             # Benchmark execution log
├── tests/
│   ├── test_pipeline.py          # Data pipeline unit tests (leakage, scalers, invalid inputs)
│   ├── test_profiler.py          # Hardware profiler unit tests (TDP model, non-negative RAM delta)
│   ├── test_models.py            # 6-model binary and multiclass unit tests
│   └── test_green_index.py       # GAI formulation, Pareto rules, and alpha sweep unit tests
├── run_experiments.py            # CLI entrypoint for running empirical experiments
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Requirements & Virtual Environment
Ensure Python 3.10+ is installed. This project uses a dedicated virtual environment at `.venv`:

```powershell
# Navigate to project directory
cd "C:\Users\Administrator\.gemini\antigravity\scratch\dynamic-green-ai-framework"

# Install dependencies into .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Verify Unit Tests
Execute the comprehensive test suite (15 tests covering all pipeline, profiler, model, and index logic):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 🔬 Running Empirical Experiments (CLI)

Run the full benchmark across all three canonical datasets (`adult_census`, `telco_churn`, `breast_cancer`):

```powershell
.\.venv\Scripts\python.exe run_experiments.py --dataset all --runs 1 --alpha 0.5
```

### CLI Arguments:
- `--dataset`: `adult_census`, `telco_churn`, `breast_cancer`, or `all` (default: `all`)
- `--runs`: Number of experimental runs for statistical averaging (default: `1`)
- `--alpha`: Trade-off parameter between Performance ($\alpha$) and Sustainability ($1 - \alpha$) (default: `0.5`)
- `--output`: Directory to store generated tables and figures (default: `results`)

Outputs are automatically saved to:
- `results/tables/adult_census_table.tex`: Ready-to-compile IEEE LaTeX table.
- `results/tables/telco_churn_table.tex`: Ready-to-compile IEEE LaTeX table.
- `results/tables/breast_cancer_table.tex`: Ready-to-compile IEEE LaTeX table.
- `results/tables/aggregated_summary.csv`: Aggregated cross-dataset summary.
- `results/tables/pareto_summary.csv`: Pareto frontier optimality classifications.
- `results/metadata.json`: Machine-readable hardware and experiment metadata.
- `results/figures/pareto_frontier.png`: Pareto frontier plot (Performance vs Energy).
- `results/figures/radar_chart.png`: Multi-criteria Spider/Radar chart.
- `results/figures/gai_leaderboard.png`: Green AI Index ranking breakdown.
- `results/figures/cross_dataset_comparison.png`: Cross-dataset GAI comparison.
- `results/figures/sensitivity_analysis.png`: Parameter $\alpha$-sweep curves.

---

## 🌐 Launching the Interactive Web Dashboard

Launch the interactive Streamlit application to explore real-time trade-offs and upload custom datasets:

**Method A (Direct Command):**
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

**Method B (One-Click / One-Command):**
- In PowerShell: `.\run_app.ps1`
- Or in CMD / File Explorer: double-click `run_app.bat`

Features inside the dashboard:
- **Interactive $\alpha$ slider**: Dynamically observe how model rankings re-order when prioritizing accuracy vs. energy.
- **Custom CSV upload**: Drop any tabular CSV file and evaluate the 6 models immediately.
- **Model selector & Feature Scaler**: Select specific algorithms and scalers (`standard`, `minmax`, `robust`, `none`).
- **Sub-weight tuning**: Adjust relative importance of CPU, RAM, Latency, and Energy.
- **Instant LaTeX & CSV Download**: One-click export for paper writing.

---

## 📊 Benchmark Findings Summary

### UCI Adult Census Income ($N=25,000$, $\alpha=0.50$)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Avg CPU (%) | Peak RAM (MB) | Train Time (s) | Energy (Joules) | GAI ($\alpha=0.5$) | Pareto Optimal |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Decision Tree** | 0.8596 | 0.8570 | 0.8596 | 0.8581 | 14.51% | 297.80 | 0.2096 | **4.8366** | **0.9261** | **Yes** |
| **2** | **LightGBM** | 0.8782 | 0.8743 | 0.8782 | 0.8751 | 56.40% | 299.10 | 0.1977 | **11.3034** | **0.8512** | **Yes** |
| **3** | **Logistic Regression** | 0.8568 | 0.8509 | 0.8568 | 0.8520 | 28.12% | 302.40 | 0.3672 | **8.5443** | 0.8456 | No |
| **4** | **XGBoost** | **0.8792** | **0.8753** | **0.8792** | **0.8759** | 62.40% | 300.50 | 0.4982 | 28.6686 | 0.7243 | **Yes** |
| **5** | **Random Forest** | 0.8688 | 0.8638 | 0.8688 | 0.8615 | 51.30% | 315.80 | 0.7765 | 30.1454 | 0.5370 | No |
| **6** | **CatBoost** | 0.8710 | 0.8661 | 0.8710 | 0.8664 | 42.10% | 319.20 | 1.1561 | 40.4047 | 0.5368 | No |

### Key Academic Insights:
1. **The Diminishing Returns of Red AI**: XGBoost consumed **$5.93\times$ more energy** and CatBoost consumed **$8.35\times$ more energy** than Decision Tree for a marginal gain of **$<1.8\%$** in F1-score.
2. **Eco-Efficiency Champion**: **LightGBM** achieved $99.9\%$ of XGBoost's F1-score while utilizing **$60.6\%$ less energy** and completing in less than half the training time.
3. **Pareto Dominance**: Only Decision Tree, LightGBM, and XGBoost occupy the non-dominated Pareto frontier on Adult Census.

---

## ⚙️ Methodology & Limitations

1. **Hardware Power Model**: Power is estimated using dynamic CPU utilization scaling against Thermal Design Power (TDP) coupled with dynamic DRAM power modeling:
   $$P(t) = P_{idle} + \left(\frac{U_{cpu}(t)}{100}\right) \cdot (P_{tdp} - P_{idle}) + P_{dram}$$
   Energy in Joules is computed via numerical Riemann integration over execution duration:
   $$E = \int_0^T P(t) dt \approx \sum P(t_k) \Delta t_k$$
2. **CPU-Only Scope**: The framework profiles CPU tabular workloads. GPU accelerators are neither utilized nor claimed.
3. **Estimation vs. Direct Measurement**: Energy values reflect empirical modeling and should not be construed as physical shunt resistor / wattmeter measurements.
4. **Carbon Intensity**: Estimated using a global average grid factor of $475\text{ gCO}_2\text{eq/kWh}$.
5. **Online Retail II & Engineered Target Limitations (`IsCancellation`)**:
   - **Target Provenance**: The binary target `IsCancellation` is an *engineered* task derived from the transaction invoice code prefix (invoices prefixed with `'C'` denote cancellations). It is **not** an original, ground-truth label recorded during prospective e-commerce operations.
   - **Direct Label Leakage Elimination**: Any field containing invoice numbers (`Invoice`, `InvoiceNo`) directly encodes the cancellation prefix `'C'`. The pipeline automatically identifies and drops invoice fields to prevent direct target leakage.
   - **Indirect Target Leakage (`Quantity` & `Price`)**: In the Online Retail II ledger, cancellations were logged post-hoc using negative item quantities (e.g., $Q < 0$). These negative quantities serve as an indirect retrospective proxy for cancellation rather than predictive precursor signals. Evaluators should recognize that unadjusted negative quantities yield artificially inflated predictive scores.
   - **High-Cardinality Scaling & Memory Safety**: Online Retail II contains approximately 1 million transaction records and high-cardinality features (thousands of `StockCode`, `Description`, and `CustomerID` values). Without protection, one-hot encoding creates $>56,000$ columns, attempting a catastrophic $\approx 360\text{ GiB}$ dense allocation. The framework safeguards execution via:
     - Automatic entity ID dropping (`CustomerID`, transaction indices) with explicit reason logging.
     - Infrequent category grouping into `'Other'` bounded by `max_categories_onehot`.
     - Reproducible stratified sampling ($10,000$, $25,000$, $50,000$, or Full).
     - Memory safety estimation checking budget headroom before encoding and model training.

---

## 📄 Academic Research Paper
- `paper/research_paper.tex`: Standard IEEE conference formatting ready for compilation.
- `paper/research_paper.md`: Complete Markdown copy with formulas and citations.

---

## 📜 Citation
```bibtex
@inproceedings{greenai2026dynamic,
  title={A Dynamic Green AI Evaluation Framework: Multi-Criteria Trade-Off Analysis of Predictive Performance and Computational Sustainability in Machine Learning},
  author={Anonymous},
  booktitle={Proceedings of the Sustainable Computing and Green AI Conference},
  year={2026}
}
```
