"""
Scientific Visualization Script: Architecture & Workflow Diagrams
Generates 7 publication-quality technical diagrams for B.Tech Major Project Thesis:
1. Overall System Architecture
2. Data Preprocessing Workflow
3. Machine Learning Model Evaluation Workflow
4. Computational Resource Profiling Workflow
5. Green AI Index Calculation Workflow
6. Pareto Frontier Analysis Workflow
7. Complete End-to-End Project Data Flow

Outputs:
- 300 DPI publication PNGs
- Scalable Vector Graphics (SVG)
- Editable Mermaid (.mmd) source files
Color Palette: Restrained Academic Green, Teal, Slate/Gray, and White
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import matplotlib.patheffects as patheffects

# Setup output directory
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentation", "figures", "architecture"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication styling defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette Tokens
C_DARK_GREEN = "#064e3b"   # Headers / Dark accents
C_PRIMARY_GREEN = "#059669" # Main highlights
C_LIGHT_GREEN = "#ecfdf5"   # Card background (Green)
C_BORDER_GREEN = "#a7f3d0"  # Card border (Green)

C_DARK_TEAL = "#0f766e"    # Teal Headers
C_PRIMARY_TEAL = "#0d9488"  # Teal highlights
C_LIGHT_TEAL = "#f0fdfa"    # Card background (Teal)
C_BORDER_TEAL = "#99f6e4"   # Card border (Teal)

C_DARK_SLATE = "#1e293b"   # Slate Headers / Main Text
C_MID_SLATE = "#475569"    # Subtitles / Secondary Text
C_LIGHT_SLATE = "#f8fafc"  # Neutral background
C_BORDER_SLATE = "#cbd5e1" # Neutral border

C_WHITE = "#ffffff"
C_ARROW = "#334155"


def draw_box(ax, x, y, w, h, title, lines=None, bg_color=C_WHITE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE, box_style="round,pad=0.2,rounding_size=0.15", line_color=C_MID_SLATE, title_weight="bold", title_size=9.5, text_size=8.2, badge=None):
    """Draws a standardized academic workflow box with title and bullet details."""
    bbox = FancyBboxPatch((x, y), w, h, boxstyle=box_style,
                          facecolor=bg_color, edgecolor=border_color, linewidth=1.2, zorder=2)
    ax.add_patch(bbox)
    
    # Title
    t_y = y + h - 0.22 if lines else y + h/2.0
    ax.text(x + w/2.0, t_y, title, ha="center", va="center", fontsize=title_size,
            fontweight=title_weight, color=title_color, zorder=3)
    
    # Optional Badge (e.g. tag or layer indicator)
    if badge:
        b_box = FancyBboxPatch((x + 0.1, y + h - 0.22), 0.7, 0.16, boxstyle="round,pad=0.04,rounding_size=0.04",
                               facecolor=border_color, edgecolor="none", zorder=4)
        ax.add_patch(b_box)
        ax.text(x + 0.45, y + h - 0.14, badge, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=title_color, zorder=5)

    # Bullet / Subtitle lines
    if lines:
        n_lines = len(lines)
        avail_h = h - 0.45
        step = avail_h / max(1, n_lines)
        start_y = y + h - 0.45
        for i, line in enumerate(lines):
            curr_y = start_y - i * step
            ax.text(x + 0.15, curr_y, line, ha="left", va="center", fontsize=text_size,
                    color=line_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0.1), color=C_ARROW, style="->", rad=0.0, lw=1.2, ls="-", label_size=7.5):
    """Draws an academic connection arrow with optional label."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style,
                            connectionstyle=f"arc3,rad={rad}",
                            color=color, linewidth=lw, linestyle=ls,
                            mutation_scale=12, zorder=4)
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2.0 + label_offset[0]
        my = (y1 + y2) / 2.0 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=label_size,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=C_WHITE, edgecolor=C_BORDER_SLATE, lw=0.6, alpha=0.95),
                zorder=5)


# =========================================================================
# DIAGRAM 1: OVERALL SYSTEM ARCHITECTURE
# =========================================================================
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Layer Backgrounds (Swimlanes)
    layers = [
        ("1. Presentation & Interaction Layer", 8.2, 1.45, C_LIGHT_SLATE, C_BORDER_SLATE),
        ("2. Data Ingestion & Preprocessing Layer", 5.9, 2.05, C_LIGHT_GREEN, C_BORDER_GREEN),
        ("3. Model Benchmarking & Hardware Profiling Layer", 3.2, 2.45, C_LIGHT_TEAL, C_BORDER_TEAL),
        ("4. Multi-Criteria Green AI & Analytics Engine", 0.6, 2.35, C_LIGHT_SLATE, C_BORDER_SLATE),
    ]
    for title, y, h, bg, border in layers:
        lane = FancyBboxPatch((0.4, y), 13.2, h, boxstyle="round,pad=0.1,rounding_size=0.1",
                              facecolor=bg, edgecolor=border, linewidth=1.0, alpha=0.6, zorder=1)
        ax.add_patch(lane)
        ax.text(0.65, y + h - 0.22, title, fontsize=9.5, fontweight="bold", color=C_DARK_SLATE, zorder=2)

    # --- Layer 1 Boxes ---
    draw_box(ax, 0.8, 8.35, 3.8, 1.05, "Streamlit Web Application",
             ["- Interactive GUI (app/dashboard.py)", "- Dynamic Alpha & Weight Sliders", "- Custom Tabular CSV Upload"],
             bg_color=C_WHITE, border_color="#94a3b8", title_color=C_DARK_GREEN)
    
    draw_box(ax, 5.1, 8.35, 3.8, 1.05, "CLI Evaluation Runner",
             ["- Command-line entry (run_experiments.py)", "- Multi-run statistical batch sweeps", "- Automated logging & CLI options"],
             bg_color=C_WHITE, border_color="#94a3b8", title_color=C_DARK_GREEN)

    draw_box(ax, 9.4, 8.35, 3.8, 1.05, "Publication & Export Engine",
             ["- IEEE / Springer LaTeX tables (.tex)", "- Summary metrics CSV & JSON metadata", "- 300 DPI publication figure exports"],
             bg_color=C_WHITE, border_color="#94a3b8", title_color=C_DARK_GREEN)

    # --- Layer 2 Boxes ---
    draw_box(ax, 0.8, 6.05, 3.8, 1.6, "Data Ingestion & Integrity",
             ["- Standard Benchmarks (Adult, Telco, Cancer)", "- Target Column Auto-Inference", "- Duplicate Row Dropping", "- Missing Target Cleaning"],
             bg_color=C_WHITE, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 5.1, 6.05, 3.8, 1.6, "Leakage & Identifier Filtering",
             ["- Direct Leakage Removal (Invoice / ID)", "- High-Cardinality Entity Dropping", "- Reason Logging & Target Limitations", "- Numerical vs Categorical Inference"],
             bg_color=C_WHITE, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 9.4, 6.05, 3.8, 1.6, "Safe Transformation & Encoding",
             ["- Stratified Split (80/20 train/test)", "- Infrequent Category Bounding ('Other')", "- Median / Constant SimpleImputer", "- Standard/MinMax Scaler + OneHotEncoder", "- CSR Sparse Matrix / Dense Guard"],
             bg_color=C_WHITE, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # --- Layer 3 Boxes ---
    draw_box(ax, 0.8, 3.35, 3.8, 2.05, "Model Registry & Factory",
             ["- Decision Tree (Scikit-Learn)", "- Random Forest (100 trees, parallel)", "- XGBoost (Tree depth 6, lr 0.1)", "- LightGBM (Leaf-wise gradient boost)", "- CatBoost (Ordered boosting)", "- Logistic Regression (L-BFGS)"],
             bg_color=C_WHITE, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 5.1, 3.35, 3.8, 2.05, "Asynchronous Resource Profiler",
             ["- 20ms High-Frequency Daemon Thread", "- Process & System CPU% (psutil)", "- Baseline, Peak, & Delta RAM (MB)", "- Microsecond Execution Latency", "- Pre-training Memory Budget Guard"],
             bg_color=C_WHITE, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 9.4, 3.35, 3.8, 2.05, "Empirical Hardware Power Model",
             ["- Dynamic TDP Power Modeling (Watts):", "  P(t) = P_idle + (U/100)(P_tdp - P_idle) + P_dram", "- Riemann Numerical Energy Integration (J)", "- Carbon Footprint: 475 gCO2eq/kWh", "- Offline CodeCarbon Cross-Check"],
             bg_color=C_WHITE, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # --- Layer 4 Boxes ---
    draw_box(ax, 0.8, 0.75, 3.8, 1.95, "Predictive Metrics Engine",
             ["- Accuracy Score", "- Precision (Weighted, zero-safe)", "- Recall (Weighted, zero-safe)", "- F1-Score (Weighted macro-tradeoff)", "- Latency per 1k test samples"],
             bg_color=C_WHITE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    draw_box(ax, 5.1, 0.75, 3.8, 1.95, "Green AI Index (GAI) Formulation",
             ["- Performance Composite (P in [0,1])", "- Normalized Resource Cost (C in [0,1])", "- Resource Efficiency (R = 1 - C)", "- Linear GAI: alpha*P + (1 - alpha)*R", "- Harmonic GAI: 2*P*R / (P + R + eps)"],
             bg_color=C_WHITE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    draw_box(ax, 9.4, 0.75, 3.8, 1.95, "Pareto & Decision Optimization",
             ["- Strict Pairwise Dominance Criterion", "- Convex Pareto Frontier Extraction", "- Eco-Efficiency Ratio (F1 per Joule)", "- Alpha Sensitivity Sweep ([0.0, 1.0])", "- Deterministic Multi-Key Ranking"],
             bg_color=C_WHITE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Connections between layers
    draw_arrow(ax, 2.7, 8.35, 2.7, 7.65, "Dataset / Config", color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.6, 6.85, 5.1, 6.85, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 8.9, 6.85, 9.4, 6.85, color=C_PRIMARY_GREEN)

    draw_arrow(ax, 11.3, 6.05, 11.3, 5.7, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 11.3, 5.7, 2.7, 5.7, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 2.7, 5.7, 2.7, 5.4, "X_train, y_train, X_test, y_test", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 4.6, 4.35, 5.1, 4.35, "Active Fit/Predict", color=C_PRIMARY_TEAL)
    draw_arrow(ax, 8.9, 4.35, 9.4, 4.35, "CPU Telemetry", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 2.7, 3.35, 2.7, 2.7, "Predictions", color=C_MID_SLATE)
    draw_arrow(ax, 7.0, 3.35, 7.0, 2.7, "Resource Metrics", color=C_MID_SLATE)
    draw_arrow(ax, 11.3, 3.35, 11.3, 2.7, "Joules & CO2", color=C_MID_SLATE)

    draw_arrow(ax, 4.6, 1.7, 5.1, 1.7, "Score P", color=C_MID_SLATE)
    draw_arrow(ax, 8.9, 1.7, 9.4, 1.7, "Trade-off", color=C_MID_SLATE)

    # Upward link to Export
    draw_arrow(ax, 11.3, 2.7, 12.8, 2.7, color=C_DARK_GREEN)
    draw_arrow(ax, 12.8, 2.7, 12.8, 8.35, color=C_DARK_GREEN)
    draw_arrow(ax, 12.8, 8.35, 12.2, 8.35, "Tables & Visuals", color=C_DARK_GREEN)

    plt.suptitle("Overall System Architecture: Dynamic Green AI Evaluation Framework",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "system_architecture.png")
    svg_path = os.path.join(OUTPUT_DIR, "system_architecture.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# DIAGRAM 2: DATA PREPROCESSING WORKFLOW
# =========================================================================
def generate_data_preprocessing_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Column 1: Input & Sanitization
    draw_box(ax, 0.6, 7.6, 3.6, 1.4, "1. Raw Dataset Ingestion",
             ["- Benchmark CSV / User Upload", "- Column name whitespace stripping", "- Duplicate row removal (optional)", "- Null target elimination"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    draw_box(ax, 0.6, 5.3, 3.6, 1.7, "2. Leakage & ID Filtering",
             ["- Target-defining column check", "  (e.g. Invoice / InvoiceNo for cancellation)", "- Identifier detection (CustomerID, UUID)", "- High-uniqueness key removal (>95%)", "- Explicit reason logging dictionary"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 0.6, 3.0, 3.6, 1.7, "3. Target Engineering Check",
             ["- Indirect leakage audit (e.g. Quantity < 0)", "- Price <= 0 accounting anomaly notice", "- Documentation of task limitations", "- Multi-class / Binary label encoding"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 0.6, 0.7, 3.6, 1.7, "4. Dataset Sampling & Split",
             ["- Reproducible sampling (sample_size)", "- Stratified class preservation", "- Zero-leakage train/test split (80/20)", "- Random state seed reproducibility"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Column 2: Transformation & Memory Safeguard
    draw_box(ax, 5.2, 7.3, 3.6, 1.7, "5. Categorical Bounding",
             ["- Training set frequency profiling", "- Top-K frequent categories retained", "  (Bounded by max_categories_onehot)", "- Tail infrequent categories -> 'Other'", "- Test set mapping (unseen -> 'Other')"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 5.2, 4.8, 3.6, 1.85, "6. Memory Safety Budget",
             ["- Pre-encoding allocation estimation:", "  Mem = N_train * D_expected * 8 bytes", "- Threshold check (max_memory_budget_mb)", "- Prevents dense combinatorial blowout", "- Actionable error recommendations if exceeded"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 5.2, 2.3, 3.6, 1.9, "7. Scikit-Learn Pipeline",
             ["- Numerical Features Pipeline:", "  SimpleImputer(median) + Scaler", "  (Standard / MinMax / Robust / Passthrough)", "- Categorical Features Pipeline:", "  SimpleImputer('Missing') + OneHotEncoder", "- ColumnTransformer(remainder='drop')"],
             bg_color=C_WHITE, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Column 3: Output & Matrix Management
    draw_box(ax, 9.8, 6.2, 3.6, 1.9, "8. Matrix Representation",
             ["- Sparse-aware pipeline transformation", "- Safe Dense Conversion Check:", "  If Dense Size <= max_dense_memory_mb:", "    Convert to dense NumPy array", "  Else:", "    Maintain SciPy CSR Sparse matrix"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 9.8, 3.2, 3.6, 2.3, "9. Final Artifact Output",
             ["- X_train, y_train (Train arrays/matrices)", "- X_test, y_test (Test arrays/matrices)", "- Preprocessing Metadata Dictionary:", "  - Initial vs Cleaned vs Sampled rows", "  - Dropped columns log & justifications", "  - Active vs Generated features", "  - Class balance distributions", "  - Sparse flag & footprint (MB)"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Connections
    draw_arrow(ax, 2.4, 7.6, 2.4, 7.0, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 2.4, 5.3, 2.4, 4.7, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 2.4, 3.0, 2.4, 2.4, color=C_PRIMARY_GREEN)

    draw_arrow(ax, 4.2, 1.55, 4.7, 1.55, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.7, 1.55, 4.7, 8.15, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.7, 8.15, 5.2, 8.15, "X_train, X_test", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 7.0, 7.3, 7.0, 6.65, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 7.0, 4.8, 7.0, 4.2, "Budget OK", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 8.8, 3.25, 9.3, 3.25, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 9.3, 3.25, 9.3, 7.15, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 9.3, 7.15, 9.8, 7.15, "Transformed X", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 11.6, 6.2, 11.6, 5.5, color=C_PRIMARY_GREEN)

    plt.suptitle("Data Preprocessing Workflow: Leakage Prevention & Memory-Safe Transformation",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "data_preprocessing_workflow.png")
    svg_path = os.path.join(OUTPUT_DIR, "data_preprocessing_workflow.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# DIAGRAM 3: MACHINE LEARNING MODEL EVALUATION WORKFLOW
# =========================================================================
def generate_model_evaluation_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Step 1: Model Selection
    draw_box(ax, 0.6, 7.4, 3.6, 1.6, "1. Model Selection & Config",
             ["- ModelFactory instantiation:", "  Decision Tree, Random Forest, XGBoost,", "  LightGBM, CatBoost, Logistic Regression", "- Reproducible random_state seed", "- Algorithmic authenticity enforcement"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Step 2: Pre-Training Memory Check
    draw_box(ax, 0.6, 4.7, 3.6, 1.9, "2. Pre-Training Memory Check",
             ["- Data footprint calculation:", "  data_mb = X_train bytes / (1024^2)", "- Overhead buffer check (3.0 * data_mb)", "- System available physical RAM check", "- Protects against OS out-of-memory crashes"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Step 3: Phase 1 Profiling (Training)
    draw_box(ax, 0.6, 1.8, 3.6, 2.2, "3. Training Phase Profiling",
             ["- Start ResourceEnergyProfiler", "- Launch background polling daemon (20ms)", "- Execute model.fit(X_train, y_train)", "- Stop profiler on fit completion", "- Capture metrics:", "  Duration (s), CPU%, Peak RAM, Train Joules"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Step 4: Phase 2 Profiling (Inference)
    draw_box(ax, 5.2, 6.8, 3.6, 2.2, "4. Inference Phase Profiling",
             ["- Start dedicated inference profiler", "- Execute y_pred = model.predict(X_test)", "- Stop profiler on prediction completion", "- Capture metrics:", "  Inference latency (s), Pred Energy (J),", "  Prediction CPU%, Latency per 1k samples"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Step 5: Predictive Performance Metrics
    draw_box(ax, 5.2, 3.9, 3.6, 2.2, "5. Predictive Metrics",
             ["- Evaluate against ground truth y_test:", "- Accuracy: accuracy_score()", "- Precision: weighted, zero_division=0", "- Recall: weighted, zero_division=0", "- F1-Score: weighted harmonic mean", "- Multi-class & binary support"],
             bg_color=C_WHITE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Step 6: Multi-Run Aggregation
    draw_box(ax, 5.2, 1.0, 3.6, 2.2, "6. Multi-Run Aggregation",
             ["- Multi-run statistical confidence:", "  Configurable runs (n_runs in [1, 5])", "- Randomized seed pertubation", "- Computes Mean +/- Std Dev for:", "  Accuracy, F1, Latency, Energy, CO2", "- Statistical confidence for publication"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Step 7: Consolidated Telemetry Output
    draw_box(ax, 9.8, 3.2, 3.6, 3.8, "7. Consolidated Telemetry",
             ["- Unified Model Evaluation Vector:", "  - Model Name & Hyperparameters", "  - Accuracy, Precision, Recall, F1", "  - Average CPU % & Peak CPU %", "  - Baseline, Peak & Delta RAM (MB)", "  - Training & Inference Duration (s)", "  - Latency per 1k test samples (s)", "  - Training & Inference Energy (J)", "  - Total Energy (Joules & kWh)", "  - Operational CO2 Emissions (gCO2eq)", "  - Measurement Methodology Label", "  - Power Model TDP Specification (W)"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Connect arrows
    draw_arrow(ax, 2.4, 7.4, 2.4, 6.6, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 2.4, 4.7, 2.4, 4.0, "Headroom OK", color=C_PRIMARY_GREEN)
    
    draw_arrow(ax, 4.2, 2.9, 4.7, 2.9, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.7, 2.9, 4.7, 7.9, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.7, 7.9, 5.2, 7.9, "Fitted Model", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 7.0, 6.8, 7.0, 6.1, "y_pred, y_test", color=C_MID_SLATE)
    draw_arrow(ax, 7.0, 3.9, 7.0, 3.2, "Single-Run Metrics", color=C_MID_SLATE)

    draw_arrow(ax, 8.8, 2.1, 9.3, 2.1, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 9.3, 2.1, 9.3, 5.1, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 9.3, 5.1, 9.8, 5.1, "Telemetry Vector", color=C_PRIMARY_TEAL)

    plt.suptitle("Machine Learning Model Evaluation Workflow: Dual Performance & Hardware Profiling",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "model_evaluation_workflow.png")
    svg_path = os.path.join(OUTPUT_DIR, "model_evaluation_workflow.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# DIAGRAM 4: COMPUTATIONAL RESOURCE PROFILING WORKFLOW
# =========================================================================
def generate_resource_profiling_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Column 1: Initialization & Baseline
    draw_box(ax, 0.6, 6.8, 3.6, 2.2, "1. Hardware Model Calibration",
             ["- Detect CPU core topology (psutil)", "- Infer processor TDP (Watts):", "  TDP = max(35, min(125, cores * 12.5))", "- Calibrate Base Idle Power:", "  P_idle = 0.20 * P_tdp", "- Calibrate Dynamic DRAM Power:", "  P_dram = RAM_total_GB * 0.375 W/GB"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    draw_box(ax, 0.6, 3.6, 3.6, 2.4, "2. Profiler Start & Telemetry",
             ["- Prime psutil CPU percentage counters", "- Measure process baseline RSS (MB)", "- Initiate threading.Event() stop flag", "- High-precision start timestamp:", "  t_start = time.perf_counter()", "- Spawn background daemon thread", "- Optional CodeCarbon Offline tracker"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 0.6, 0.8, 3.6, 2.1, "3. Asynchronous Sampling Loop",
             ["- High-frequency interval (dt = 20ms)", "- Capture tuple (t, cpu_sys, cpu_proc,", "                 mem_rss, sys_mem_pct):", "  - System CPU% & Process CPU%", "  - Process RSS Memory (MB)", "  - Virtual RAM utilization %", "- Append to volatile thread samples list"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Column 2: Code Execution & Stop Event
    draw_box(ax, 5.2, 5.8, 3.6, 2.5, "4. Workload Execution & Stop",
             ["- Target code executes concurrently", "  (model.fit() or model.predict())", "- Execution ends -> call profiler.stop()", "- High-precision stop timestamp:", "  t_end = time.perf_counter()", "- Set stop_event flag & join thread", "- Duration: T = max(1e-4, t_end - t_start)"],
             bg_color=C_WHITE, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 5.2, 1.8, 3.6, 3.2, "5. Telemetry Extraction",
             ["- Compute CPU utilization statistics:", "  - Average System CPU %: mean(cpu_sys)", "  - Peak System CPU %: max(cpu_sys)", "  - Process CPU %: mean(cpu_proc)", "- Compute Memory Footprint metrics:", "  - Peak RAM: max(mem_rss) [MB]", "  - Delta RAM: max(0, Peak - Baseline)", "  - System Total Physical RAM (MB)", "- Stop CodeCarbon tracker & extract kWh"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Column 3: Numerical Integration & Environmental Conversion
    draw_box(ax, 9.8, 5.5, 3.6, 3.5, "6. Power Model Integration",
             ["- Instantaneous Power Equation (Watts):", "  P(t) = P_idle + (U_cpu(t)/100) *", "         (P_tdp - P_idle) + P_dram", "- Riemann Numerical Sum:", "  Energy = sum_{k} P(t_k) * dt_k", "- Dimensionally correct integration:", "  Watts * Seconds = Joules (J)", "- Short execution fallback check", "- Unit Conversion:", "  Energy (kWh) = Energy (J) / (3.6e6)"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 9.8, 1.2, 3.6, 3.5, "7. Carbon & Methodology Output",
             ["- Grid Carbon Intensity Factor:", "  I_grid = 475 gCO2eq/kWh", "- Operational Carbon Footprint:", "  CO2 (g) = Energy (kWh) * I_grid", "- Methodology Transparency Output:", "  - power_measurement_method:", "    'TDP-calibrated empirical integration'", "  - is_energy_estimated: True", "  - gpu_scope: 'CPU-only profiling'", "  - tdp_watts: calibrated wattage"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Connections
    draw_arrow(ax, 2.4, 6.8, 2.4, 6.0, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 2.4, 3.6, 2.4, 2.9, color=C_PRIMARY_GREEN)

    draw_arrow(ax, 4.2, 1.85, 4.7, 1.85, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.7, 1.85, 4.7, 7.05, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 4.7, 7.05, 5.2, 7.05, "Active Execution", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 7.0, 5.8, 7.0, 5.0, "Samples Collected", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 8.8, 3.4, 9.3, 3.4, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 9.3, 3.4, 9.3, 7.25, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 9.3, 7.25, 9.8, 7.25, "CPU Telemetry List", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 11.6, 5.5, 11.6, 4.7, "Energy (J & kWh)", color=C_MID_SLATE)

    plt.suptitle("Computational Resource Profiling Workflow: Asynchronous Telemetry & TDP Power Model",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "resource_profiling_workflow.png")
    svg_path = os.path.join(OUTPUT_DIR, "resource_profiling_workflow.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# DIAGRAM 5: GREEN AI INDEX CALCULATION WORKFLOW
# =========================================================================
def generate_green_index_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Column 1: Performance & Resource Inputs
    draw_box(ax, 0.6, 5.6, 3.6, 3.2, "1. Predictive Performance Score (P)",
             ["- Performance Sub-Weights (w_perf):", "  - Accuracy: w_acc = 0.25", "  - Precision: w_prec = 0.20", "  - Recall: w_rec = 0.20", "  - F1-Score: w_f1 = 0.35", "- Normalized Weight Sum: sum(w) = 1.0", "- Composite Formulation:", "  P = w_acc*Acc + w_prec*Prec +", "      w_rec*Rec + w_f1*F1", "- Output: P in [0.0, 1.0]"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 0.6, 0.8, 3.6, 4.2, "2. Resource Cost Normalization (C)",
             ["- Raw Computational Metrics:", "  - Energy (Joules)", "  - Training Time (s)", "  - Prediction Latency (s)", "  - Average CPU %", "  - Peak RAM (MB)", "- Min-Max Rescaling to [0.0, 1.0]:", "  C_tilde = (X - X_min) / (X_max - X_min + eps)", "- Cost Sub-Weights (w_cost):", "  - Energy: 0.35, Train Time: 0.20,", "  - Pred Time: 0.15, CPU: 0.15, RAM: 0.15", "- Composite Cost: C = sum(w_j * C_tilde_j)"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Column 2: Resource Efficiency & GAI Formulations
    draw_box(ax, 5.2, 6.0, 3.6, 2.4, "3. Resource Efficiency (R)",
             ["- Linear Complement Formulation:", "  R = 1.0 - C", "- Properties:", "  - Bounded in [0.0, 1.0]", "  - R = 1.0 -> Minimum Resource Footprint", "  - R = 0.0 -> Maximum Resource Footprint", "  - Higher is Greener"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 5.2, 1.0, 3.6, 4.4, "4. GAI Index Syntheses",
             ["- Trade-Off Parameter alpha in [0.0, 1.0]:", "  - alpha = 0.50 (Default balanced)", "  - alpha = 1.00 (Pure accuracy / Red AI)", "  - alpha = 0.00 (Pure sustainability / Green)", "- Linear Green AI Index:", "  GAI_linear = alpha*P + (1 - alpha)*R", "- Harmonic GAI (Green F-Measure):", "  GAI_harmonic = (2 * P * R) / (P + R + eps)", "  Penalizes severe imbalance in either axis", "- Eco-Efficiency Ratio:", "  EcoEff = (F1 / Energy_Joules) * 100"],
             bg_color=C_WHITE, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Column 3: Multi-Criteria Ranking & Sensitivity Sweep
    draw_box(ax, 9.8, 5.2, 3.6, 3.4, "5. Deterministic Ranking",
             ["- Multi-Key Deterministic Tie-Breaking:", "  Sort Order:", "  1. GAI_linear (Descending)", "  2. F1-Score (Descending)", "  3. Energy_Joules (Ascending)", "  4. Model Name (Alphabetical)", "- Output Rank Fields:", "  - rank_gai: Primary ranking", "  - rank_f1: Performance ranking", "  - rank_energy: Sustainability ranking"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    draw_box(ax, 9.8, 1.0, 3.6, 3.6, "6. Alpha Sensitivity Sweep",
             ["- Sensitivity Analysis Sweep:", "  alpha in np.linspace(0.0, 1.0, 21)", "- Re-calculates GAI & rankings", "- Evaluates Rank Robustness:", "  - Identifies rank transition thresholds", "  - Verifies stability across preferences", "- Visualized in sensitivity curve plot", "- Supports dynamic slider tuning"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Connections
    draw_arrow(ax, 4.2, 7.2, 5.2, 7.2, "P score", color=C_PRIMARY_GREEN)
    draw_arrow(ax, 2.4, 0.8, 2.4, 0.4, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 2.4, 0.4, 7.0, 0.4, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 7.0, 0.4, 7.0, 1.0, "C cost", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 7.0, 6.0, 7.0, 5.4, "R efficiency", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 8.8, 3.2, 9.3, 3.2, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 9.3, 3.2, 9.3, 6.9, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 9.3, 6.9, 9.8, 6.9, "GAI Scores", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 11.6, 5.2, 11.6, 4.6, "Rank Matrix", color=C_MID_SLATE)

    plt.suptitle("Green AI Index (GAI) Calculation Workflow: Multi-Attribute Mathematical Formulation",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "green_ai_index_workflow.png")
    svg_path = os.path.join(OUTPUT_DIR, "green_ai_index_workflow.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# DIAGRAM 6: PARETO FRONTIER ANALYSIS WORKFLOW
# =========================================================================
def generate_pareto_analysis_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Column 1: Objectives Definition & Extraction
    draw_box(ax, 0.6, 6.0, 3.6, 2.8, "1. Objective Space Formulation",
             ["- Objective 1 (Maximize):", "  Predictive Performance (F1-Score / P)", "  f1 in [0.0, 1.0]", "- Objective 2 (Minimize):", "  Energy Consumption (Joules)", "  E in [E_min, E_max]", "- Extraction from benchmark results", "- Multi-objective trade-off framing"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    draw_box(ax, 0.6, 1.2, 3.6, 4.2, "2. Pairwise Dominance Criterion",
             ["- Strict Mathematical Rule:", "  Model j dominates Model i (j succ i) iff:", "  (f1_j >= f1_i  AND  E_j <= E_i)", "  AND", "  (f1_j > f1_i   OR   E_j < E_i)", "- Double-nested comparison loop (N x N)", "- Evaluates all candidate model pairs", "- No tolerance for surrogate dominance"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Column 2: Classification & Convex Frontier Extraction
    draw_box(ax, 5.2, 5.2, 3.6, 3.6, "3. Optimality Classification",
             ["- Iterative Dominance Evaluation:", "  For model i in 1..N:", "    is_pareto[i] = True", "    For model j != i:", "      If j dominates i:", "        is_pareto[i] = False", "        Break", "- Partition into sets:", "  - Pareto Optimal: Non-dominated", "  - Dominated: Suboptimal alternatives"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 5.2, 0.8, 3.6, 3.8, "4. Frontier Curve Construction",
             ["- Filter Non-Dominated Models:", "  pareto_df = df[pareto_optimal == True]", "- Sort by Energy ascending:", "  pareto_df.sort_values('energy_joules')", "- Coordinates trace Pareto Frontier:", "  (E_1, f1_1) -> (E_2, f1_2) -> ...", "- Monotonically increasing F1 and Energy", "- Convex efficiency boundary"],
             bg_color=C_WHITE, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Column 3: Eco-Efficiency & Decision Synthesis
    draw_box(ax, 9.8, 5.0, 3.6, 3.8, "5. Trade-Off Synthesis",
             ["- Marginal Performance Analysis:", "  Delta F1 / Delta Energy", "  Evaluates additional accuracy purchased", "  per Joule of energy consumed", "- Diminishing Returns Quantified:", "  e.g. XGBoost vs LightGBM: 3.8x energy", "  for 0.14% gain in F1-score", "- Eco-Efficiency Ratio:", "  F1 per Joule * 100"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 9.8, 0.8, 3.6, 3.6, "6. Multi-Dataset Synthesis",
             ["- Pareto Summaries across benchmarks:", "  - Adult Census: LightGBM, XGBoost", "  - Telco Churn: Logistic Regression", "  - Breast Cancer: Logistic Regression", "- Export to pareto_summary.csv", "- Plot 300 DPI Scatter & Star Annotations", "- Interactive Plotly visualization in app"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Connections
    draw_arrow(ax, 2.4, 6.0, 2.4, 5.4, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 4.2, 3.3, 4.7, 3.3, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 4.7, 3.3, 4.7, 7.0, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 4.7, 7.0, 5.2, 7.0, "(f1, Energy)", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 7.0, 5.2, 7.0, 4.6, "Optimality Flags", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 8.8, 2.7, 9.3, 2.7, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 9.3, 2.7, 9.3, 6.9, color=C_PRIMARY_TEAL)
    draw_arrow(ax, 9.3, 6.9, 9.8, 6.9, "Frontier Points", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 11.6, 5.0, 11.6, 4.4, "Optimal Profiles", color=C_MID_SLATE)

    plt.suptitle("Pareto Frontier Analysis Workflow: Bi-Objective Multi-Attribute Dominance Engine",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "pareto_analysis_workflow.png")
    svg_path = os.path.join(OUTPUT_DIR, "pareto_analysis_workflow.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# DIAGRAM 7: COMPLETE END-TO-END PROJECT DATA FLOW
# =========================================================================
def generate_end_to_end_data_flow():
    fig, ax = plt.subplots(figsize=(14.0, 9.8), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    # Column 1: Input Data Sources
    draw_box(ax, 0.5, 7.2, 3.2, 2.4, "Stage 1: Data Sources",
             ["- Canonical Tabular Benchmarks:", "  - UCI Adult Census (25k rows)", "  - Telco Customer Churn (7k rows)", "  - Breast Cancer Diagnostic (569 rows)", "- Custom User CSV Upload", "- Automatic Schema & Target Detection"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Column 1 Bottom: Preprocessing
    draw_box(ax, 0.5, 2.4, 3.2, 4.3, "Stage 2: Safe Pipeline",
             ["- DynamicDataPipeline Execution:", "  - Duplicate dropping", "  - Stratified sampling (sample_size)", "  - Direct target leakage removal", "    (e.g. Invoice / InvoiceNo)", "  - Customer / UUID ID dropping", "  - Categorical tail bounding ('Other')", "  - Memory budget headroom validation", "  - Stratified 80/20 train/test split", "  - SimpleImputer (median / 'Missing')", "  - Scaler (Standard/MinMax/Robust)", "  - Safe sparse CSR / dense array"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    # Column 2 Top: Model Benchmarking
    draw_box(ax, 4.3, 5.8, 3.2, 3.8, "Stage 3: ML Benchmarking",
             ["- 6 Canonical Classifiers:", "  1. Decision Tree (Depth 10)", "  2. Random Forest (100 estimators)", "  3. XGBoost (Depth 6, lr 0.1)", "  4. LightGBM (Leaf 31, lr 0.1)", "  5. CatBoost (Depth 6, 100 iter)", "  6. Logistic Regression (L-BFGS)", "- Pre-training memory overhead guard", "- Multi-run statistical looping"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Column 2 Bottom: Profiler & Hardware Power Model
    draw_box(ax, 4.3, 0.8, 3.2, 4.5, "Stage 4: Hardware Profiler",
             ["- ResourceEnergyProfiler Daemon:", "  - 20ms background polling thread", "  - Process & System CPU% (psutil)", "  - Baseline, Peak, & Delta RAM", "  - Execution Duration (seconds)", "- Empirical Hardware Power Model:", "  - TDP-calibrated wattage envelope", "  - P(t) = P_idle + (U/100)*(P_tdp - P_idle)", "          + P_dram", "  - Riemann Numerical Sum -> Joules", "  - Grid Carbon Intensity -> gCO2eq", "  - CodeCarbon Offline cross-check"],
             bg_color=C_WHITE, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    # Column 3: Green AI Synthesis & Pareto Engine
    draw_box(ax, 8.1, 4.8, 3.2, 4.8, "Stage 5: Analytics Engine",
             ["- GreenAIIndexCalculator:", "  - Predictive Performance Score P", "    (Accuracy, Precision, Recall, F1)", "  - Min-Max Cost Normalization C", "    (Energy, Time, CPU, RAM)", "  - Resource Efficiency R = 1 - C", "  - Linear GAI: alpha*P + (1-alpha)*R", "  - Harmonic GAI (Green F-Measure):", "    2*P*R / (P + R + eps)", "  - Eco-Efficiency (F1 / Joules)", "- Pareto Frontier Optimization:", "  - Pairwise dominance extraction", "  - Alpha sweep sensitivity analysis"],
             bg_color=C_LIGHT_GREEN, border_color=C_BORDER_GREEN, title_color=C_DARK_GREEN)

    draw_box(ax, 8.1, 0.8, 3.2, 3.5, "Stage 6: Artifact Storage",
             ["- Structured File Repository:", "  - results/tables/ (CSV summaries)", "  - results/tables/ (*_table.tex)", "  - results/figures/ (300 DPI PNGs)", "  - results/metadata.json", "    (Hardware specs & CPU model)", "  - results/execution.log"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Column 4: Delivery / Interfaces
    draw_box(ax, 11.9, 5.2, 2.3, 4.4, "Stage 7A: Web App",
             ["- Streamlit Dashboard:", "  - Header KPI cards", "  - Overview & Leaderboard", "  - Telemetry breakdowns", "  - Interactive Pareto plot", "  - Dynamic alpha slider", "  - Sub-weight tuners", "  - CSV & LaTeX export"],
             bg_color=C_LIGHT_TEAL, border_color=C_BORDER_TEAL, title_color=C_DARK_TEAL)

    draw_box(ax, 11.9, 0.8, 2.3, 3.8, "Stage 7B: Paper / Thesis",
             ["- Academic Thesis:", "  - Chapter 4: Architecture", "  - Chapter 6: Dashboard", "  - Chapter 7: Results", "- IEEE LaTeX paper:", "  - Ready-to-compile .tex", "  - 300 DPI Figures", "  - Reproducible findings"],
             bg_color=C_LIGHT_SLATE, border_color=C_BORDER_SLATE, title_color=C_DARK_SLATE)

    # Trace Arrows
    draw_arrow(ax, 2.1, 7.2, 2.1, 6.7, "Raw CSV", color=C_PRIMARY_GREEN)
    
    draw_arrow(ax, 3.7, 5.0, 4.3, 7.5, "X, y splits", color=C_PRIMARY_TEAL)
    draw_arrow(ax, 3.7, 3.5, 4.3, 2.5, "Data arrays", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 5.9, 5.8, 5.9, 5.3, "Model Fit/Predict", color=C_PRIMARY_TEAL)

    draw_arrow(ax, 7.5, 7.7, 8.1, 7.7, "Predictions", color=C_PRIMARY_GREEN)
    draw_arrow(ax, 7.5, 3.0, 7.8, 3.0, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 7.8, 3.0, 7.8, 6.2, color=C_PRIMARY_GREEN)
    draw_arrow(ax, 7.8, 6.2, 8.1, 6.2, "Hardware Telemetry", color=C_PRIMARY_GREEN)

    draw_arrow(ax, 9.7, 4.8, 9.7, 4.3, "Summary Data", color=C_MID_SLATE)

    draw_arrow(ax, 11.3, 7.5, 11.9, 7.5, "Live UI", color=C_PRIMARY_TEAL)
    draw_arrow(ax, 11.3, 2.5, 11.9, 2.5, "Thesis & Paper", color=C_DARK_SLATE)

    plt.suptitle("Complete End-to-End Project Data Flow: Ingestion to Telemetry, Synthesis & Publication",
                 fontsize=14, fontweight="bold", color=C_DARK_SLATE, y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "end_to_end_data_flow.png")
    svg_path = os.path.join(OUTPUT_DIR, "end_to_end_data_flow.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {png_path} & {svg_path}")


# =========================================================================
# MERMAID SOURCE FILES GENERATOR
# =========================================================================
def generate_mermaid_sources():
    """Generates editable Mermaid (.mmd) files for all 7 diagrams."""

    # 1. Overall System Architecture
    mmd_arch = """flowchart TD
    %% Overall System Architecture - Dynamic Green AI Evaluation Framework
    subgraph PresentationLayer["1. Presentation & Interaction Layer"]
        DASH["Streamlit Interactive Dashboard<br/>(app/dashboard.py)"]
        CLI["CLI Evaluation Runner<br/>(run_experiments.py)"]
        EXPORT["Publication Export Engine<br/>(LaTeX tables, CSVs, Figures)"]
    end

    subgraph DataLayer["2. Data Ingestion & Preprocessing Layer (DynamicDataPipeline)"]
        INGEST["Data Ingestion & Integrity<br/>(Adult, Telco, Cancer, Custom CSV)"]
        LEAKAGE["Leakage & Identifier Filtering<br/>(Drop Invoice, CustomerID, UUID)"]
        ENCODE["Safe Transformation & Encoding<br/>(Stratified Split, 'Other' Grouping, Scaler, CSR/Dense)"]
    end

    subgraph BenchmarkingLayer["3. Model Benchmarking & Hardware Profiling Layer"]
        MODELS["Model Registry & Factory (ModelFactory)<br/>(Decision Tree, RF, XGBoost, LightGBM, CatBoost, LogReg)"]
        PROFILER["Asynchronous Telemetry Profiler<br/>(20ms daemon thread, psutil CPU & RAM)"]
        POWER["Empirical Hardware Power Model<br/>(TDP-calibrated integration, Joules & CO2eq)"]
    end

    subgraph AnalyticsLayer["4. Multi-Criteria Green AI & Analytics Engine"]
        PRED_METRICS["Predictive Metrics Engine<br/>(Accuracy, Precision, Recall, F1-Score)"]
        GAI_CALC["Green AI Index (GAI) Engine<br/>(Linear GAI, Harmonic GAI, Eco-Efficiency)"]
        PARETO_ENGINE["Pareto Frontier & Sensitivity Engine<br/>(Dominance testing, alpha sweeps)"]
    end

    INGEST --> LEAKAGE --> ENCODE
    ENCODE -->|X_train, y_train, X_test, y_test| MODELS
    MODELS <-->|Active Profiling| PROFILER
    PROFILER --> POWER
    MODELS --> PRED_METRICS
    POWER --> GAI_CALC
    PRED_METRICS --> GAI_CALC
    GAI_CALC --> PARETO_ENGINE
    PARETO_ENGINE --> DASH
    PARETO_ENGINE --> EXPORT
    CLI --> INGEST
    DASH --> INGEST
"""
    with open(os.path.join(OUTPUT_DIR, "system_architecture.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_arch)

    # 2. Data Preprocessing Workflow
    mmd_prep = """flowchart TD
    %% Data Preprocessing Workflow (DynamicDataPipeline)
    A[Raw Dataset CSV / User Upload] --> B[Clean Column Names & Drop Null Targets]
    B --> C{Duplicates Dropping?}
    C -->|Yes| C1[Drop Duplicate Rows]
    C -->|No| D[Stratified / Random Dataset Sampling]
    C1 --> D
    D --> E[Filter Direct Target Leakage & High-Cardinality IDs]
    E --> F[Audit Indirect Target Leakage & Quantity Artifacts]
    F --> G[Infer Numerical vs Categorical Feature Types]
    G --> H[Zero-Leakage Stratified 80/20 Train-Test Split]
    H --> I[Group Infrequent Categories into 'Other' on Train]
    I --> J{Memory Budget Check<br/>Estimated Dense RAM <= Budget?}
    J -->|No| J_ERR[Raise MemoryBudgetExceededError with Resolution Steps]
    J -->|Yes| K[Construct ColumnTransformer Pipeline]
    K --> L[Fit Scaler & OneHotEncoder Strictly on Train]
    L --> M[Transform Train and Test Sets]
    M --> N{Dense Conversion Memory <= max_dense_memory_mb?}
    N -->|Yes| O[Convert to Safe Dense NumPy Arrays]
    N -->|No| P[Maintain SciPy CSR Sparse Matrix Representation]
    O --> Q[Return X_train, X_test, y_train, y_test and Metadata Log]
    P --> Q
"""
    with open(os.path.join(OUTPUT_DIR, "data_preprocessing_workflow.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_prep)

    # 3. Model Evaluation Workflow
    mmd_eval = """flowchart TD
    %% Machine Learning Model Evaluation Workflow (GreenAIEvaluator)
    A[Initialize Benchmark Configuration & Target Model List] --> B[ModelFactory.get_model name, random_state]
    B --> C{Pre-Training Memory Check<br/>3x Data MB <= Budget & System RAM OK?}
    C -->|No| C_ERR[Raise MemoryBudgetExceededError]
    C -->|Yes| D[Launch Asynchronous ResourceEnergyProfiler for Training]
    D --> E[Execute model.fit X_train, y_train]
    E --> F[Stop Profiler: Record Training Time, CPU %, RAM Delta, Training Joules]
    F --> G[Launch Asynchronous ResourceEnergyProfiler for Inference]
    G --> H[Execute y_pred = model.predict X_test]
    H --> I[Stop Profiler: Record Prediction Time, Latency per 1k, Inference Joules]
    I --> J[Compute Predictive Metrics: Accuracy, Precision, Recall, F1-Score]
    J --> K{Multi-Run Evaluation Enabled?}
    K -->|Yes| L[Repeat over N Randomized Seeds: Calculate Mean +/- Std Dev]
    K -->|No| M[Consolidate Single-Run Telemetry & Predictive Metrics]
    L --> N[Structured Model Evaluation Record Output]
    M --> N
"""
    with open(os.path.join(OUTPUT_DIR, "model_evaluation_workflow.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_eval)

    # 4. Computational Resource Profiling Workflow
    mmd_prof = """flowchart TD
    %% Computational Resource Profiling Workflow (ResourceEnergyProfiler)
    A[Profiler Instantiation] --> B[Calibrate HardwarePowerModel: P_idle = 0.20*TDP, P_dram = 0.375 W/GB]
    B --> C[profiler.start: Prime psutil CPU counters & Measure Baseline RSS MB]
    C --> D[Spawn High-Frequency Background Sampling Daemon dt = 20ms]
    D --> E{Stop Event Triggered?}
    E -->|No| F[Sample: cpu_sys, cpu_proc, mem_rss, sys_mem_pct]
    F --> E
    E -->|Yes| G[profiler.stop: Record t_end & Join Sampling Daemon]
    G --> H[Compute Execution Duration: T = t_end - t_start]
    H --> I[Calculate Resource Statistics: Avg CPU %, Peak CPU %, Peak RAM, Delta RAM MB]
    I --> J[Riemann Numerical Integration: Energy = sum P_k * dt_k]
    J --> K[Calculate Energy in kWh = Joules / 3,600,000]
    K --> L[Calculate Carbon Footprint: CO2_g = kWh * 475 gCO2eq/kWh]
    L --> M[Return Complete Telemetry Dictionary with Methodology Transparency Flags]
"""
    with open(os.path.join(OUTPUT_DIR, "resource_profiling_workflow.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_prof)

    # 5. Green AI Index Workflow
    mmd_gai = """flowchart TD
    %% Green AI Index (GAI) Calculation Workflow (GreenAIIndexCalculator)
    A[Model Evaluation Results DataFrame] --> B[Compute Performance Score P]
    B --> B1[P = 0.25*Acc + 0.20*Prec + 0.20*Rec + 0.35*F1]
    A --> C[Normalize Resource Costs to [0, 1] via Min-Max Scaling]
    C --> C1[Norm Energy, Norm Train Time, Norm Pred Time, Norm CPU %, Norm Peak RAM]
    C1 --> D[Compute Composite Resource Cost C]
    D --> D1[C = sum w_cost * Norm_Metric]
    D1 --> E[Compute Resource Efficiency R = 1.0 - C]
    B1 --> F[Calculate Linear Green AI Index: GAI_linear = alpha*P + 1 - alpha*R]
    E --> F
    B1 --> G[Calculate Harmonic Green AI Index: GAI_harmonic = 2*P*R / P + R + eps]
    E --> G
    A --> H[Calculate Eco-Efficiency Ratio: F1 / Energy_Joules * 100]
    F --> I[Deterministic Multi-Key Sorting: GAI desc, F1 desc, Energy asc, Name asc]
    G --> I
    I --> J[Assign Discrete Model Ranks: rank_gai, rank_f1, rank_energy]
    J --> K[Alpha Sensitivity Analysis Sweep: alpha in [0.0, 1.0] across 21 steps]
"""
    with open(os.path.join(OUTPUT_DIR, "green_index_workflow.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_gai)

    # 6. Pareto Frontier Analysis Workflow
    mmd_pareto = """flowchart TD
    %% Pareto Frontier Analysis Workflow
    A[Model Results: F1-Score Objective 1 & Energy Joules Objective 2] --> B[Initialize is_pareto = True for all N models]
    B --> C[Outer Loop: Model i in 1..N]
    C --> D[Inner Loop: Model j != i in 1..N]
    D --> E{Does Model j dominate Model i?<br/>F1_j >= F1_i AND E_j <= E_i<br/>AND at least one strictly better?}
    E -->|Yes| F[is_pareto_i = False<br/>Break Inner Loop]
    E -->|No| G[Continue Inner Loop]
    G --> D
    F --> H{More Models in Outer Loop?}
    D --> H
    H -->|Yes| C
    H -->|No| I[Partition Models: Pareto Optimal vs Dominated]
    I --> J[Extract Pareto Optimal Subset & Sort by Energy Ascending]
    J --> K[Construct Non-Dominated Convex Frontier Curve]
    K --> L[Calculate Marginal F1 Gain per Joule Investment]
    L --> M[Export to pareto_summary.csv & Generate Publication Visualizations]
"""
    with open(os.path.join(OUTPUT_DIR, "pareto_analysis_workflow.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_pareto)

    # 7. Complete End-to-End Project Data Flow
    mmd_e2e = """flowchart LR
    %% Complete End-to-End Project Data Flow
    subgraph S1["1. Data Ingestion"]
        D1[(Adult Census)]
        D2[(Telco Churn)]
        D3[(Breast Cancer)]
        D4[(Custom CSV)]
    end

    subgraph S2["2. Pipeline & Safeguards"]
        P1[Feature Inference & ID Filtering]
        P2[Leakage Detection & Category Bounding]
        P3[Stratified 80/20 Train/Test Split]
        P4[Safe Scaler & Encoding CSR/Dense]
    end

    subgraph S3["3. Model Dispatch & Profiling"]
        M[6 Canonical Classifiers]
        PROF[20ms Telemetry Daemon]
        TDP[TDP Power Integration Model]
    end

    subgraph S4["4. Green AI & Decision Engine"]
        GAI[Green AI Index Formulations]
        PAR[Pareto Dominance Engine]
        RANK[Deterministic Ranker & Alpha Sweep]
    end

    subgraph S5["5. Publication & Deliverables"]
        UI[Streamlit Interactive App]
        TAB[IEEE LaTeX & CSV Tables]
        FIG[300 DPI Publication Visuals]
        THESIS[B.Tech Major Project Thesis]
    end

    D1 & D2 & D3 & D4 --> P1 --> P2 --> P3 --> P4
    P4 -->|X_train, y_train| M
    P4 -->|X_test, y_test| M
    M <-->|High-Frequency Telemetry| PROF --> TDP
    M -->|Predictions| GAI
    TDP -->|Energy Joules & CO2| GAI
    GAI --> PAR --> RANK
    RANK --> UI
    RANK --> TAB & FIG --> THESIS
"""
    with open(os.path.join(OUTPUT_DIR, "end_to_end_data_flow.mmd"), "w", encoding="utf-8") as f:
        f.write(mmd_e2e)

    print("Generated all 7 Mermaid (.mmd) source files successfully!")


if __name__ == "__main__":
    print("Generating 7 publication-quality technical diagrams...")
    generate_system_architecture()
    generate_data_preprocessing_workflow()
    generate_model_evaluation_workflow()
    generate_resource_profiling_workflow()
    generate_green_index_workflow()
    generate_pareto_analysis_workflow()
    generate_end_to_end_data_flow()
    generate_mermaid_sources()
    print("All architecture and workflow deliverables generated successfully!")
