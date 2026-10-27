"""
Interactive Dynamic Green AI Evaluation Framework Dashboard (Streamlit)
A research-grade sustainability analytics platform benchmarking predictive performance,
computational resource consumption, and environmental footprint for machine learning models.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.data_pipeline import DynamicDataPipeline, MemoryBudgetExceededError
from src.evaluator import GreenAIEvaluator
from src.green_index import GreenAIIndexCalculator
from src.models import ModelFactory

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Dynamic Green AI Evaluation Framework",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- THEME STYLING & CUSTOM CSS -----------------
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #334155;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header Branding */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #064e3b;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-ready {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .status-active {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    .pulse-dot-blue {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #3b82f6;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    /* KPI Metric Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Sidebar Section Headers */
    .sidebar-section-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #065f46;
        padding-bottom: 4px;
        margin-top: 14px;
        margin-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Information Callout Banner */
    .info-banner {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #059669;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    
    /* Welcome Hero Card */
    .welcome-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .welcome-hero-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #064e3b;
        margin-bottom: 8px;
    }
    .welcome-hero-text {
        font-size: 0.98rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .pillar-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        height: 100%;
    }
    .pillar-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pillar-desc {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.5;
    }
    
    /* Educational Panel */
    .edu-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    /* Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-size: 0.92rem;
        font-weight: 600;
        color: #64748b;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #065f46 !important;
        border-bottom: 2px solid #059669 !important;
        background-color: #f0fdf4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "dataset_meta" not in st.session_state:
    st.session_state["dataset_meta"] = None
if "cached_uploaded_file" not in st.session_state:
    st.session_state["cached_uploaded_file"] = {"name": None, "size": 0, "df": None, "meta": None}

# ----------------- HELPER FUNCTIONS -----------------
def format_file_size(num_bytes: int) -> str:
    """Formats byte counts into human-readable strings."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"

def inspect_dataframe_quick(df: pd.DataFrame, filename: str, file_size_bytes: int) -> dict:
    """Safe, fast metadata inspection for dataset upload preview."""
    nrows, ncols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_count = int(df.isna().sum().sum())
    missing_pct = (missing_count / max(1, nrows * ncols)) * 100.0
    
    # Detect potential high cardinality categorical features
    high_card_cols = [c for c in cat_cols if df[c].nunique(dropna=True) > 30]
    
    return {
        "filename": filename,
        "size_str": format_file_size(file_size_bytes),
        "size_bytes": file_size_bytes,
        "nrows": nrows,
        "ncols": ncols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "high_card_cols": high_card_cols
    }

# ----------------- HEADER & BRANDING -----------------
results_active = st.session_state["results_df"] is not None
status_badge_html = (
    f'<div class="status-badge status-active"><span class="pulse-dot-blue"></span> Benchmark Active &bull; {len(st.session_state["results_df"])} Models</div>'
    if results_active
    else '<div class="status-badge status-ready"><span class="pulse-dot"></span> Framework Ready</div>'
)

st.markdown(f"""
<div class="header-container">
    <div>
        <div class="brand-title">🌿 Dynamic Green AI Evaluation Framework</div>
        <div class="brand-subtitle">Benchmark predictive performance, computational cost, and environmental sustainability.</div>
    </div>
    <div>
        {status_badge_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION PANEL -----------------
st.sidebar.markdown('<div class="sidebar-section-header">1. Dataset Configuration</div>', unsafe_allow_html=True)

dataset_choice = st.sidebar.selectbox(
    "Select Benchmark Dataset",
    ["Adult Census Income (UCI)", "Telco Customer Churn", "Breast Cancer Diagnostic", "Upload Custom CSV"],
    index=0,
    help="Select a standard benchmark or upload your custom tabular CSV."
)

data_file_map = {
    "Adult Census Income (UCI)": (os.path.join(root_dir, "data", "adult_census.csv"), "income"),
    "Telco Customer Churn": (os.path.join(root_dir, "data", "telco_churn.csv"), "Churn"),
    "Breast Cancer Diagnostic": (os.path.join(root_dir, "data", "breast_cancer.csv"), "target")
}

custom_uploaded = None
target_col = None
df_custom_loaded = None

if dataset_choice == "Upload Custom CSV":
    custom_uploaded = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help="Upload tabular CSV. Safe high-cardinality protection and sampling safeguards are applied automatically."
    )
    if custom_uploaded:
        # Check cache to avoid reloading file repeatedly
        cache = st.session_state["cached_uploaded_file"]
        if cache["name"] == custom_uploaded.name and cache["size"] == custom_uploaded.size and cache["df"] is not None:
            df_custom_loaded = cache["df"]
            file_meta = cache["meta"]
        else:
            try:
                custom_uploaded.seek(0)
                df_custom_loaded = pd.read_csv(custom_uploaded)
                file_meta = inspect_dataframe_quick(df_custom_loaded, custom_uploaded.name, custom_uploaded.size)
                st.session_state["cached_uploaded_file"] = {
                    "name": custom_uploaded.name,
                    "size": custom_uploaded.size,
                    "df": df_custom_loaded,
                    "meta": file_meta
                }
            except Exception as e:
                st.sidebar.error(f"Error reading CSV file: {e}")
                df_custom_loaded = None

        if df_custom_loaded is not None and file_meta is not None:
            st.sidebar.caption(
                f"📄 **{file_meta['filename']}** ({file_meta['size_str']})  \n"
                f"📐 {file_meta['nrows']:,} rows &times; {file_meta['ncols']} cols | "
                f"Missing: {file_meta['missing_count']} ({file_meta['missing_pct']:.1f}%)"
            )
            
            # Select target column
            cols = list(df_custom_loaded.columns)
            default_idx = len(cols) - 1
            # Auto-detect common target names
            for i, c in enumerate(cols):
                if c.lower() in ["target", "label", "iscancellation", "churn", "income", "class"]:
                    default_idx = i
                    break
                    
            target_col = st.sidebar.selectbox("Select Target Column", cols, index=default_idx)
            
            # Show class distribution & checks
            if target_col in df_custom_loaded.columns:
                target_s = df_custom_loaded[target_col].dropna()
                n_classes = target_s.nunique()
                if n_classes < 2:
                    st.sidebar.error("⚠️ Target has fewer than 2 classes. Classification requires &ge; 2 classes.")
                else:
                    top_classes = target_s.value_counts(normalize=True).head(3)
                    dist_summary = ", ".join([f"{k}: {v*100:.1f}%" for k, v in top_classes.items()])
                    st.sidebar.caption(f"🎯 **Target Classes ({n_classes}):** {dist_summary}")
                    
                    # Class imbalance warning
                    min_prop = target_s.value_counts(normalize=True).min()
                    if min_prop < 0.10:
                        st.sidebar.warning(f"⚠️ Imbalanced target detected (minority: {min_prop*100:.1f}%). Stratification is enabled.")
            
            # Large dataset sampling prompt
            if file_meta["nrows"] > 100000:
                st.sidebar.info("💡 Large dataset detected. We recommend using a training sample below for fast benchmarking.")
else:
    csv_path, target_col = data_file_map[dataset_choice]
    st.sidebar.caption(f"📌 Benchmark: **{dataset_choice}** (Target: `{target_col}`)")

# ----------------- SECTION 2: PREPROCESSING & SAMPLING -----------------
st.sidebar.markdown('<div class="sidebar-section-header">2. Preprocessing & Sampling</div>', unsafe_allow_html=True)

scaler_choice = st.sidebar.selectbox(
    "Feature Scaler",
    ["standard", "minmax", "robust", "none"],
    index=0,
    help="Standard: Zero-mean unit-variance. MinMax: [0, 1] range. Robust: Median/IQR (outlier resistant)."
)

sample_choice = st.sidebar.selectbox(
    "Training Sample Size",
    ["Full Dataset", "10,000", "25,000", "50,000", "Custom"],
    index=0,
    help="Reproducible stratified sampling prevents out-of-memory errors on massive datasets like Online Retail II."
)

sample_size = None
if sample_choice == "10,000":
    sample_size = 10000
elif sample_choice == "25,000":
    sample_size = 25000
elif sample_choice == "50,000":
    sample_size = 50000
elif sample_choice == "Custom":
    sample_size = int(st.sidebar.number_input("Custom Sample Size", min_value=100, max_value=2000000, value=25000, step=1000))

sampling_seed = int(st.sidebar.number_input(
    "Sampling Seed",
    min_value=0, max_value=999999, value=42, step=1,
    help="Random seed ensuring reproducible stratified splits across runs."
))

with st.sidebar.expander("🛠️ Memory Safety & Tuning", expanded=False):
    max_categories = st.slider(
        "Max Categories per Feature",
        min_value=5, max_value=100, value=30, step=5,
        help="Infrequent categories beyond this limit are safely grouped into 'Other' to prevent feature explosion."
    )
    memory_budget_mb = st.number_input(
        "Memory Safety Budget (MB)",
        min_value=256, max_value=32768, value=2048, step=256,
        help="Execution halts safely if estimated memory exceeds this budget, preventing system crashes."
    )
    max_dense_mb = st.number_input(
        "Sparse-to-Dense Threshold (MB)",
        min_value=128, max_value=8192, value=512, step=128,
        help="High-dimensional encoded data exceeding this size is preserved in memory-efficient SciPy CSR sparse matrix format."
    )

# ----------------- SECTION 3: MODEL SELECTION -----------------
st.sidebar.markdown('<div class="sidebar-section-header">3. Model Selection</div>', unsafe_allow_html=True)

selected_models = st.sidebar.multiselect(
    "Select Algorithms",
    ModelFactory.SUPPORTED_MODELS,
    default=ModelFactory.SUPPORTED_MODELS,
    help="Choose the machine learning architectures to evaluate."
)

model_count = len(selected_models)
total_supported = len(ModelFactory.SUPPORTED_MODELS)
st.sidebar.caption(f"🤖 **Selected:** {model_count} of {total_supported} algorithms")

if model_count == 0:
    st.sidebar.warning("⚠️ Please select at least one algorithm to run evaluation.")

# ----------------- SECTION 4: SUSTAINABILITY PREFERENCES -----------------
st.sidebar.markdown('<div class="sidebar-section-header">4. Sustainability Preferences</div>', unsafe_allow_html=True)

alpha = st.sidebar.slider(
    "Tradeoff Parameter (α)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="α = 1.0 (Pure Accuracy Priority) | α = 0.5 (Balanced Green AI) | α = 0.0 (Pure Sustainability Priority)"
)

# Intuitive visual feedback for alpha
if alpha == 0.5:
    st.sidebar.caption("⚖️ **Balanced Green AI:** Equal weight on predictive quality and hardware efficiency.")
elif alpha > 0.5:
    st.sidebar.caption(f"🎯 **Performance-Biased:** {alpha*100:.0f}% predictive accuracy, {(1-alpha)*100:.0f}% sustainability.")
else:
    st.sidebar.caption(f"🍃 **Sustainability-Biased:** {(1-alpha)*100:.0f}% sustainability, {alpha*100:.0f}% predictive accuracy.")

with st.sidebar.expander("⚖️ Sub-Weight Customization", expanded=False):
    st.markdown("**Sustainability Cost Sub-Weights:**")
    w_energy = st.slider("Energy Weight (Joules)", 0.0, 1.0, 0.35, 0.05)
    w_t_train = st.slider("Train Time Weight", 0.0, 1.0, 0.20, 0.05)
    w_t_pred = st.slider("Inference Latency Weight", 0.0, 1.0, 0.15, 0.05)
    w_cpu = st.slider("CPU Usage Weight", 0.0, 1.0, 0.15, 0.05)
    w_ram = st.slider("Peak RAM Weight", 0.0, 1.0, 0.15, 0.05)

    st.markdown("**Predictive Performance Sub-Weights:**")
    w_acc = st.slider("Accuracy Weight", 0.0, 1.0, 0.25, 0.05)
    w_f1 = st.slider("F1-Score Weight", 0.0, 1.0, 0.35, 0.05)
    w_prec = st.slider("Precision Weight", 0.0, 1.0, 0.20, 0.05)
    w_rec = st.slider("Recall Weight", 0.0, 1.0, 0.20, 0.05)

cost_weights = {
    "energy_joules": w_energy,
    "training_time_s": w_t_train,
    "prediction_time_s": w_t_pred,
    "cpu_avg_pct": w_cpu,
    "ram_peak_mb": w_ram
}
perf_weights = {
    "accuracy": w_acc,
    "f1_score": w_f1,
    "precision": w_prec,
    "recall": w_rec
}

# ----------------- SECTION 5: EVALUATION ACTIONS -----------------
st.sidebar.markdown('<div class="sidebar-section-header">5. Evaluation Actions</div>', unsafe_allow_html=True)

n_runs = st.sidebar.slider("Experimental Runs (Mean ± Std)", 1, 5, 1, help="Run multiple randomized seeds for publication-grade statistical confidence.")

run_btn = st.sidebar.button("🚀 Run Green AI Evaluation", type="primary", use_container_width=True)

if st.sidebar.button("↺ Reset Configuration", use_container_width=True):
    for key in ["results_df", "dataset_meta", "cached_uploaded_file"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.sidebar.caption("⚡ **Telemetry Calibration:** TDP power integration (CPU + DRAM). Carbon intensity calibrated at 475 gCO2eq/kWh. Polling rate: 100ms.")

# ----------------- EXECUTION CONTROLLER -----------------
if run_btn:
    if not selected_models:
        st.error("Please select at least one machine learning model to evaluate.")
        st.stop()

    with st.spinner("Executing dynamic data preprocessing, memory profiling, and model benchmarking..."):
        try:
            # 1. Load Dataset
            if dataset_choice == "Upload Custom CSV":
                if custom_uploaded is None:
                    st.error("Please upload a CSV dataset first.")
                    st.stop()
                if df_custom_loaded is not None:
                    df_input = df_custom_loaded.copy()
                else:
                    custom_uploaded.seek(0)
                    df_input = pd.read_csv(custom_uploaded)
            else:
                df_input = pd.read_csv(csv_path)

            # 2. Preprocess with sampling & memory protection
            pipeline = DynamicDataPipeline(
                target_column=target_col,
                scaler_type=scaler_choice,
                test_size=0.2,
                random_state=42,
                max_categories_onehot=max_categories,
                max_memory_budget_mb=memory_budget_mb,
                max_dense_memory_mb=max_dense_mb,
                sample_size=sample_size,
                sampling_seed=sampling_seed
            )
            X_train, X_test, y_train, y_test, meta = pipeline.prepare(df_input)
            st.session_state["dataset_meta"] = meta

            # 3. Model Evaluation & Hardware Telemetry
            evaluator = GreenAIEvaluator(
                alpha=alpha,
                perf_weights=perf_weights,
                cost_weights=cost_weights,
                max_memory_budget_mb=memory_budget_mb
            )

            if n_runs > 1:
                results_df, _ = evaluator.evaluate_multi_run(
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    models=selected_models,
                    n_runs=n_runs
                )
            else:
                results_df = evaluator.evaluate_all(
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    models=selected_models
                )

            st.session_state["results_df"] = results_df
            st.toast("Evaluation completed successfully!", icon="✅")
            st.rerun()

        except MemoryBudgetExceededError as mem_err:
            st.error(f"🛑 **Memory Budget Safeguard ({mem_err.stage.upper()})**\n\n{mem_err.message}")
            if mem_err.recommendations:
                st.info("💡 **Recommended Resolution Steps:**\n\n" + "\n".join(f"- {rec}" for rec in mem_err.recommendations))
            st.session_state["results_df"] = None

        except Exception as err:
            import traceback
            tb_str = traceback.format_exc()
            st.error(f"Execution Error: {err}")
            with st.expander("🔍 Complete Technical Traceback", expanded=False):
                st.code(tb_str, language="text")
            st.session_state["results_df"] = None

# =========================================================================
# MAIN CONTENT AREA
# =========================================================================

# ----------------- SCENARIO A: WELCOME / EMPTY STATE -----------------
if st.session_state["results_df"] is None:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-hero-title">Welcome to the Green AI Evaluation Framework</div>
        <div class="welcome-hero-text">
            Modern machine learning development prioritizes predictive accuracy at any computational cost ("Red AI").
            This framework establishes a scientific benchmark to evaluate algorithms across both dimensions:
            <b>Predictive Accuracy</b> versus <b>Hardware Resource Footprint & Environmental Impact ("Green AI")</b>.
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
            <div class="pillar-card">
                <div class="pillar-title">🎯 Predictive Performance</div>
                <div class="pillar-desc">
                    Comprehensive multi-class and binary metrics including Accuracy, Precision, Recall, and F1-Score,
                    evaluated with strict data-leakage safeguards.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-title">⚡ Hardware Telemetry</div>
                <div class="pillar-desc">
                    Continuous asynchronous profiling capturing empirical Joules via Thermal Design Power (TDP) integration,
                    peak memory allocation, CPU utilization, and inference latency.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-title">⚖️ Dynamic Green AI Index (GAI)</div>
                <div class="pillar-desc">
                    Multi-attribute decision synthesis uniting accuracy and normalized resource efficiency into linear and
                    harmonic indices with real-time &alpha;-tradeoff tuning.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_w1, col_w2 = st.columns([3, 2])
    with col_w1:
        st.markdown("""
        ### 🚀 How to Run Your First Benchmark
        1. **Select or Upload a Dataset**: Choose from benchmark datasets (*Adult Census*, *Telco Churn*, *Breast Cancer*) or upload your custom CSV in the sidebar.
        2. **Configure Preprocessing & Sampling**: Select feature scaling and choose a sample size (e.g. 10,000 rows for massive datasets) to ensure rapid, memory-safe execution.
        3. **Choose Algorithms & Sustainability Tradeoff**: Select which of the 6 supported architectures to evaluate and adjust the &alpha; slider to express your policy priority.
        4. **Execute**: Click **"Run Green AI Evaluation"** in the sidebar to generate interactive Pareto frontiers, telemetry breakdowns, and publication tables.
        """)
    with col_w2:
        st.markdown("""
        <div class="edu-panel" style="border-left: 4px solid #10b981;">
            <h4 style="margin: 0 0 8px 0; color: #064e3b;">🌿 The Green AI Paradigm</h4>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                <em>"Red AI focuses exclusively on accuracy with exponentially increasing computational costs.
                Green AI advocates for reporting computational efficiency as a first-class evaluation metric alongside accuracy."</em>
                <br><br>
                <strong>— Roy Schwartz et al. (Allen Institute for AI, 2020)</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- SCENARIO B: RESULTS AVAILABLE -----------------
else:
    results_df = st.session_state["results_df"].copy()
    meta = st.session_state["dataset_meta"]

    # Re-calculate GAI dynamically if sliders changed without needing full model re-training
    try:
        calc = GreenAIIndexCalculator(alpha=alpha, perf_weights=perf_weights, cost_weights=cost_weights)
        results_df = calc.calculate_gai(results_df)
    except Exception as e:
        st.warning(f"GAI Weight Recalculation Notice: {e}")

    # 1. Dataset Profile & Data Integrity Banner
    if meta is not None:
        sampling_info = meta.get("sampling", {})
        is_sampled = sampling_info.get("is_sampled", False)
        total_rows = sampling_info.get("total_rows", meta.get("initial_samples", "N/A"))
        eval_rows = sampling_info.get("sampled_rows", meta.get("cleaned_samples", "N/A"))
        seed_used = sampling_info.get("sampling_seed", 42)
        strat_used = "Stratified Split" if sampling_info.get("stratified", False) else "Random Split"
        matrix_repr = "Sparse CSR Matrix" if meta.get("is_sparse", False) else "Dense NumPy Array"
        mem_mb = meta.get("memory_footprint_mb", 0.0)

        st.markdown(f"""
        <div class="info-banner">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px; font-size: 0.9rem;">
                <div>
                    📊 <b>Dataset Profile:</b> Total Samples: <b>{total_rows:,}</b> &rarr; Evaluated: <b>{eval_rows:,}</b>
                    <span style="background: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; margin-left: 6px;">
                        {'Sampled (' + str(eval_rows) + ')' if is_sampled else 'Full Dataset'}
                    </span>
                    &nbsp;|&nbsp; Seed: <b>{seed_used}</b> ({strat_used})
                </div>
                <div>
                    📐 <b>Dimensions:</b> Raw Features: <b>{meta.get('active_features')}</b> &rarr; Processed: <b>{meta.get('processed_features')}</b>
                    &nbsp;|&nbsp; Memory: <b>{matrix_repr} ({mem_mb:.2f} MB)</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dropped Columns & Data Leakage Expanders
        if meta.get("dropped_columns_log"):
            with st.expander(f"📋 Feature Cleaning & Filtering Log ({len(meta['dropped_columns_log'])} dropped)", expanded=False):
                drop_df = pd.DataFrame(meta["dropped_columns_log"])
                st.dataframe(drop_df, use_container_width=True)

        if meta.get("leakage_warnings") or meta.get("dataset_limitations"):
            with st.expander("⚠️ Target Engineering & Feature Safeguard Notes", expanded=False):
                if meta.get("dataset_limitations"):
                    st.markdown(f"**Methodology Limitation:** {meta['dataset_limitations']}")
                if meta.get("leakage_warnings"):
                    for warn in meta["leakage_warnings"]:
                        st.markdown(f"- ⚠️ {warn}")

    # 2. Top KPI Metric Cards (5 Cards)
    best_gai = results_df.iloc[0]
    best_f1 = results_df.sort_values("f1_score", ascending=False).iloc[0]
    best_energy = results_df.sort_values("energy_joules", ascending=True).iloc[0]
    fastest = results_df.sort_values("prediction_time_s", ascending=True).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #059669;">
            <div class="kpi-label">🏆 Green AI Champion</div>
            <div class="kpi-value">{best_gai['model_name']}</div>
            <div class="kpi-subtext">GAI: <b>{best_gai['gai_linear']:.3f}</b> &bull; F1: {best_gai['f1_score']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #2563eb;">
            <div class="kpi-label">🎯 Top Predictor</div>
            <div class="kpi-value">{best_f1['model_name']}</div>
            <div class="kpi-subtext">F1: <b>{best_f1['f1_score']:.3f}</b> &bull; Acc: {best_f1['accuracy']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #0d9488;">
            <div class="kpi-label">🍃 Lowest Energy</div>
            <div class="kpi-value">{best_energy['model_name']}</div>
            <div class="kpi-subtext">Energy: <b>{best_energy['energy_joules']:.2f} J</b> &bull; {best_energy['co2_emissions_g']:.4f}g CO₂</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #d97706;">
            <div class="kpi-label">⚡ Lowest Latency</div>
            <div class="kpi-value">{fastest['model_name']}</div>
            <div class="kpi-subtext">Latency: <b>{fastest['prediction_time_s']*1000:.1f} ms</b> &bull; Total: {fastest['total_time_s']:.3f}s</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #64748b;">
            <div class="kpi-label">📊 Evaluation Scope</div>
            <div class="kpi-value">{len(results_df)} Models</div>
            <div class="kpi-subtext">Sampled: <b>{eval_rows if 'eval_rows' in locals() else len(results_df):,}</b> rows</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Interactive Upgraded Results Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & Leaderboard",
        "🍃 Sustainability Telemetry",
        "🌐 Pareto Frontier Analysis",
        "📋 Detailed Results & Export",
        "🔬 Methodology & Formulations"
    ])

    # ---------------- TAB 1: OVERVIEW & LEADERBOARD ----------------
    with tab1:
        st.markdown("### Model Leaderboard & Composite Green AI Index")
        
        # Leaderboard Table
        disp_cols = [
            "rank_gai", "model_name", "gai_linear", "gai_harmonic",
            "f1_score", "accuracy", "energy_joules",
            "training_time_s", "prediction_time_s", "cpu_avg_pct", "ram_peak_mb",
            "pareto_optimal"
        ]
        lb_df = results_df[[c for c in disp_cols if c in results_df.columns]].copy()
        lb_df.columns = [
            "Rank", "Model", "GAI (Linear)", "GAI (Harmonic)",
            "F1-Score", "Accuracy", "Energy (J)",
            "Train Time (s)", "Pred Time (s)", "CPU (%)", "Peak RAM (MB)",
            "Pareto Optimal"
        ][:len(lb_df.columns)]

        st.dataframe(
            lb_df.style.highlight_max(subset=["GAI (Linear)", "GAI (Harmonic)", "F1-Score", "Accuracy"], color="#d1fae5")
                       .highlight_min(subset=["Energy (J)", "Train Time (s)", "Pred Time (s)"], color="#e0f2fe"),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns([3, 2])

        with col_t1:
            st.markdown("#### Green AI Index (GAI) Model Ranking")
            
            # Interactive Plotly horizontal bar chart
            sorted_chart_df = results_df.sort_values("gai_linear", ascending=True)
            fig_gai = go.Figure()
            
            # Linear GAI Bar
            fig_gai.add_trace(go.Bar(
                y=sorted_chart_df["model_name"],
                x=sorted_chart_df["gai_linear"],
                name="GAI (Linear)",
                orientation="h",
                marker=dict(color="#059669", line=dict(color="#047857", width=1)),
                text=[f"{v:.3f}" for v in sorted_chart_df["gai_linear"]],
                textposition="auto"
            ))
            
            # Harmonic GAI Bar
            fig_gai.add_trace(go.Bar(
                y=sorted_chart_df["model_name"],
                x=sorted_chart_df["gai_harmonic"],
                name="GAI (Harmonic)",
                orientation="h",
                marker=dict(color="#0d9488", line=dict(color="#0f766e", width=1)),
                text=[f"{v:.3f}" for v in sorted_chart_df["gai_harmonic"]],
                textposition="auto"
            ))
            
            fig_gai.update_layout(
                barmode="group",
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Green AI Index Score (Higher = Superior Green AI Balance)", range=[0, 1.05]),
                yaxis=dict(title="")
            )
            st.plotly_chart(fig_gai, use_container_width=True)

        with col_t2:
            st.markdown("#### Academic Trade-Off Synthesis")
            
            # Dynamic calculation of key trade-off metrics
            champ_model = best_gai["model_name"]
            top_perf_model = best_f1["model_name"]
            
            f1_diff = ((best_f1["f1_score"] - best_gai["f1_score"]) / max(1e-6, best_f1["f1_score"])) * 100.0
            energy_saved = ((best_f1["energy_joules"] - best_gai["energy_joules"]) / max(1e-6, best_f1["energy_joules"])) * 100.0
            
            st.markdown(f"""
            <div class="edu-panel" style="border-left: 4px solid #059669;">
                <h5 style="margin: 0 0 6px 0; color: #064e3b;">📌 Key Empirical Findings (&alpha; = {alpha:.2f})</h5>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 18px;">
                    <li><b>Green AI Champion:</b> <code>{champ_model}</code> achieves the top composite score 
                        (GAI: <b>{best_gai['gai_linear']:.3f}</b>).</li>
                    <li><b>Comparative Efficiency:</b> Compared to the top accuracy model (<code>{top_perf_model}</code>), 
                        the champion retains <b>{100.0 - f1_diff:.1f}%</b> of the predictive F1-score while 
                        reducing energy consumption by <b>{max(0.0, energy_saved):.1f}%</b>.</li>
                    <li><b>Multi-Criteria Synthesis:</b> Linear GAI captures direct proportional reward, 
                        whereas Harmonic GAI penalizes models that severely sacrifice either performance or resource efficiency.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick summary metrics table
            st.markdown("**Leaderboard Summary:**")
            st.dataframe(
                results_df[["model_name", "gai_linear", "f1_score", "energy_joules", "pareto_optimal"]].rename(
                    columns={"model_name": "Model", "gai_linear": "GAI", "f1_score": "F1", "energy_joules": "Joules", "pareto_optimal": "Pareto"}
                ),
                use_container_width=True,
                height=180
            )

    # ---------------- TAB 2: SUSTAINABILITY TELEMETRY ----------------
    with tab2:
        st.markdown("### Empirical Sustainability & Hardware Profiling")
        
        st.markdown("""
        <div style="background: #f1f5f9; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-size: 0.86rem; color: #475569;">
            🔬 <b>Measurement Protocol:</b> 
            <b>CPU & RAM</b> are measured via high-frequency OS kernel telemetry hooks &bull; 
            <b>Energy (Joules)</b> is integrated via Thermal Design Power (TDP) empirical model &bull; 
            <b>CO₂ Emissions</b> are estimated using regional electric grid factor (475 gCO₂eq/kWh).
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### Energy Consumption (Joules)")
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Bar(
                x=results_df["model_name"],
                y=results_df["train_energy_joules"],
                name="Training Energy",
                marker_color="#059669"
            ))
            fig_energy.add_trace(go.Bar(
                x=results_df["model_name"],
                y=results_df["pred_energy_joules"],
                name="Inference Energy",
                marker_color="#14b8a6"
            ))
            fig_energy.update_layout(
                barmode="stack",
                template="plotly_white",
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Energy Consumed (Joules)")
            )
            st.plotly_chart(fig_energy, use_container_width=True)

        with col_s2:
            st.markdown("#### Estimated Carbon Emissions (g CO₂eq)")
            fig_co2 = px.bar(
                results_df,
                x="model_name",
                y="co2_emissions_g",
                color="co2_emissions_g",
                color_continuous_scale=["#10b981", "#d97706", "#dc2626"],
                labels={"co2_emissions_g": "CO₂ (grams)", "model_name": "Model"},
                template="plotly_white",
                height=340
            )
            fig_co2.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_co2, use_container_width=True)

        col_s3, col_s4 = st.columns(2)
        
        with col_s3:
            st.markdown("#### Hardware Utilization: CPU vs Peak RAM")
            fig_hw = go.Figure()
            fig_hw.add_trace(go.Bar(
                x=results_df["model_name"],
                y=results_df["cpu_avg_pct"],
                name="Avg CPU (%)",
                marker_color="#3b82f6"
            ))
            fig_hw.add_trace(go.Bar(
                x=results_df["model_name"],
                y=results_df["ram_peak_mb"],
                name="Peak RAM (MB)",
                marker_color="#8b5cf6"
            ))
            fig_hw.update_layout(
                barmode="group",
                template="plotly_white",
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Resource Magnitude")
            )
            st.plotly_chart(fig_hw, use_container_width=True)

        with col_s4:
            st.markdown("#### Execution Latency: Training vs Inference")
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                x=results_df["model_name"],
                y=results_df["training_time_s"],
                name="Training Time (s)",
                marker_color="#f59e0b"
            ))
            fig_lat.add_trace(go.Bar(
                x=results_df["model_name"],
                y=results_df["prediction_time_s"] * 1000.0,
                name="Inference Latency (ms)",
                marker_color="#ec4899"
            ))
            fig_lat.update_layout(
                barmode="group",
                template="plotly_white",
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Duration (seconds / milliseconds)")
            )
            st.plotly_chart(fig_lat, use_container_width=True)

    # ---------------- TAB 3: PARETO FRONTIER ANALYSIS ----------------
    with tab3:
        st.markdown("### Multi-Objective Optimization: Predictive F1-Score vs. Energy (Joules)")
        
        col_p1, col_p2 = st.columns([3, 2])
        
        with col_p1:
            # Interactive Pareto Scatter Plot
            pareto_df = results_df.copy()
            
            # Sort Pareto-optimal points by energy to draw the frontier curve
            pareto_pts = pareto_df[pareto_df["pareto_optimal"]].sort_values("energy_joules")
            
            fig_pareto = go.Figure()
            
            # Draw Pareto Frontier curve
            if len(pareto_pts) > 1:
                fig_pareto.add_trace(go.Scatter(
                    x=pareto_pts["energy_joules"],
                    y=pareto_pts["f1_score"],
                    mode="lines",
                    name="Pareto Frontier",
                    line=dict(color="#10b981", width=2, dash="dash"),
                    hoverinfo="skip"
                ))
            
            # Non-Pareto Models
            non_pareto = pareto_df[~pareto_df["pareto_optimal"]]
            if not non_pareto.empty:
                fig_pareto.add_trace(go.Scatter(
                    x=non_pareto["energy_joules"],
                    y=non_pareto["f1_score"],
                    mode="markers+text",
                    name="Dominated Models",
                    text=non_pareto["model_name"],
                    textposition="top center",
                    marker=dict(size=11, color="#64748b", symbol="circle"),
                    hovertemplate="<b>%{text}</b><br>F1-Score: %{y:.4f}<br>Energy: %{x:.2f} J<br>Status: Dominated<extra></extra>"
                ))
            
            # Pareto-Optimal Models (Starred)
            fig_pareto.add_trace(go.Scatter(
                x=pareto_pts["energy_joules"],
                y=pareto_pts["f1_score"],
                mode="markers+text",
                name="Pareto Optimal ⭐",
                text=pareto_pts["model_name"],
                textposition="bottom right",
                marker=dict(size=16, color="#059669", symbol="star", line=dict(color="#047857", width=1.5)),
                hovertemplate="<b>%{text} ⭐</b><br>F1-Score: %{y:.4f}<br>Energy: %{x:.2f} J<br>Status: Pareto Optimal<extra></extra>"
            ))
            
            fig_pareto.update_layout(
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Energy Consumption (Joules) — Lower is Greener", zeroline=False),
                yaxis=dict(title="Predictive Performance (F1-Score) — Higher is Better", zeroline=False)
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

        with col_p2:
            st.markdown("""
            <div class="edu-panel" style="border-left: 4px solid #0d9488;">
                <h4 style="margin: 0 0 6px 0; color: #064e3b;">🌐 Pareto Optimality Criteria</h4>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin-bottom: 12px;">
                    In multi-objective optimization, a model <i>A</i> <b>dominates</b> model <i>B</i> (<i>A &succ; B</i>) if:
                    <br>&bull; <i>F1(A) &ge; F1(B)</i> and <i>Energy(A) &le; Energy(B)</i>
                    <br>&bull; With strict inequality in at least one objective.
                </p>
                <p style="font-size: 0.86rem; color: #475569; line-height: 1.5; margin: 0;">
                    <b>Non-Dominated Models (Frontier Candidates):</b> Models marked with a star (⭐) cannot be improved in accuracy without consuming additional energy.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Pareto Frontier Candidates:**")
            pareto_disp = pareto_pts[["model_name", "f1_score", "energy_joules", "eco_efficiency_f1_per_joule"]].rename(
                columns={
                    "model_name": "Model",
                    "f1_score": "F1-Score",
                    "energy_joules": "Energy (J)",
                    "eco_efficiency_f1_per_joule": "Eco-Efficiency (F1/J)"
                }
            )
            st.dataframe(pareto_disp, use_container_width=True)

    # ---------------- TAB 4: DETAILED RESULTS & EXPORT ----------------
    with tab4:
        st.markdown("### Comprehensive Evaluation Matrix & Publication Export")
        
        # Searchable and sortable full telemetry matrix
        st.dataframe(results_df, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Export Formatted Results")
        
        # Format IEEE-ready LaTeX table
        cols_latex = [
            "model_name", "accuracy", "precision", "recall", "f1_score",
            "cpu_avg_pct", "ram_peak_mb", "training_time_s", "energy_joules",
            "gai_linear", "rank_gai"
        ]
        latex_sub = results_df[[c for c in cols_latex if c in results_df.columns]].copy()
        latex_sub.columns = [
            "Model", "Accuracy", "Precision", "Recall", "F1",
            "CPU (\\%)", "RAM (MB)", "Train Time (s)", "Energy (J)",
            "GAI", "Rank"
        ][:len(latex_sub.columns)]

        latex_code = latex_sub.to_latex(
            index=False, float_format="%.4f",
            caption=f"Green AI Evaluation Empirical Results ({dataset_choice})",
            label="tab:green_ai_results"
        )
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Full Telemetry (.csv)",
                data=results_df.to_csv(index=False),
                file_name=f"green_ai_{dataset_choice.lower().replace(' ', '_')}_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                label="📥 Download IEEE LaTeX Table (.tex)",
                data=latex_code,
                file_name=f"green_ai_{dataset_choice.lower().replace(' ', '_')}.tex",
                mime="text/plain",
                use_container_width=True
            )

        with st.expander("📄 View Formatted IEEE LaTeX Code", expanded=False):
            st.code(latex_code, language="latex")

    # ---------------- TAB 5: METHODOLOGY & FORMULATIONS ----------------
    with tab5:
        st.markdown("### Mathematical Formulations & Profiling Methodology")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### 1. Green AI Index Formulation")
            st.latex(r"P = \sum_{k} w_{k}^{\text{perf}} \cdot \text{Metric}_{k}")
            st.latex(r"C = \sum_{j} w_{j}^{\text{cost}} \cdot \tilde{C}_{j}, \quad \tilde{C}_{j} \in [0, 1]")
            st.latex(r"R = 1.0 - C \quad (\text{Resource Efficiency})")
            st.latex(r"\text{GAI}_{\text{linear}} = \alpha \cdot P + (1 - \alpha) \cdot R")
            st.latex(r"\text{GAI}_{\text{harmonic}} = \frac{2 \cdot P \cdot R}{P + R + \epsilon}")
            st.markdown("""
            - **Trade-Off Parameter (α):** Bounds policy preference between pure accuracy ($\alpha=1$) and pure environmental sustainability ($\alpha=0$).
            - **Harmonic Formulation:** The Green F-Measure penalizes architectures that achieve high accuracy at catastrophic energy costs.
            """)

        with col_m2:
            st.markdown("#### 2. Resource Telemetry Model")
            st.latex(r"P(t) = P_{\text{idle}} + (P_{\text{TDP}} - P_{\text{idle}}) \times u(t)")
            st.latex(r"E = \int_{0}^{T} P(t) \, dt \approx \sum_{i} P(t_{i}) \cdot \Delta t")
            st.latex(r"\text{CO}_2 = E_{\text{kWh}} \times I_{\text{grid}} \quad (I_{\text{grid}} = 475 \, \text{gCO}_2/\text{kWh})")
            st.markdown(r"""
            - **Background Sampling Daemon:** High-precision daemon thread polls CPU processor load $u(t)$ and process resident set size (RSS) every 100 milliseconds.
            - **Thermal Design Power (TDP):** Baseline calibrated to $65\,\text{W}$ CPU envelope.
            """)

        st.markdown("---")
        st.markdown("#### 3. Data Pipeline & Memory Safeguards")
        st.markdown("""
        - **Frequency-Based Categorical Bounding:** Automatically caps one-hot encoded categories at a configurable limit (default: 30) per feature; tail categories are grouped into `'Other'`.
        - **Zero-Copy Sparse Representation:** Retains high-dimensional preprocessed matrices in memory-efficient SciPy CSR sparse format when dense memory exceeds threshold.
        - **Data Leakage Safeguards:** Explicitly drops identifier columns (e.g. `CustomerID`, `InvoiceNo`) and handles missing values using strictly split-safe transformers.
        """)
