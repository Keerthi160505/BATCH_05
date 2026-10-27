# A Dynamic Green AI Evaluation Framework: Multi-Criteria Trade-Off Analysis of Predictive Performance and Computational Sustainability in Machine Learning

**Authors**: Anonymous (Final Year Undergraduate Capstone Research)  
**Affiliation**: Department of Computer Science & Engineering  
**Target Venue**: IEEE / Springer Sustainable Computing & AI Conferences / Journals  

---

## Abstract
Machine learning (ML) models are traditionally evaluated almost exclusively using accuracy, precision, recall, and F1-score. However, these conventional measures fail to quantify computational overhead, including CPU utilization, peak memory footprint, execution latency, and carbon-emitting energy consumption. In response to the growing environmental concerns surrounding "Red AI", this paper introduces a **Dynamic Green AI Evaluation Framework** designed to benchmark machine learning algorithms across both predictive fidelity and environmental sustainability. The framework automates tabular dataset ingestion, missing value imputation, categorical encoding, feature scaling, and zero-leakage stratified splitting. It rigorously profiles six foundational algorithms: **Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, and Logistic Regression**. For each algorithm, comprehensive quantitative metrics are captured: Accuracy, Precision, Recall, F1-score, Average CPU Usage, Peak RAM, Training Latency, Inference Latency, and Energy Consumption (Joules). These metrics are harmonized through a novel **Green AI Index (GAI)** that computes Pareto-optimal tradeoffs and models decision-maker sustainability preferences. Empirical evaluation across canonical benchmarks (UCI Adult Census, Telco Churn, and Breast Cancer) reveals that complex gradient boosting ensembles consume between **$5.9\times$ and $8.4\times$ more energy** to gain less than **$1.8\%$ improvement in F1-score** compared to lightweight decision trees, establishing the practical utility of GAI for sustainable AI deployment.

---

## 1. Introduction
Over the past decade, advancements in machine learning have been predominantly driven by the "Red AI" paradigm, wherein marginal improvements in benchmark accuracy are obtained at the expense of exponential increases in computational complexity, power consumption, and environmental carbon emissions. Modern engineering practices routinely select top-performing models based strictly on predictive metrics such as F1-score, Area Under the ROC Curve (AUC), or Classification Accuracy. However, these metrics remain blind to operational footprints: a model requiring hundreds of Joules per inference cycle is treated as strictly superior to a lightweight counterpart that consumes a fraction of the energy while delivering virtually identical predictive utility.

This asymmetry is particularly detrimental in edge computing, embedded systems, Internet of Things (IoT), and high-throughput enterprise systems, where energy budgets, hardware constraints, and cooling costs dominate operational realities. Furthermore, as international environmental regulations and corporate ESG (Environmental, Social, and Governance) commitments intensify, data science practitioners require standardized, objective, and reproducible methodologies to quantify the eco-efficiency of their predictive pipelines.

To bridge this critical gap, this paper presents a **Dynamic Green AI Evaluation Framework**. The primary contributions of this work are as follows:
1. **Dataset-Independent Automated Pipeline**: A unified preprocessing engine that automatically detects feature schemas, imputes missing observations, applies one-hot encoding, and standardizes continuous distributions without statistical data leakage.
2. **High-Frequency Hardware Profiling**: An asynchronous, non-intrusive runtime monitor tracking system-level and process-level CPU percentages, peak RAM consumption (MB), sub-millisecond execution times, and numerical Riemann integration of energy draw (Joules and kWh) using Thermal Design Power (TDP) modeling and CodeCarbon tracking.
3. **Formal Green AI Index (GAI)**: A mathematically grounded multi-criteria metric combining predictive accuracy and resource efficiency through linear parameter sweeping and harmonic mean formulations, accompanied by Pareto frontier extraction.
4. **Empirical Benchmarking across Six Canonical Architectures**: Comprehensive evaluation of Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, and Logistic Regression on stable real-world benchmarks.

---

## 2. Methodology & Mathematical Formulation

### 2.1 Automated Data Pipeline
Given an arbitrary tabular dataset $D = \{(x_i, y_i)\}_{i=1}^N$ with feature vector $x_i \in \mathbb{R}^d$ and label $y_i \in \mathcal{Y}$, the pipeline executes:
- **Type Inference & Cleaning**: Continuous numerical features $\mathcal{F}_{num}$ and discrete categorical features $\mathcal{F}_{cat}$ are segregated. Identification columns are detected via uniqueness heuristics ($>95\%$ unique) and removed.
- **Missing Value Imputation**: Continuous features are imputed using median estimation: $\hat{x}_{j} = \text{median}(\{x_{i,j}\})$, while categorical features are filled using mode estimation.
- **One-Hot Encoding & Standardization**: Categorical attributes are converted to binary vectors via dummy encoding. Continuous attributes are standardized to zero mean and unit variance:
  $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
  All transformations are fitted strictly on the training partition $D_{train}$ and applied to $D_{test}$ without leakage.

### 2.2 Asynchronous Hardware Profiling
An asynchronous daemon runs at interval $\Delta t = 20\text{ ms}$ sampling process and system metrics:
- **CPU Utilization ($U_{cpu}$)**: Monitored via system kernel counters.
- **Memory Footprint ($M_{rss}$)**: Peak Resident Set Size in MB and non-negative delta allocation $\Delta M = \max(0.0, \max_t M(t) - M(0))$.
- **Power and Energy ($E$)**: Instantaneous power $P(t)$ in Watts is modeled via:
  $$P(t) = P_{idle} + \left(\frac{U_{cpu}(t)}{100}\right) \cdot (P_{tdp} - P_{idle}) + P_{dram}$$
  Total energy consumed in Joules:
  $$E = \int_{0}^T P(t) \, dt \approx \sum_{k=1}^K P(t_k) \cdot \Delta t_k$$
  Equivalent carbon emission in grams of $\text{CO}_2$:
  $$m_{CO_2} = \left(\frac{E}{3.6 \times 10^6}\right) \times CI_{grid}$$

### 2.3 Mathematical Formulation of the Green AI Index (GAI)
Let $\mathcal{M} = \{m_1, m_2, \dots, m_K\}$ represent the candidate algorithms.

1. **Performance Composite Score ($P_m$)**:
   $$P_m = w_{acc} \text{Acc}_m + w_{prec} \text{Prec}_m + w_{rec} \text{Rec}_m + w_{f1} \text{F1}_m$$
   where $\sum w_i = 1.0$ and $P_m \in [0, 1]$.

2. **Normalized Resource Cost ($C_m$)**:
   Each computational metric $X \in \{E, T_{train}, T_{pred}, U_{cpu}, M_{rss}\}$ is Min-Max normalized across candidate models:
   $$\widetilde{X}_m = \frac{X_m - \min_{k} X_k}{\max_{k} X_k - \min_{k} X_k + \epsilon} \in [0, 1]$$
   The aggregated resource cost is:
   $$C_m = \lambda_e \widetilde{E}_m + \lambda_{tr} \widetilde{T}_{train,m} + \lambda_{pr} \widetilde{T}_{pred,m} + \lambda_{cpu} \widetilde{U}_{cpu,m} + \lambda_{ram} \widetilde{M}_{rss,m}$$

3. **Sustainability Efficiency Score ($R_m$)**:
   $$R_m = 1.0 - C_m \in [0, 1]$$

4. **Parametric Green AI Index ($GAI_{\alpha}$)**:
   $$GAI_{\alpha}(m) = \alpha \cdot P_m + (1 - \alpha) \cdot R_m$$
   where $\alpha \in [0, 1]$ parameterizes stakeholder trade-off priorities.

5. **Harmonic Green AI Index ($GAI_{harm}$)**:
   $$GAI_{harm}(m) = \frac{2 \cdot P_m \cdot R_m}{P_m + R_m + \epsilon}$$

6. **Pareto Dominance**:
   A model $m_A$ Pareto-dominates model $m_B$ if $P_{m_A} \ge P_{m_B}$ and $E_{m_A} \le E_{m_B}$, with at least one strict inequality.

---

## 3. Empirical Experimental Results

### 3.1 UCI Adult Census Income Benchmark ($N=25,000$, 104 Processed Features, $\alpha=0.50$)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Avg CPU (%) | Peak RAM (MB) | Train Time (s) | Energy (Joules) | GAI ($\alpha=0.5$) | Pareto Optimal |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Decision Tree** | 0.8596 | 0.8570 | 0.8596 | 0.8581 | 14.51% | 297.80 | 0.2096 | **4.8366** | **0.9261** | **Yes** |
| **2** | **LightGBM** | 0.8782 | 0.8743 | 0.8782 | 0.8751 | 56.40% | 299.10 | 0.1977 | **11.3034** | **0.8512** | **Yes** |
| **3** | **Logistic Regression** | 0.8568 | 0.8509 | 0.8568 | 0.8520 | 28.12% | 302.40 | 0.3672 | **8.5443** | 0.8456 | No |
| **4** | **XGBoost** | **0.8792** | **0.8753** | **0.8792** | **0.8759** | 62.40% | 300.50 | 0.4982 | 28.6686 | 0.7243 | **Yes** |
| **5** | **Random Forest** | 0.8688 | 0.8638 | 0.8688 | 0.8615 | 51.30% | 315.80 | 0.7765 | 30.1454 | 0.5370 | No |
| **6** | **CatBoost** | 0.8710 | 0.8661 | 0.8710 | 0.8664 | 42.10% | 319.20 | 1.1561 | 40.4047 | 0.5368 | No |

### 3.2 Telco Customer Churn Benchmark ($N=7,043$, 45 Processed Features, $\alpha=0.50$)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Avg CPU (%) | Peak RAM (MB) | Train Time (s) | Energy (Joules) | GAI ($\alpha=0.5$) | Pareto Optimal |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Logistic Regression** | **0.8055** | **0.7963** | **0.8055** | **0.8002** | 25.60% | 319.50 | 0.0805 | **1.8314** | **0.8834** | **Yes** |
| **2** | **Decision Tree** | 0.7559 | 0.7547 | 0.7559 | 0.7552 | 10.20% | 318.40 | 0.0810 | **1.6883** | **0.8610** | **Yes** |
| **3** | **LightGBM** | 0.7913 | 0.7801 | 0.7913 | 0.7847 | 48.30% | 320.10 | 0.3228 | 7.0559 | 0.7375 | No |
| **4** | **XGBoost** | 0.7963 | 0.7858 | 0.7963 | 0.7902 | 54.10% | 322.40 | 0.7950 | 15.4784 | 0.6246 | No |
| **5** | **CatBoost** | 0.8041 | 0.7925 | 0.8041 | 0.7972 | 43.50% | 330.60 | 1.1316 | 22.2242 | 0.5302 | No |
| **6** | **Random Forest** | 0.7984 | 0.7891 | 0.7984 | 0.7929 | 50.80% | 326.20 | 0.6713 | 14.3319 | 0.5085 | No |

### 3.3 UCI Breast Cancer Diagnostic Benchmark ($N=569$, 30 Continuous Features, $\alpha=0.50$)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Avg CPU (%) | Peak RAM (MB) | Train Time (s) | Energy (Joules) | GAI ($\alpha=0.5$) | Pareto Optimal |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Logistic Regression** | **0.9825** | **0.9827** | **0.9825** | **0.9825** | 18.20% | 291.80 | 0.0179 | **0.0302** | **0.9672** | **Yes** |
| **2** | **Decision Tree** | 0.9035 | 0.9058 | 0.9035 | 0.9045 | 8.40% | 290.40 | 0.0164 | 0.1566 | 0.9487 | No |
| **3** | **LightGBM** | 0.9649 | 0.9654 | 0.9649 | 0.9647 | 44.50% | 292.60 | 0.1185 | 2.5096 | 0.8669 | No |
| **4** | **Random Forest** | 0.9561 | 0.9563 | 0.9561 | 0.9560 | 46.20% | 298.50 | 0.4626 | 9.6759 | 0.7522 | No |
| **5** | **XGBoost** | 0.9474 | 0.9470 | 0.9474 | 0.9471 | 51.10% | 297.10 | 0.6792 | 14.1658 | 0.6934 | No |
| **6** | **CatBoost** | 0.9561 | 0.9558 | 0.9561 | 0.9558 | 40.70% | 305.20 | 1.2252 | 23.9141 | 0.5610 | No |

---

## 4. Key Academic Findings & Discussion
1. **The Cost of Marginal Accuracy**: On Adult Census, XGBoost achieved the highest F1-score ($0.8759$), but consumed $5.93\times$ more energy than Decision Tree ($28.67\text{ J}$ vs $4.84\text{ J}$) for a modest $1.78\%$ accuracy improvement. CatBoost required $8.35\times$ more energy ($40.40\text{ J}$).
2. **LightGBM as the Eco-Efficiency Leader**: LightGBM attained virtually identical predictive accuracy to XGBoost ($F1 = 0.8751$ vs $0.8759$) while consuming **$60.6\%$ less energy** ($11.30\text{ J}$ vs $28.67\text{ J}$) and training $2.5\times$ faster.
3. **Pareto Dominance**: On Adult Census, only Decision Tree, LightGBM, and XGBoost lie on the Pareto efficiency frontier. Random Forest and CatBoost are strictly dominated in this tabular setting.
4. **Sensitivity Crossover**: Under low to moderate $\alpha \le 0.65$, Decision Tree and LightGBM dominate the Green AI Index rankings. At extreme accuracy weights ($\alpha > 0.68$), XGBoost takes first place.

---

## 5. Threats to Validity & Methodological Limitations
1. **TDP-Calibrated Power Modeling vs. Physical Measurement**: Energy consumption is estimated using Thermal Design Power (TDP) scaling combined with dynamic memory power models. While standard in software-based Green AI research, it represents an empirical estimation rather than direct hardware wattmeter readings.
2. **CPU-Only Scope**: The framework benchmarks standard CPU tabular model training and inference. GPU acceleration is neither used nor claimed.
3. **Operating System Scheduling Noise**: Multi-tenant operating systems introduce non-deterministic process scheduling. Performing multi-run averages mitigates this variance.
4. **Carbon Grid Intensity Factor**: The baseline grid intensity of $475\text{ gCO}_2\text{eq/kWh}$ represents a global average and fluctuates geographically.

---

## 6. Conclusion
The Dynamic Green AI Evaluation Framework introduces an objective, multi-criteria paradigm for machine learning model selection. By unifying predictive fidelity with hardware and carbon footprint measurement, the Green AI Index empowers researchers and practitioners to systematically identify sustainable, high-performing algorithms.
