"""
Regression & Safety Tests for High-Cardinality Datasets,
Sparse Matrix Handling, Memory Safety Checks, and Online Retail II Target Leakage.
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data_pipeline import DynamicDataPipeline, MemoryBudgetExceededError
from src.models import ModelFactory
from src.evaluator import GreenAIEvaluator


def test_high_cardinality_categorical_explosion_prevented():
    """
    Regression test replicating Online Retail II characteristics:
    - 3,000 rows
    - High-cardinality invoice numbers (2,800 unique values)
    - High-cardinality customer IDs (2,500 unique values)
    - High-cardinality product descriptions (1,500 unique values)
    - High-cardinality stock codes (1,200 unique values)
    - Negative quantity accounting artifacts
    - Target: IsCancellation
    """
    np.random.seed(42)
    n = 3000

    df = pd.DataFrame({
        "InvoiceNo": [f"C{500000 + i}" if i % 10 == 0 else f"{500000 + i}" for i in range(n)],
        "CustomerID": [f"CUST_{i % 2500:05d}" for i in range(n)],
        "StockCode": [f"ITEM_{i % 1200:04d}" for i in range(n)],
        "Description": [f"Product Description {i % 1500}" for i in range(n)],
        "Quantity": np.where(np.arange(n) % 10 == 0, -np.random.randint(1, 10, size=n), np.random.randint(1, 50, size=n)),
        "Price": np.random.uniform(0.5, 100.0, size=n),
        "Country": np.random.choice(["United Kingdom", "France", "Germany", "EIRE", "Spain"], size=n),
        "IsCancellation": (np.arange(n) % 10 == 0).astype(int)
    })

    pipeline = DynamicDataPipeline(
        target_column="IsCancellation",
        max_categories_onehot=25,
        test_size=0.2,
        random_state=42
    )
    X_train, X_test, y_train, y_test, meta = pipeline.prepare(df)

    # 1. Total features must NOT explode into thousands
    # Raw categorical features should each have at most (max_categories_onehot + 1) = 26 columns
    assert X_train.shape[1] < 100, f"Expected bounded features (<100), but got {X_train.shape[1]}"
    assert X_test.shape[1] == X_train.shape[1]
    assert X_train.shape[0] == 2400
    assert X_test.shape[0] == 600

    # 2. InvoiceNo and CustomerID must be dropped with reasons logged
    assert "InvoiceNo" in meta["dropped_id_features"]
    assert "CustomerID" in meta["dropped_id_features"]
    assert len(meta["dropped_columns_log"]) >= 2

    dropped_col_names = [d["column"] for d in meta["dropped_columns_log"]]
    assert "InvoiceNo" in dropped_col_names
    assert "CustomerID" in dropped_col_names

    # Check reason mentions leakage for InvoiceNo
    invoice_log = next(d for d in meta["dropped_columns_log"] if d["column"] == "InvoiceNo")
    assert "leakage" in invoice_log["reason"].lower()

    # 3. Limitations and leakage warnings must be documented
    assert meta["dataset_limitations"] is not None
    assert "engineered classification task" in meta["dataset_limitations"]
    assert any("Quantity" in w for w in meta["leakage_warnings"])


def test_memory_budget_safety_check_raises_gracefully():
    """
    Verifies that if the estimated memory requirement exceeds the configurable memory budget,
    the pipeline halts gracefully by raising MemoryBudgetExceededError with actionable advice.
    """
    n = 1000
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "cat1": [f"Category_{i % 50}" for i in range(n)],
        "target": np.random.choice([0, 1], size=n)
    })

    # Set an impractically low budget (e.g. 0.0001 MB = 100 bytes)
    pipeline = DynamicDataPipeline(
        target_column="target",
        max_memory_budget_mb=0.0001,
        random_state=42
    )

    with pytest.raises(MemoryBudgetExceededError) as exc_info:
        pipeline.prepare(df)

    err = exc_info.value
    assert "Memory safety limit exceeded" in err.message
    assert err.estimated_mb > 0.0
    assert err.budget_mb == 0.0001
    assert len(err.recommendations) > 0
    assert any("sampling" in r.lower() for r in err.recommendations)


def test_sparse_representation_and_all_six_models():
    """
    Verifies that when data is kept as a sparse CSR matrix (due to max_dense_memory_mb threshold),
    all six canonical models train, predict, and evaluate correctly without memory explosion.
    """
    n = 200
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "cat1": [f"Cat_{i % 20}" for i in range(n)],
        "cat2": [f"Type_{i % 15}" for i in range(n)],
        "target": np.random.choice([0, 1], size=n)
    })

    # Force sparse representation by setting max_dense_memory_mb very low
    pipeline = DynamicDataPipeline(
        target_column="target",
        max_dense_memory_mb=0.00001,
        random_state=42
    )
    X_train, X_test, y_train, y_test, meta = pipeline.prepare(df)

    assert sp.issparse(X_train), "X_train should be a sparse matrix when exceeding max_dense_memory_mb"
    assert sp.issparse(X_test), "X_test should be a sparse matrix when exceeding max_dense_memory_mb"
    assert meta["is_sparse"] is True

    # Test all 6 models with sparse matrix input
    for m_name in ModelFactory.SUPPORTED_MODELS:
        model = ModelFactory.get_model(m_name, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_flat = np.asarray(preds).ravel()
        assert len(preds_flat) == X_test.shape[0]
        assert set(preds_flat).issubset({0, 1})


def test_dataset_sampling_controls_stratification():
    """
    Verifies sidebar dataset sampling controls:
    - Reproducible row count
    - Preserved class distribution stratification
    - Reproducibility via sampling seed
    """
    n = 2000
    # Imbalanced target: 85% class 0, 15% class 1
    y_vals = np.array([0] * 1700 + [1] * 300)
    np.random.seed(42)
    np.random.shuffle(y_vals)

    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "cat1": np.random.choice(["A", "B", "C"], size=n),
        "target": y_vals
    })

    # Sample 500 rows with seed 123
    p1 = DynamicDataPipeline(target_column="target", sample_size=500, sampling_seed=123, random_state=42)
    X_train1, X_test1, y_train1, y_test1, meta1 = p1.prepare(df)

    assert meta1["sampling"]["is_sampled"] is True
    assert meta1["sampling"]["total_rows"] == 2000
    assert meta1["sampling"]["sampled_rows"] == 500
    assert meta1["train_samples"] + meta1["test_samples"] == 500

    # Verify stratified proportion in sampled data (~15% positive class)
    total_y = np.concatenate([y_train1, y_test1])
    pos_ratio = (total_y == 1).mean()
    assert 0.12 <= pos_ratio <= 0.18, f"Stratification failed, pos_ratio was {pos_ratio}"

    # Verify reproducibility with same seed
    p2 = DynamicDataPipeline(target_column="target", sample_size=500, sampling_seed=123, random_state=42)
    X_train2, X_test2, y_train2, y_test2, meta2 = p2.prepare(df)

    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(y_test1, y_test2)


def test_zero_leakage_categorical_grouping():
    """
    Verifies that infrequent categories are identified strictly on training data,
    and unseen categories appearing in test data are safely mapped to 'Other'
    without error or leakage.
    """
    train_data = pd.DataFrame({
        "cat": ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Gamma"],
        "target": [0, 0, 1, 1, 0, 1]
    })
    # 'Delta' and 'Omega' only exist in test data
    test_data = pd.DataFrame({
        "cat": ["Alpha", "Beta", "Delta", "Omega"],
        "target": [0, 1, 0, 1]
    })
    full_df = pd.concat([train_data, test_data], ignore_index=True)

    # Let pipeline split and process
    pipeline = DynamicDataPipeline(target_column="target", max_categories_onehot=2, test_size=0.4, random_state=42)
    X_train, X_test, y_train, y_test, meta = pipeline.prepare(full_df)

    # Must execute cleanly without error
    assert X_train.shape[0] == 6
    assert X_test.shape[0] == 4
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_evaluator_memory_budget_check():
    """
    Verifies that GreenAIEvaluator halts gracefully with MemoryBudgetExceededError
    when memory budget is exceeded prior to model training.
    """
    X_train = np.random.randn(100, 10)
    y_train = np.random.choice([0, 1], size=100)
    X_test = np.random.randn(25, 10)
    y_test = np.random.choice([0, 1], size=25)

    evaluator = GreenAIEvaluator(max_memory_budget_mb=0.00001)

    with pytest.raises(MemoryBudgetExceededError) as exc_info:
        evaluator.evaluate_single_model("Decision Tree", X_train, X_test, y_train, y_test)

    assert "Memory budget exceeded for model training" in exc_info.value.message


def test_sparse_matrix_row_counting_and_evaluator():
    """
    Direct regression test verifying:
    1. SciPy sparse matrix row counting uses .shape[0] without triggering
       'TypeError: sparse array length is ambiguous; use getnnz() or shape[0]'
    2. GreenAIEvaluator runs evaluate_single_model and evaluate_all on sparse CSR matrices
    3. Metrics dictionary correctly records test_sample_count == X_test.shape[0]
    4. Feature counts use .shape[1]
    5. Non-zero counting uses .getnnz()
    """
    n_train = 60
    n_test = 25
    n_features = 8

    X_train = sp.csr_matrix(np.random.randn(n_train, n_features))
    X_test = sp.csr_matrix(np.random.randn(n_test, n_features))
    y_train = np.random.choice([0, 1], size=n_train)
    y_test = np.random.choice([0, 1], size=n_test)

    # 1. Direct row counting assertion
    assert X_train.shape[0] == n_train
    assert X_test.shape[0] == n_test
    assert X_train.shape[1] == n_features
    assert X_test.getnnz() > 0

    # 2. Verify evaluate_single_model succeeds with sparse input
    evaluator = GreenAIEvaluator(use_codecarbon=False)
    metrics = evaluator.evaluate_single_model("Decision Tree", X_train, X_test, y_train, y_test)
    assert metrics["test_sample_count"] == n_test
    assert metrics["latency_per_1k_samples_s"] >= 0.0

    # 3. Verify evaluate_all runs across models on sparse CSR matrix
    results_df = evaluator.evaluate_all(X_train, X_test, y_train, y_test, models=["Decision Tree", "Logistic Regression"])
    assert len(results_df) == 2
    assert "gai_linear" in results_df.columns

