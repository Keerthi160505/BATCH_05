"""
Comprehensive Unit Tests for DynamicDataPipeline
Validates:
- Synthetic dataset preparation with numerical, categorical, and ID columns
- Data leakage prevention (verifying fitted parameters derive strictly from X_train)
- All four scaler options (standard, minmax, robust, none)
- Duplicate handling
- Error handling on invalid/unsupported datasets (empty df, missing target, single class)
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_pipeline import DynamicDataPipeline


def test_pipeline_synthetic_dataset():
    # Create synthetic dataset with numeric, categorical, missing values, and an ID column
    n = 200
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(n)],
        "num_feat1": np.random.randn(n),
        "num_feat2": np.random.uniform(10, 50, size=n),
        "cat_feat": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n)
    })
    # Inject missing values
    df.loc[10, "num_feat1"] = np.nan
    df.loc[20, "cat_feat"] = np.nan

    pipeline = DynamicDataPipeline(target_column="target", scaler_type="standard", test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test, meta = pipeline.prepare(df)

    assert X_train.shape[0] == 150
    assert X_test.shape[0] == 50
    assert y_train.shape[0] == 150
    assert y_test.shape[0] == 50
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert meta["processed_features"] >= 4
    assert "id" in meta["dropped_id_features"]
    assert meta["classes"] == 2


def test_pipeline_scalers():
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    for s_type in ["standard", "minmax", "robust", "none"]:
        p = DynamicDataPipeline(target_column="target", scaler_type=s_type, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test, meta = p.prepare(df)
        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert meta["scaler_applied"] == s_type


def test_data_leakage_prevention():
    """
    Verifies that the fitted transformer parameters (e.g. mean, scale) are computed
    STRICTLY on the training split, and not altered by the test split.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "x": np.arange(100, dtype=float),
        "target": np.random.choice([0, 1], size=100)
    })

    p = DynamicDataPipeline(target_column="target", scaler_type="standard", test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test, meta = p.prepare(df)

    # Scaler must be strictly centered and scaled on training set
    assert np.isclose(np.mean(X_train), 0.0, atol=1e-7), "X_train must have zero mean after fitting strictly on train"
    assert np.isclose(np.std(X_train), 1.0, atol=1e-7), "X_train must have unit variance after fitting strictly on train"
    
    # Test set transformed with train parameters should not have zero mean
    assert not np.isclose(np.mean(X_test), 0.0, atol=1e-3), "X_test should not have zero mean because scaler was fitted exclusively on X_train"


def test_duplicate_removal():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    p = DynamicDataPipeline(target_column="target", drop_duplicates=True, test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test, meta = p.prepare(df)
    assert meta["duplicates_removed"] == 2
    assert meta["cleaned_samples"] == 6


def test_invalid_and_unsupported_inputs():
    # Empty DataFrame
    p = DynamicDataPipeline(target_column="target")
    with pytest.raises(ValueError, match="empty or None"):
        p.prepare(pd.DataFrame())

    # Missing target column
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    p_wrong = DynamicDataPipeline(target_column="nonexistent")
    with pytest.raises(ValueError, match="not found in DataFrame"):
        p_wrong.prepare(df)

    # Single-class target
    df_single = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": [1, 1, 1, 1, 1, 1]})
    with pytest.raises(ValueError, match="at least 2 distinct"):
        p.prepare(df_single, target_col="target")

    # Insufficient samples
    df_small = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match="Insufficient samples"):
        p.prepare(df_small, target_col="target")
