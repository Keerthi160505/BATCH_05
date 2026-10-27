"""
Unit Tests for Model Factory
Validates:
- Training and inference of all 6 canonical algorithms on binary classification
- Multiclass classification support across all 6 models
- Error handling on unsupported model requests
"""

import pytest
import numpy as np
from src.models import ModelFactory


def test_all_six_models_train_and_predict_binary():
    X_train = np.random.randn(50, 4)
    y_train = np.random.choice([0, 1], size=50)
    X_test = np.random.randn(20, 4)

    expected_models = [
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Logistic Regression"
    ]

    for m_name in expected_models:
        model = ModelFactory.get_model(m_name, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_flat = np.asarray(preds).ravel()
        assert len(preds_flat) == 20
        assert set(preds_flat).issubset({0, 1})


def test_all_six_models_multiclass():
    # 3-class synthetic classification
    X_train = np.random.randn(60, 4)
    y_train = np.random.choice([0, 1, 2], size=60)
    X_test = np.random.randn(15, 4)

    for m_name in ModelFactory.SUPPORTED_MODELS:
        model = ModelFactory.get_model(m_name, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_flat = np.asarray(preds).ravel()
        assert len(preds_flat) == 15
        assert set(preds_flat).issubset({0, 1, 2})


def test_unsupported_model_exception():
    with pytest.raises(ValueError, match="Unsupported model"):
        ModelFactory.get_model("NonExistentNeuralNetwork")
