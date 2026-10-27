"""
Model Registry & Factory Module
Supports the six canonical algorithms evaluated by the Dynamic Green AI Evaluation Framework:
1. Decision Tree (Scikit-Learn)
2. Random Forest (Scikit-Learn)
3. XGBoost (XGBClassifier)
4. LightGBM (LGBMClassifier)
5. CatBoost (CatBoostClassifier)
6. Logistic Regression (Scikit-Learn)

Enforces strict algorithmic authenticity: no model is ever silently replaced with a surrogate.
If an optional native binary is unavailable, a clear, actionable dependency error is raised.
"""

import warnings
from typing import Dict, Any, List
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier as NativeCatBoostClassifier
    CATBOOST_NATIVE_AVAILABLE = True
except (ImportError, Exception):
    NativeCatBoostClassifier = None
    CATBOOST_NATIVE_AVAILABLE = False


class ModelFactory:
    """
    Factory providing standardized, reproducible instances of the 6 canonical benchmark models.
    """
    SUPPORTED_MODELS = [
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Logistic Regression"
    ]

    @staticmethod
    def get_model(name: str, random_state: int = 42, **kwargs) -> Any:
        """
        Instantiate a classification model by name with standardized defaults.
        Parameters:
        -----------
        name : str
            One of the 6 canonical model names.
        random_state : int
            Reproducibility seed.
        **kwargs :
            Additional model-specific hyperparameters.
        """
        name_clean = name.strip()

        if name_clean == "Decision Tree":
            params = {
                "max_depth": kwargs.get("max_depth", 10),
                "min_samples_split": kwargs.get("min_samples_split", 5),
                "random_state": random_state
            }
            return DecisionTreeClassifier(**params)

        elif name_clean == "Random Forest":
            params = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 12),
                "n_jobs": kwargs.get("n_jobs", -1),
                "random_state": random_state
            }
            return RandomForestClassifier(**params)

        elif name_clean == "XGBoost":
            params = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 6),
                "learning_rate": kwargs.get("learning_rate", 0.1),
                "random_state": random_state,
                "n_jobs": kwargs.get("n_jobs", -1),
                "verbosity": 0
            }
            # Allow custom eval_metric if specified
            if "eval_metric" in kwargs:
                params["eval_metric"] = kwargs["eval_metric"]
            return XGBClassifier(**params)

        elif name_clean == "LightGBM":
            params = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "max_depth": kwargs.get("max_depth", 6),
                "num_leaves": kwargs.get("num_leaves", 31),
                "learning_rate": kwargs.get("learning_rate", 0.1),
                "random_state": random_state,
                "n_jobs": kwargs.get("n_jobs", -1),
                "verbose": -1
            }
            return LGBMClassifier(**params)

        elif name_clean == "CatBoost":
            if not CATBOOST_NATIVE_AVAILABLE or NativeCatBoostClassifier is None:
                raise ImportError(
                    "Native CatBoost is not available in the current environment. "
                    "Install with `pip install catboost`. Substitution with an alternate algorithm is prohibited."
                )
            params = {
                "iterations": kwargs.get("iterations", 100),
                "depth": kwargs.get("depth", 6),
                "learning_rate": kwargs.get("learning_rate", 0.1),
                "random_seed": random_state,
                "verbose": 0,
                "thread_count": kwargs.get("thread_count", -1)
            }
            return NativeCatBoostClassifier(**params)

        elif name_clean == "Logistic Regression":
            params = {
                "max_iter": kwargs.get("max_iter", 1000),
                "solver": kwargs.get("solver", "lbfgs"),
                "random_state": random_state
            }
            return LogisticRegression(**params)

        else:
            raise ValueError(
                f"Unsupported model: '{name}'. Supported models: {ModelFactory.SUPPORTED_MODELS}"
            )

    @staticmethod
    def get_all_models(random_state: int = 42) -> Dict[str, Any]:
        """Returns instantiated dictionary of all 6 models."""
        return {
            name: ModelFactory.get_model(name, random_state=random_state)
            for name in ModelFactory.SUPPORTED_MODELS
        }
