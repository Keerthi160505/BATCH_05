"""
Dynamic Green AI Data Pipeline
Provides automated, dataset-independent preparation:
- Automatic feature type inference (numerical vs categorical vs ID)
- Data cleaning (handling missing values, invalid entries, duplicates)
- High-cardinality categorical explosion prevention & 'Other' grouping
- Direct & indirect target leakage detection (Online Retail II IsCancellation)
- Pre-encoding and pre-training memory safety checks
- Sparse-compatible matrix management with dense memory guards
- Zero-leakage stratified train-test splitting and dataset sampling
"""

import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Dict, Any, Optional, List, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class MemoryBudgetExceededError(ValueError):
    """
    Raised when estimated memory requirement for preprocessing or model training
    exceeds the configured memory budget.
    """
    def __init__(
        self,
        message: str,
        estimated_mb: float = 0.0,
        budget_mb: float = 0.0,
        stage: str = "preprocessing",
        recommendations: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.estimated_mb = float(estimated_mb)
        self.budget_mb = float(budget_mb)
        self.stage = stage
        self.recommendations = recommendations or [
            "Enable dataset sampling (e.g., 10,000, 25,000, or 50,000 rows) in the sidebar or via sample_size.",
            "Reduce 'max_categories_onehot' (e.g., set to 20 or 30) to group infrequent categories into 'Other'.",
            "Drop high-cardinality identifier columns (e.g., CustomerID, InvoiceNo).",
            "Increase the memory budget if sufficient system RAM is available."
        ]


class DynamicDataPipeline:
    """
    Automated, dataset-independent data preprocessor.
    Guarantees zero data leakage by fitting transformers strictly on training sets.
    Features robust high-cardinality protection, sparse matrix awareness, and memory safety checks.
    """
    def __init__(
        self,
        target_column: Optional[str] = None,
        scaler_type: str = "standard",
        test_size: float = 0.2,
        random_state: int = 42,
        drop_duplicates: bool = False,
        max_categories_onehot: int = 30,
        min_category_frequency: Optional[Union[int, float]] = None,
        max_memory_budget_mb: float = 2048.0,
        max_dense_memory_mb: float = 512.0,
        sample_size: Optional[int] = None,
        sampling_seed: int = 42
    ):
        """
        Parameters:
        -----------
        target_column : str, optional
            Name of target column. If None, the last column is used.
        scaler_type : str
            One of 'standard', 'minmax', 'robust', or 'none'.
        test_size : float
            Fraction of dataset to reserve for evaluation (default 0.20).
        random_state : int
            Reproducibility seed for train/test split.
        drop_duplicates : bool
            Whether to drop duplicate rows during preparation (default False).
        max_categories_onehot : int
            Maximum number of top frequent categories to preserve per categorical feature.
            All infrequent categories beyond this threshold are grouped into 'Other'.
        min_category_frequency : int or float, optional
            Minimum occurrence count or frequency threshold for retaining a category.
        max_memory_budget_mb : float
            Maximum estimated RAM (in MB) allowed for preprocessing before halting safely.
        max_dense_memory_mb : float
            Maximum memory size (in MB) for converting sparse matrices to dense arrays.
            Sparse matrices exceeding this threshold remain sparse.
        sample_size : int, optional
            Reproducible sample size to draw before preprocessing (e.g. 10000, 25000, 50000).
        sampling_seed : int
            Reproducibility seed for dataset sampling.
        """
        self.target_column = target_column
        self.scaler_type = scaler_type.lower()
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.drop_duplicates = bool(drop_duplicates)
        self.max_categories_onehot = int(max_categories_onehot)
        self.min_category_frequency = min_category_frequency
        self.max_memory_budget_mb = float(max_memory_budget_mb)
        self.max_dense_memory_mb = float(max_dense_memory_mb)
        self.sample_size = int(sample_size) if sample_size is not None else None
        self.sampling_seed = int(sampling_seed)

        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.dropped_cols: List[str] = []
        self.dropped_all_nan_cols: List[str] = []
        self.dropped_columns_log: List[Dict[str, str]] = []
        self.leakage_warnings: List[str] = []
        self.dataset_limitations: Optional[str] = None
        self.retained_categories_: Dict[str, List[Any]] = {}
        self.preprocessor: Optional[ColumnTransformer] = None
        self.dataset_meta: Dict[str, Any] = {}

    def _get_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "none":
            return "passthrough"
        else:
            raise ValueError(f"Unknown scaler_type: '{self.scaler_type}'. Choose 'standard', 'minmax', 'robust', or 'none'.")

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect and return dataset structural metrics."""
        if df is None or df.empty:
            return {"num_rows": 0, "num_cols": 0, "columns": [], "missing_values": 0, "dtypes": {}}
        return {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
        }

    def _apply_sampling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies reproducible stratified sampling if sample_size is specified and smaller than df.
        Preserves class balance for classification targets when possible.
        """
        initial_len = len(df)
        if self.sample_size is None or self.sample_size >= initial_len:
            return df, {
                "is_sampled": False,
                "total_rows": initial_len,
                "sampled_rows": initial_len,
                "sampling_seed": self.sampling_seed,
                "stratified": False
            }

        target_series = df[self.target_column]
        class_counts = target_series.value_counts()
        can_stratify = (class_counts.min() >= 2) and (len(class_counts) <= self.sample_size)

        if can_stratify:
            try:
                sampled_df, _ = train_test_split(
                    df,
                    train_size=self.sample_size,
                    stratify=target_series,
                    random_state=self.sampling_seed
                )
                stratified_used = True
            except Exception:
                sampled_df = df.sample(n=self.sample_size, random_state=self.sampling_seed)
                stratified_used = False
        else:
            sampled_df = df.sample(n=self.sample_size, random_state=self.sampling_seed)
            stratified_used = False

        sampled_df = sampled_df.reset_index(drop=True)
        sampling_info = {
            "is_sampled": True,
            "total_rows": initial_len,
            "sampled_rows": len(sampled_df),
            "sampling_seed": self.sampling_seed,
            "stratified": stratified_used
        }
        return sampled_df, sampling_info

    def _detect_and_drop_unusable_columns(self, X: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        Detects and drops:
        1. All-NaN columns
        2. Direct target-defining columns (e.g. Invoice / InvoiceNo for IsCancellation)
        3. Identifier-like columns (CustomerID, transaction IDs, high-cardinality unique keys)
        Every dropped column is logged with an explicit justification.
        """
        self.dropped_cols = []
        self.dropped_all_nan_cols = []
        self.dropped_columns_log = []

        cols_to_drop = []
        target_lower = target_name.lower()
        is_cancellation_task = "cancel" in target_lower

        # 1. Drop 100% NaN columns
        for col in X.columns:
            if X[col].isna().all():
                self.dropped_all_nan_cols.append(col)
                cols_to_drop.append(col)
                self.dropped_columns_log.append({
                    "column": col,
                    "reason": "Dropped 100% missing (all-NaN) column."
                })

        # 2. Identify target-defining & identifier columns
        remaining_cols = [c for c in X.columns if c not in cols_to_drop]
        id_keywords = ["id", "uuid", "guid", "code", "index", "invoice", "customer", "cust", "trans", "order", "receipt"]

        for col in remaining_cols:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            clean_name = re.sub(r"[^a-z0-9]", "", col_lower)
            n_unique = X[col].nunique(dropna=True)
            n_rows = len(X)
            unique_ratio = n_unique / max(1, n_rows)

            # Check direct target leakage: Invoice numbers for IsCancellation
            if is_cancellation_task and ("invoice" in clean_name or clean_name in ["invoiceno", "invoice"]):
                self.dropped_cols.append(col)
                cols_to_drop.append(col)
                self.dropped_columns_log.append({
                    "column": col,
                    "reason": (
                        f"Direct target leakage: '{col}' contains invoice identifiers from which "
                        f"'{target_name}' was engineered (e.g. 'C' prefix indicates cancellation in Online Retail II). "
                        f"Retaining it would cause 100% artificial label leakage."
                    )
                })
                continue

            # Check entity identifiers (Customer ID, Order ID, etc.)
            has_id_keyword = any(kw in clean_name for kw in id_keywords)
            is_string_or_cat = (
                X[col].dtype == object
                or isinstance(X[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(X[col])
            )

            # Customer ID detection: often numeric float or integer with thousands of distinct values
            is_customer_col = any(k in clean_name for k in ["customer", "custid", "customerid"])
            if is_customer_col and (n_unique > 50 or unique_ratio > 0.01):
                self.dropped_cols.append(col)
                cols_to_drop.append(col)
                self.dropped_columns_log.append({
                    "column": col,
                    "reason": (
                        f"Identifier column: '{col}' represents individual customer entities ({n_unique} unique IDs). "
                        f"Entity identifiers do not generalize to unseen populations and induce severe feature explosion."
                    )
                })
                continue

            # Invoice / Transaction identifier when not already caught by leakage rule
            is_invoice_or_trans = any(k in clean_name for k in ["invoice", "transid", "orderid", "transaction"])
            if is_invoice_or_trans and (n_unique > 50 or unique_ratio > 0.05):
                self.dropped_cols.append(col)
                cols_to_drop.append(col)
                self.dropped_columns_log.append({
                    "column": col,
                    "reason": (
                        f"Transaction identifier: '{col}' is a per-transaction identifier with {n_unique} unique values. "
                        f"Transaction IDs do not provide predictive generalization."
                    )
                })
                continue

            # General high-cardinality ID keyword match
            if has_id_keyword:
                if (unique_ratio > 0.90 and n_rows > 50) or (is_string_or_cat and n_unique > 500 and unique_ratio > 0.10):
                    self.dropped_cols.append(col)
                    cols_to_drop.append(col)
                    self.dropped_columns_log.append({
                        "column": col,
                        "reason": (
                            f"Identifier column: '{col}' matches identifier patterns with high cardinality "
                            f"({n_unique} unique values, {unique_ratio*100:.1f}% uniqueness ratio)."
                        )
                    })
                    continue

            # Pure unique string/key (even without ID keyword)
            if is_string_or_cat and unique_ratio > 0.95 and n_rows > 50:
                self.dropped_cols.append(col)
                cols_to_drop.append(col)
                self.dropped_columns_log.append({
                    "column": col,
                    "reason": (
                        f"Unique key column: '{col}' has {unique_ratio*100:.1f}% unique string values across rows, "
                        f"functioning as an arbitrary record key."
                    )
                })
                continue

        return X.drop(columns=cols_to_drop)

    def _investigate_indirect_leakage(self, X: pd.DataFrame, target_name: str):
        """
        Investigates indirect target leakage, specifically for Online Retail II / cancellation tasks:
        - Quantity < 0: Post-cancellation transaction logs in retail systems
        - Price <= 0: Accounting adjustments and manual administrative items
        Documents the limitations of engineered targets.
        """
        self.leakage_warnings = []
        target_lower = target_name.lower()
        if not ("cancel" in target_lower or "cancellation" in target_lower):
            return

        self.dataset_limitations = (
            f"The target '{target_name}' is an engineered classification task derived from invoice prefixes ('C'), "
            "rather than an original ground-truth label recorded in prospective real-time conditions. "
            "In transaction datasets such as Online Retail II, post-hoc adjustments (e.g. negative quantities or manual accounting credits) "
            "correlate strongly with cancellations because they were logged simultaneously with or after the cancellation event. "
            "Evaluation results on this engineered task demonstrate tabular processing efficiency rather than prospective commercial forecasting."
        )

        for col in X.columns:
            clean = re.sub(r"[^a-z0-9]", "", str(col).lower())
            if "quantity" in clean and pd.api.types.is_numeric_dtype(X[col]):
                neg_count = int((X[col] < 0).sum())
                if neg_count > 0:
                    self.leakage_warnings.append(
                        f"Indirect Leakage Warning: Column '{col}' contains {neg_count} negative entries. "
                        "In Online Retail II, cancellations were logged post-hoc using negative quantities (e.g. -1, -5). "
                        "Models utilizing raw negative quantities may achieve artificially inflated accuracy."
                    )
            elif any(p in clean for p in ["price", "unitprice"]) and pd.api.types.is_numeric_dtype(X[col]):
                non_pos_count = int((X[col] <= 0).sum())
                if non_pos_count > 0:
                    self.leakage_warnings.append(
                        f"Pricing Anomaly Notice: Column '{col}' contains {non_pos_count} zero or negative entries, "
                        "which typically denote manual accounting adjustments or fee waivers."
                    )

    def _group_infrequent_categories(
        self,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strictly fits category frequency distribution on X_train_df.
        Retains at most `max_categories_onehot` top frequent categories per column.
        Groups all infrequent categories into 'Other'.
        Applies the exact same mapping to X_test_df, routing any unseen test categories into 'Other'.
        Guarantees zero data leakage and prevents combinatorial one-hot explosion.
        """
        X_train_grouped = X_train_df.copy()
        X_test_grouped = X_test_df.copy()
        self.retained_categories_ = {}

        for col in self.categorical_cols:
            # Clean string values strictly on train
            train_vals = X_train_grouped[col].astype(str).str.strip()
            counts = train_vals.value_counts(dropna=False)

            # Determine retained categories
            if self.min_category_frequency is not None:
                if isinstance(self.min_category_frequency, float) and 0.0 < self.min_category_frequency < 1.0:
                    min_count = self.min_category_frequency * len(X_train_grouped)
                else:
                    min_count = float(self.min_category_frequency)
                frequent_categories = counts[counts >= min_count].index.tolist()
            else:
                frequent_categories = counts.index.tolist()

            # Limit to max_categories_onehot
            top_k_categories = frequent_categories[:self.max_categories_onehot]
            retained_set = set(top_k_categories)
            self.retained_categories_[col] = top_k_categories

            # Map infrequent categories on train to 'Other'
            X_train_grouped[col] = train_vals.apply(lambda val: val if val in retained_set else "Other")

            # Apply identical mapping to test (unseen or rare values automatically become 'Other')
            test_vals = X_test_grouped[col].astype(str).str.strip()
            X_test_grouped[col] = test_vals.apply(lambda val: val if val in retained_set else "Other")

        return X_train_grouped, X_test_grouped

    def _estimate_and_check_memory_safety(self, n_samples: int, expected_features: int, stage: str = "preprocessing"):
        """
        Estimates the memory requirement for the dense matrix representation.
        If it exceeds max_memory_budget_mb, raises MemoryBudgetExceededError with actionable advice.
        """
        # Dense float64 matrix: 8 bytes per cell
        dense_bytes = n_samples * expected_features * 8
        dense_mb = dense_bytes / (1024 * 1024)
        dense_gb = dense_mb / 1024

        if dense_mb > self.max_memory_budget_mb:
            raise MemoryBudgetExceededError(
                message=(
                    f"Memory safety limit exceeded during {stage}. "
                    f"Estimated memory for dense array with shape ({n_samples}, {expected_features}) "
                    f"is {dense_mb:.1f} MB ({dense_gb:.2f} GiB), exceeding the memory budget of {self.max_memory_budget_mb:.1f} MB."
                ),
                estimated_mb=dense_mb,
                budget_mb=self.max_memory_budget_mb,
                stage=stage,
                recommendations=[
                    f"Enable reproducible dataset sampling (e.g. 10,000, 25,000, or 50,000 rows). Currently evaluated on {n_samples} samples.",
                    f"Decrease 'max_categories_onehot' (currently {self.max_categories_onehot}) to limit categorical cardinality.",
                    "Verify that high-cardinality transaction or customer identifiers have been dropped.",
                    f"Increase 'max_memory_budget_mb' (currently {self.max_memory_budget_mb:.0f} MB) if the workstation has sufficient physical RAM."
                ]
            )

    def prepare(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Tuple[Union[np.ndarray, sp.csr_matrix], Union[np.ndarray, sp.csr_matrix], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepares the dataset from end-to-end:
        1. Clean and identify target
        2. Optional reproducible stratified dataset sampling
        3. Drop direct target-defining features (Online Retail II leakage) and entity identifiers
        4. Investigate indirect leakage (negative quantity artifacts)
        5. Infer numerical vs categorical features
        6. Zero-leakage stratified train-test splitting
        7. Group infrequent categories into 'Other' on training data
        8. Memory safety validation
        9. Sparse-aware pipeline transformation with safe dense representation
        10. Return X_train, X_test, y_train, y_test, metadata
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        df = df.copy()

        # Clean column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        # Handle duplicates if requested
        initial_len = len(df)
        duplicates_removed = 0
        if self.drop_duplicates:
            duplicates_removed = int(df.duplicated().sum())
            if duplicates_removed > 0:
                df = df.drop_duplicates().reset_index(drop=True)

        # Identify target column
        if target_col is not None:
            self.target_column = str(target_col).strip()
        elif self.target_column is None:
            self.target_column = df.columns[-1]
        else:
            self.target_column = str(self.target_column).strip()

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame. Available columns: {list(df.columns)}")

        # Drop rows with null target
        df = df.dropna(subset=[self.target_column]).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"Target column '{self.target_column}' contains only missing (NaN) values.")

        if len(df) < 5:
            raise ValueError(f"Insufficient samples ({len(df)}) for train-test splitting.")

        # Apply dataset sampling if configured
        df, sampling_meta = self._apply_sampling(df)

        # Separate X and y
        y_raw = df[self.target_column]
        X_raw = df.drop(columns=[self.target_column])

        # Validate that target has at least 2 distinct classes
        unique_targets = pd.Series(y_raw).dropna().unique()
        if len(unique_targets) < 2:
            raise ValueError(f"Target column must contain at least 2 distinct classes for classification. Found: {list(unique_targets)}")

        # Check for direct leakage and drop unusable / identifier columns
        X_clean = self._detect_and_drop_unusable_columns(X_raw, self.target_column)
        if X_clean.shape[1] == 0:
            raise ValueError("No feature columns remaining after target selection and identifier filtering.")

        # Investigate indirect leakage (e.g. Quantity < 0 in Online Retail II)
        self._investigate_indirect_leakage(X_clean, self.target_column)

        # Identify numerical and categorical columns
        self.numerical_cols = []
        self.categorical_cols = []

        for col in X_clean.columns:
            # Check if column can be coerced to numeric (e.g. TotalCharges in Telco Churn)
            if X_clean[col].dtype == object or pd.api.types.is_string_dtype(X_clean[col]):
                coerced = pd.to_numeric(X_clean[col].astype(str).str.strip(), errors='coerce')
                # If less than 10% are NaN after coercion, treat as numeric
                if coerced.isna().sum() / len(coerced) < 0.10:
                    X_clean[col] = coerced
                    self.numerical_cols.append(col)
                    continue

            if pd.api.types.is_numeric_dtype(X_clean[col]):
                self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)

        # Encode target variable
        class_mapping: Dict[str, int] = {}
        if (
            y_raw.dtype == object 
            or isinstance(y_raw.dtype, pd.CategoricalDtype) 
            or pd.api.types.is_string_dtype(y_raw) 
            or not pd.api.types.is_numeric_dtype(y_raw)
        ):
            y = self.label_encoder.fit_transform(y_raw.astype(str).str.strip())
            class_mapping = {str(c): int(idx) for idx, c in enumerate(self.label_encoder.classes_)}
        else:
            unique_vals = np.sort(np.unique(y_raw))
            if np.array_equal(unique_vals, np.arange(len(unique_vals))):
                y = y_raw.values.astype(int)
                class_mapping = {str(val): int(val) for val in unique_vals}
            else:
                y = self.label_encoder.fit_transform(y_raw)
                class_mapping = {str(c): int(idx) for idx, c in enumerate(self.label_encoder.classes_)}

        # Stratified Train-Test Split (zero data leakage)
        class_counts = pd.Series(y).value_counts()
        stratify = y if class_counts.min() >= 2 else None

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_clean, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

        # Group infrequent categories strictly using training distribution
        if self.categorical_cols:
            X_train_df, X_test_df = self._group_infrequent_categories(X_train_df, X_test_df)

        # Memory Safety Estimation before One-Hot Encoding
        n_train = len(X_train_df)
        expected_cat_features = 0
        if self.categorical_cols:
            for col in self.categorical_cols:
                n_uniq = len(self.retained_categories_.get(col, []))
                # Add 1 for the 'Other' category if any values were grouped
                expected_cat_features += n_uniq + 1
        expected_total_features = len(self.numerical_cols) + expected_cat_features

        # Check memory safety
        self._estimate_and_check_memory_safety(n_train, expected_total_features, stage="feature encoding")

        # Build Scikit-Learn Preprocessing Pipeline
        transformers = []

        if self.numerical_cols:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', self._get_scaler())
            ])
            transformers.append(('num', num_pipe, self.numerical_cols))

        if self.categorical_cols:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
            ])
            transformers.append(('cat', cat_pipe, self.categorical_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            sparse_threshold=0.3
        )

        # Fit preprocessor strictly on train set, then transform both train and test sets
        X_train_proc = self.preprocessor.fit_transform(X_train_df)
        X_test_proc = self.preprocessor.transform(X_test_df)

        # Manage matrix representation: avoid unnecessary dense conversion
        is_sparse = sp.issparse(X_train_proc)
        if is_sparse:
            # Estimate dense conversion requirement
            dense_mb = (X_train_proc.shape[0] * X_train_proc.shape[1] * 8) / (1024 * 1024)
            if dense_mb <= self.max_dense_memory_mb:
                # Convert to dense safely when within memory threshold
                X_train = X_train_proc.toarray()
                X_test = X_test_proc.toarray()
                is_sparse = False
            else:
                # Maintain memory-efficient sparse CSR matrix
                X_train = X_train_proc.tocsr()
                X_test = X_test_proc.tocsr()
        else:
            X_train = X_train_proc
            X_test = X_test_proc

        # Extract generated feature names
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols:
            try:
                cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
                cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols)
                feature_names.extend(list(cat_feature_names))
            except Exception:
                feature_names.extend([f"cat_{i}" for i in range(X_train.shape[1] - len(self.numerical_cols))])

        self.feature_names = feature_names

        # Compute memory footprint
        if sp.issparse(X_train):
            mem_footprint_mb = (X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes) / (1024 * 1024)
        else:
            mem_footprint_mb = X_train.nbytes / (1024 * 1024)

        # Metadata dictionary
        self.dataset_meta = {
            "initial_samples": initial_len,
            "cleaned_samples": len(df),
            "duplicates_removed": duplicates_removed,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "original_features": len(X_raw.columns) + len(self.dropped_cols) + len(self.dropped_all_nan_cols),
            "active_features": len(X_clean.columns),
            "processed_features": int(X_train.shape[1]),
            "numerical_features": len(self.numerical_cols),
            "categorical_features": len(self.categorical_cols),
            "dropped_id_features": self.dropped_cols,
            "dropped_all_nan_features": self.dropped_all_nan_cols,
            "dropped_columns_log": self.dropped_columns_log,
            "leakage_warnings": self.leakage_warnings,
            "dataset_limitations": self.dataset_limitations,
            "classes": len(np.unique(y)),
            "class_mapping": class_mapping,
            "class_distribution_train": {int(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
            "class_distribution_test": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
            "scaler_applied": self.scaler_type,
            "is_sparse": is_sparse,
            "memory_footprint_mb": round(mem_footprint_mb, 2),
            "sampling": sampling_meta
        }

        return X_train, X_test, y_train, y_test, self.dataset_meta

    @classmethod
    def load_and_prepare_csv(
        cls,
        csv_path: str,
        target_column: Optional[str] = None,
        scaler_type: str = "standard",
        test_size: float = 0.2,
        random_state: int = 42,
        drop_duplicates: bool = False,
        max_categories_onehot: int = 30,
        sample_size: Optional[int] = None,
        sampling_seed: int = 42
    ) -> Tuple[Union[np.ndarray, sp.csr_matrix], Union[np.ndarray, sp.csr_matrix], np.ndarray, np.ndarray, Dict[str, Any]]:
        """Convenience loader from CSV file with input validation."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Unable to read CSV file '{csv_path}': {e}")

        pipeline = cls(
            target_column=target_column,
            scaler_type=scaler_type,
            test_size=test_size,
            random_state=random_state,
            drop_duplicates=drop_duplicates,
            max_categories_onehot=max_categories_onehot,
            sample_size=sample_size,
            sampling_seed=sampling_seed
        )
        return pipeline.prepare(df)
