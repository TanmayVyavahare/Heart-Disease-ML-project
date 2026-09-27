"""
HeartGuard Data Transformation Component
Handles preprocessing and feature engineering using scikit-learn Pipelines.

IMPORTANT: Preprocessing is fitted ONLY on training data to prevent data leakage.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.heartguard.logger import get_logger
from src.heartguard.exception import PreprocessingError
from src.heartguard.config import path_config, data_config
from src.heartguard.utils.common import save_object

logger = get_logger(__name__)


class DataTransformation:
    """
    Builds and applies a scikit-learn ColumnTransformer pipeline.
    
    Numerical features: SimpleImputer (median) → StandardScaler
    Categorical features: SimpleImputer (most_frequent) → OneHotEncoder
    
    Feature engineering is applied BEFORE the sklearn pipeline
    to keep engineered features part of the data flow.
    """

    def __init__(self, preprocessor_path: str = None):
        self.preprocessor_path = preprocessor_path or path_config.preprocessor_path
        self.numerical_features = data_config.numerical_features
        self.categorical_features = data_config.categorical_features

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a small number of meaningful engineered features.
        Each feature has a documented clinical rationale.
        
        Features added:
        - age_group: Binned age into clinically relevant groups
          (Young/Middle/Senior/Elderly). Age is a known risk factor
          and grouping can capture non-linear effects.
        - hr_reserve: Difference between max heart rate achieved
          and a rough age-predicted maximum (220 - age).
          Low heart rate reserve is associated with cardiovascular risk.
        - bp_category: Categorized resting blood pressure into
          clinical hypertension stages (Normal/Elevated/High).
        """
        df = df.copy()

        # Age group: clinically meaningful age brackets
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 40, 55, 65, 120],
                labels=[0, 1, 2, 3],  # Young, Middle, Senior, Elderly
            ).astype(float)

        # Heart rate reserve: how close to age-predicted max HR
        if "thalach" in df.columns and "age" in df.columns:
            df["hr_reserve"] = df["thalach"] - (220 - df["age"])

        # Blood pressure category (hypertension staging)
        if "trestbps" in df.columns:
            df["bp_category"] = pd.cut(
                df["trestbps"],
                bins=[0, 120, 140, 300],
                labels=[0, 1, 2],  # Normal, Elevated, High
            ).astype(float)

        return df

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Build the sklearn ColumnTransformer pipeline.
        
        Returns:
            Configured ColumnTransformer (unfitted)
        """
        # Include engineered numerical features
        num_features = self.numerical_features + ["hr_reserve"]
        cat_features = self.categorical_features + ["age_group", "bp_category"]

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",  # Drop any columns not explicitly handled
        )

        return preprocessor

    def initiate_data_transformation(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple:
        """
        Apply feature engineering and preprocessing to train/test data.
        
        IMPORTANT: The preprocessor is fit ONLY on training data.
        
        Args:
            train_df: Training DataFrame (with target)
            test_df: Test DataFrame (with target)
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed, 
                      y_train, y_test, preprocessor_path)
        """
        logger.info("Data transformation started")

        try:
            target_col = data_config.target_column

            # Separate features and target
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col].values
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col].values

            # Apply feature engineering to both sets
            X_train = self._add_engineered_features(X_train)
            X_test = self._add_engineered_features(X_test)

            logger.info(f"Engineered features added. Columns: {list(X_train.columns)}")

            # Build and fit preprocessor on TRAINING data only
            preprocessor = self.get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logger.info(
                f"Preprocessing completed. "
                f"Train shape: {X_train_transformed.shape}, "
                f"Test shape: {X_test_transformed.shape}"
            )

            # Save the fitted preprocessor
            save_object(preprocessor, self.preprocessor_path)

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.preprocessor_path,
            )

        except Exception as e:
            raise PreprocessingError(
                f"Data transformation failed: {e}", error_detail=sys.exc_info()
            )
