"""
Tests for HeartGuard preprocessing/data transformation.
"""

import pytest
import pandas as pd
import numpy as np

from src.heartguard.components.data_transformation import DataTransformation


@pytest.fixture
def sample_train_df():
    """Minimal training DataFrame."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "age": np.random.randint(30, 80, n),
        "sex": np.random.choice([0, 1], n),
        "cp": np.random.choice([0, 1, 2, 3], n),
        "trestbps": np.random.randint(90, 180, n),
        "chol": np.random.randint(150, 400, n),
        "fbs": np.random.choice([0, 1], n),
        "restecg": np.random.choice([0, 1, 2], n),
        "thalach": np.random.randint(100, 200, n),
        "exang": np.random.choice([0, 1], n),
        "oldpeak": np.round(np.random.uniform(0, 5, n), 1),
        "slope": np.random.choice([0, 1, 2], n),
        "ca": np.random.choice([0, 1, 2, 3], n),
        "thal": np.random.choice([0, 1, 2], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def sample_test_df():
    """Minimal test DataFrame."""
    np.random.seed(99)
    n = 15
    return pd.DataFrame({
        "age": np.random.randint(30, 80, n),
        "sex": np.random.choice([0, 1], n),
        "cp": np.random.choice([0, 1, 2, 3], n),
        "trestbps": np.random.randint(90, 180, n),
        "chol": np.random.randint(150, 400, n),
        "fbs": np.random.choice([0, 1], n),
        "restecg": np.random.choice([0, 1, 2], n),
        "thalach": np.random.randint(100, 200, n),
        "exang": np.random.choice([0, 1], n),
        "oldpeak": np.round(np.random.uniform(0, 5, n), 1),
        "slope": np.random.choice([0, 1, 2], n),
        "ca": np.random.choice([0, 1, 2, 3], n),
        "thal": np.random.choice([0, 1, 2], n),
        "target": np.random.choice([0, 1], n),
    })


class TestDataTransformation:
    """Test suite for preprocessing pipeline."""

    def test_feature_engineering_adds_columns(self, sample_train_df):
        """Feature engineering should add age_group, hr_reserve, bp_category."""
        transformer = DataTransformation()
        df = transformer._add_engineered_features(sample_train_df.drop(columns=["target"]))
        assert "age_group" in df.columns
        assert "hr_reserve" in df.columns
        assert "bp_category" in df.columns

    def test_preprocessor_output_shape(self, sample_train_df, sample_test_df, tmp_path):
        """Transformation should produce numeric arrays with consistent columns."""
        transformer = DataTransformation(preprocessor_path=str(tmp_path / "prep.pkl"))
        X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(
            sample_train_df, sample_test_df
        )
        assert X_train.shape[0] == len(sample_train_df)
        assert X_test.shape[0] == len(sample_test_df)
        # Both should have the same number of features
        assert X_train.shape[1] == X_test.shape[1]

    def test_no_nan_after_transformation(self, sample_train_df, sample_test_df, tmp_path):
        """Preprocessor should handle missing values - no NaNs in output."""
        # Introduce some NaNs
        sample_train_df.loc[0, "age"] = np.nan
        sample_train_df.loc[2, "chol"] = np.nan

        transformer = DataTransformation(preprocessor_path=str(tmp_path / "prep.pkl"))
        X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(
            sample_train_df, sample_test_df
        )
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))

    def test_target_separation(self, sample_train_df, sample_test_df, tmp_path):
        """Target column should be separated correctly."""
        transformer = DataTransformation(preprocessor_path=str(tmp_path / "prep.pkl"))
        X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(
            sample_train_df, sample_test_df
        )
        assert len(y_train) == len(sample_train_df)
        assert set(np.unique(y_train)).issubset({0, 1})

    def test_original_data_unchanged(self, sample_train_df, sample_test_df):
        """Feature engineering should not modify the original DataFrame."""
        original_cols = list(sample_train_df.columns)
        transformer = DataTransformation()
        transformer._add_engineered_features(sample_train_df.drop(columns=["target"]))
        # Original should be unchanged
        assert list(sample_train_df.columns) == original_cols
