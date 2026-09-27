"""
Tests for HeartGuard data validation component.
"""

import pytest
import pandas as pd
import numpy as np

from src.heartguard.components.data_validation import DataValidation, ValidationReport
from src.heartguard.exception import InvalidDatasetError


@pytest.fixture
def valid_df():
    """Create a minimal valid dataset for testing."""
    return pd.DataFrame({
        "age": [55, 45, 60, 50, 65],
        "sex": [1, 0, 1, 0, 1],
        "cp": [0, 1, 2, 3, 0],
        "trestbps": [130, 120, 140, 110, 150],
        "chol": [250, 200, 300, 180, 270],
        "fbs": [1, 0, 1, 0, 0],
        "restecg": [0, 1, 0, 2, 1],
        "thalach": [150, 170, 130, 180, 120],
        "exang": [0, 1, 0, 1, 0],
        "oldpeak": [1.0, 0.5, 2.3, 0.0, 3.1],
        "slope": [0, 1, 2, 0, 1],
        "ca": [0, 1, 2, 0, 3],
        "thal": [1, 2, 0, 1, 2],
        "target": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def validator():
    return DataValidation()


class TestDataValidation:
    """Test suite for data validation checks."""

    def test_valid_dataset_passes(self, validator, valid_df):
        """A properly formatted dataset should pass validation."""
        report = validator.validate(valid_df)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_empty_dataset_fails(self, validator):
        """An empty DataFrame should fail validation."""
        empty_df = pd.DataFrame()
        report = validator.validate(empty_df)
        assert not report.is_valid

    def test_missing_columns_detected(self, validator, valid_df):
        """Missing required columns should be flagged as errors."""
        df_missing = valid_df.drop(columns=["age", "chol"])
        report = validator.validate(df_missing)
        assert not report.is_valid
        assert any("Missing required columns" in e for e in report.errors)

    def test_missing_target_detected(self, validator, valid_df):
        """Missing target column should fail validation."""
        df_no_target = valid_df.drop(columns=["target"])
        report = validator.validate(df_no_target)
        assert not report.is_valid

    def test_non_binary_target_detected(self, validator, valid_df):
        """Non-binary target values should fail validation."""
        valid_df["target"] = [0, 1, 2, 3, 4]
        report = validator.validate(valid_df)
        assert not report.is_valid
        assert any("binary" in e.lower() for e in report.errors)

    def test_missing_values_warned(self, validator, valid_df):
        """Missing values should produce a warning (not error)."""
        valid_df.loc[0, "age"] = np.nan
        report = validator.validate(valid_df)
        assert report.is_valid  # Warnings don't fail validation
        assert any("Missing values" in w for w in report.warnings)

    def test_duplicate_rows_warned(self, validator, valid_df):
        """Duplicate rows should produce a warning."""
        df_with_dups = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
        report = validator.validate(df_with_dups)
        assert any("duplicate" in w.lower() for w in report.warnings)

    def test_validate_or_raise_raises(self, validator):
        """validate_or_raise should raise InvalidDatasetError on failure."""
        with pytest.raises(InvalidDatasetError):
            validator.validate_or_raise(pd.DataFrame())

    def test_out_of_range_values_warned(self, validator, valid_df):
        """Out-of-range clinical values should produce a warning."""
        valid_df.loc[0, "age"] = 200  # Impossible age
        report = validator.validate(valid_df)
        assert any("age" in w and "outside" in w for w in report.warnings)
