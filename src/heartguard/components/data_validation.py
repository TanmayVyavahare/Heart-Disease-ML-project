"""
HeartGuard Data Validation Component
Validates dataset schema, types, and quality before processing.
"""

import sys
import pandas as pd
from typing import List, Optional

from src.heartguard.logger import get_logger
from src.heartguard.exception import InvalidDatasetError
from src.heartguard.config import data_config

logger = get_logger(__name__)


class ValidationReport:
    """Stores results of all validation checks."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str):
        self.errors.append(message)
        logger.error(f"Validation ERROR: {message}")

    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.warning(f"Validation WARNING: {message}")

    def add_info(self, message: str):
        self.info.append(message)
        logger.info(f"Validation INFO: {message}")

    def summary(self) -> str:
        lines = ["=" * 50, "DATA VALIDATION REPORT", "=" * 50]
        lines.append(f"Status: {'PASSED' if self.is_valid else 'FAILED'}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")

        if self.errors:
            lines.append("\n--- Errors ---")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
        if self.warnings:
            lines.append("\n--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        if self.info:
            lines.append("\n--- Info ---")
            for i in self.info:
                lines.append(f"  [INFO] {i}")

        lines.append("=" * 50)
        return "\n".join(lines)


class DataValidation:
    """
    Validates the dataset against expected schema and quality criteria.
    
    Checks:
    - Required columns exist
    - Target column is present and binary
    - Numeric columns contain valid values
    - Missing values are reported
    - Duplicate rows are reported
    - Dataset is not empty
    """

    def __init__(self):
        self.required_columns = data_config.required_columns
        self.target_column = data_config.target_column
        self.numerical_features = data_config.numerical_features
        self.categorical_features = data_config.categorical_features

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks on the dataframe.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationReport with all findings
        """
        report = ValidationReport()
        logger.info("Data validation started")

        self._check_not_empty(df, report)
        if not report.is_valid:
            return report

        self._check_required_columns(df, report)
        self._check_target_column(df, report)
        self._check_numeric_columns(df, report)
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_data_ranges(df, report)

        report.add_info(f"Dataset shape: {df.shape}")
        report.add_info(f"Data types:\n{df.dtypes.to_string()}")

        logger.info(f"Data validation completed: {'PASSED' if report.is_valid else 'FAILED'}")
        return report

    def validate_or_raise(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate and raise InvalidDatasetError if validation fails.
        """
        report = self.validate(df)
        if not report.is_valid:
            raise InvalidDatasetError(
                f"Dataset validation failed with {len(report.errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in report.errors)
            )
        return report

    def _check_not_empty(self, df: pd.DataFrame, report: ValidationReport):
        if df.empty:
            report.add_error("Dataset is empty (0 rows)")
        elif len(df) < 10:
            report.add_warning(f"Dataset has very few rows ({len(df)})")

    def _check_required_columns(self, df: pd.DataFrame, report: ValidationReport):
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            report.add_error(f"Missing required columns: {sorted(missing)}")
        else:
            report.add_info("All required columns present")

    def _check_target_column(self, df: pd.DataFrame, report: ValidationReport):
        if self.target_column not in df.columns:
            report.add_error(f"Target column '{self.target_column}' not found")
            return

        unique_values = df[self.target_column].dropna().unique()
        if not set(unique_values).issubset({0, 1}):
            report.add_error(
                f"Target column must contain binary values (0, 1). "
                f"Found: {sorted(unique_values)}"
            )
        else:
            counts = df[self.target_column].value_counts()
            report.add_info(f"Target distribution: {counts.to_dict()}")

            # Check for class imbalance
            ratio = counts.min() / counts.max()
            if ratio < 0.3:
                report.add_warning(
                    f"Significant class imbalance detected (ratio: {ratio:.2f})"
                )

    def _check_numeric_columns(self, df: pd.DataFrame, report: ValidationReport):
        for col in self.numerical_features:
            if col not in df.columns:
                continue
            non_numeric = df[col].apply(
                lambda x: not isinstance(x, (int, float)) and pd.notna(x)
            ).sum()
            if non_numeric > 0:
                report.add_warning(
                    f"Column '{col}' has {non_numeric} non-numeric value(s)"
                )

    def _check_missing_values(self, df: pd.DataFrame, report: ValidationReport):
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            cols_with_missing = missing[missing > 0]
            report.add_warning(
                f"Missing values found ({total_missing} total):\n"
                + cols_with_missing.to_string()
            )
        else:
            report.add_info("No missing values detected")

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport):
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            report.add_warning(f"{n_duplicates} duplicate row(s) found")
        else:
            report.add_info("No duplicate rows detected")

    def _check_data_ranges(self, df: pd.DataFrame, report: ValidationReport):
        """Check for obviously out-of-range clinical values."""
        range_checks = {
            "age": (0, 120),
            "trestbps": (50, 250),
            "chol": (50, 600),
            "thalach": (50, 250),
            "oldpeak": (-5, 10),
        }
        for col, (low, high) in range_checks.items():
            if col not in df.columns:
                continue
            out_of_range = ((df[col] < low) | (df[col] > high)).sum()
            if out_of_range > 0:
                report.add_warning(
                    f"Column '{col}' has {out_of_range} value(s) outside "
                    f"expected range [{low}, {high}]"
                )
