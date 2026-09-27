"""
HeartGuard Custom Exception Module
Provides meaningful, context-rich exceptions for the ML pipeline.
"""

import sys
from typing import Optional


class HeartGuardException(Exception):
    """Base exception for all HeartGuard errors."""

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(message)
        self.message = message
        if error_detail is not None:
            _, _, exc_tb = error_detail
            if exc_tb is not None:
                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.line_number = exc_tb.tb_lineno
                self.message = (
                    f"Error in [{self.file_name}] at line [{self.line_number}]: {message}"
                )

    def __str__(self) -> str:
        return self.message


class DatasetNotFoundError(HeartGuardException):
    """Raised when the dataset file cannot be found."""
    pass


class InvalidDatasetError(HeartGuardException):
    """Raised when the dataset fails validation checks."""
    pass


class ModelNotFoundError(HeartGuardException):
    """Raised when a saved model file cannot be found."""
    pass


class PredictionError(HeartGuardException):
    """Raised when prediction fails due to invalid input or model error."""
    pass


class PreprocessingError(HeartGuardException):
    """Raised when data preprocessing fails."""
    pass


class TrainingError(HeartGuardException):
    """Raised when model training encounters an error."""
    pass
