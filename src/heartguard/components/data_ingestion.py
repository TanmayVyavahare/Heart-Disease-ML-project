"""
HeartGuard Data Ingestion Component
Handles loading, initial cleaning, and train/test splitting of the dataset.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.heartguard.logger import get_logger
from src.heartguard.exception import DatasetNotFoundError, InvalidDatasetError
from src.heartguard.config import path_config, data_config, training_config

logger = get_logger(__name__)


class DataIngestion:
    """
    Responsible for:
    1. Reading the raw CSV dataset
    2. Applying column name mapping for compatibility
    3. Basic malformed-data handling
    4. Saving processed copies
    5. Creating stratified train/test splits
    """

    def __init__(self):
        self.raw_data_path = path_config.raw_data_path
        self.train_data_path = path_config.train_data_path
        self.test_data_path = path_config.test_data_path

    def _apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names using the configurable mapping.
        This allows datasets with different column names to work
        without changing code everywhere.
        """
        # Lowercase all column names first
        df.columns = df.columns.str.strip().str.lower()

        # Apply mapping
        rename_map = {}
        for original, standard in data_config.column_mapping.items():
            if original in df.columns and standard not in df.columns:
                rename_map[original] = standard

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Applied column mapping: {rename_map}")

        return df

    def _handle_malformed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle obvious data quality issues."""
        initial_shape = df.shape

        # Remove completely empty rows
        df = df.dropna(how="all")

        # Convert columns that should be numeric but have stray strings
        for col in data_config.numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure target is numeric binary
        if data_config.target_column in df.columns:
            df[data_config.target_column] = pd.to_numeric(
                df[data_config.target_column], errors="coerce"
            )
            # Map multi-class target (some datasets use 0-4) to binary
            df[data_config.target_column] = (
                df[data_config.target_column].apply(lambda x: 1 if x >= 1 else 0)
            )

        final_shape = df.shape
        if initial_shape != final_shape:
            logger.info(
                f"Malformed data handling: shape changed from {initial_shape} to {final_shape}"
            )

        return df

    def initiate_data_ingestion(self) -> tuple:
        """
        Execute the full data ingestion pipeline.
        
        Returns:
            Tuple of (train_data_path, test_data_path)
        """
        logger.info("Data ingestion started")

        try:
            # 1. Validate file exists
            if not os.path.exists(self.raw_data_path):
                raise DatasetNotFoundError(
                    f"Dataset not found at: {self.raw_data_path}\n"
                    f"Please place your heart disease CSV at this path, or set the "
                    f"HEART_DATA_PATH environment variable."
                )

            # 2. Read CSV
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Dataset loaded: shape={df.shape}")

            if df.empty:
                raise InvalidDatasetError("Dataset is empty.")

            # 3. Apply column mapping
            df = self._apply_column_mapping(df)

            # 4. Handle malformed data
            df = self._handle_malformed_data(df)

            # 5. Create output directories
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            # 6. Stratified train/test split
            train_df, test_df = train_test_split(
                df,
                test_size=training_config.test_size,
                random_state=training_config.random_state,
                stratify=df[data_config.target_column],
            )

            # 7. Save splits
            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logger.info(f"Train set: {train_df.shape}, saved to {self.train_data_path}")
            logger.info(f"Test set: {test_df.shape}, saved to {self.test_data_path}")
            logger.info("Data ingestion completed successfully")

            return self.train_data_path, self.test_data_path

        except (DatasetNotFoundError, InvalidDatasetError):
            raise
        except Exception as e:
            raise InvalidDatasetError(
                f"Data ingestion failed: {e}", error_detail=sys.exc_info()
            )
