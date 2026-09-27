"""
HeartGuard Predict Pipeline
Handles inference on new patient data.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from src.heartguard.config import path_config
from src.heartguard.utils.common import load_object
from src.heartguard.components.data_transformation import DataTransformation
from src.heartguard.exception import ModelNotFoundError, PredictionError
from src.heartguard.logger import get_logger

logger = get_logger(__name__)


class PredictPipeline:
    """Production prediction pipeline for HeartGuard."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.transformer = DataTransformation()

    def load_artifacts(self, force_reload: bool = False):
        """Load model and preprocessor artifacts."""
        if self.model is None or force_reload:
            if not os.path.exists(path_config.model_path):
                raise ModelNotFoundError(f"Model not found at {path_config.model_path}")
            self.model = load_object(path_config.model_path)
            logger.info("Model loaded")

        if self.preprocessor is None or force_reload:
            if not os.path.exists(path_config.preprocessor_path):
                raise ModelNotFoundError(f"Preprocessor not found at {path_config.preprocessor_path}")
            self.preprocessor = load_object(path_config.preprocessor_path)
            logger.info("Preprocessor loaded")

    def predict(self, features_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction on raw patient features.
        
        Args:
            features_df: DataFrame containing the 13 raw clinical features.
            
        Returns:
            Tuple of (prediction: int, probability: float)
        """
        try:
            self.load_artifacts()

            # Apply engineered features
            engineered_df = self.transformer._add_engineered_features(features_df)

            # Transform features
            transformed_features = self.preprocessor.transform(engineered_df)

            # Predict
            prediction = int(self.model.predict(transformed_features)[0])
            probability = float(self.model.predict_proba(transformed_features)[0][1])

            return prediction, probability

        except Exception as e:
            logger.error(f"Prediction failed in pipeline: {e}")
            raise PredictionError(f"Prediction failed: {e}", error_detail=sys.exc_info())
