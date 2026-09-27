"""
HeartGuard Configuration Module
Centralizes all configurable paths, parameters, and constants.
Uses environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ─────────────────────────── Project Root ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# ─────────────────────────── Path Config ────────────────────────────

@dataclass
class PathConfig:
    """All filesystem paths used by the project."""
    # Data
    raw_data_path: str = os.environ.get(
        "HEART_DATA_PATH",
        os.path.join(PROJECT_ROOT, "data", "raw", "heart.csv")
    )
    processed_data_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    train_data_path: str = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

    # Artifacts
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts")
    model_dir: str = os.path.join(PROJECT_ROOT, "artifacts", "models")
    model_path: str = os.environ.get(
        "MODEL_PATH",
        os.path.join(PROJECT_ROOT, "artifacts", "models", "model.pkl")
    )
    preprocessor_path: str = os.path.join(PROJECT_ROOT, "artifacts", "preprocessing", "preprocessor.pkl")
    plots_dir: str = os.path.join(PROJECT_ROOT, "artifacts", "plots")
    metrics_path: str = os.path.join(PROJECT_ROOT, "artifacts", "models", "metrics.json")
    comparison_path: str = os.path.join(PROJECT_ROOT, "artifacts", "models", "model_comparison.csv")

    def create_dirs(self):
        """Create all necessary directories."""
        for d in [self.processed_data_dir, self.model_dir, self.plots_dir,
                  os.path.dirname(self.preprocessor_path)]:
            os.makedirs(d, exist_ok=True)


# ─────────────────────────── Data Config ────────────────────────────

@dataclass
class DataConfig:
    """Dataset schema and column configuration."""
    target_column: str = "target"
    
    # Column name mapping: maps common alternative names → standard names
    # Allows datasets with different column names to be normalized
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "condition": "target",
        "num": "target",
        "heart_disease": "target",
        "output": "target",
    })
    
    # Expected feature columns (after mapping)
    numerical_features: List[str] = field(default_factory=lambda: [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ])
    
    @property
    def all_features(self) -> List[str]:
        return self.numerical_features + self.categorical_features
    
    @property
    def required_columns(self) -> List[str]:
        return self.all_features + [self.target_column]


# ─────────────────────────── Training Config ────────────────────────

@dataclass
class TrainingConfig:
    """Model training parameters."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    primary_metric: str = "roc_auc"
    
    # Hyperparameter search spaces (kept small for fast training)
    param_grids: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "min_samples_split": [2, 5],
        },
    })


# ─────────────────────────── App Config ─────────────────────────────

@dataclass
class AppConfig:
    """Flask application configuration."""
    port: int = int(os.environ.get("PORT", 5000))
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    host: str = "0.0.0.0"


# ─────────────────────────── Global Instances ───────────────────────

path_config = PathConfig()
data_config = DataConfig()
training_config = TrainingConfig()
app_config = AppConfig()
