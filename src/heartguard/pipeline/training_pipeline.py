"""
HeartGuard Training Pipeline
Orchestrates the complete ML lifecycle from data loading to model saving.

Usage:
    python -m src.heartguard.pipeline.training_pipeline
"""

import sys
import os
import pandas as pd
from typing import Optional

from src.heartguard.logger import get_logger
from src.heartguard.config import path_config, data_config
from src.heartguard.components.data_ingestion import DataIngestion
from src.heartguard.components.data_validation import DataValidation
from src.heartguard.components.data_transformation import DataTransformation
from src.heartguard.components.model_trainer import ModelTrainer
from src.heartguard.components.model_evaluation import ModelEvaluation
from src.heartguard.components.model_explanation import ModelExplanation
from src.heartguard.utils.common import load_object, save_json

logger = get_logger(__name__)


def run_training_pipeline():
    """
    Execute the complete HeartGuard training pipeline:
    
    1.  Load data
    2.  Validate data
    3.  Split data (train/test)
    4.  Preprocess data
    5.  Train candidate models
    6.  Run cross-validation
    7.  Tune selected models
    8.  Compare models
    9.  Select final model
    10. Evaluate final model
    11. Generate explanations
    12. Save model + preprocessor + metrics + plots
    """
    print("\n" + "=" * 60)
    print("  HEARTGUARD - TRAINING PIPELINE")
    print("  Explainable Heart Disease Risk Prediction")
    print("=" * 60)

    try:
        # ─── Step 1: Data Ingestion ───
        print("\n[1/7] Data Ingestion...")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        print("  [OK] Train/test split saved")

        # ─── Step 2: Data Validation ───
        print("\n[2/7] Data Validation...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        validator = DataValidation()
        train_report = validator.validate_or_raise(train_df)
        test_report = validator.validate(test_df)
        print(f"  [OK] Validation passed (train: {train_df.shape}, test: {test_df.shape})")

        # ─── Step 3: Data Transformation ───
        print("\n[3/7] Data Transformation (Feature Engineering + Preprocessing)...")
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = (
            transformation.initiate_data_transformation(train_df, test_df)
        )
        print("  [OK] Preprocessor fitted on training data")
        print(f"  [OK] Features: {X_train.shape[1]} (after transformation)")

        # ─── Step 4: Model Training ───
        print("\n[4/7] Model Training, Cross-Validation & Hyperparameter Tuning...")
        trainer = ModelTrainer()
        training_results = trainer.initiate_model_training(
            X_train, X_test, y_train, y_test
        )

        best_model = training_results["best_model"]
        best_model_name = training_results["best_model_name"]
        best_metrics = training_results["best_metrics"]

        # ─── Step 5: Model Evaluation ───
        print("\n[5/7] Model Evaluation (Plots & Metrics)...")
        evaluator = ModelEvaluation()
        eval_results = evaluator.evaluate(best_model, X_test, y_test, best_model_name)
        print("  [OK] Confusion matrix saved")
        print("  [OK] ROC curve saved")
        print("  [OK] Precision-recall curve saved")
        print("  [OK] Threshold analysis saved")

        # ─── Step 6: Model Explainability ───
        print("\n[6/7] Model Explainability...")
        preprocessor = load_object(preprocessor_path)
        explainer = ModelExplanation()
        explanation = explainer.explain(
            best_model, preprocessor, X_test, best_model_name
        )
        if explanation.get("feature_importances"):
            print("  [OK] Feature importance plot saved")
        else:
            print("  [WARN] Feature importance not available for this model type")

        # ─── Step 7: MLflow Logging (Optional) ───
        print("\n[7/7] Experiment Tracking...")
        _try_mlflow_logging(best_model_name, best_metrics, training_results)

        # ─── Final Summary ───
        _print_summary(best_model_name, best_metrics, eval_results)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\n[FAILED] Pipeline failed: {e}")
        raise


def _try_mlflow_logging(
    best_model_name: str, best_metrics: dict, training_results: dict
):
    """
    Log experiment to MLflow if available.
    The project works without MLflow - this is optional.
    """
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("HeartGuard")

        with mlflow.start_run(run_name=f"best_{best_model_name}"):
            # Log parameters
            mlflow.log_param("best_model", best_model_name)

            # Log metrics
            for metric_name, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)

            # Log model
            mlflow.sklearn.log_model(
                training_results["best_model"], "model"
            )

            # Log plots as artifacts
            plots_dir = path_config.plots_dir
            if os.path.exists(plots_dir):
                mlflow.log_artifacts(plots_dir, "plots")

        print("  [OK] Logged to MLflow (experiment: HeartGuard)")
        logger.info("MLflow logging completed")

    except ImportError:
        print("  [INFO] MLflow not installed - skipping experiment tracking")
        logger.info("MLflow not installed, skipping")
    except Exception as e:
        print(f"  [WARN] MLflow logging failed (non-critical): {e}")
        logger.warning(f"MLflow logging failed: {e}")


def _print_summary(model_name: str, metrics: dict, eval_results: dict):
    """Print the final training summary."""
    print("\n" + "=" * 60)
    print("  HEARTGUARD TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n  Best Model: {model_name}")
    print(f"\n  Accuracy:   {metrics.get('accuracy', 'N/A')}")
    print(f"  Precision:  {metrics.get('precision', 'N/A')}")
    print(f"  Recall:     {metrics.get('recall', 'N/A')}")
    print(f"  F1 Score:   {metrics.get('f1', 'N/A')}")
    print(f"  ROC-AUC:    {metrics.get('roc_auc', 'N/A')}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={eval_results['tn']}  FP={eval_results['fp']}")
    print(f"    FN={eval_results['fn']}  TP={eval_results['tp']}")
    print(f"\n  Artifacts saved to: artifacts/")
    print(f"  Plots saved to:     artifacts/plots/")
    print(f"\n  Model saved successfully.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_training_pipeline()
