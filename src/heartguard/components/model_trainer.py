"""
HeartGuard Model Trainer Component
Trains multiple models, performs cross-validation and hyperparameter tuning,
compares models, and selects the best one.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from src.heartguard.logger import get_logger
from src.heartguard.exception import TrainingError
from src.heartguard.config import path_config, training_config
from src.heartguard.utils.common import save_object, save_json

logger = get_logger(__name__)


class ModelTrainer:
    """
    Handles the full model training lifecycle:
    1. Train multiple candidate models
    2. Cross-validate each model
    3. Hyperparameter-tune top candidates
    4. Compare all models on a held-out test set
    5. Select the best model based on ROC-AUC
    6. Save the final model
    """

    def __init__(self):
        self.model_path = path_config.model_path
        self.metrics_path = path_config.metrics_path
        self.comparison_path = path_config.comparison_path
        self.random_state = training_config.random_state
        self.cv_folds = training_config.cv_folds
        self.primary_metric = training_config.primary_metric
        self.param_grids = training_config.param_grids

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Define candidate models with default parameters."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "SVM": SVC(
                probability=True, random_state=self.random_state
            ),
        }

    def _evaluate_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics for a trained model."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }

    def _cross_validate(
        self, model, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Stratified K-Fold cross-validation.
        
        Returns:
            Tuple of (mean_score, std_score) using ROC-AUC
        """
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc"
        )
        return round(scores.mean(), 4), round(scores.std(), 4)

    def _hyperparameter_tune(
        self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray
    ):
        """
        Perform GridSearchCV on a model if a parameter grid is defined.
        
        Returns:
            Tuned model (or original if no grid defined)
        """
        if model_name not in self.param_grids:
            logger.info(f"No hyperparameter grid for {model_name}, skipping tuning")
            return model

        logger.info(f"Hyperparameter tuning: {model_name}")
        param_grid = self.param_grids[model_name]

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Execute the full model training pipeline.
        
        Args:
            X_train: Preprocessed training features
            X_test: Preprocessed test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary with best_model, best_model_name, metrics,
            comparison_df, and all trained models
        """
        logger.info("Model training started")

        try:
            candidate_models = self._get_candidate_models()
            comparison_results = []
            trained_models = {}

            # ─── Phase 1: Train and cross-validate all candidates ───
            print("\n" + "=" * 60)
            print("PHASE 1: MODEL TRAINING & CROSS-VALIDATION")
            print("=" * 60)

            for name, model in candidate_models.items():
                logger.info(f"Training: {name}")
                print(f"\n  Training {name}...")

                # Cross-validate first (on training data only)
                cv_mean, cv_std = self._cross_validate(model, X_train, y_train)
                print(f"    CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

                # Fit on full training data
                model.fit(X_train, y_train)

                # Evaluate on test set
                metrics = self._evaluate_model(model, X_test, y_test)
                metrics["cv_roc_auc_mean"] = cv_mean
                metrics["cv_roc_auc_std"] = cv_std

                comparison_results.append({"Model": name, **metrics})
                trained_models[name] = {"model": model, "metrics": metrics}

                print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Test ROC-AUC:  {metrics['roc_auc']:.4f}")

            # ─── Phase 2: Hyperparameter tuning on top candidates ───
            print("\n" + "=" * 60)
            print("PHASE 2: HYPERPARAMETER TUNING")
            print("=" * 60)

            # Identify models that have tuning grids
            tunable_models = [
                name for name in candidate_models if name in self.param_grids
            ]

            for name in tunable_models:
                base_model = self._get_candidate_models()[name]
                tuned_model = self._hyperparameter_tune(
                    name, base_model, X_train, y_train
                )

                # Re-evaluate tuned model
                metrics = self._evaluate_model(tuned_model, X_test, y_test)
                cv_mean, cv_std = self._cross_validate(tuned_model, X_train, y_train)
                metrics["cv_roc_auc_mean"] = cv_mean
                metrics["cv_roc_auc_std"] = cv_std

                tuned_name = f"{name} (Tuned)"
                comparison_results.append({"Model": tuned_name, **metrics})
                trained_models[tuned_name] = {"model": tuned_model, "metrics": metrics}

                print(f"\n  {tuned_name}:")
                print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Test ROC-AUC:  {metrics['roc_auc']:.4f}")

            # ─── Phase 3: Model comparison and selection ───
            print("\n" + "=" * 60)
            print("PHASE 3: MODEL COMPARISON & SELECTION")
            print("=" * 60)

            comparison_df = pd.DataFrame(comparison_results)
            comparison_df = comparison_df.sort_values("roc_auc", ascending=False)
            print(f"\n{comparison_df.to_string(index=False)}\n")

            # Select best model by primary metric (ROC-AUC)
            best_row = comparison_df.iloc[0]
            best_model_name = best_row["Model"]
            best_model = trained_models[best_model_name]["model"]
            best_metrics = trained_models[best_model_name]["metrics"]

            logger.info(f"Best model selected: {best_model_name}")
            logger.info(f"Best metrics: {best_metrics}")

            # ─── Phase 4: Save artifacts ───

            # Save the best model
            save_object(best_model, self.model_path)
            logger.info(f"Best model saved to {self.model_path}")

            # Save comparison table
            comparison_df.to_csv(self.comparison_path, index=False)
            logger.info(f"Comparison table saved to {self.comparison_path}")

            # Save metrics
            save_json(
                {
                    "best_model": best_model_name,
                    "metrics": best_metrics,
                    "all_models": {
                        name: info["metrics"]
                        for name, info in trained_models.items()
                    },
                },
                self.metrics_path,
            )

            return {
                "best_model": best_model,
                "best_model_name": best_model_name,
                "best_metrics": best_metrics,
                "comparison_df": comparison_df,
                "trained_models": trained_models,
            }

        except Exception as e:
            raise TrainingError(
                f"Model training failed: {e}", error_detail=sys.exc_info()
            )
