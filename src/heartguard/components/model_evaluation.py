"""
HeartGuard Model Evaluation Component
Generates evaluation metrics, confusion matrix, ROC curve, 
and precision-recall curve for the final model.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

from src.heartguard.logger import get_logger
from src.heartguard.exception import HeartGuardException
from src.heartguard.config import path_config

logger = get_logger(__name__)


class ModelEvaluation:
    """
    Generates comprehensive evaluation artifacts for the final model:
    - Confusion matrix plot
    - Classification report
    - ROC curve plot
    - Precision-Recall curve plot
    - Threshold analysis
    """

    def __init__(self):
        self.plots_dir = path_config.plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Best Model"
    ) -> dict:
        """
        Run full evaluation and save plots.
        
        Args:
            model: Trained sklearn model
            X_test: Test features (preprocessed)
            y_test: True test labels
            model_name: Name for plot titles
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Model evaluation started")

        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # 1. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            self._plot_confusion_matrix(cm, model_name)

            # 2. Classification Report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_str = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report_str}")

            # 3. ROC Curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            self._plot_roc_curve(fpr, tpr, roc_auc_val, model_name)

            # 4. Precision-Recall Curve
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
                y_test, y_proba
            )
            avg_precision = average_precision_score(y_test, y_proba)
            self._plot_precision_recall_curve(
                precision_vals, recall_vals, avg_precision, model_name
            )

            # 5. Threshold Analysis
            self._plot_threshold_analysis(
                fpr, tpr, roc_thresholds,
                precision_vals, recall_vals, pr_thresholds,
                model_name
            )

            results = {
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "roc_auc": round(roc_auc_val, 4),
                "average_precision": round(avg_precision, 4),
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            }

            logger.info("Model evaluation completed. Plots saved to artifacts/plots/")
            return results

        except Exception as e:
            raise HeartGuardException(
                f"Model evaluation failed: {e}", error_detail=sys.exc_info()
            )

    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Generate and save confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            annot_kws={"size": 14},
        )
        plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        plt.ylabel("Actual", fontsize=12)
        plt.xlabel("Predicted", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    def _plot_roc_curve(
        self, fpr: np.ndarray, tpr: np.ndarray, roc_auc_val: float,
        model_name: str
    ):
        """Generate and save ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#2563eb", lw=2,
                 label=f"{model_name} (AUC = {roc_auc_val:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
                 label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "roc_curve.png"), dpi=150)
        plt.close()

    def _plot_precision_recall_curve(
        self, precision: np.ndarray, recall: np.ndarray,
        avg_precision: float, model_name: str
    ):
        """Generate and save Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="#dc2626", lw=2,
                 label=f"{model_name} (AP = {avg_precision:.4f})")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "precision_recall_curve.png"), dpi=150)
        plt.close()

    def _plot_threshold_analysis(
        self, fpr, tpr, roc_thresholds,
        precision_vals, recall_vals, pr_thresholds,
        model_name: str
    ):
        """
        Threshold analysis: show how precision and recall change
        with different classification thresholds.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(pr_thresholds, precision_vals[:-1], color="#2563eb", lw=2,
                 label="Precision")
        plt.plot(pr_thresholds, recall_vals[:-1], color="#dc2626", lw=2,
                 label="Recall")
        plt.xlabel("Classification Threshold", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Precision & Recall vs. Classification Threshold",
                  fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "threshold_analysis.png"), dpi=150)
        plt.close()
