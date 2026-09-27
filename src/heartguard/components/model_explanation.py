"""
HeartGuard Model Explanation Component
Provides feature importance and optional SHAP-based explanations.

NOTE: Feature importance shows which features the model relied on
most for predictions. It does NOT prove causation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from src.heartguard.logger import get_logger
from src.heartguard.exception import HeartGuardException
from src.heartguard.config import path_config

logger = get_logger(__name__)


class ModelExplanation:
    """
    Generates model explainability outputs:
    
    1. Feature importance (tree-based) or coefficient magnitude (linear)
    2. Feature importance bar chart
    3. Optional SHAP support (graceful fallback if not installed)
    """

    def __init__(self):
        self.plots_dir = path_config.plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def get_feature_names(self, preprocessor) -> List[str]:
        """Extract feature names from a fitted ColumnTransformer."""
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            elif hasattr(transformer[-1], "get_feature_names_out"):
                # Pipeline: get from last step
                names = transformer[-1].get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
        return list(feature_names)

    def explain(
        self,
        model,
        preprocessor,
        X_test: Optional[np.ndarray] = None,
        model_name: str = "Model",
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate feature importance explanations.
        
        Args:
            model: Trained sklearn model
            preprocessor: Fitted ColumnTransformer
            X_test: Test data (for SHAP, optional)
            model_name: Model name for plot titles
            top_n: Number of top features to display
            
        Returns:
            Dictionary with feature importances
        """
        logger.info("Model explanation started")

        try:
            feature_names = self.get_feature_names(preprocessor)
            importances = self._get_feature_importance(model, feature_names)

            if importances is not None:
                self._plot_feature_importance(importances, model_name, top_n)
                self._try_shap(model, X_test, feature_names, model_name)

                logger.info("Model explanation completed")

                return {
                    "feature_importances": importances.to_dict(),
                    "top_features": importances.head(top_n).to_dict(),
                }

            logger.warning("Could not extract feature importance from this model type")
            return {"feature_importances": None}

        except Exception as e:
            logger.error(f"Model explanation encountered an error: {e}")
            return {"feature_importances": None, "error": str(e)}

    def _get_feature_importance(self, model, feature_names: List[str]) -> Optional[pd.Series]:
        """
        Extract feature importance from the model.
        
        - Tree-based models: feature_importances_
        - Linear models: coefficient magnitude
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            return None

        # Align names with importances
        n_features = len(importances)
        names = feature_names[:n_features] if len(feature_names) >= n_features else [
            f"feature_{i}" for i in range(n_features)
        ]

        importance_series = pd.Series(importances, index=names)
        importance_series = importance_series.sort_values(ascending=False)

        return importance_series

    def _plot_feature_importance(
        self, importances: pd.Series, model_name: str, top_n: int
    ):
        """Generate and save feature importance bar chart."""
        top = importances.head(top_n)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
        top.sort_values().plot(kind="barh", color=colors)
        plt.title(
            f"Top {top_n} Feature Importances — {model_name}",
            fontsize=14, fontweight="bold"
        )
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "feature_importance.png"), dpi=150
        )
        plt.close()

        logger.info(f"Top {top_n} features:\n{top.to_string()}")

    def _try_shap(
        self, model, X_test: Optional[np.ndarray],
        feature_names: List[str], model_name: str
    ):
        """
        Attempt SHAP analysis. Gracefully skips if shap is not installed
        or encounters errors.
        """
        if X_test is None:
            return

        try:
            import shap

            logger.info("SHAP analysis started")

            # Use a small sample to keep SHAP fast
            sample_size = min(100, len(X_test))
            X_sample = X_test[:sample_size]

            # Choose appropriate explainer
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(
                    model.predict_proba, X_sample[:20]
                )

            shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Summary plot
            plt.figure(figsize=(10, 8))
            n_features = X_sample.shape[1]
            names = feature_names[:n_features] if len(feature_names) >= n_features else [
                f"feature_{i}" for i in range(n_features)
            ]
            shap.summary_plot(
                shap_values, X_sample,
                feature_names=names,
                show=False,
            )
            plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plots_dir, "shap_summary.png"), dpi=150
            )
            plt.close()

            logger.info("SHAP analysis completed")

        except ImportError:
            logger.info("SHAP not installed — skipping SHAP analysis. "
                       "Install with: pip install shap")
        except Exception as e:
            logger.warning(f"SHAP analysis failed (non-critical): {e}")
