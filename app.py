"""
HeartGuard Flask Application
REST API and web interface for heart disease risk prediction.

Endpoints:
    GET  /         → Web form for clinical input
    POST /predict  → Returns prediction, probability, and explanation
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

from src.heartguard.config import path_config, data_config, app_config
from src.heartguard.utils.common import load_object
from src.heartguard.exception import ModelNotFoundError, PredictionError
from src.heartguard.logger import get_logger
from src.heartguard.components.data_transformation import DataTransformation

logger = get_logger(__name__)

app = Flask(__name__)

# ─────────────────────────── Model Loading ──────────────────────────

_model = None
_preprocessor = None


def get_model(force_reload: bool = False):
    """Lazy-load the trained model."""
    global _model
    if _model is None or force_reload:
        if not os.path.exists(path_config.model_path):
            raise ModelNotFoundError(
                f"No trained model found at: {path_config.model_path}\n"
                "Run the training pipeline first:\n"
                "  python -m src.heartguard.pipeline.training_pipeline"
            )
        _model = load_object(path_config.model_path)
        logger.info("Model loaded successfully")
    return _model


def get_preprocessor(force_reload: bool = False):
    """Lazy-load the fitted preprocessor."""
    global _preprocessor
    if _preprocessor is None or force_reload:
        if not os.path.exists(path_config.preprocessor_path):
            raise ModelNotFoundError(
                f"No preprocessor found at: {path_config.preprocessor_path}\n"
                "Run the training pipeline first."
            )
        _preprocessor = load_object(path_config.preprocessor_path)
        logger.info("Preprocessor loaded successfully")
    return _preprocessor


# ─────────────────────────── Feature Schema ─────────────────────────

# Human-readable labels, types, and valid ranges for the web form
FEATURE_SCHEMA = [
    {"name": "age", "label": "Age", "type": "number", "min": 1, "max": 120, "step": 1, "placeholder": "e.g. 55"},
    {"name": "sex", "label": "Sex", "type": "select", "options": [("1", "Male"), ("0", "Female")]},
    {"name": "cp", "label": "Chest Pain Type", "type": "select",
     "options": [("0", "Typical Angina"), ("1", "Atypical Angina"),
                 ("2", "Non-Anginal Pain"), ("3", "Asymptomatic")]},
    {"name": "trestbps", "label": "Resting Blood Pressure (mm Hg)", "type": "number",
     "min": 50, "max": 250, "step": 1, "placeholder": "e.g. 130"},
    {"name": "chol", "label": "Serum Cholesterol (mg/dl)", "type": "number",
     "min": 50, "max": 600, "step": 1, "placeholder": "e.g. 250"},
    {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl", "type": "select",
     "options": [("0", "No"), ("1", "Yes")]},
    {"name": "restecg", "label": "Resting ECG Results", "type": "select",
     "options": [("0", "Normal"), ("1", "ST-T Wave Abnormality"),
                 ("2", "Left Ventricular Hypertrophy")]},
    {"name": "thalach", "label": "Maximum Heart Rate Achieved", "type": "number",
     "min": 50, "max": 250, "step": 1, "placeholder": "e.g. 150"},
    {"name": "exang", "label": "Exercise Induced Angina", "type": "select",
     "options": [("0", "No"), ("1", "Yes")]},
    {"name": "oldpeak", "label": "ST Depression (Oldpeak)", "type": "number",
     "min": -5, "max": 10, "step": 0.1, "placeholder": "e.g. 1.0"},
    {"name": "slope", "label": "Slope of Peak Exercise ST Segment", "type": "select",
     "options": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")]},
    {"name": "ca", "label": "Number of Major Vessels (0-3)", "type": "select",
     "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")]},
    {"name": "thal", "label": "Thalassemia", "type": "select",
     "options": [("0", "Normal"), ("1", "Fixed Defect"), ("2", "Reversible Defect")]},
]


def validate_and_parse_input(form_data: dict) -> pd.DataFrame:
    """
    Validate form input and return a single-row DataFrame.
    
    Args:
        form_data: Dictionary of feature name → value
        
    Returns:
        DataFrame with one row of numeric feature values
        
    Raises:
        PredictionError: If any required field is missing or invalid
    """
    parsed = {}
    errors = []

    for feature in FEATURE_SCHEMA:
        name = feature["name"]
        value = form_data.get(name)

        if value is None or str(value).strip() == "":
            errors.append(f"Missing required field: {feature['label']}")
            continue

        try:
            parsed[name] = float(value)
        except (ValueError, TypeError):
            errors.append(f"Invalid value for {feature['label']}: '{value}'")

    if errors:
        raise PredictionError("Input validation failed:\n" + "\n".join(errors))

    return pd.DataFrame([parsed])


def get_top_features(model, preprocessor, n: int = 5) -> list:
    """Get top contributing feature names from the model."""
    try:
        from src.heartguard.components.model_explanation import ModelExplanation
        explainer = ModelExplanation()
        feature_names = explainer.get_feature_names(preprocessor)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            return []

        n_features = min(len(importances), len(feature_names))
        pairs = sorted(
            zip(feature_names[:n_features], importances[:n_features]),
            key=lambda x: x[1], reverse=True
        )
        return [name for name, _ in pairs[:n]]
    except Exception:
        return []


# ─────────────────────────── Routes ─────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Render the web form."""
    return render_template("index.html", features=FEATURE_SCHEMA)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept clinical feature values and return a prediction.
    
    Supports both form submission (returns HTML) and JSON API.
    """
    logger.info("Prediction requested")

    try:
        # Determine input source
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()

        # Validate and parse input
        input_df = validate_and_parse_input(form_data)

        # Apply feature engineering (same as training)
        transformer = DataTransformation()
        input_df = transformer._add_engineered_features(input_df)

        # Load model and preprocessor
        model = get_model()
        preprocessor = get_preprocessor()

        # Preprocess input using the saved preprocessor
        input_transformed = preprocessor.transform(input_df)

        # Handle mismatch gracefully by reloading if artifacts were updated
        if hasattr(model, "n_features_in_") and input_transformed.shape[1] != model.n_features_in_:
            logger.warning("Feature count mismatch detected, refreshing artifacts from disk...")
            model = get_model(force_reload=True)
            preprocessor = get_preprocessor(force_reload=True)
            input_transformed = preprocessor.transform(input_df)

        # Make prediction
        prediction = int(model.predict(input_transformed)[0])
        probability = float(model.predict_proba(input_transformed)[0][1])

        # Get top features
        top_features = get_top_features(model, preprocessor)

        # Format result
        if prediction == 1:
            label = "Higher Predicted Risk"
            message = ("The model predicts a higher likelihood of heart disease "
                      "based on the provided inputs.")
        else:
            label = "Lower Predicted Risk"
            message = ("The model predicts a lower likelihood of heart disease "
                      "based on the provided inputs.")

        result = {
            "prediction": prediction,
            "label": label,
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 1),
            "message": message,
            "top_features": top_features,
        }

        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")

        # Return JSON for API calls, HTML for form submissions
        if request.is_json:
            return jsonify(result)
        return render_template("result.html", result=result)

    except PredictionError as e:
        logger.warning(f"Prediction error: {e}")
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return render_template("error.html", error=str(e)), 400

    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e}")
        if request.is_json:
            return jsonify({"error": str(e)}), 503
        return render_template("error.html", error=str(e)), 503

    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        error_msg = "An unexpected error occurred. Please check your inputs and try again."
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        return render_template("error.html", error=error_msg), 500


# ─────────────────────────── Main ───────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=app_config.host,
        port=app_config.port,
        debug=app_config.debug,
    )
