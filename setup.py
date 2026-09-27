from setuptools import setup, find_packages

setup(
    name="heartguard",
    version="1.0.0",
    description="Explainable Heart Disease Risk Prediction - End-to-End ML Pipeline",
    author="HeartGuard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "flask>=3.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "mlflow>=2.8.0",
            "dvc>=3.30.0",
            "shap>=0.43.0",
        ],
    },
)
