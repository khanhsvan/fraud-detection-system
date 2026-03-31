# Fraud Detection System

Production-style fraud detection system built with Python, scikit-learn, FastAPI, and Streamlit. The project trains tuned fraud classifiers on the Kaggle Credit Card Fraud Detection dataset, logs training activity in real time, generates explainability artifacts with SHAP, produces evaluation reports, and serves the best model through an API.

## Project Structure

```text
fraud-detection/
|-- data/
|   `-- creditcard.csv
|-- model/
|   |-- best_model.pkl
|   `-- scaler.pkl
|-- reports/
|   |-- training_report.txt
|   |-- metrics.json
|   `-- shap_summary.png
|-- logs/
|   `-- training.log
|-- dashboard/
|   `-- app.py
|-- train.py
|-- monitor.py
|-- app.py
|-- requirements.txt
`-- README.md
```

## Overview

This system is designed to feel like a realistic ML workflow rather than a single training script. It covers:

- Data loading, scaling, stratified splitting, and SMOTE-based imbalance handling
- Baseline model training and tuned model selection
- Hyperparameter search for Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- Persisted artifacts for model serving
- Training logs and monitoring utilities
- Metrics and report generation
- Feature importance and SHAP-based explainability
- GPU-aware training path for XGBoost with automatic CPU fallback
- FastAPI inference endpoint
- Streamlit dashboard for operational visibility

## Tech Stack

- Python
- pandas
- scikit-learn
- imbalanced-learn
- FastAPI
- Uvicorn
- joblib
- matplotlib
- seaborn
- SHAP
- Streamlit
- tqdm
- XGBoost

## Dataset

Place the Kaggle Credit Card Fraud Detection dataset at:

```text
data/creditcard.csv
```

Expected columns:

- Features: `Time`, `V1` through `V28`, `Amount`
- Target: `Class`

## Training Pipeline

`train.py` performs the following:

1. Loads the dataset with pandas
2. Splits features and target
3. Creates a stratified train/test split
4. Fits and saves a `StandardScaler`
5. Applies SMOTE on the training split only
6. Trains baseline models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost when installed
7. Tunes all candidate models with `RandomizedSearchCV`
8. Compares baseline vs tuned metrics
9. Selects the best model by fraud-class F1 score
10. Saves:
   - `model/best_model.pkl`
   - `model/scaler.pkl`
   - `reports/metrics.json`
   - `reports/training_report.txt`
   - `reports/shap_summary.png`
   - `logs/training.log`

## Monitoring

### Real-Time Log Monitor

Run:

```bash
python monitor.py
```

This tails the latest training logs and prints a compact metrics summary if `reports/metrics.json` exists.

### Streamlit Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

Dashboard capabilities:

- View training logs
- View tuned model metrics
- Inspect the confusion matrix
- Compare baseline vs tuned models
- Review feature importance
- Display SHAP summary plot
- Auto refresh every few seconds

## FastAPI Inference Service

Run:

```bash
uvicorn app:app --reload
```

Endpoints:

- `GET /`
  Returns service health, selected model name, and expected feature count
- `POST /predict`
  Scores a transaction and returns prediction plus fraud probability

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"features\": [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]}"
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.0187,
  "model_name": "random_forest"
}
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

After training completes, the project will generate artifacts under `model/`, `reports/`, and `logs/`.

## Explainability

The training pipeline generates:

- Random Forest feature importance ranking
- SHAP summary plot saved to `reports/shap_summary.png`

These artifacts help explain which features most strongly influence fraud predictions.

## Notes

- Tuning is optimized around the fraud-class F1 score to better reflect class imbalance.
- SMOTE is applied only to the training split to avoid leakage.
- scikit-learn models remain CPU-based, so the pipeline limits CPU workers to reduce full-core saturation.
- XGBoost automatically uses `device="cuda"` when an NVIDIA GPU is detected and falls back to CPU otherwise.
- If the best model is not tree-based, top feature importance remains sourced from the tuned Random Forest report.
- The API and dashboard read the saved artifacts directly, so no code changes are needed after training.
