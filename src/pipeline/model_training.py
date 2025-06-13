import os
from math import sqrt
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.common import read_yaml
from src.utils.logger import logger


def train_model():
    """
    Train both XGBoost and LightGBM models and log them to MLflow.
    """
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    config = read_yaml(Path("src/config/config.yaml"))
    output_path = Path(config.data_transformation.output_data_path)

    df = pd.read_csv(output_path)
    logger.info("File loaded successfully")

    # Features & target
    target_col = "Sales"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("sales_prediction")

    # Define param grids
    xgb_params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100,
        "objective": "reg:squarederror",
    }

    lgbm_params = {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_estimators": 100,
        "objective": "regression",
    }

    # Train XGBoost
    with mlflow.start_run(run_name="XGBoost"):
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = sqrt(mean_squared_error(y_val, preds))

        mlflow.log_params(xgb_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")

    # Train LightGBM
    with mlflow.start_run(run_name="LightGBM"):
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = sqrt(mean_squared_error(y_val, preds))

        mlflow.log_params(lgbm_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(model, artifact_path="lgbm_model")


if __name__ == "__main__":
    logger.info("Start training models")
    train_model()
    logger.info("Model training completed and logged to MLflow.")
