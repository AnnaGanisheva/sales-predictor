import os
from functools import partial
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split

from src.utils.common import read_yaml
from src.utils.logger import logger


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna to optimize XGBoost hyperparameters.
    """

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    preds = model.predict(dvalid)
    rmse = root_mean_squared_error(y_val, preds)

    # mask devide 0
    mask = y_val != 0
    mape = mean_absolute_percentage_error(y_val[mask], preds[mask]) * 100

    # Log with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("mape", round(mape, 2))

    return rmse


def tune_model():
    """
    Tune XGBoost model hyperparameters using Optuna and
    log the best model to MLflow.
    """
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("xgboost-optuna")

    config = read_yaml(Path("src/config/config.yaml"))
    output_path = Path(config.data_transformation.output_data_path)

    # Load data
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

    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction="minimize")
    objective_func = partial(
        objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )
    study.optimize(objective_func, n_trials=30)
    logger.info("Hyperparameter tuning completed.")

    # Best model with best params
    best_params = study.best_params
    logger.info("Best params:", best_params)

    # Save best model
    dtrain_full = xgb.DMatrix(X, label=y)
    final_model = xgb.train(
        best_params, dtrain_full, num_boost_round=study.best_trial.number
    )

    mlflow.xgboost.log_model(
        final_model,
        artifact_path="model",
        registered_model_name="xgboost_sales_forecaster"
    )
    logger.info("Final XGBoost model logged and registered in MLflow.")


if __name__ == "__main__":
    logger.info("Start tuning XGBoost model.")
    tune_model()
    logger.info("Done.")
