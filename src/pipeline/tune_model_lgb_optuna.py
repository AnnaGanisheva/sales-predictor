import os
from functools import partial
from pathlib import Path

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.common import read_yaml
from src.utils.logger import logger


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna to optimize LightGBM hyperparameters.
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "seed": 42,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params, n_estimators=200)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(20)],
    )

    best_iteration = model.best_iteration_

    preds = model.predict(X_val)

    mask = y_val != 0
    mape = mean_absolute_percentage_error(y_val[mask], preds[mask]) * 100
    rmse = root_mean_squared_error(y_val, preds)
    logger.info(f"Trial {trial.number}: RMSE={rmse}, MAPE={mape}")

    trial.set_user_attr("best_iteration", model.best_iteration_)

    # log to mlflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_param("best_iteration", best_iteration)
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("mape", round(mape, 2))

    return rmse


def run_optuna_lgb():
    """
    Tune LG model hyperparameters using Optuna and
    log the best model to MLflow.
    """
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("lightgbm-experiment")

    config = read_yaml(Path("src/config/config.yaml"))
    output_path = Path(config.data_transformation.output_train_path)

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

    logger.info("Starting LightGBM tuning with Optuna...")
    study = optuna.create_study(direction="minimize")
    objective_func = partial(
        objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )

    study.optimize(objective_func, n_trials=30)
    logger.info("Hyperparameter tuning completed.")

    # Best model with best params
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs["best_iteration"]

    logger.info(f"Best params: {best_params}")

    logger.info("Train final model on all data")
    final_model = lgb.LGBMRegressor(**best_params, n_estimators=best_iteration)
    final_model.fit(X, y)

    X_sample = X.sample(n=100, random_state=42)
    y_sample = final_model.predict(X_sample)
    signature = infer_signature(X_sample, y_sample)

    input_example = X_sample.iloc[0:1]

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_param("best_iteration", best_iteration)

        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="lightgbm_sales_forecaster",
            signature=signature,
            input_example=input_example,
        )
    mlflow.end_run()
    logger.info("Final LightGBM model logged and registered in MLflow.")


if __name__ == "__main__":
    logger.info("Start tuning LightGBM model.")
    run_optuna_lgb()
    logger.info("Done.")
