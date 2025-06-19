from src.utils.logger import logger
from src.pipeline.data_ingestion import load_train_data
from src.pipeline.data_split import split_data_train_val
from src.pipeline.data_transformation import transform_train_data
from src.pipeline.tune_model_xgb_optuna import run_optuna_xgb
from src.pipeline.tune_model_lgb_optuna import run_optuna_lgb


def main():
    logger.info("Loading raw training data...")
    load_train_data()

    logger.info("Splitting into train/validation sets...")
    split_data_train_val()

    logger.info("Transforming training and validation data...")
    transform_train_data()

    logger.info("Starting Optuna tuning for XGBoost...")
    run_optuna_xgb()

    logger.info("Starting Optuna tuning for LightGBM...")
    run_optuna_lgb()

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
