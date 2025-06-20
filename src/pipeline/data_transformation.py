from pathlib import Path

import pandas as pd

from src.utils.common import read_yaml
from src.utils.logger import logger

config = read_yaml(Path("src/config/config.yaml"))


def transform_data(processed_data_path, output_path):
    """
    This function is responsible for transforming the data into a format
    suitable for model training. It includes feature engineering, handling
    categorical features, and saving the transformed data.
    """

    # Load data and change data types for specific column
    try:
        logger.info(f"Reading data from {processed_data_path}")
        df = pd.read_csv(processed_data_path, dtype={"StateHoliday": str})
        logger.info("File read successfully")
        df = drop_missing_values_columns(df)
        df = handle_date_features(df)
        df = handle_categorical_features(df)

        logger.info("Data transformation completed successfully")
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Transformed data saved to {output_path}")

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error in data transformation: {e}")
        raise e


def drop_missing_values_columns(df):
    """
    Drop columns with missing values.
    """
    cols_to_drop = [
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    ]
    logger.info(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    return df


def handle_date_features(df):
    """
    Handle date features by extracting year, month, day,
     dayOfWeek, WeekOfYear, IsWeekend from the date column.
    """
    logger.info("Handling date features")

    df["Date"] = pd.to_datetime(df["Date"])
    # Create new time-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["IsWeekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)

    # Drop the original 'Date' column
    df.drop("Date", axis=1, inplace=True)

    return df


def handle_categorical_features(df):
    """
    Handle categorical features by converting them to numerical values.
    """
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"Handling categorical features: {categorical_cols}")
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


def transform_train_data():
    logger.info("Transforming training data...")
    val_path = Path(config.data_transformation.input_train_path)
    output_path = Path(config.data_transformation.output_train_path)
    transform_data(val_path, output_path)


def transform_val_data():
    logger.info("Transforming validation data...")
    val_path = Path(config.data_transformation.input_val_path)
    output_path = Path(config.data_transformation.output_val_path)
    transform_data(val_path, output_path)


if __name__ == "__main__":
    transform_train_data()
