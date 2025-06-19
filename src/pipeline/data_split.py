from pathlib import Path

import pandas as pd

from src.utils.common import read_yaml
from src.utils.logger import logger


def split_train_val_by_last_month(
    input_path: Path, train_output: Path, val_output: Path
) -> None:
    df = pd.read_csv(input_path, low_memory=False)

    # Convert the 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Identify the last available month in the dataset
    last_month = df["Date"].max().to_period("M")

    # Create a YearMonth column to simplify filtering
    df["YearMonth"] = df["Date"].dt.to_period("M")

    # Validation set: all records from the last month
    val_df = df[df["YearMonth"] == last_month]

    # Training set: all records before the last month
    train_df = df[df["YearMonth"] < last_month]

    # Ensure output directories exist
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    # Save the train and validation datasets
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    logger.info(f"Train set: {train_df.shape}, saved to {train_output}")
    logger.info(f"Val set: {val_df.shape}, saved to {val_output}")


def split_data_train_val():
    """
    Split the dataset into training and validation sets based on the last month.
    The last month of data will be used for validation, and all previous months
    will be used for training.
    """
    config = read_yaml(Path("src/config/config.yaml"))
    input_path = Path(config.data_paths.processed_merged_file)
    train_output_path = Path(config.data_paths.processed_train_file)
    val_output_path = Path(config.data_paths.processed_val_file)

    split_train_val_by_last_month(input_path, train_output_path, val_output_path)


if __name__ == "__main__":
    split_data_train_val()
