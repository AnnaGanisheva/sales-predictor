from pathlib import Path

import pandas as pd

from src.utils.common import read_yaml
from src.utils.logger import logger


def ingest_data():
    """
    This function is responsible for ingesting raw data from CSV files,
    merging them, and saving the processed data to a specified output path.
    It reads configuration settings from a YAML file to determine file paths.
    """

    config = read_yaml(Path("src/config/config.yaml"))
    raw_dir = Path(config.data_paths.raw_data_dir)

    train_path = raw_dir / config.data_paths.train_file
    store_path = raw_dir / config.data_paths.store_file
    output_path = Path(config.data_paths.processed_file)

    # load raw data
    logger.info(f"Loading data from {train_path} and {store_path}")
    train_df = pd.read_csv(train_path, low_memory=False)
    store_df = pd.read_csv(store_path)

    # merge
    merged_df = train_df.merge(store_df, on="Store", how="left")

    # save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    ingest_data()
