from pathlib import Path

import pandas as pd

from src.utils.common import read_yaml
from src.utils.logger import logger

config = read_yaml(Path("src/config/config.yaml"))


def ingest_data(data_path: Path, output_path: Path) -> None:
    """
    Ingests data from CSV files, merges them, and saves the processed data.
    """

    raw_dir = Path(config.data_paths.raw_data_dir)
    data_path = raw_dir / data_path
    store_path = raw_dir / config.data_paths.store_file

    logger.info(f"Loading data from {data_path} and {store_path}")
    data_df = pd.read_csv(data_path, low_memory=False)
    store_df = pd.read_csv(store_path)

    merged_df = data_df.merge(store_df, on="Store", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def load_train_data():
    logger.info("Loading amd merde row data...")
    data_path = Path(config.data_paths.train_file)
    output_path = Path(config.data_paths.processed_merged_file)
    ingest_data(data_path, output_path)


# TODO: Uncomment and change output path accordingly
# def load_prediction_data():
#     logger.info("Loading prediction data...")
#     data_path = Path(config.data_paths.predict_file)
#     output_path = Path()
#     ingest_data(data_path, output_path)


if __name__ == "__main__":
    load_train_data()
