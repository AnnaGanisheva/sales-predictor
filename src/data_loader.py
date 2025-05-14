import pandas as pd

def load_train_data(path: str = "./data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def load_store_data(path: str = "./data/store.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def merge_data(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(train_df, store_df, on="Store", how="left")