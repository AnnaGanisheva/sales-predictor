import pandas as pd

from src import data_loader


def test_merge_data_on_fake_data():
    train = pd.DataFrame({"Store": [1, 2, 3], "Sales": [5000, 6000, 7000]})

    store = pd.DataFrame({"Store": [1, 2, 3], "StoreType": ["a", "b", "c"]})

    merged = data_loader.merge_data(train, store)

    assert merged.shape[0] == train.shape[0]
    assert "StoreType" in merged.columns
