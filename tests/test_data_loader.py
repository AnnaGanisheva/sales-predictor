from src import data_loader

def test_merge_shapes():
    train = data_loader.load_train_data()
    store = data_loader.load_store_data()
    merged = data_loader.merge_data(train, store)
    assert merged.shape[0] == train.shape[0]