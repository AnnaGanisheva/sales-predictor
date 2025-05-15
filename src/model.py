from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model on the training data.
    Returns:
    model: The trained Random Forest Regressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    Parameters:
    model: The trained Random Forest Regressor model.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The target variable for testing.

    Returns:
    float: The RMSE of the model on the test data.
    """
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    return rmse


def save_model(model, filename):
    """
    Save the trained model to a file.
    Parameters:
    model: The trained Random Forest Regressor model.
    filename (str): The filename to save the model to.
    """
    import pickle

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
