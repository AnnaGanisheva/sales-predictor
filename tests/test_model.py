import pandas as pd

from src import model


def test_train_and_evaluate():
    X = pd.DataFrame(
        {
            "DayOfWeek": [1, 2, 3],
            "Promo": [0, 1, 0],
            "SchoolHoliday": [0, 0, 1],
        }
    )
    y = pd.Series([4500, 7000, 4000])

    m = model.train_model(X, y)
    rmse = model.evaluate_model(m, X, y)

    assert rmse >= 0
