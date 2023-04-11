from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import pandas as pd
from typing import Tuple
import random


class RFModel:

    def __init__(self, training_X: pd.DataFrame, training_y: pd.DataFrame,
                 max_depth: int = 100, n_estimators: int = 100,
                 random_state: int = random.randint(0, 100),
                 criterion: str = "squared_error"):
        self.clf = RandomForestRegressor(max_depth=max_depth,
                                         random_state=random_state,
                                         n_estimators=n_estimators,
                                         criterion=criterion)
        self.clf.fit(training_X, training_y)

    def test(self, testing_X: pd.DataFrame, testing_y: pd.DataFrame,
             to_std_out: bool = False) -> Tuple[float, float, float, np.array,
                                                np.array]:
        predicted = self.clf.predict(testing_X)

        mape, rmse, mbe = self.validate(predicted, testing_y)

        if to_std_out:
            s = f"| MAPE: {mape} | RMSE: {rmse} | MBE: {mbe} |"
            string_len = len(s)
            s = "="*string_len + "\n" + s + "\n" + "="*string_len + "\n"
            print(s)
        return mape, rmse, mbe, predicted, testing_y

    def validate(self, predicted: np.array,
                 realized: np.array) -> Tuple[float, float, float]:
        mape = mean_absolute_percentage_error(realized, predicted)
        rmse = mean_squared_error(realized, predicted)
        mbe = np.mean(realized - predicted)

        return mape, rmse, mbe
