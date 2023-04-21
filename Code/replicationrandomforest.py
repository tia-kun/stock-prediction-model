from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


class ReplicationRFModel:

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame,
                 start_index: int, stop_index: int, verbose: bool):
        self.clf = RandomForestRegressor()
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.start_index = start_index
        self.stop_index = stop_index
        self.verbose = verbose

    def run(self):
        output_length = self.stop_index + 1 - self.start_index
        predicted_output = np.empty(output_length)

        for i in range(self.start_index, self.stop_index + 1):
            predicted_output[i - self.start_index] = self.run_one_iteration(i)

        return predicted_output, (self.y[self.start_index:
                                         self.stop_index + 1])

    def run_one_iteration(self, index):
        X = self.X[:index]
        y = self.y[:index]
        self.clf.fit(X, y)
        output = self.clf.predict(self.X[index].reshape(1, -1))
        if self.verbose:
            print(index)

        return output

    def test(self):
        predicted, actual = self.run()

        rmse = mean_squared_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)
        mbe = np.mean(actual, predicted)

        return actual, predicted, rmse, mape, mbe
