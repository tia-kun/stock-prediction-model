from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

class RFModel:
    def __init__(self, training_X, training_y, max_depth, random_state):
        self.clf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        self.clf.fit(training_X, training_y)


    def test(self, testing_X, testing_y):
        output = self.clf.predict(testing_X)

        mape, rmse, mbe = self.validate(predicted, realized)

        s = f"MAPE: {mape} | RMSE: {rmse} | MBE: {mbe}"
        l = len(s)
        s = "="*l + "\n" + s + "\n" + "="*l + "\n"
        

    def validate(self, predicted, realized):
        mape = mean_absolute_percentage_error(realized, predicted)
        rmse = mean_squared_error(realized, predicted)
        mbe = self.calc_mbe(predicted, realized)

        return mape, rmse, mbe


    def calc_mbe(self, predicted, realized):
        return np.mean(realized - predicted)
