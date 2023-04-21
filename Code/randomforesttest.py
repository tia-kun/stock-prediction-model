from dataparser import DataParser
from timeseriesforest import TSRFModel
# import matplotlib.pyplot as plt
import pandas as pd

import datetime


def main():
    ticker = "PFE"
    offset_dates = True
    TRAINING_DATA_FILE = (
            f"../Corporation Data/Training Data/{ticker}_Training.csv")
    TESTING_DATA_FILE = (
            f"../Corporation Data/Test Data/{ticker}_Testing.csv")

    testing_dp = DataParser(TESTING_DATA_FILE, autoparse=True)
    training_dp = DataParser(TRAINING_DATA_FILE, autoparse=True)

    testing_data = testing_dp.get_data()
    training_data = training_dp.get_data()

    if offset_dates:
        training_data = (pd
                         .concat([training_data, testing_data
                                  [:datetime.date(2018, 4, 11)]]))

        testing_data = testing_data[datetime.date(2018, 4, 11):]

    features = ["Volume", "H-L", "O-C", "7_day_ma", "14_day_ma",
                "21_day_ma", "7_day_std"]

    training_X = training_data[features]
    training_y = training_data["next_day_close"]
    testing_X = testing_data[features]
    testing_y = testing_data["next_day_close"]

    tsrf = TSRFModel(training_X, training_y, n_estimators=1000, min_interval=3)

    rmse, mape, mbe, predicted = tsrf.test_plot(testing_X, testing_y,
                                                to_std_out=True,
                                                save_img=False)


if __name__ == '__main__':
    main()
