from dataparser import DataParser
from randomforest import RFModel
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import datetime


def main():
    ticker = "JNJ"
    k_splits = 5
    offset_dates = False
    TRAINING_DATA_FILE = (
            f"../Corporation Data/Training Data/{ticker}_Training.csv")
    TESTING_DATA_FILE = (
            f"../Corporation Data/Test Data/{ticker}_Testing.csv")

    tss = TimeSeriesSplit(n_splits=k_splits)

    testing_dp = DataParser(TESTING_DATA_FILE, autoparse=True)
    training_dp = DataParser(TRAINING_DATA_FILE, autoparse=True)

    testing_data = testing_dp.get_data()
    training_data = training_dp.get_data()

    if offset_dates:
        training_data = (pd
                         .concat([training_data, testing_data
                                  [:datetime.date(2018, 4, 11)]]))

        testing_data = testing_data[datetime.date(2018, 4, 11):]

    '''
    ticker = "JNJ"
    TRAINING_DATA_DIR = (
            f"../Corporation Data/Training Data/{ticker}_Training.csv")

    TESTING_DATA_DIR = f"../Corporation Data/Test Data/{ticker}_Testing.csv"



    testing_data_raw = DataParser(TESTING_DATA_DIR, autoparse=True).get_data()

    testing_data = testing_data_raw[datetime.date(2018, 4, 11):]

    initial_factors = ["Volume"]

    computed_factors = ["H-L", "O-C", "7_day_ma", "14_day_ma", "21_day_ma",
                        "7_day_std"]

    factors = initial_factors + computed_factors
    training_data = pd.concat([DataParser(TRAINING_DATA_DIR, autoparse=True)
                              .get_data(),
                              testing_data_raw[:datetime.date(2018, 4, 11)]])

    training_X = training_data[factors]
    training_y = training_data["next_day_close"]
    testing_X = testing_data[factors]
    testing_y = testing_data["next_day_close"]

    rf = RFModel(training_X,
                 training_y,
                 max_depth=100,
                 n_estimators=100,
                 criterion="squared_error",
                 min_samples_leaf=1,
                 max_features="sqrt")

    MAPE, RMSE, MBE, predicted, actual = rf.test(testing_X, testing_y,
                                                 to_std_out=True)

    x = testing_y.index
    y = predicted
    plt.plot(x, y, marker='o', color='b', label="Predicted Price")

    y = actual
    plt.plot(x, y, marker='o', color='r', label="Actual Price")

    plt.legend()

    plt.show()

    '''


if __name__ == '__main__':
    main()
