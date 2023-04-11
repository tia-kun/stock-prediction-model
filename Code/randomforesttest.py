from dataparser import DataParser
from randomforest import RFModel
import matplotlib.pyplot as plt
import datetime


def main():
    TRAINING_DATA_DIR = "../Corporation Data/Training Data/GS_Training.csv"
    TESTING_DATA_DIR = "../Corporation Data/Test Data/GS_Testing.csv"

    testing_data = DataParser(TESTING_DATA_DIR,
                              autoparse=True).get_data()[datetime.date(2018,
                                                                       4, 11):]
    training_data = DataParser(TRAINING_DATA_DIR, autoparse=True).get_data()
    training_X = training_data[["H-L", "O-C", "7_day_ma", "14_day_ma",
                                "21_day_ma", "7_day_std"]]
    training_y = training_data["next_day_close"]
    testing_X = testing_data[["H-L", "O-C", "7_day_ma", "14_day_ma",
                              "21_day_ma", "7_day_std"]]
    testing_y = testing_data["next_day_close"]

    rf = RFModel(training_X, training_y, max_depth=150, n_estimators=200)

    MAPE, RMSE, MBE, predicted, actual = rf.test(testing_X, testing_y,
                                                 to_std_out=True)

    x = testing_y.index
    y = predicted
    plt.plot(x, y, marker='o', color='b', label="Predicted Price")

    y = actual
    plt.plot(x, y, marker='o', color='r', label="Actual Price")

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
