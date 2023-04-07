from dataparser import DataParser
from randomforest import RFModel
import matplotlib.pyplot as plt


def main():
    TRAINING_DATA_DIR = "../Corporation Data/Training Data/GS_Training.csv"
    TESTING_DATA_DIR = "../Corporation Data/Test Data/GS_Testing.csv"

    testing_data = DataParser(TESTING_DATA_DIR, autoparse=True).get_data()
    training_data = DataParser(TRAINING_DATA_DIR, autoparse=True).get_data()

    training_X = training_data.drop(columns=["Close"])
    training_y = training_data["Close"]
    testing_X = testing_data.drop(columns=["Close"])
    testing_y = testing_data["Close"]

    rf = RFModel(training_X, training_y, max_depth=100)

    MAPE, RMSE, MBE, predicted, actual = rf.test(testing_X, testing_y,
                                                 to_std_out=True)

    x = testing_y.index
    y = predicted
    plt.plot(x, y, marker='o')

    y = actual
    plt.plot(x, y, marker='o')

    plt.show()


if __name__ == '__main__':
    main()
