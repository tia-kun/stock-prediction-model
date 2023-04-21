import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from randomforest import RFModel
from dataparser import DataParser
from replicationrandomforest import ReplicationRFModel

tickers = ["NKE", "PFE", "GS", "JPM", "JNJ"]
plots = ["Naive Plot", "Plot with Shuffled Data", "Plot with Retraining"]
FEATURES = ["Volume", "H-L", "O-C", "7_day_ma", "14_day_ma", "21_day_ma", "7_day_std"]
split_index = 1992
PLOT_WIDTH = 13
PLOT_HEIGHT = 10


def save_plot(predicted, actual, x, title, xaxis, yaxis, filename=None,
              show=False):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.plot(x, predicted, color="r", marker="o", label="Predicted Price")
    plt.plot(x, actual, color="b", marker="s", label="Realized Price")

    plt.legend()
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

    if show:
        plt.show()
    else:
        plt.savefig(filename)

    plt.clf()


for index, plot in enumerate(plots):
    for ticker in tickers:
        TRAINING_DATA_FILE = f"../Corporation Data/Training Data/{ticker}_Training.csv"
        TESTING_DATA_FILE = f"../Corporation Data/Test Data/{ticker}_Testing.csv"
        FEATURES = ["Volume", "H-L", "O-C", "7_day_ma", "14_day_ma",
                    "21_day_ma", "7_day_std"]

        data = pd.concat([pd.read_csv(TRAINING_DATA_FILE),
                          pd.read_csv(TESTING_DATA_FILE)]).reset_index()
        data_dp = DataParser(data=data, autoparse=True)
        processed_data = data_dp.get_data().drop("index", axis=1)

        training_data = processed_data.iloc[:split_index + 1]
        testing_data = processed_data.iloc[split_index:]

        if index == 0:
            training_y = training_data["next_day_close"]
            training_X = training_data[FEATURES]
            testing_y = testing_data["next_day_close"]
            testing_X = testing_data[FEATURES]

            rf = RFModel(training_X, training_y)

            mape, rmse, mbe, predicted, actual = rf.test(testing_X, testing_y)

            x = testing_y.index
            xaxis = "Date"
            yaxis = "Price"
            title = f"{plot}: {ticker}"

            save_plot(predicted, actual, x, title, xaxis, yaxis,
                      filename=f"naive_plot_{ticker}.png", show=False)

        elif index == 1:

            X = processed_data[FEATURES]
            y = processed_data["next_day_close"]

            training_X, _, training_y, _ = train_test_split(X, y, test_size=.2)

            rf = RFModel(training_X, training_y)

            testing_X = testing_data[FEATURES]
            testing_y = testing_data["next_day_close"]

            mape, rmse, mbe, predicted, actual = rf.test(testing_X, testing_y)

            x = testing_y.index
            xaxis = "Date"
            yaxis = "Price"
            title = f"{plot}: {ticker}"

            save_plot(predicted, actual, x, title, xaxis, yaxis,
                      filename=f"shuffled_plot_{ticker}.png", show=False)

        else:

            rf = ReplicationRFModel(processed_data
                                    .drop("next_day_close", axis=1),
                                    processed_data["next_day_close"],
                                    1992, 2494, verbose=True)

            print(ticker)
            actual, predicted, rmse, mape, mbe = rf.test()

            x = testing_y.index
            xaxis = "Date"
            yaxis = "Price"
            title = f"{plot}: {ticker}"

            save_plot(predicted, actual, x, title, xaxis, yaxis,
                      filename=f"replication_plot_{ticker}.png", show=False)

        with open("programout.txt", "a") as pout:
            out = f"{plot} {ticker} MAPE: {mape} RMSE: {rmse} MBE: {mbe}\n"
            pout.write(out)
