import matplotlib.pyplot as plt

from dataparser import DataParser


def main():
    tickers = ["GS", "NKE", "JNJ", "JPM", "PFE"]
    factor = "High"

    fig, ax = plt.subplots(nrows=len(tickers), ncols=2)

    for index, ticker in enumerate(tickers):

        training_data_file = (
            f"../Corporation Data/Training Data/{ticker}_Training.csv")

        dp = DataParser(training_data_file, autoparse=True)
        y = dp.get_data()[factor]
        x = dp.get_data().index
        ax[index, 0].plot(x, y)
        ax[index, 0].title.set_text(ticker + " Training")

    for index, ticker in enumerate(tickers):

        testing_data_file = (
            f"../Corporation Data/Test Data/{ticker}_Testing.csv")

        dp = DataParser(testing_data_file, autoparse=True)
        y = dp.get_data()[factor]
        x = dp.get_data().index
        ax[index, 1].plot(x, y)
        ax[index, 1].title.set_text(ticker + " Testing")

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
