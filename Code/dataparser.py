import pandas as pd
import os


class DataParser:
    def __init__(self, data_filename: str, autoparse=False):

        if os.path.isfile(data_filename):
            self.__data = pd.read_csv(data_filename)
        else:
            raise OSError(f"{data_filename} is not a real file")

        if autoparse:
            self.parse_all()

    def parse_all(self):
        moving_average_symbol = "Close"

        self.__data["H-L"] = self.calc_HLrange()
        self.__data["O-C"] = self.calc_OCrange()
        self.__data["7_day_ma"] = self.calc_MA(7, moving_average_symbol)
        self.__data["14_day_ma"] = self.calc_MA(14, moving_average_symbol)
        self.__data["21_day_ma"] = self.calc_MA(21, moving_average_symbol)
        self.__data["7_day_std"] = self.calc_STD(7, moving_average_symbol)

        self.__data = self.__data.dropna()
        self.__data["Date"] = pd.to_datetime(self.__data["Date"],
                                             format="%Y-%m-%d")
        self.__data.set_index("Date", inplace=True)

    def parse_dates(self) -> pd.Series:
        return pd.to_datetime(self.__data["Date"], format='%Y-%m-%d')

    def calc_HLrange(self) -> pd.Series:
        return self.__data["High"] - self.__data["Low"]

    def calc_OCrange(self) -> pd.Series:
        return self.__data["Open"] - self.__data["Close"]

    def calc_MA(self, days: int, symbol: str):
        return self.__data[symbol].rolling(window=days).mean()

    def calc_STD(self, days: int, symbol: str) -> pd.Series:
        return self.__data[symbol].rolling(window=days).std()

    def get_data(self):
        return self.__data
