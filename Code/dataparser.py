import pandas as pd
import numpy as np
import datetime


class DataParser:
    def __init__(self, data: pd.DataFrame, autoparse=False):
        self.data = data

        if autoparse:
            self.parse_all()


    def parse_all(self):
        self.data["Date"] = self.parse_dates()
        self.data.set_index("Date")

        moving_average_symbol = "Close"

        self.data["H-L"] = self.parse_HLrange()
        self.data["O-C"] = self.parse_OCrange()
        self.data["7_day_ma"] = self.calc_MA(7, moving_average_symbol)
        self.data["14_day_ma"] = self.calc_MA(14, moving_average_symbol)
        self.data["21_day_ma"] = self.calc_MA(21, moving_average_symbol)
        self.data["7_day_std"] = self.calc_STD()


    def parse_dates(self) -> pd.Series:
        return pd.to_datetime(self.data["Date"], format='%Y-%m-%d')


    def calc_HLrange(self) -> pd.Series:
        return self.data["High"] - self.data["Low"]


    def calc_OCrange(self) -> pd.Series:
        return self.data["Open"] - self.data["Close"]


    def calc_MA(self, days: int, symbol: str) -> pd.Series:
        delta = datetime.timedelta(days=days)
        mindate = self.data.index[0]

        return np.vectorize(self.__calc_MA_helper)(self.data, mindate, symbol)

        
    def calc_STD(self, days: int, symbol: str) -> pd.Series:
        delta = datetime.timedelta(days=days)
        mindate = self.data.index[0]
        
        return np.vectorize(self.__calc_STD_helper)(self.data, mindate, symbol)


    def __calc_STD_helper(self, row, delta, mindate, symbol):
        minimum = max(data - delta, mindate)

        subtable = self.data.iloc[minimum : row.index][symbol]

        return np.std(subtable)


    def __calc_MA_helper(self, row, delta, mindate, symbol):
        minimum = max(data - delta, mindate)

        subtable = self.data.iloc[minimum : row.index][symbol]

        return np.mean(subtable)
