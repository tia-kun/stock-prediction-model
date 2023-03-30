import pandas as pd
import datetime


class DataParser:
    def __init__(self, data):
        self.data = data
        self.data["Date"] = self.parse_dates()
        self.data.set_index("Date")


    def parse_dates(self):
        return pd.to_datetime(self.data["Date"], format='%Y-%m-%d')


    def calc_HLrange(self):
        return self.data["High"] - self.data["Low"]


    def calc_OCrange(self):
        return self.data["Open"] - self.data["Close"]


    def calc_MA(self, timeframe, mindate):
        delta = datetime.timedelta(days=timeframe)
        
        
    def __calc_MA_helper(self, date, delta, mindate):
        minimum = max(data - delta, mindate)

        return self.data.iloc[minimum : date]

