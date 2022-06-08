import pandas as pd
import numpy as np


class Indicators:

    def norma(self, data, mini, maxi):
        z = (data - mini) / (maxi - mini)
        return z

    def desnorma(self, data, mini, maxi):
        d = (data * (maxi - mini)) - mini
        return d

    def simple_ma(self, data, period):
        return data.rolling(period).mean()

    def rsi(self, data, period):
        ## ver np.where para acelerar
        change = data.diff()
        gain = pd.Series(list(map(lambda x: x if x > 0 else 0, change)))
        loss = pd.Series(list(map(lambda x: abs(x) if x < 0 else 0, change)))
        rs = gain.rolling(period).mean() / loss.rolling(period).mean()
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def stochastic_oscillator(self, data_close, data_low, data_high, period):
        so = 100 * ((data_close - data_low.rolling(period).min()) /
                    (data_close - data_high.rolling(period).max() - data_close - data_low.rolling(period).min()))
        return so

    def williams(self, data_close, data_low, data_high, period):
        r = ((data_high.rolling(period).max() - data_close) /
             (data_close - data_high.rolling(period).max() - data_close - data_low.rolling(period).min())) * -100
        return r

    def macd(self, data_close, p, q, r):
        # 12,26,9
        signal_line = data_close.ewm(span=p, adjust=False).mean() - data_close.ewm(span=q, adjust=False).mean()
        return signal_line.ewm(span=r, adjust=False).mean()

    def obv(self, data_close, data_volumen):
        data_close = np.array(data_close)
        obv = np.zeros(data_close.shape)
        obv[0] = data_volumen[0]
        for i in range(1, data_close.shape[0]):
            if data_close[i] - data_close[i - 1] > 0:
                obv[i] = obv[i - 1] + data_volumen[i]
            elif data_close[i] - data_close[i - 1] < 0:
                obv[i] = obv[i - 1] - data_volumen[i]
            else:
                obv[i] = obv[i - 1]
        return obv

    def add_ma(self, data_set, name_data_set):
        data_set['ma_5 ' + str(name_data_set)] = self.simple_ma(data_set['Close'], 5)
        data_set['ma_10 ' + str(name_data_set)] = self.simple_ma(data_set['Close'], 10)
        data_set['ma_20 ' + str(name_data_set)] = self.simple_ma(data_set['Close'], 20)
        data_set['ma_50 ' + str(name_data_set)] = self.simple_ma(data_set['Close'], 50)
        data_set['ma_100 ' + str(name_data_set)] = self.simple_ma(data_set['Close'], 100)
        data_set['ma_200 ' + str(name_data_set)] = self.simple_ma(data_set['Close'], 200)
        return data_set

    def add_features(self, data_set):
        data_set['MACD'] = self.macd(data_set['Close'], 12, 26, 9)
        data_set['RSI'] = self.rsi(data_set['Close'], 21)
        data_set['MA_14'] = self.simple_ma(data_set['Close'], 14)
        data_set['MA_21'] = self.simple_ma(data_set['Close'], 21)
        data_set['MA_100'] = self.simple_ma(data_set['Close'], 100)
        data_set = data_set.sort_values('Date')
        return data_set

    def add_lags_close(self, data_set, name_data_set):
        data_set['1-day_close ' + str(name_data_set)] = data_set['Close'].shift(1)
        data_set['2-day_close ' + str(name_data_set)] = data_set['Close'].shift(2)
        data_set['3-day_close ' + str(name_data_set)] = data_set['Close'].shift(3)
        data_set['4-day_close ' + str(name_data_set)] = data_set['Close'].shift(4)
        data_set['5-day_close ' + str(name_data_set)] = data_set['Close'].shift(5)
        data_set['6-day_close ' + str(name_data_set)] = data_set['Close'].shift(6)
        data_set['7-day_close ' + str(name_data_set)] = data_set['Close'].shift(7)
        data_set['8-day_close ' + str(name_data_set)] = data_set['Close'].shift(8)
        return data_set

    def volume_difference(self, data_set):
        # data_set['1-day_volume'] = (data_set['Volume'].shift(2)-data_set['Volume'].shift(1))/data_set['Volume'].shift(1)
        data_set['1-day_volume'] = data_set['Volume'].pct_change()
        return data_set

    def append_indicators(self, data_set, name_data_set):
        data_0 = data_set.drop(['Date'], axis=1)
        # data_0_n = self.norma(data_0, np.min(data_0), np.max(data_0))
        # data_1 = self.add_ma(data_0, name_data_set) ## Add MA

        data_1 = self.add_ma(data_0, name_data_set)  ## Add MA
        data_1 = self.add_indicators(data_1, name_data_set)  ## Add other indicators
        data_1 = self.add_lags_close(data_1, name_data_set)  ## Add lags
        return data_1

    def add_initial_cut(self, data_set, cut_dimensions):
        data_set = data_set[cut_dimensions:len(data_set)]
        data_set = data_set.reset_index()
        return data_set

