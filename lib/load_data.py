import pandas as pd
from .technical_analysis import Indicators

indicators = Indicators()

class load_data:
    def __init__(self):
        self.N_ASSETS = 4
        self.SHARES_NAME = ["AAPL", "MCD", "MMM", "XOM"]

    def data_preprocessing(self, data_sets, initial_cut):
        for i in range(0, self.N_ASSETS):
            data_sets[i] = indicators.add_features(data_sets[i])
            data_sets[i] = indicators.add_initial_cut(data_sets[i], initial_cut)
        return data_sets

    def training_data(self):
        df_a = pd.read_csv('data/AAPL_train.csv')
        df_b = pd.read_csv('data/MCD_train.csv')
        df_c = pd.read_csv('data/MMM_train.csv')
        df_d = pd.read_csv('data/XOM_train.csv')
        dfs_training = [df_a, df_b, df_c, df_d]
        dfs_training = self.data_preprocessing(dfs_training, 100)
        return dfs_training

    def testing_data(self):
        df_test_a = pd.read_csv('data/AAPL_test.csv')
        df_test_b = pd.read_csv('data/MCD_test.csv')
        df_test_c = pd.read_csv('data/MMM_test.csv')
        df_test_d = pd.read_csv('data/XOM_test.csv')
        dfs_testing = [df_test_a, df_test_b, df_test_c, df_test_d]
        dfs_testing = self.data_preprocessing(dfs_testing, 253)
        return dfs_testing

    def test(self):
        return 2

