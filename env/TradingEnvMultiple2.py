# Daniel Gonzalez Cortes - 2022
import gym
from gym import spaces
import numpy as np
import pandas as pd

np.random.seed(2)

INITIAL_BALANCE = 1000000
COMMISSION_RATE = 0.005  # 0.5% - 0.005

MAX_STEPS = 1000

class TradingEnviroment(gym.Env):
    transaction_n: int

    def __init__(self, data_sets, shares_name, n_assets):
        super(TradingEnviroment, self).__init__()
        self.shares_name = shares_name
        self.N_ASSETS = n_assets
        self.type_action = ['BUY', 'SELL']

        self.data_sets = data_sets

        self.data_sets_norma = [self.norma(sets.copy()) for sets in self.data_sets]

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Cambiar la estructura de los inputs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N_ASSETS, 10, 5), dtype=np.float16)

        self.DF_results = pd.DataFrame(columns=["Episode", "Date", "Prices", "Weights", "n Shares held", "Value",
                                                "n Shares bought", "n Shares sold", "Portfolio",
                                                "Cash", "Commissions", "Balance"])
        self.DF_transactions = pd.DataFrame(
            columns=["Episode", "Number", "Date", "Initial Cash", "Type", "Instrument", "Price", "Amount", "Value",
                     "Commission", "Cash"])
        self.initial_shares = 0
        self.initial_cash_rem = INITIAL_BALANCE
        self.episode_count_variable = 0
        self.total_commissions = 0

        self.cash2 = 0

    def norma(self, data):
        min_max = lambda col: (col - col.min()) / (col.max() - col.min())
        for column in data.columns:
            if column != "Date":
                data[column] = min_max(data[column].values)
        return data

    def add_episode(self):
        self.episode_count_variable += 1

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_BALANCE
        # self.net_worth = INITIAL_BALANCE
        self.pl = 0
        self.n_shares = 0
        self.portfolio = 0
        self.pl_p = 0
        self.total_commissions = 0

        self.cash = INITIAL_BALANCE
        self.step_t = np.random.randint(4, len(self.data_sets[0].loc[:, 'Open'].values))
        self.step_i = 0
        self.shares_held = []
        self.positions = []
        self.transaction_n = 0
        self.cash2 = 0
        return self.observation()

    def register_trade(self, date_t, prices_t, weights_t, shares_held, values_positions, n_shares_bought,
                       n_shares_sold, commissions):

        # append row to the dataframe
        self.DF_results = self.DF_results.append({"Episode": self.episode_count_variable,
                                                  "Date": date_t,
                                                  "Prices": prices_t,
                                                  "Weights": weights_t,
                                                  "n Shares held": shares_held,
                                                  "Value": values_positions,
                                                  "n Shares bought": n_shares_bought,
                                                  "n Shares sold": n_shares_sold,
                                                  "Portfolio": self.portfolio,
                                                  "Cash": self.cash,
                                                  "Commissions": commissions,
                                                  "Balance": self.balance},
                                                 ignore_index=True)

    def register_transaction_initial(self, date_t, cash_i, price_t, amount_t):
        cash_int, cash_f = cash_i, cash_i
        for i in range(0, self.N_ASSETS):
            self.transaction_n += 1
            cash_int = cash_f
            value_t = price_t[i] * amount_t[i]
            commission_t = value_t * COMMISSION_RATE
            cash_f -= (value_t + commission_t)
            self.DF_transactions = self.DF_transactions.append({"Episode": self.episode_count_variable,
                                                                "Number": self.transaction_n, "Date": date_t,
                                                                "Initial Cash": cash_int, "Type": self.type_action[0],
                                                                "Instrument": self.shares_name[i],
                                                                "Price": price_t[i], "Amount": amount_t[i],
                                                                "Value": value_t, "Commission": commission_t,
                                                                "Cash": cash_f}, ignore_index=True)

    def register_transaction(self, date_t, i, n_price, n_cash, n_size, n_action):

        self.transaction_n += 1
        value_t = n_price * n_size
        commission_t = value_t * COMMISSION_RATE
        cash_f = n_cash

        if n_action == 1:
            cash_f += (value_t - commission_t)

        else:
            cash_f -= (value_t + commission_t)

        self.DF_transactions = self.DF_transactions.append({"Episode": self.episode_count_variable,
                                                            "Number": self.transaction_n, "Date": date_t,
                                                            "Initial Cash": n_cash, "Type": self.type_action[n_action],
                                                            "Instrument": self.shares_name[i],
                                                            "Price": n_price, "Amount": n_size,
                                                            "Value": value_t, "Commission": commission_t,
                                                            "Cash": cash_f}, ignore_index=True)

    def get_results_csv(self):
        self.DF_results.to_csv('result.csv', index=False)

    def get_transactions_csv(self):
        self.DF_transactions.to_csv('transactions.csv', index=False)

    def observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        # Esta data debe ser normalizada
        lagged = 5
        obs = []

        for i in range(self.N_ASSETS):
            obs_t = np.array([
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'Open'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'High'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'Low'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'Close'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'Volume'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'MACD'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'RSI'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'MA_14'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'MA_21'].values,
                self.data_sets_norma[i].loc[self.step_t - lagged: self.step_t - 1, 'MA_100'].values,
            ])
            obs.append(obs_t)
        obs = np.array(obs)
        return obs

    def get_prices(self, data_sets):
        price_t = []
        for i in range(len(self.data_sets)):
            price_t.append(
                np.random.uniform(data_sets[i].loc[self.step_t, "Open"], data_sets[i].loc[self.step_t, "Close"]))
        return np.array(price_t)

    def initial_buy(self, n_weights, initial_balance, initial_prices):
        if len(n_weights) == len(initial_prices):
            ##initial_n_shares = np.array(initial_balance * n_weights) / initial_prices
            initial_n_shares = np.array(initial_balance * n_weights) / (initial_prices * (1 + COMMISSION_RATE))
            initial_n_shares = initial_n_shares.astype(int)
            initial_positions = initial_n_shares * initial_prices
            initial_commissions = sum(initial_positions * COMMISSION_RATE)
            initial_cash = initial_balance - sum(initial_positions) - initial_commissions
            return initial_n_shares, initial_positions, initial_cash, initial_commissions
        else:
            print("Not the same dimensions")

    def rebalancing(self, n_shares, n_weights, n_prices,
                    remaining_cash):  # n_shares, n_weights, n_prices, remaining_cash
        # print(self.step_i, action)
        # What do we have?
        n = len(n_shares)
        # positions
        value_positions = np.array(n_shares * n_prices)
        total_value_portfolio = sum(value_positions) + remaining_cash
        # calculate the difference needed
        diff_positions = np.array(value_positions - (total_value_portfolio * n_weights))
        st = np.argsort(diff_positions)[::-1]
        # how many stocks?
        diff_n_stocks = np.array(np.abs(diff_positions) / n_prices).astype(int)
        date_t = self.data_sets[0].loc[self.step_t, "Date"]
        for i in st:
            if diff_positions[i] > 0:
                n_shares[i] -= diff_n_stocks[i]
                value_position = diff_n_stocks[i] * n_prices[i]
                commission = value_position * COMMISSION_RATE
                remaining_cash += value_position
                remaining_cash -= commission
                #self.register_transaction(date_t, i, n_prices[i], self.cash2.copy(), diff_n_stocks[i], 1)
                self.cash2 = remaining_cash.copy()
            else:
                if remaining_cash > diff_n_stocks[i] * n_prices[i]:  # Buy if I have money
                    n_shares[i] += diff_n_stocks[i]
                    value_position = diff_n_stocks[i] * n_prices[i]
                    commission = value_position * COMMISSION_RATE
                    remaining_cash -= value_position
                    remaining_cash -= commission
                    #self.register_transaction(date_t, i, n_prices[i], self.cash2, diff_n_stocks[i], 0)
                    self.cash2 = remaining_cash.copy()
                else:
                    diff_n_stocks_new = int(remaining_cash / n_prices[i])
                    n_shares[i] += diff_n_stocks_new
                    value_position = diff_n_stocks_new * n_prices[i]
                    commission = value_position * COMMISSION_RATE
                    remaining_cash -= value_position
                    remaining_cash -= commission
                    #self.register_transaction(date_t, i, n_prices[i], self.cash2, diff_n_stocks_new, 0)
                    self.cash2 = remaining_cash.copy()

        value_positions = np.array(n_shares * n_prices)
        return n_shares, value_positions, remaining_cash.copy()

    def trade(self, action):
        prices_t = self.get_prices(self.data_sets)
        n_shares_bought, n_shares_sold = np.zeros(self.N_ASSETS), np.zeros(self.N_ASSETS)
        commissions = 0
        date_t = self.data_sets[0].loc[self.step_t, "Date"]
        if self.step_i == 0:
            self.shares_held, self.positions, self.cash, commissions = self.initial_buy(action, self.cash, prices_t)
            self.portfolio = sum(self.shares_held * prices_t)
            self.balance = self.portfolio + self.cash
            #self.register_transaction_initial(date_t, INITIAL_BALANCE, prices_t, self.shares_held.copy())
            self.cash2 = self.cash.copy()

        else:
            initial_shares = self.shares_held.copy()
            self.shares_held, self.positions, self.cash = self.rebalancing(self.shares_held, action, prices_t,
                                                                           self.cash)

            n_shares_bought = np.array(self.shares_held - initial_shares).astype(int)
            n_shares_sold = np.array(initial_shares - self.shares_held).astype(int)

            n_shares_bought[n_shares_bought < 0] = 0
            n_shares_sold[n_shares_sold < 0] = 0

            commissions_bought = n_shares_bought * prices_t * COMMISSION_RATE
            commissions_sold = n_shares_sold * prices_t * COMMISSION_RATE
            commissions += round((sum(commissions_bought) + sum(commissions_sold)), 2)

            self.portfolio = sum(self.shares_held * prices_t)

            self.balance = self.portfolio + self.cash
            self.total_commissions += commissions

        #self.register_trade(date_t, prices_t, action, self.shares_held.copy(), self.shares_held * prices_t,
                            #n_shares_bought, n_shares_sold, commissions)

    def step(self, action):
        self.trade(action)
        self.step_t += 1
        self.step_i += 1


        delay_modifier = (self.step_t / MAX_STEPS)
        # print("self.step_t" ,self.step_t ,"delay modifier ",delay_modifier)

        # reward = self.balance * delay_modifier
        # reward = self.balance
        reward = self.balance * delay_modifier
        obs = self.observation()

        if self.step_t > len(self.data_sets[0].loc[:, 'Open'].values) - 5:
            done = True
        #elif self.balance <= INITIAL_BALANCE * 0.8:
            #done = True
        else:
            done = False

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        # print(f'Step: {self.current_step}')
        print(self.data_set_norma)

##################################
##################################
