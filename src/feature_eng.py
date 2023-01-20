import pandas as pd
from bisect import bisect_left
from scipy.stats import mode
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from genetic_selection import GeneticSelectionCV
features = {
    "keep_features": ["bid_ask_qty_diff_diff_lag_5", "up_down_rolling_std_5", "spread_diff_rolling_mean_20", "spread_diff_rolling_mean_5s", "bid_price_rolling_std_1s", "bid_advance_time_rolling_mean_1s", "ask_qty_diff_rolling_max_10s", "ask_price_diff_rolling_std_3s", "ask_qty_rolling_std_10s", "bid_ask_qty_diff_rolling_std_20", "ask_advance_time_lag_2", "bid_ask_qty_total_rolling_max_10", "bid_ask_qty_diff_rolling_sum_5", "bid_qty_rolling_min_5", "bid_ask_qty_diff_diff_rolling_sum_3s", "sum_trade_1s_rolling_std_1s", "spread_rolling_mean_1s", "trade_price_diff_rolling_sum_10", "ask_qty_diff_rolling_sum_10s", "ask_price_diff_rolling_mean_5s", "sum_trade_1s_diff_rolling_sum_20", "bid_price_lag_5", "sum_trade_1s_rolling_mean_5", "bid_ask_qty_diff_rolling_min_5", "bid_ask_qty_diff_diff_rolling_std_3s", "bid_ask_qty_total_rolling_min_5", "bid_advance_time_diff_lag_2", "trade_price_compare", "bid_ask_qty_diff_diff_rolling_mean_20", "trade_price_diff_rolling_sum_3s", "bid_ask_qty_diff_rolling_sum_1s", "bid_qty", "ask_advance_time_rolling_mean_5s", "spread_diff_rolling_std_1s", "trade_price_compare_diff_rolling_std_1s", "bid_ask_qty_diff", "ask_qty_lag_1", "ask_qty_diff_rolling_sum_1s", "trade_price_compare_diff_rolling_sum_5", "spread", "bid_qty_lag_1", "bid_ask_qty_diff_rolling_mean_10", "bid_qty_lag_2", "bid_price_lag_3", "ask_qty_rolling_min_3s", "ask_advance_time_lag_4", "spread_diff_rolling_std_3s",
 "bid_qty_rolling_max_20", "ask_qty_lag_3", "bid_qty_diff_lag_5", 
 "bid_price_diff_rolling_sum_5s", "trade_price_compare_diff_lag_4", 
 "bid_price_diff_lag_4", "bid_qty_diff_rolling_sum_1s", 
 "bid_ask_qty_diff_diff_rolling_max_1s", "bid_advance_time_rolling_mean_3s",
  "ask_advance_time_diff_lag_1", "ask_qty_rolling_min_5", "spread_rolling_std_3s",
   "bid_advance_time_rolling_std_20", "ask_qty_diff_rolling_min_20", 
   "sum_trade_1s_rolling_mean_10", "spread_diff_rolling_std_20", "ask_qty_rolling_mean_5", 
   "bid_qty_rolling_min_10", "trade_price_compare_diff_lag_5", "bid_price_rolling_std_5", 
   "trade_price_rolling_mean_10", "sum_trade_1s_diff_rolling_std_10",
    "bid_advance_time_diff_rolling_sum_5s", "ask_qty_lag_2", "trade_price_pos_diff_rolling_std_10s", 
    "ask_advance_time_diff_rolling_mean_5", "ask_qty_rolling_min_10", "sum_trade_1s_diff_lag_5", 
    "last_trade_time_diff_lag_4", "bid_qty_diff_rolling_std_5", "bid_price_diff_lag_3", 
    "ask_advance_time_lag_3", "ask_qty_rolling_mean_20", "ask_qty_diff_rolling_mean_5",
     "bid_ask_qty_diff_diff_rolling_sum_10s", "bid_advance_time_rolling_mean_5s", "sum_trade_1s_lag_1", 
     "bid_qty_rolling_min_3s", "bid_qty_rolling_max_5s", "sum_trade_1s_diff_lag_2", 
     "bid_ask_qty_total_rolling_max_10s", "bid_qty_rolling_mean_10", "bid_advance_time_lag_1",
      "bid_ask_qty_diff_lag_1", "bid_ask_qty_diff_diff_rolling_min_1s", "bid_qty_diff_rolling_std_10s",
       "bid_price_rolling_std_5s", "ask_qty_diff_rolling_std_5s", "bid_qty_diff_rolling_max_10",
        "last_trade_time", "ask_qty_diff_rolling_mean_1s", "trade_price_pos_diff_rolling_mean_3s", 
        "bid_ask_qty_total_diff_rolling_max_3s", "ask_qty_diff_rolling_sum_3s", "last_trade_time_diff_rolling_mean_5s", 
        "bid_ask_qty_total_diff_rolling_max_10", "bid_qty_rolling_mean_5", "ask_qty", 
        "bid_ask_qty_diff_diff_rolling_mean_5s", "bid_ask_qty_total_diff_rolling_sum_5",
         "bid_qty_rolling_min_20", "last_trade_time_diff_rolling_sum_5", "bid_price_rolling_mean_10s", 
         "ask_advance_time_diff_rolling_mean_1s", "sum_trade_1s_diff"], "correlation_remove": ["ask_price"]}



class feature_eng:
    timestamp = None
    max_lag = 5
    num_window = [5, 10, 20]
    sec_window = [1, 3, 5, 10]
    rolling_sum_cols = []
    rolling_mean_cols = []
    rolling_max_cols = []
    rolling_min_cols = []
    rolling_std_cols = []

    @staticmethod
    def bid_ask_spread(data):
        data['spread'] = data['ask_price'] - data['bid_price']

    @staticmethod
    def bid_ask_qty_comb(data):
        data['bid_ask_qty_total'] = data['ask_qty'] + data['bid_qty']
        data['bid_ask_qty_diff'] = data['ask_qty'] - data['bid_qty']

    @staticmethod
    def trade_price_feature(data):
        data['trade_price_compare'] = 0  # when trade price between current bid and ask price
        data.loc[data['trade_price'] <= data[
            'bid_price'], 'trade_price_compare'] = -1  # when trade price on current bid side
        data.loc[data['trade_price'] >= data[
            'ask_price'], 'trade_price_compare'] = 1  # when trade price on current sell side

        # whether trade price happens on bid side or ask side during the time it happens
        last_trade_timestamp = data['timestamp'] - pd.to_timedelta(data['last_trade_time'], unit='s')
        idx_list = [bisect_left(data['timestamp'], i) for i in list(last_trade_timestamp)]
        trade_price_pos = []
        for i, index in enumerate(idx_list):
            index1 = index
            index2 = index1 + 1 if index1 < data.shape[0] - 1 else index1
            bid1 = data['bid_price'][index1]
            bid2 = data['bid_price'][index2]
            ask1 = data['ask_price'][index1]
            ask2 = data['ask_price'][index2]
            trade_price = data['trade_price'][i]
            if (bid1 <= trade_price <= bid2) or (bid2 <= trade_price <= bid1):
                trade_price_pos.append(-1)  # happen on bid side
            elif (ask1 <= trade_price <= ask2) or (ask2 <= trade_price <= ask1):
                trade_price_pos.append(1)  # happen on sell side
            else:
                trade_price_pos.append(0)  # unknown case
        data['trade_price_pos'] = trade_price_pos

    @staticmethod
    def diff_feature(data):
        for i in set(data.columns) - {'timestamp'}:
            new_name = '{}_diff'.format(i)
            data[new_name] = data[i] - data[i].shift(1)

    @staticmethod
    def up_or_down(data):
        data['up_down'] = 0
        data.loc[data['bid_price_diff'] < 0, 'up_down'] = -1
        data.loc[data['ask_price_diff'] > 0, 'up_down'] = 1

    @staticmethod
    def lag_feature(data, col, lag):
        new_col_name = '{}_lag_{}'.format(col, lag)
        data[new_col_name] = data[col].shift(lag)

    @staticmethod
    def rolling_feature(data, col, window, feature):
        rolling = data[col].rolling(window=window)
        new_col = '{}_rolling_{}_{}'.format(col, feature, window)

        if feature == 'sum':
            data[new_col] = rolling.sum()
        elif feature == 'mean':
            data[new_col] = rolling.mean()
        elif feature == 'max':
            data[new_col] = rolling.max()
        elif feature == 'min':
            data[new_col] = rolling.min()
        elif feature == 'std':
            data[new_col] = rolling.std()
        elif feature == 'mode':
            data[new_col] = rolling.apply(lambda x: mode(x)[0])

    @classmethod
    def basic_features(cls, data):
        data = data.copy()
        cls.timestamp = data['timestamp']

        cls.bid_ask_spread(data)
        cls.bid_ask_qty_comb(data)
        cls.trade_price_feature(data)
        cls.diff_feature(data)
        cls.up_or_down(data)

        data = data.drop('timestamp', axis=1)
        return data

    @classmethod
    def lag_rolling_features(cls, data):
        data = data.copy()

        # get lag and rolling feature based on previous n records
        rolling_cols = set(data.columns) - {'trade_price_compare', 'trade_price_pos'}
        cls.rolling_sum_cols = [i for i in rolling_cols if 'diff' in i or 'up_down' in i]
        cls.rolling_mean_cols = rolling_cols
        cls.rolling_max_cols = [i for i in rolling_cols if 'bid_qty' in i or 'ask_qty' in i]
        cls.rolling_min_cols = [i for i in rolling_cols if 'bid_qty' in i or 'ask_qty' in i]
        cls.rolling_std_cols = rolling_cols

        for col in rolling_cols:
            for lag in range(1, cls.max_lag + 1):
                cls.lag_feature(data, col, lag)

        for col in rolling_cols:
            for num_window in cls.num_window:
                if col in cls.rolling_sum_cols:
                    cls.rolling_feature(data, col, num_window, 'sum')
                if col in cls.rolling_mean_cols:
                    cls.rolling_feature(data, col, num_window, 'mean')
                if col in cls.rolling_max_cols:
                    cls.rolling_feature(data, col, num_window, 'max')
                if col in cls.rolling_min_cols:
                    cls.rolling_feature(data, col, num_window, 'min')
                if col in cls.rolling_std_cols:
                    cls.rolling_feature(data, col, num_window, 'std')

        # get rolling feature based on previous n seconds
        data.index = cls.timestamp
        for col in rolling_cols:
            for sec_window in cls.sec_window:
                sec_window = '{}s'.format(sec_window)
                if col in cls.rolling_sum_cols:
                    cls.rolling_feature(data, col, sec_window, 'sum')
                if col in cls.rolling_mean_cols:
                    cls.rolling_feature(data, col, sec_window, 'mean')
                if col in cls.rolling_max_cols:
                    cls.rolling_feature(data, col, sec_window, 'max')
                if col in cls.rolling_min_cols:
                    cls.rolling_feature(data, col, sec_window, 'min')
                if col in cls.rolling_std_cols:
                    cls.rolling_feature(data, col, sec_window, 'std')
                if col in ['up_down', 'trade_price_compare', 'trade_price_pos']:
                    cls.rolling_feature(data, col, sec_window, 'mode')

        return data

    @staticmethod
    def remove_na(x, y):
        x = x.reset_index(drop=True)
        x = x.dropna()
        y = y.loc[x.index, :].reset_index(drop=True)
        x = x.reset_index(drop=True)
        return x, y


class feature_selection:
    '''feature selection combining feature importance ranking and GA optimization based on random forest model'''

    @classmethod
    def select(cls, x, y):
        rf_imp_features = cls.rf_imp_features(x, y)
        ga_features = cls.GA_features(x, y)
        features = set(rf_imp_features) | set(ga_features)

        return list(features)

    @classmethod
    def rf_imp_features(cls, x, y, top_perc=0.05):
        '''select top features based on feature importance ranking among all the features'''
        feature_imp = cls.rf_importance_selection(x, y)
        perc_threshold = np.percentile(feature_imp['avg_importance'], int((1 - top_perc) * 100))
        features = list(feature_imp.loc[feature_imp['avg_importance'] >= perc_threshold, 'feature'])

        return features

    @staticmethod
    def rf_importance_selection(x, y, iter_time=3):
        feature_imp = pd.DataFrame(np.zeros((x.shape[1], iter_time + 2)))
        feature_imp.columns = ['feature'] + ['importance_{}'.format(i) for i in range(1, iter_time + 1)] + [
            'avg_importance']
        for col in feature_imp.columns:
            feature_imp[col] = list(x.columns)

        for i in range(1, iter_time + 1):
            col = 'importance_{}'.format(i)
            rf = RandomForestClassifier(n_estimators=10, max_depth=8)
            rf.fit(x, y)
            feature_imp_dict = dict(zip(x.columns, rf.feature_importances_))
            feature_imp[col] = feature_imp[col].replace(feature_imp_dict)

        feature_imp['avg_importance'] = feature_imp.iloc[:, 1:-1].mean(axis=1)
        return feature_imp

    @staticmethod
    def GA_features(x, y):
        rf = RandomForestClassifier(max_depth=8, n_estimators=10)
        selector = GeneticSelectionCV(
            rf,
            cv=TimeSeriesSplit(n_splits=4),
            verbose=1,
            scoring="accuracy",
            max_features=80,
            n_population=200,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=100,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=5,
            caching=True,
            n_jobs=-1
        )
        selector = selector.fit(x, y)
        features = x.columns[selector.support_]

        return features