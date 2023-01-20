import pandas as pd
import numpy as np
def preprocessing(data):
    '''align data type and time order'''
    float_list = [
        'bid_price',
        'bid_qty',
        'ask_price',
        'ask_qty',
        'trade_price',
        'sum_trade_1s',
        'bid_advance_time',
        'ask_advance_time',
        'last_trade_time',
    ]

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    for i in float_list:
        data[i] = data[i].astype(float)

    data = data.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    return data


def check_null(data):
    '''check null values in dataframe'''
    data = data.copy()
    have_null_cols = list(data.columns[data.isnull().any()])
    print('Columns with null values are {}'.format(', '.join(have_null_cols)))
    for i in have_null_cols:
        print('number of rows that column {} is null: {}'.format(i, data[i].isnull().sum()))
        print('null percentage is {}'.format(round(data[i].isnull().sum() / data.shape[0], 2)))

    stat1 = data['sum_trade_1s'][data['last_trade_time'].isnull()].notnull().sum()
    stat2 = data['last_trade_time'][data['sum_trade_1s'].isnull()].notnull().sum()
    stat3 = data['sum_trade_1s'][data['last_trade_time'] >= 1].isnull().sum()
    stat4 = stat3 / data['sum_trade_1s'].isnull().sum()
    print('number of rows sum_trade_1s is not null when last_trade_time is not: {}'.format(stat1))
    print('number of rows last_trade_time is null when sum_trade_1s is not: {}'.format(stat2))
    print('number of rows sum_trade_1s null at last_trade_time > 1: {}, percentage: {}'.format(stat3, round(stat4, 2)))


def fill_null(data):
    '''
    based on the null check and basic logic, most of the sum_trade_1s null value happens when last_trade_time larger
    than 1 sec (in this case sum_trade_1s should be 0). Therefore, we make an assumption that all the sum_trade_1s null
    value could be filled with 0. Based on such assumption, last_trade_time can be filled with last_trade_time of the
    previous record plus a time movement if record interval is smaller than 1 sec.
    '''

    class last_trade_time_filler:
        prev_last_trade_time = None
        prev_timestamp = None

        @classmethod
        def fill(cls, index):
            last_trade_time = data.loc[index, 'last_trade_time']
            timestamp = data.loc[index, 'timestamp']

            if pd.isnull(last_trade_time):
                time_interval = (timestamp - cls.prev_timestamp).microseconds / (1e+6)
                if time_interval <= 1:
                    last_trade_time = cls.prev_last_trade_time + time_interval
                else:
                    last_trade_time = np.nan

            cls.prev_last_trade_time = last_trade_time
            cls.prev_timestamp = timestamp

            return last_trade_time

    data = data.copy()
    data.loc[data['sum_trade_1s'].isnull(), 'sum_trade_1s'] = 0
    data['last_trade_time'] = data.index.map(last_trade_time_filler.fill)
    print('number of null columns is: {} now'.format(len(list(data.columns[data.isnull().any()]))))

    return data


def x_y_split(data):
    label_cols = ['_1s_side', '_3s_side', '_5s_side']
    feature_cols = list(set(data.columns) - set(label_cols))
    y = data[label_cols].copy()
    x = data[feature_cols].copy()

    return x, y


class correlation_filter:
    remove_cols = []

    @classmethod
    def filter(cls, x, threshold=0.99):
        x = x.copy()
        index2col = {i: col for i, col in enumerate(x.columns)}
        corr = np.array(x.corr())
        correlated_pairs = list(zip(*np.where(np.abs(corr) >= threshold)))
        to_be_delete = []
        for i, j in correlated_pairs:
            former = index2col[i]
            latter = index2col[j]
            if former != latter:
                add = True
                for i, del_set in enumerate(to_be_delete):
                    has_intersect = ({former, latter} & del_set) != {}
                    if has_intersect:
                        add = False
                        to_be_delete[i] = del_set | {former, latter}
                if add:
                    to_be_delete.append({former, latter})

        for i in to_be_delete:
            delete_set = i.copy()
            delete_set.pop()
            x = x.drop(list(delete_set), axis=1)
            cls.remove_cols += list(delete_set)

        return x

