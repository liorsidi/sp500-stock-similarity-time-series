

import  pandas as pd
import  numpy as np
import os
from dtw import dtw
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

home_path = 'C:\\Users\\Lior\\StockSimilarity'

TIME = 'Date'
ENTITY = 'Name'
TARGET = 'Close'


def fix_stock_len(stock1, stock2):
    diff_len = len(stock1) - len(stock2)
    if diff_len > 0:
        add_values = [stock2[0]] * diff_len
        stock2 = add_values + stock2
    elif diff_len < 0:
        add_values = [stock1[0]] * abs(diff_len)
        stock1 = add_values + stock1
    return stock1, stock2


def apply_dtw(stock1, stock2):
    stock1, stock2 = fix_stock_len(stock1, stock2)
    return dtw(stock1, stock2, dist=lambda x, y: abs(x - y))[0]


def get_similarity(df_stocks, stock_to_compare, stock_names, similarity_func, force = False):
    file_name = "similarities_" + stock_to_compare + "_" + similarity_func.__name__ + ".pkl"
    similarities_path = os.path.join(home_path, file_name)
    if (not os.path.isfile(similarities_path)) or force:
        similarities = [
            similarity_func(df_stocks[df_stocks[ENTITY] == stock_to_compare][TARGET].tolist(),
                          df_stocks[df_stocks[ENTITY] == stock_name][TARGET].tolist())
                for stock_name in stock_names
                ]
        with open(similarities_path, 'wb') as f:
            pickle.dump(similarities, f)
    with open(similarities_path, 'rb') as f:
        similarities = pickle.load(f)
    return similarities


def get_data(file_path):
    all_stocks = pd.read_csv(file_path)
    all_stocks[TIME] = pd.to_datetime(all_stocks[TIME])
    all_stocks = all_stocks.dropna(axis=0)
    return all_stocks


def get_top_k(df_stocks, stock_names, similarities, k):
    s = np.array(similarities)
    idx = np.argpartition(s, k)
    names_top_k = np.array(stock_names)[idx[:k]]
    df_top_k = df_stocks[df_stocks[ENTITY].isin(names_top_k)]
    return df_top_k, names_top_k


def prepare_stock_windows(X, stock_name, window, slide, next_t):
    stock_X = X[X[ENTITY] == stock_name]

    X_ = pd.DataFrame()
    Y_ = pd.DataFrame()
    i = 0

    #y_column_names = [TIME] + next_t
    while i < len(stock_X[TIME]) - window:
        X_.append(stock_X[i:i + window])
        y_ti = i + window + next_t
        curr_time = stock_X[-1][TIME]
        next_targets = np.array(stock_X[TARGET].tolist())[y_ti]
        y_vals = [curr_time] + next_targets
        y_ = pd.Series()
        Y_.append(stock_X[stock_X[TIME].isin(next_dates)])
        i += slide
    return X_, Y_


def prepare_rolling_folds(df_top_k,names_top_k, start_period_train, end_period_train, start_period_test, end_period_test, window, slide, next_t):
    train_X = df_top_k[(start_period_train <= df_top_k[TIME]) & (df_top_k[TIME] <= end_period_train)]
    test_X = df_top_k[(start_period_test <= df_top_k[TIME]) & (df_top_k[TIME] <= end_period_test)]
    train_windows_x, train_windows_y, test_windows_x, test_windows_y = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    #prepare windows per stock
    for stock_name in names_top_k:
        stock_train_windows_x, stock_train_windows_y = prepare_stock_windows(train_X, stock_name,window,slide,next_t)
        stock_test_windows_x, stock_test_windows_y = prepare_stock_windows(test_X, stock_name, window, slide, next_t)

        train_windows_x.append(stock_train_windows_x)
        train_windows_y.append(stock_train_windows_y)
        test_windows_x.append(stock_test_windows_x)
        test_windows_y.append(stock_test_windows_y)

    return train_windows_x, train_windows_y, test_windows_x, test_windows_y


def simple_profit_evaluation(curr_price, window, predicted_price):
    in_position = False
    profit = 0
    last_buy = 0
    for i in range(len(curr_price)-window):
        if curr_price[i] < predicted_price and not in_position:
            in_position = True
            last_buy = curr_price[i]
        if curr_price[i] > predicted_price and in_position:
            in_position = False
            profit = profit + (curr_price[i]-last_buy)
    return profit


def long_short_profit_evaluation(curr_price, window, predicted_price):
    is_long = None
    profit = 0
    last_buy = 0

    for i in range(len(curr_price)-window):
        #go long
        if curr_price[i] < predicted_price:
            # if short position - close it and go long
            if not is_long:
                profit += last_buy - curr_price[i]
                last_buy = curr_price[i]
                is_long = True
            #first time
            elif is_long is None:
                last_buy = curr_price[i]
                is_long = True

        #go short
        if curr_price[i] > predicted_price:
            # if long position - close it and go short
            if is_long:
                profit += curr_price[i] - last_buy
                last_buy = curr_price[i]
                is_long = False
            # first time
            elif is_long is None:
                last_buy = curr_price[i]
                is_long = False
    return profit


def evaluate_model(df_top_k, stock_to_compare, names_top_k, model,window, slide,next_t, folds, eval_funcs, profit_eval_funcs):

    stock_times = df_top_k[df_top_k[ENTITY] == stock_to_compare][TIME].tolist()
    period_len = abs(len(stock_times)/folds)
    for f in range(folds-1):
        start_period_train = stock_times[f*period_len]
        end_period_train = stock_times[f*period_len + period_len]
        start_period_test = stock_times[f*period_len + period_len + 1]
        if f == folds-1 :
            end_period_test = stock_times[-1]
        else:
            end_period_test = stock_times[f*period_len + period_len + 1 + period_len]

        train_windows_x, train_windows_y, test_windows_x, test_windows_y =\
            prepare_rolling_folds(df_top_k, names_top_k, start_period_train, end_period_train, start_period_test,
                      end_period_test, window, slide, next_t)


def main():
    file_path = os.path.join(home_path, 'sandp500\\all_stocks_1yr.csv')
    df_stocks = get_data(file_path)
    stock_to_compare = "GOOG"
    stock_names = df_stocks[ENTITY].unique()
    similarities = get_similarity(df_stocks,stock_to_compare,stock_names,apply_dtw,force = False)

    k=10
    df_top_k, names_top_k = get_top_k(df_stocks,stock_names,similarities,k)

    window = 10
    slide = 1
    next_t = np.asarray([1, 3, 7])
    folds = 5
    models = [RandomForestRegressor,GradientBoostingRegressor]
    evaluations = [evaluate_model(df_top_k, stock_to_compare, names_top_k, model, window, slide, next_t, folds, [mean_squared_error], [simple_profit_evaluation]) for model in models]

main()

# for i in range(len(stock_names)):
#     for j in range(i,len(stock_names)):
#         stock0 = all_stocks[all_stocks['Name'] == stock_names[i]][['Open', 'Date']]
#         stock1 = all_stocks[all_stocks['Name'] == stock_names[j]][['Open', 'Date']]
#         x = (stock0['Open'].tolist())
#         y = (stock1['Open'].tolist())
#         dist, cost, acc, path = dtw(x, x , dist=lambda x, y: abs(x - y))


#x = np.array(x)
# all_stocks = all_stocks[:252*25]
# stock_names = stock_names[:25]
#similarities = \
   # [
   #      [
   #          dtw(all_stocks[all_stocks['Name'] == stock_name_1][['Open', 'Date']]['Open'].tolist(),
   #              all_stocks[all_stocks['Name'] == stock_name_2][['Open', 'Date']]['Open'].tolist(),
   #              dist=lambda x, y: abs(x - y))
   #          for stock_name_2 in stock_names
   #          ]
   #      for stock_name_1 in stock_names
   #  ]

#min_len = all_stocks.groupby(['Name'],axis=0)['Name'].count()
#min_len = min_len[min_len == min_len[min_len.loc[stock]]].index.values.tolist()




