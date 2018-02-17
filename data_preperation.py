from copy import copy

import pandas as pd
import os
import pickle

from utils.financial_features import *


def get_data(data_period):
    file_path = os.path.join(home_path, 'sandp500', 'all_stocks_' + data_period + '.csv')
    all_stocks = pd.read_csv(file_path)
    all_stocks[TIME] = pd.to_datetime(all_stocks[TIME], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index(TIME, drop=False)
    return all_stocks

from Constants import *

# data modeling

def combine_df(data, name, cols_names, idx_name, idx_col, transformation):
    if transformation.__class__.__name__== 'PCA':
        cols_names_new = [ 'pc_' + str(i) for i in range(data.shape[1])]
        data_df = pd.DataFrame(data, columns=cols_names_new)
        data_df[idx_name] = idx_col
    else:
        cols_names_new = [s + name for s in cols_names]
        data_df = pd.DataFrame(data, columns=cols_names_new)
        data_df[idx_name] = idx_col
    data_df = data_df.set_index(TIME)
    return data_df


def preprocess_stock_features(stocks_df, stock_name, features_selection, finance_features, normalization,
                              transformation, y_col, to_fit=True, **kwargs):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # selected_numeric_cols = stocks_df[features].select_dtypes(include=numerics).columns.tolist()
    selected_numeric_cols = stocks_df.select_dtypes(include=numerics).columns.tolist()

    stock_X_raw = stocks_df[stocks_df[ENTITY] == stock_name]

    stock_X = stock_X_raw[selected_numeric_cols]  # TODO check id still index
    stock_X_prep = []
    stock_X_finance = pd.DataFrame()
    stock_X_finance[TIME] = stock_X_raw[TIME]
    stock_X_finance = stock_X_finance.set_index(TIME)
    # print stock_name
    if finance_features:
        stock_X_finance['Close_proc'] = stock_X_raw['Close'].pct_change()
        stock_X_finance['Close_proc'].iloc[0] = 0
        # stock_X_finance_df = combine_df(stock_X_finance.values, "_proc", stock_X.columns, TIME, stock_X.index)
        stock_X_finance['rsi'] = rsiFunc(stock_X_raw['Close'])
        stock_X_finance['MACD'] = computeMACD(stock_X_raw['Close'])[2]
        stock_X_finance['Open_Close_diff'] = stock_X_raw['Open'] - stock_X_raw['Close']
        stock_X_finance['High_Low_diff'] = stock_X_raw['High'] - stock_X_raw['Low']
        stock_X_prep.append(stock_X_finance)

    if to_fit:
        normalization = copy(normalization)
        normalization.fit(stock_X)
    stock_X_norm = normalization.transform(stock_X)
    stock_X_norm_df = combine_df(stock_X_norm, "_norm", stock_X.columns, TIME, stock_X.index,normalization)

    stock_X_prep.append(stock_X_norm_df)

    if len(stock_X_prep) > 1:
        stock_X_prep_df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), stock_X_prep)
    else:
        stock_X_prep_df = stock_X

    if transformation is not None:
        if to_fit:
            transformation = copy(transformation)
            transformation.fit(stock_X_prep_df)
        stock_X_transform = transformation.transform(stock_X_prep_df)
        stock_X_norm_df = combine_df(stock_X_transform, "_transform", stock_X_prep_df.columns, TIME, stock_X.index, transformation)
        stock_X_prep_df = pd.merge(stock_X_prep_df, stock_X_norm_df, left_index=True, right_index=True)

    # stock_X_raw_keep = stock_X_raw[[ENTITY, TARGET]]

    if y_col in stock_X_prep_df.columns:
        stock_X_prep_df[TARGET_PREP] = stock_X_prep_df[y_col]
    else:
        raise Exception('no y cols to evaluate')

    if TARGET not in stock_X_prep_df.columns:
        stock_X_prep_df[TARGET] = stock_X_raw[TARGET]

    stock_X_prep_df[ENTITY] = stock_X_raw[ENTITY]

    if isinstance(features_selection, tuple):
        features = [fe for fe in features_selection[1]]
    else:
        features = [fe for fe in features_selection]

    # features += [TARGET_PREP]

    features_names = stock_X_prep_df[features].columns

    return stock_X_prep_df, normalization, transformation, features_names


def calculate_features_all_stocks(path, features_selection, finance_features, force, normalization, prev_stocks_names,
                                  train_X, test_X, transformation, y_col):
    print "calc features all stocks"
    file_path = path + "_" + features_selection[0] + "_" + y_col + "_" + transformation.__class__.__name__
    features_names_file_path = file_path + '_features_names.pkl'
    train_data_file_path = file_path + '_train_stocks_processed.csv'
    test_data_file_path = file_path + '_test_stocks_processed.csv'
    if (not os.path.isfile(train_data_file_path)) or force:
        train_X_processed = []
        test_X_processed = []
        features_names = []
        for stock in prev_stocks_names:
            train_stock_X, normalization, transformation, features_names = preprocess_stock_features(
                train_X, stock, features_selection, finance_features, normalization, transformation, y_col)
            train_X_processed.append(train_stock_X)

            if test_X is None:
                test_X_processed = [pd.DataFrame()]
            else:
                test_stock_X, _, _, features_names = preprocess_stock_features(test_X, stock, features_selection,
                                                                               finance_features, normalization,
                                                                               transformation, y_col, to_fit=False)
                test_X_processed.append(test_stock_X)

        train_X_processed_df = pd.concat(train_X_processed)
        test_X_processed_df = pd.concat(test_X_processed)
        train_X_processed_df.to_csv(train_data_file_path)
        test_X_processed_df.to_csv(test_data_file_path)

        with open(features_names_file_path, 'wb') as f:
            pickle.dump(features_names.tolist(), f)
    with open(features_names_file_path, 'rb') as f:
        features_names = pickle.load(f)
    train_X_processed_df = pd.read_csv(train_data_file_path).set_index(TIME)
    test_X_processed_df = pd.read_csv(test_data_file_path)
    if test_X is not None:
        test_X_processed_df = test_X_processed_df.set_index(TIME)

    return train_X_processed_df, test_X_processed_df, features_names


def prepare_stock_windows(stock_X, features_names, window_len, slide, next_t, to_pivot=None, y_col=None):
    """
    prepare stock data for classifcation as follows: each window of series data is an instance, the instances are computed in sliding window manner,
    for each instance a next future targets values are computed
    :param stock_X: that data of stock
    :param window_len: the window size
    :param slide: the sliding movenet between windows calculation
    :param next_t: the future time targets
    :param column_to_window: columns names to be calculated and flatten per window
    :to_pivot: in case of flatten the data set for regular regression or classification
    :return: DataFrames: the dataset instances, the values to predict and the original prices to be evaluated by profit
    """

    i = 0
    y_column_names = next_t
    X_stocks_windows = []
    Y_ = []
    Y_price = []
    y = np.array(stock_X[TARGET_PREP].tolist())

    while i < len(stock_X[ENTITY]) - window_len - max(next_t):
        y_ti = i + window_len - 1 + np.asarray(next_t)
        stock_X_window = stock_X[i:i + window_len]
        stock_X_window.insert(0, 't', range(window_len))
        window_time = stock_X_window.index.values[-1]

        stock_X_window_flat = stock_X_window[features_names + [ENTITY] + ['t']].pivot(index=ENTITY,
                                                                                      columns='t')  # .iloc[0].to_dict()
        stock_X_window_flat = stock_X_window_flat.iloc[0].to_dict()
        stock_X_window_flat[TIME] = window_time
        X_stocks_windows.append(stock_X_window_flat)

        next_y = y[y_ti]
        y_vals = next_y.tolist()
        y_ = {}
        for c in range(len(y_column_names)):
            y_[str(y_column_names[c])] = y_vals[c]
        Y_.append(y_)

        # the current window last price
        y_price = {}
        y_price[TARGET] = np.array(stock_X[TARGET].tolist())[i + window_len].tolist()
        Y_price.append(y_price)

        i += slide
    windows_X = pd.DataFrame(X_stocks_windows).set_index(TIME)
    windows_y = pd.DataFrame(Y_, index=windows_X.index)
    windows_price = pd.DataFrame(Y_price, index=windows_X.index)[TARGET]
    windows_curr_target_prep = stock_X[TARGET_PREP].loc[windows_price.index]

    return windows_X, windows_y, windows_curr_target_prep, windows_price


def prepare_stock_y_points(stock_X, next_t, y_col=None):
    Y_ = []
    Y_price = []
    i = 0
    y = np.array(stock_X[TARGET_PREP].tolist())  # TODO chande to y_col for generic y
    while i < len(stock_X[ENTITY]) - max(next_t):
        #        point_time = stock_X.index.values[i]
        y_ti = i + np.asarray(next_t)
        next_y = y[y_ti]
        y_vals = next_y.tolist()
        y_ = {}
        for c in range(len(next_t)):
            y_[str(next_t[c])] = y_vals[c]
        Y_.append(y_)

        # the current window last price
        y_price = {}
        y_price[TARGET] = np.array(stock_X[TARGET].tolist())[i].tolist()
        Y_price.append(y_price)

        i += 1

    points_y = pd.DataFrame(Y_, index=stock_X.index.values[:len(stock_X[ENTITY]) - max(next_t)])
    price = pd.DataFrame(Y_price, index=stock_X.index.values[:len(stock_X[ENTITY]) - max(next_t)])[TARGET]
    points_curr_target_prep = stock_X[TARGET_PREP].loc[price.index]

    return points_y, price, points_curr_target_prep

# spliting data for evaluation

def prepare_train_test_points(stock_to_compare, train_X, test_X, next_t,
                              top_stock_w, weighted_sampleing, y_col):
    train_points_x = train_X[train_X[ENTITY] == stock_to_compare]

    train_points_y, _, train_points_curr_target_prep = prepare_stock_y_points(train_points_x, next_t, y_col)
    train_points_x = train_points_x.loc[train_points_y.index]

    if test_X is not None:
        test_points_x = test_X[test_X[ENTITY] == stock_to_compare]
        test_points_y, test_price, test_points_curr_target_prep = prepare_stock_y_points(test_points_x, next_t, y_col)
        test_points_x = test_points_x.loc[test_points_y.index]
    else:
        test_points_x, test_points_y, test_price, test_points_curr_target_prep = None, None, None, None

    # prepare windows per stock
    if top_stock_w is not None:
        for stock_name in top_stock_w.keys():
            train_points_x_i = train_X[train_X[ENTITY] == stock_name]
            if len(train_points_x_i) < len(train_X[train_X[ENTITY] == stock_to_compare]) * 2 / 3:
                print "stock-" + stock_name + " not in train data enogth - " + str(len(train_points_x_i))
                continue
            train_points_x_i = train_points_x_i.drop(train_points_x_i.index[:max(next_t) - 1])
            train_points_x = pd.concat([train_points_x, train_points_x_i], axis=1, join='inner')

            train_points_y = train_points_y.loc[train_points_x.index]
            train_points_curr_target_prep = train_points_curr_target_prep.loc[train_points_x.index]

            if test_X is not None:
                test_points_x_i = test_X[test_X[ENTITY] == stock_name]
                test_points_x_i = test_points_x_i.drop(test_points_x_i.index[:max(next_t) - 1])
                test_points_x = pd.concat([test_points_x, test_points_x_i], axis=1, join='inner')
                test_points_y = test_points_y.loc[test_points_x.index]
                test_points_curr_target_prep = test_points_curr_target_prep.loc[test_points_x.index]

    return train_points_x, train_points_y, \
           test_points_x, test_points_y, test_price, \
           train_points_curr_target_prep, \
           test_points_curr_target_prep


def prepare_train_test_windows(
        stock_to_compare, train_X, test_X, features_names, next_t, slide,
        to_pivot, top_stock_w, weighted_sampleing, window_len, y_col):
    train_windows_x, train_windows_y, train_windows_curr_target_prep, _ = prepare_stock_windows(
        train_X[train_X[ENTITY] == stock_to_compare],
        features_names, window_len, slide, next_t, to_pivot, y_col)

    if test_X is not None:
        test_windows_x, test_windows_y, test_windows_curr_target_prep, test_price = prepare_stock_windows(
            test_X[test_X[ENTITY] == stock_to_compare],
            features_names, window_len, slide, next_t, to_pivot, y_col)
    else:
        test_windows_x, test_windows_y, test_windows_curr_target_prep, test_price = None, None, None, None

    # prepare windows per stock
    if top_stock_w is not None:
        for stock_name in top_stock_w.keys():
            train_windows_x_i, train_windows_y_i, train_windows_curr_target_prep_i, _ = prepare_stock_windows(
                train_X[train_X[ENTITY] == stock_name],
                features_names, window_len, slide, next_t, to_pivot, y_col)

            if weighted_sampleing:
                np.random.seed(0)
                msk = np.random.rand(len(train_windows_x_i)) < top_stock_w[stock_name]
                train_windows_x_i = train_windows_x_i[msk]
                train_windows_y_i = train_windows_y_i[msk]
                train_windows_curr_target_prep_i = train_windows_curr_target_prep_i[msk]

            train_windows_x = pd.concat([train_windows_x, train_windows_x_i])
            train_windows_y = pd.concat([train_windows_y, train_windows_y_i])
            train_windows_curr_target_prep = pd.concat(
                [train_windows_curr_target_prep, train_windows_curr_target_prep_i])

    return train_windows_x, train_windows_y, \
           test_windows_x, test_windows_y, test_price, \
           train_windows_curr_target_prep, \
           test_windows_curr_target_prep


