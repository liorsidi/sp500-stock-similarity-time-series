import random
import threading

from datetime import datetime
import pandas as pd
import numpy as np
import os
from dtw import dtw
import pickle

from scipy.stats import rankdata
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error

import itertools

from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from ANN_stock import ANN_stock
from Financial_FE import rsiFunc, computeMACD

# home_path = 'C:\\Users\\Lior\\StockSimilarity'
home_path = '/home/ise/Desktop/StockSimilarity'

TIME = 'Date'
ENTITY = 'Name'
TARGET = 'Close'
FEATURES = ['Close']
TARGET_PREP = 'traget_prep'

from SAX_FILE import SAX

sax_obj = SAX()


def get_data(data_period):
    file_path = os.path.join(home_path, 'sandp500', 'all_stocks_' + data_period + '.csv')
    all_stocks = pd.read_csv(file_path)
    all_stocks[TIME] = pd.to_datetime(all_stocks[TIME], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index(TIME, drop=False)
    return all_stocks


##################  Similarities calculation ######################
def fix_stock_len(stock1, stock2):
    """
    fix 2 stoack to be in same leangth, by multiplying the first value of the shorter stock
    :param stock1:
    :param stock2:
    :return:
    """
    diff_len = len(stock1) - len(stock2)
    if diff_len > 0:
        add_values = pd.DataFrame([stock2.iloc[0]] * diff_len)
        stock2 = pd.concat([add_values, stock2])
    elif diff_len < 0:
        add_values = pd.DataFrame([stock1.iloc[0]] * abs(diff_len))
        stock1 = pd.concat([add_values, stock1])
    return stock1, stock2


def correlate_stock_len(stock1, stock2):
    """
    fix 2 stoack to be in same leangth, by multiplying the first value of the shorter stock
    :param stock1:
    :param stock2:
    :return:
    """
    stock2_ = stock2.copy()
    stock1_ = stock1.copy()

    stock2_[TIME] = stock2_.index
    stock1_[TIME] = stock1_.index

    stock2_ = stock2_[stock2_[TIME].isin(stock1_[TIME])]
    stock1_ = stock1_[stock1_[TIME].isin(stock2_[TIME])]

    del stock2_[TIME]
    del stock1_[TIME]
    return stock1_, stock2_


from fastpip import fastpip


def pip_fix(stock1, stock2, factor=10, similarity_col=TARGET):
    stock1, stock2 = correlate_stock_len(stock1, stock2)
    min_len = min(*[len(stock1[similarity_col]), len(stock2[similarity_col])])
    stock1_pairs = [(t, p) for t, p in zip(range(len(stock1[similarity_col])), stock1[similarity_col])]
    stock2_pairs = [(t, p) for t, p in zip(range(len(stock2[similarity_col])), stock2[similarity_col])]
    stock1_pairs_pip = fastpip(stock1_pairs, min_len / factor)
    stock2_pairs_pip = fastpip(stock2_pairs, min_len / factor)

    locs1 = [i[0] for i in stock1_pairs_pip]
    locs2 = [i[0] for i in stock2_pairs_pip]

    stock1_index = stock1.index[locs1]
    stock2_index = stock2.index[locs2]
    indexes = stock1_index.union(stock2_index)
    return stock1.loc[indexes], stock2.loc[indexes]


def correlate_stock_len_delay(stock1, stock2, delay=1):
    """
    fix 2 stoack to be in same leangth, by multiplying the first value of the shorter stock
    :param stock1:
    :param stock2:
    :return:
    """
    stock2_ = stock2.copy()
    stock1_ = stock1.copy()

    stock2_[TIME] = stock2_.index
    stock1_[TIME] = stock1_.index

    stock2_[TIME] = stock2_[TIME].shift(-1)
    stock2_ = stock2_[stock2_[TIME].isin(stock1_[TIME])]
    stock1_ = stock1_[stock1_[TIME].isin(stock2_[TIME])]

    del stock2_[TIME]
    del stock1_[TIME]

    return stock1_, stock2_


def apply_dtw(stock1, stock2, fix_len_func=correlate_stock_len, similarity_col=TARGET):
    """
    apply DTW distance between 2 stocks
    :param stock1:
    :param stock2:
    :return:
    """
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    return dtw(stock1[similarity_col].tolist(), stock2[similarity_col].tolist(), dist=lambda x, y: abs(x - y))[0]


def apply_pearson(stock1, stock2, fix_len_func=correlate_stock_len, similarity_col=TARGET):
    """
    apply pearson distance between 2 stocks
    :param stock1:
    :param stock2:
    :return:
    """
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    pearson = np.corrcoef(np.array(stock1[similarity_col].tolist()), np.array(stock2[similarity_col].tolist()))[0, 1]
    return abs(pearson - 1)


def apply_euclidean(stock1, stock2, fix_len_func=correlate_stock_len, similarity_col=TARGET):
    """
    apply euclidean distance between 2 stocks
    :param stock1:
    :param stock2:
    :return:
    """

    stock1, stock2 = fix_len_func(stock1, stock2)
    return np.linalg.norm(np.array(stock1[similarity_col].tolist()) - np.array(stock2[similarity_col].tolist()))


def compare_sax(stock1, stock2, fix_len_func=correlate_stock_len, similarity_col=TARGET):
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    sax_obj_ = SAX(wordSize=np.math.ceil(len(stock1)), alphabetSize=12)
    stock1_s = sax_obj_.transform(stock1[similarity_col].tolist())
    stock2_s = sax_obj_.transform(stock2[similarity_col].tolist())
    return sax_obj_.compare_strings(stock1_s, stock2_s)


def ensemble_dist(experiment_path,stock_to_compare, similarity_funcs,similarity_cols,fix_len_funcs,
                  train_X_all_prev_periods_processed, prev_stocks_names,end_period_train,end_period_test, force = False):
    for similarity_func in similarity_funcs:
        for similarity_col in similarity_cols:
            for fix_len_func in fix_len_funcs:
                similarity_path = os.path.join(experiment_path, stock_to_compare, 'similarity',
                                               'func-' + similarity_func + '_col-' + similarity_col + fix_len_func, 'fold-')
                file_name = "train_" + str(str(end_period_train.strftime("%Y-%m-%d"))) + "test_" + str(
                    str(end_period_test.strftime("%Y-%m-%d")))

                similarity_file_path =  os.path.join(similarity_path,file_name + ".pkl")
                calculate_similarity_all_stocks(train_X_all_prev_periods_processed, stock_to_compare,
                                            prev_stocks_names,
                                            similarity_func, similarity_file_path, fix_len_func, similarity_col,
                                            split_time=str(end_period_train), force=force)


class model_bases_distance(object):
    def __init__(self, model):
        self.model = model
        self.selected_numeric_cols = [u'Close_proc',
                                      u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff', u'Open_norm',
                                      u'High_norm', u'Low_norm', u'traget_prep', u'Volume_norm']
        self.selected_numeric_cols = [u'Close_proc', u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff']
        self.window_len = 5

    def fit(self, df, stock_name, similarity_col):
        self.similarity_col = similarity_col
        if self.model.__class__.__name__ == "ANN_stock":
            count_stock = df.groupby(df[ENTITY]).count().reset_index()
            prev_stocks_names = count_stock[count_stock[similarity_col] > 5 + 1 + 1 * 15][ENTITY].tolist()
            top_stock_w = {}
            for s in prev_stocks_names:
                top_stock_w[s] = 1

            top_stock_w = {stock_name: 1}

            train_x_time_model, train_y_time_model, _, _, _, _, \
            _ = prepare_train_test_windows(
                stock_name, df, None, self.selected_numeric_cols, [1], 1, False,
                top_stock_w, False, self.window_len, self.similarity_col)

            train_x_time_model_respahe = train_x_time_model.as_matrix().reshape((len(train_x_time_model),
                                                                                 self.window_len,
                                                                                 len(self.selected_numeric_cols)))
            self.model.fit(train_x_time_model_respahe, train_y_time_model)
        else:

            train_points_x, train_points_y, \
            test_points_x, test_points_y, test_price, _, _ = prepare_train_test_points(stock_name, df, None, [1], None,
                                                                                       None, similarity_col)
            self.model.fit(train_points_x[self.selected_numeric_cols], train_points_y)

    def apply_distance(self, df, stock_name):
        """
        apply euclidean distance between 2 stocks
        :param stock1:
        :param stock2:
        :return:
        """

        if stock_name not in df[ENTITY].unique():
            print stock_name
            return 1000

        top_stock_w = {stock_name: 1}
        if self.model.__class__.__name__ == "ANN_stock":
            train_x, train_y, _, _, _, _, \
            _ = prepare_train_test_windows(
                stock_name, df, None, self.selected_numeric_cols, [1], 1, False,
                top_stock_w, False, self.window_len, self.similarity_col)

            train_x_time_model_respahe = train_x.as_matrix().reshape((len(train_x),
                                                                      self.window_len,
                                                                      len(self.selected_numeric_cols)))
            preds = self.model.predict(train_x_time_model_respahe)
        else:
            train_x, train_y, \
            test_points_x, test_points_y, test_price, _, _ = prepare_train_test_points(stock_name, df, None, [1], None,
                                                                                       None, self.similarity_col)

            preds = self.model.predict(train_x[self.selected_numeric_cols])
        return mean_squared_error(train_y, preds)

    @property
    def __name__(self):
        return "model_bases_distance"


import statsmodels.tsa.stattools as ts
def coinintegration(stock1, stock2, fix_len_func=correlate_stock_len, similarity_col=TARGET):
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    oin_t, p_val, _crit = ts.coint(stock1[similarity_col].tolist(),stock2[similarity_col].tolist())
    return p_val

def calculate_similarity_all_stocks(df_stocks, stock_to_compare, stock_names, similarity_func,
                                    similarity_file_path, fix_len_func=correlate_stock_len, similarity_col=TARGET,
                                    force=False, split_time="", **kwargs):
    """
    claculate similarities between target stock to the otherts using a similarity function,
    save the similarities on disk

    :param df_stocks: all sticks
    :param stock_to_compare: target stock name
    :param stock_names: list of all stocks names
    :param similarity_func:
    :param experiment_path:
    :param force: force write new file
    :param split_time: if the similarity is on diffrent periods save the similarity with time in filename
    :return: list of similarities between the target stoack and the other in the same order of the stock_names list
    """
    print "calc similarities for " + stock_to_compare + " func " + str(similarity_func) + \
          " fix len " + str(fix_len_func) + " on column " + similarity_col
    # print " to stocks " + " ".join(stock_names)
    if (not os.path.isfile(similarity_file_path)) or force:
        if isinstance(similarity_func, model_bases_distance):
            # stock_X = df_stocks[df_stocks[ENTITY] == stock_to_compare]
            similarity_func.fit(df_stocks, stock_to_compare, similarity_col)
            similarities = [
                similarity_func.apply_distance(df_stocks, stock_name)
                for stock_name in stock_names
                ]
        else:

            similarities = [
                similarity_func(df_stocks[df_stocks[ENTITY] == stock_to_compare],  # [y_col].tolist(),
                                df_stocks[df_stocks[ENTITY] == stock_name], fix_len_func,
                                similarity_col)  # [y_col].tolist())
                for stock_name in stock_names
                ]
        with open(similarity_file_path, 'wb') as f:
            pickle.dump(similarities, f)
    with open(similarity_file_path, 'rb') as f:
        similarities = pickle.load(f)
    return similarities


def get_top_k(stock_names, similarities, k):
    """
    a function for selecting the stocks by the highest similarity - lowest distance
    :param stock_names: the name of stocks
    :param similarities: the distance measure bewteen the stocks to the target stock
    :param k: amount of stocks
    :return: list of top stocks
    """
    s = np.array(similarities)
    k = k + 1
    idx = np.argpartition(s, k)
    names_top_k = np.array(stock_names)[idx[:k]]
    sim_top_k = s[idx[:k]]
    top_stocks = {}
    for i in range(len(names_top_k)):
        top_stocks[names_top_k[i]] = sim_top_k[i]

    return top_stocks


def get_random_k(stock_names, similarities, k):
    """
    a function for selecting the stocks randomly
    :param stock_names: the name of stocks
    :param similarities: the distance measure bewteen the stocks to the target stock
    :param k: amount of stocks
    :return: list of top stocks
    """

    s = np.array(similarities)
    idx = np.argpartition(s, 1)
    name_target_stock = np.array(stock_names)[idx[:1]]
    top_stocks = {}
    top_stocks[name_target_stock[0]] = 0.0

    k -= 1
    random.seed(0)
    for i in range(k - 1):
        ind = random.choice(range(len(stock_names)))
        top_stocks[stock_names[ind]] = similarities[ind]
    return top_stocks


##################  Data preparation for Time Series #################
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


def combine_df(data, name, cols_names, idx_name, idx_col):
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
        normalization.fit(stock_X)
    stock_X_norm = normalization.transform(stock_X)
    stock_X_norm_df = combine_df(stock_X_norm, "_norm", stock_X.columns, TIME, stock_X.index)

    stock_X_prep.append(stock_X_norm_df)

    if len(stock_X_prep) > 1:
        stock_X_prep_df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), stock_X_prep)
    else:
        stock_X_prep_df = stock_X

    if transformation is not None:
        if to_fit:
            transformation.fit(stock_X_prep_df)
        stock_X_transform = transformation.transform(stock_X_prep_df)
        stock_X_norm_df = combine_df(stock_X_transform, "_transform", stock_X_prep_df.columns, TIME, stock_X.index)
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


def prepare_rolling_periods_for_top_stocks(data_period, stock_to_compare,
                                           start_period_train, end_period_train, start_period_test, end_period_test,
                                           features_selection, finance_features, normalization, transformation,
                                           to_pivot, \
                                           k, select_k_func, similarity_col, similarity_func, fix_len_func, window_len,
                                           slide, weighted_sampleing, y_col, next_t,
                                           data_path, similarity_path, force):
    """
    split the data to train period and evaluation period, the entire preprocessing parameters are computed seperatly
     per k neareast stock on the train period and then applied also on the evaluation period,
     the periods are defined by stock_to_compare
    the spliting
    for each period a dataset is constructed with windows and targets
    :param df_stocks: the daily datr
    :param stock_to_compare: the comparance stocke
    :param start_period_train:
    :param end_period_train:
    :param start_period_test:
    :param end_period_test:
    :param similarity_func:
    :param select_k_func: a function logic for selecting the top k
    :param k:
    :param preprocessing_pipeline: the preprocsiing pipeline with normalization and discretization
    :param experiment_path:
    :param window_len:
    :param slide:
    :param next_t:
    :param weighted_sampleing: id the k are sampleted by value then tue
    :return: Dataframes: train_x, train_y, test_x, test_y, test_prices
                and top_stocks selected
    """
    print "preparing rolling periods"

    # select relevant time frames data
    train_X_all_prev_periods = data_period[(data_period[TIME] < end_period_train)]
    prev_periods_stock_to_compare_ts = train_X_all_prev_periods[train_X_all_prev_periods[ENTITY] == stock_to_compare][
        TIME]
    train_X_all_prev_periods = train_X_all_prev_periods[
        train_X_all_prev_periods[TIME].isin(prev_periods_stock_to_compare_ts)]
    # calc similar stock on all previous data with atleast window len amount of recoeds

    count_stock = train_X_all_prev_periods.groupby([ENTITY]).count()[TIME].reset_index()
    prev_stocks_names = count_stock[count_stock[TIME] > window_len + max(next_t) + slide * 15][ENTITY].tolist()

    file_name = "train_" + str(str(end_period_train.strftime("%Y-%m-%d"))) + "test_" + str(
        str(end_period_test.strftime("%Y-%m-%d")))
    # calculate_features for all stocks
    train_X_all_prev_periods_processed, _, _ = calculate_features_all_stocks(
        os.path.join(data_path, 'all_prev_' + file_name),
        features_selection,
        True, force, normalization,
        prev_stocks_names, train_X_all_prev_periods,
        None, transformation,
        y_col)

    # calculate_similarity
    similarity_file_path = os.path.join(similarity_path, file_name + ".pkl")
    similarities = calculate_similarity_all_stocks(train_X_all_prev_periods_processed, stock_to_compare,
                                                   prev_stocks_names,
                                                   similarity_func, similarity_file_path, fix_len_func, similarity_col,
                                                   split_time=str(end_period_train), force=force)
    top_stocks = select_k_func(prev_stocks_names, similarities, k)
    # normalize similarity
    stocks_val = list(top_stocks.values())
    top_stock_w = {}
    sum_vals = 0
    for stock_k, v in top_stocks.items():
        if stock_k != stock_to_compare:
            top_stock_w[stock_k] = np.abs(float(v) - max(stocks_val)) / (max(stocks_val) - min(stocks_val))
            sum_vals += top_stock_w[stock_k]

    for stock_k, v in top_stock_w.items():
        top_stock_w[stock_k] = top_stock_w[stock_k] / sum_vals

    train_X = data_period[(start_period_train <= data_period[TIME]) & (data_period[TIME] < end_period_train) &
                          data_period[ENTITY].isin(prev_stocks_names)]
    train_X_stock_to_compare_ts = train_X[train_X[ENTITY] == stock_to_compare][TIME]
    train_X = train_X[train_X[TIME].isin(train_X_stock_to_compare_ts)]

    test_X = data_period[(start_period_test <= data_period[TIME]) & (data_period[TIME] < end_period_test) &
                         data_period[ENTITY].isin(prev_stocks_names)]
    test_X_stock_to_compare_ts = test_X[test_X[ENTITY] == stock_to_compare][TIME]
    test_X = test_X[test_X[TIME].isin(test_X_stock_to_compare_ts)]

    train_X_processed_df, test_X_processed_df, features_names = calculate_features_all_stocks(
        os.path.join(data_path, file_name), features_selection, finance_features, force, normalization,
        prev_stocks_names,
        train_X, test_X, transformation, y_col)

    features_names = features_selection[1]  # TODO turn to multivariate
    if window_len > 0:
        train_x_time_model, train_y_time_model, test_x_time_model, test_y_model, test_price, train_curr_target_prep, \
        test_curr_target_prep = prepare_train_test_windows(
            stock_to_compare, train_X_processed_df, test_X_processed_df, features_names, next_t, slide,
            to_pivot, top_stock_w, weighted_sampleing, window_len, y_col)

        def set_index(df):
            return df.set_index(pd.MultiIndex.from_arrays([df.index, range(len(df))], names=('number', 'color')))

        train_x_time_model = set_index(train_x_time_model)
        train_y_time_model = set_index(train_y_time_model)
        train_curr_target_prep = set_index(train_curr_target_prep.to_frame())[TARGET_PREP]

    else:
        train_x_time_model, train_y_time_model, test_x_time_model, test_y_model, test_price, train_curr_target_prep, \
        test_curr_target_prep = prepare_train_test_points(
            stock_to_compare, train_X_processed_df, test_X_processed_df, next_t,
            top_stock_w, weighted_sampleing, y_col)

    return train_x_time_model, train_y_time_model, test_x_time_model, test_y_model, test_price, \
           top_stocks, features_names, train_curr_target_prep, \
           test_curr_target_prep


def prepare_folds(data_period, stock_to_compare, n_folds, features_selection, finance_features, normalization,
                  transformation, to_pivot, \
                  k, select_k_func, similarity_col, similarity_func, fix_len_func, window_len, slide,
                  weighted_sampleing, y_col, next_t, data_path, similarity_path, force):
    """
    prepare a rolling folds evaluations
    :param df_stocks:
    :param stock_to_compare:
    :param window_len:
    :param slide:
    :param next_t:
    :param n_folds:
    :param experiment_path:
    :param force:
    :param similarity_func:
    :param select_k_func:
    :param k:
    :param preprocessing_pipeline:
    :param weighted_sampleing:
    :return:
    """
    print "preparing folds"
    stock_times = data_period[data_period[ENTITY] == stock_to_compare][TIME].tolist()

    period_len = int(abs(len(stock_times) / n_folds))
    folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test, folds_topk = [], [], [], [], [], []
    folds_curr_target_prep_train, folds_curr_target_prep_test = [], []
    # for each fold
    for f in range(n_folds - 1):
        # define rolling times by the tagets period
        start_period_train = stock_times[f * period_len]
        end_period_train = stock_times[f * period_len + period_len]
        start_period_test = stock_times[f * period_len + period_len]
        # if f == n_folds-1 :
        #     end_period_test = stock_times[-1]
        # else:
        #     end_period_test = stock_times[f*period_len + period_len/10 + period_len-1]
        end_period_test = stock_times[f * period_len + period_len + period_len / 4 - 1 + 2 * window_len]
        train_x, train_y, \
        test_x, test_y, test_price, \
        top_stocks, features_names, train_curr_target_prep, \
        test_curr_target_prep = \
            prepare_rolling_periods_for_top_stocks(data_period, stock_to_compare,
                                                   start_period_train, end_period_train, start_period_test,
                                                   end_period_test,
                                                   features_selection, finance_features, normalization, transformation,
                                                   to_pivot,
                                                   k, select_k_func, similarity_col, similarity_func, fix_len_func,
                                                   window_len, slide, weighted_sampleing, y_col, next_t
                                                   , data_path, similarity_path, force)

        folds_X_train.append(train_x), folds_Y_train.append(train_y), \
        folds_X_test.append(test_x), folds_Y_test.append(test_y), folds_price_test.append(test_price), \
        folds_topk.append(top_stocks), \
        folds_curr_target_prep_train.append(train_curr_target_prep), \
        folds_curr_target_prep_test.append(test_curr_target_prep)

    return [folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test, folds_curr_target_prep_train,
            folds_curr_target_prep_test], \
           folds_topk, features_names


################# Evaluation methods ################################
def simple_profit_evaluation(curr_price, predicted_price):
    """
    a profit evaluation that buys if stock goes up and sell of goes down
    :param curr_price:
    :param predicted_price:
    :return:
    """
    in_position = False
    profit = 0
    last_buy = 0
    profits = []
    for i in range(len(curr_price)):
        if predicted_price[i] > 0 and not in_position:
            in_position = True
            last_buy = curr_price[i]
        if predicted_price[i] < 0 and in_position:
            in_position = False
            profit = profit + (curr_price[i] - last_buy)
        if in_position:
            profits.append(profit + (curr_price[i] - last_buy))
        else:
            profits.append(profit)
    return profit, profits


def long_short_profit_evaluation(curr_price, predicted_price):
    """
    a profit evaluation that buys long if stock goes up and buys short of goes down
    :param curr_price:
    :param predicted_price:
    :return:
    """
    is_long = None
    profit = 0
    last_buy = 0
    profits = []
    position = 0
    for i in range(len(curr_price)):
        # go long
        if predicted_price[i] > 0:
            # first time
            if is_long is None:
                last_buy = curr_price[i]
                is_long = True
            # if short position - close it and go long
            elif not is_long:
                profit = profit + (last_buy - curr_price[i])
                position = profit
                last_buy = curr_price[i]
                is_long = True
            elif is_long:
                position = profit + (curr_price[i] - last_buy)

        # go short
        if predicted_price[i] < 0:
            # first time
            if is_long is None:
                last_buy = curr_price[i]
                is_long = False
            # if long position - close it and go short
            elif is_long:
                profit = profit + (curr_price[i] - last_buy)
                position = profit
                last_buy = curr_price[i]
                is_long = False
            elif not is_long:
                position = profit + (last_buy - curr_price[i])

        profits.append(position)

    return profit, profits


################# Experiments Executions ############################
def evaluate_model(window_len, folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test,
                   folds_curr_target_prep_train, folds_curr_target_prep_test, features_names, model_class, model_args,y_col):
    """
    a function that run the model on the diffrent folds and calculate metircs
    :param window_len:
    :param folds_X_train:
    :param folds_Y_train:
    :param folds_X_test:
    :param folds_Y_test:
    :param folds_price_test:
    :param model_class:
    :param model_args:
    :param evaluation_methods:
    :param profit_methods:
    :param target_discretization:
    :return: evaluations
    """
    print "evaluate model"
    print model_class.__name__
    future_ys = folds_Y_train[0].columns.tolist()
    evaluations = []
    evaluations_values = []
    if window_len > 0:
        features = [(features_name, wl) for wl in range(window_len) for features_name in features_names]
    else:
        features = features_names
    # iterate folds
    for f in range(len(folds_X_train)):
        print 'fold' + str(f)
        X_train = folds_X_train[f][features]
        X_test = folds_X_test[f][features]

        # X_train['ind'] = range(len(X_train))
        # X_train = X_train.set_index(['ind'], append=True)
        # X_test_index = X_test.index
        # X_test = X_test.set_index(range(len(X_train)),append=True)
        # folds_Y_train_index_f = folds_Y_train[f].index
        # folds_Y_train[f] = folds_Y_train[f].set_index(range(len(X_train)),append=True)
        # folds_curr_target_prep_train_index_f = folds_curr_target_prep_train[f].index
        # folds_curr_target_prep_train[f] = folds_curr_target_prep_train[f].set_index(range(len(X_train)),append=True)
        # folds_Y_test_index_f = folds_Y_test[f].index
        # folds_Y_test[f] = folds_Y_test[f].set_index(range(len(X_train)),append=True)
        # folds_curr_target_prep_test_index_f = folds_curr_target_prep_test[f].index
        # folds_curr_target_prep_test[f] = folds_curr_target_prep_test[f].set_index(range(len(X_train)),append=True)
        # folds_price_test_index_f = folds_price_test[f].index
        # folds_price_test[f] = folds_price_test[f].set_index(range(len(X_train)),append=True)
        # iterate future value to predict
        for t in future_ys:
            print 'next t' + str(t)

            y_train = folds_Y_train[f].loc[X_train.index][t]
            X_train_curr_price_prep = folds_curr_target_prep_train[f].loc[X_train.index]

            y_test = folds_Y_test[f][t].loc[X_test.index]
            X_test_curr_price_prep = folds_curr_target_prep_test[f].loc[y_test.index].tolist()
            price_test = folds_price_test[f].loc[y_test.index]

            model = model_class(**model_args)

            if isinstance(model, RegressorMixin):
                model.fit(X_train, y_train)
                y_preds_val = model.predict(X_test)
                if (y_col == 'Close_norm'):
                    y_preds_binary = np.sign(y_preds_val - X_test_curr_price_prep)
                    y_preds_binary = [1 if x == 0 else x for x in y_preds_binary]
                elif (y_col == 'Close_proc'):
                    y_preds_binary =  np.sign(y_preds_val)

            else:
                if (y_col == 'Close_norm'):
                    y_train_binary = np.sign(y_train - X_train_curr_price_prep)
                    y_train_binary = [1 if x == 0 else x for x in y_train_binary]
                elif (y_col == 'Close_proc'):
                    y_train_binary =  np.sign(y_train)

                model.fit(X_train, y_train_binary)
                y_preds_binary = model.predict(X_test)

            fold_eval = {}
            fold_eval["fold"] = f
            fold_eval["model"] = model_class.__name__
            fold_eval["next_t"] = t

            eval_values = pd.DataFrame()
            eval_values['curr_price'] = price_test
            eval_values['preds'] = y_preds_binary

            if (y_col == 'Close_norm'):
                y_test_binary = np.sign(y_test - X_test_curr_price_prep)
                y_test_binary = [1 if x == 0 else x for x in y_test_binary]
            elif (y_col == 'Close_proc'):
                y_test_binary = np.sign(y_test)

            eval_values['y'] = y_test_binary
            # eval_values['curr_price2'] = folds_price_test[f][t].values
            for k1, v in fold_eval.items():
                eval_values[k1] = v

            evals = dict(fold_eval)
            evals['accuracy_score'] = accuracy_score(y_test_binary, y_preds_binary)
            evals['f1_score'] = f1_score(y_test_binary, y_preds_binary, average='macro')
            evals['precision_score'] = precision_score(y_test_binary, y_preds_binary, average='macro')

            if not isinstance(model, RegressorMixin):
                # y_proba = model.predict_proba(X_test)
                try:
                    evals['roc_auc_score'] = roc_auc_score(y_test_binary, y_preds_binary, average='macro')
                except:
                    evals['roc_auc_score'] = -1
            else:
                evals['roc_auc_score'] = 0

            evals["long_short_profit"], eval_values["long_short_profit"] = long_short_profit_evaluation(
                price_test.tolist(), y_preds_binary)
            evals["sharp_ratio"] = np.mean(eval_values["long_short_profit"]) / (
            np.std(eval_values["long_short_profit"]) + 0.0001)

            evaluations.append(evals)
            evaluations_values.append(eval_values)

    return pd.DataFrame(evaluations), pd.concat(evaluations_values)


def get_index_product(params):
    """
    a function the calculate all the combinations of different experiments
    :param params:
    :return: list dict, each dict is args an experiment
    """
    i = 0
    params_index = {}
    for k, v in params.items():
        params_index[k] = i
        i += 1
    params_list = [None] * len(params_index.values())
    for name, loc in params_index.items():
        params_list[loc] = params[name]

    params_product = list(itertools.product(*params_list))
    params_product_dicts = []
    for params_value in params_product:
        params_dict = {}
        for param_name, param_index in params_index.items():
            params_dict[param_name] = params_value[param_index]
        params_product_dicts.append(params_dict)

    return params_product_dicts


def save_evaluations(evaluations, results_path, folds_topk, processing_params, windowing_params):
    model_evals = []
    model_values_evals = []
    model_params = {}
    model_params.update(processing_params)
    model_params.update(windowing_params)

    for model_i in range(len(evaluations)):
        model_eval_df = evaluations[model_i][0]
        model_values_eval = evaluations[model_i][1]
        for k1, v1 in model_params.items():
            if isinstance(v1, tuple):
                v1 = v1[0]
            model_eval_df[k1] = v1
            model_values_eval[k1] = v1
        model_evals.append(model_eval_df)
        model_values_evals.append(model_values_eval)

    # pd.concat(model_evals).to_csv(os.path.join(results_path, 'models_evaluations.csv'), mode='a')
    pd.concat(model_values_evals).to_csv(os.path.join(results_path, 'models_values_evaluations.csv'), mode='a')

    similar_stock_eval_folds = []
    for f in range(processing_params['n_folds'] - 1):
        similar_stock_eval_fold = {}
        similar_stock_eval_fold.update(processing_params)
        for name, distance in folds_topk[f].items():
            stock_f = dict(similar_stock_eval_fold)
            stock_f['name'] = name
            stock_f['distance'] = distance
            similar_stock_eval_folds.append(stock_f)

    sim_eval_path = os.path.join(results_path, 'similarity_evaluations.csv')
    pd.DataFrame(similar_stock_eval_folds).to_csv(sim_eval_path, mode='a')


def run_experiment(data_period, stock_to_compare, n_folds, features_selection, finance_features, normalization,
                   transformation, to_pivot, \
                   k, select_k_func, similarity_col, similarity_func, fix_len_func, window_len, slide,
                   weighted_sampleing, y_col, next_t, models, models_arg, force):
    """
    run the entire experiments on the diffrent parameters and saves the data to csv.
    :param data_period: 1yr / 5yr
    :param stock_to_compare:
    :param preprocessing_pipeline:
    :param similarity_func:
    :param k:
    :param select_k_func:
    :param window_len:
    :param slide:
    :param next_t:
    :param n_folds:
    :param models:
    :param models_arg:
    :param target_discretization:
    :param weighted_sampleing:
    :param force:
    :return:
    """

    experiment_path = os.path.join(home_path, 'experiments', data_period + "_folds-" + str(n_folds))
    data_path = os.path.join(experiment_path, stock_to_compare, 'data',
                             'fs-' + features_selection[0] + '_finance_fe-' + str(
                                 finance_features) + "_norm-" + normalization
                             + "_transform-" + transformation, 'fold-')
    similarity_path = os.path.join(experiment_path, stock_to_compare, 'similarity',
                                   'func-' + similarity_func + '_col-' + similarity_col + fix_len_func, 'fold-')
    if not os.path.exists(os.path.dirname(experiment_path)):
        os.makedirs(experiment_path)
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(data_path)
    if not os.path.exists(os.path.dirname(similarity_path)):
        os.makedirs(similarity_path)

    df_stocks = get_data(data_period)

    # preapare data with slide, time window as fold -> test + train X features + targets
    folds_loaded, folds_topk, features_names = prepare_folds(df_stocks, stock_to_compare, n_folds, features_selection,
                                                             finance_features,
                                                             normalizations[normalization],
                                                             transformations[transformation], to_pivot, k,
                                                             select_k_funcs[select_k_func], similarity_col,
                                                             similarity_funcs[similarity_func],
                                                             fix_len_funcs[fix_len_func], window_len, slide,
                                                             weighted_sampleing, y_col, next_t,
                                                             data_path, similarity_path, force)

    folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test, folds_curr_target_prep_train, folds_curr_target_prep_test = \
        folds_loaded[0], folds_loaded[1], folds_loaded[2], folds_loaded[3], folds_loaded[4], folds_loaded[5], \
        folds_loaded[6]

    evaluations = [
        evaluate_model(window_len, folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test,
                       folds_curr_target_prep_train, folds_curr_target_prep_test,
                       features_names, model, models_arg[model.__name__],y_col)
        for model in models]

    processing_params = {'data_period': data_period,
                         'stock_to_compare': stock_to_compare,
                         'n_folds': n_folds,
                         'features_selection': features_selection,
                         'finance_features': finance_features,
                         'normalization': normalization,
                         'transformation': transformation,
                         'k': k,
                         'select_k_func': select_k_func,
                         'similarity_col': similarity_col,
                         'similarity_func': similarity_func}

    windowing_params = {'window_len': window_len,
                        'slide': slide,
                        'weighted_sampleing': weighted_sampleing,
                        'y_col': y_col}
    # 'next_t' : next_t}

    save_evaluations(evaluations, experiment_path, folds_topk, processing_params, windowing_params)
    return experiment_path


transformations = {'None': None,
                   'SAX': SAX(),
                   'PCA': PCA()}

normalizations = {'Standard': StandardScaler()}

select_k_funcs = {'get_random_k': get_random_k,
                  'get_top_k': get_top_k}


similarity_funcs = {'cointegration': coinintegration,
    'model_based_LSTM': model_bases_distance(ANN_stock()),
                    'sax' : compare_sax,
                    'model_based_RFR': model_bases_distance(RandomForestRegressor(n_estimators = 25, random_state=0)),
                    'euclidean' : apply_euclidean,
                    'dtw' : apply_dtw,
                    'pearson' : apply_pearson
                    }

fix_len_funcs = {'simple_fix': fix_stock_len,
                 'time_corr': correlate_stock_len,
                 'delay_1': correlate_stock_len_delay,
                 'pip_fix': pip_fix
                 }


def calc_similarites(data_name):
    stocks_df = get_data(data_name)
    # stocks_compare_eval = stocks_df[ENTITY].unique()
    stocks_to_compare = ['JPM', "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"]
    all_stocks_names = stocks_df[ENTITY].unique()

    preprocess_params = dict(
        stocks_df=stocks_df,
        features_selection=[u'Close_proc', u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff',
                            u'Open_norm', u'High_norm', u'Low_norm', u'Close_norm', u'Volume_norm',
                            u'traget_prep', u'Close', u'Name'],
        finance_features=True,
        normalization=StandardScaler(),
        transformation=None,
        to_fit=True,
        y_col='Close_norm'
    )
    stock_X_prep_dfs = []
    for stock in all_stocks_names:
        preprocess_params['stock_name'] = stock
        # stocks_df, stock_name,features_selection, finance_features, normalization, transformation,y_col, to_fit = True,
        stock_X_prep_df, _, _, _ = preprocess_stock_features(**preprocess_params)
        stock_X_prep_dfs.append(stock_X_prep_df)
        del preprocess_params['stock_name']

    stock_X_prep_dfs = pd.concat(stock_X_prep_dfs)
    similarity_params = dict(
        similarity_col=[TARGET_PREP, 'rsi', 'Close_proc', 'MACD'],  # , 'Volume_norm'],
        y_col=[TARGET],
        similarity_func_name= similarity_funcs.keys(),
        fix_len_func_name=fix_len_funcs.keys(),
        stock_to_compare=stocks_to_compare
    )
    all_sim_path = os.path.join(home_path, 'experiments', 'similarities', data_name)
    similarity_params_product = get_index_product(similarity_params)
    sim_results = []
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=len(similarity_params_product))
    for similarity_param in similarity_params_product:
        sim_results.append(run_sim_analysis(all_sim_path, all_stocks_names, similarity_param, stock_X_prep_dfs))
        # async_result = pool.apply_async(run_sim_analysis, (all_sim_path, all_stocks_names, similarity_param, stock_X_prep_dfs))
    # sim_results = async_result.get()
    pd.concat(sim_results).to_csv(os.path.join(home_path, 'experiments', 'similarities.csv'), mode='a')


def run_sim_analysis(all_sim_path, all_stocks_names, similarity_param, stock_X_prep_dfs):
    similarity_param['similarity_file_path'] = all_sim_path + '_'.join(similarity_param.values()) + \
                                               '_stocks' + str(len(all_stocks_names))
    similarity_param['similarity_func'] = similarity_funcs[similarity_param['similarity_func_name']]
    similarity_param['fix_len_func'] = fix_len_funcs[similarity_param['fix_len_func_name']]
    similarity_param['stock_names'] = all_stocks_names
    res = pd.DataFrame()
    if (similarity_param['similarity_func_name'] == 'model_based_RFR' and similarity_param[
        'fix_len_func_name'] == 'time_corr') \
            or (similarity_param['similarity_func_name'] != 'model_based_RFR'):
        similarities = calculate_similarity_all_stocks(stock_X_prep_dfs, **similarity_param)
        del similarity_param['similarity_file_path']
        del similarity_param['similarity_func']
        del similarity_param['fix_len_func']
        del similarity_param['stock_names']

        res['distance'] = similarities
        res['rank'] = rankdata(similarities, method='ordinal')
        res['norm_similarity'] = (np.abs(np.array(similarities) - max(similarities))) / (max(similarities) - min(similarities))
        res['stock_name'] = all_stocks_names
        for kp, v in similarity_param.items():
            res[kp] = v
            # sim_results.append(res)
    return res


def main():
    experiment_predict_params = \
        {
            'next_t': [1, 3, 7],
            'to_pivot': True,
            'models': [GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                       GradientBoostingClassifier],
            'models_arg': {RandomForestClassifier.__name__: {'n_estimators': 100, 'random_state': 0},
                           RandomForestRegressor.__name__: {'n_estimators': 100, 'random_state': 0},
                           GradientBoostingClassifier.__name__: {'learning_rate': 0.02, 'random_state': 0},
                           GradientBoostingRegressor.__name__: {'learning_rate': 0.02, 'random_state': 0}},
            'force': False
        }


    experiment_params_base = {
        'data_period': ['5yr'
                        ],
        # tech, finance, service, health, consumer, Industrial
        'stock_to_compare': ["GE", "JPM", "KO", "DIS", "JNJ", "MMM", "GOOG"],
        'n_folds': [6],
        'finance_features': [True],
        'normalization': ['Standard'],
        'select_k_func': ['get_top_k'],
        'slide': [1]
    }



    experiment_params_1 = {
        'features_selection': [
            ('univariate', [u'Close_norm']),
            ('multivariate',
             [u'Close_proc', u'Close_norm',
              u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff', u'Volume_norm']
             ),
        ],
        'transformation': ['PCA','SAX', 'None'], #TODO SAX for uni PCA for multi
        'k': [10],
        'similarity_col': ['Close_norm'],
        'similarity_func': ['euclidean'],
        'fix_len_func': ['time_corr'],
        'window_len': [0,5,10],
        'weighted_sampleing': [True, False],
        'y_col': ['Close_proc','Close_norm'],
    }
    experiment_params_1.update(experiment_params_base)

    iterate_exp(experiment_params_1, experiment_predict_params)

    experiment_params_2 ={

        'features_selection': [
            ('univariate', [u'Close_norm'])
        ],
        'finance_features': [True],
        'normalization': ['Standard'],
        'transformation': ['None'],
        'k': [10,25,50],
        'similarity_col': ['Close_norm','Close_proc'],
        'similarity_func': similarity_funcs.keys(),
        'fix_len_func': fix_len_funcs.keys()
    }

    experiment_params_3 ={
        'select_k_func': ['get_top_k','get_random_k'],
        'k': [0],
    }

    experiment_params_MLP = {

        'features_selection': [
            ('multivariate',
             [u'Close_proc', u'Close_norm',
              u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff', u'Volume_norm']
             ),
        ],
        'k': [5],
        'similarity_col': ['Close_norm','Close_proc'],
        'similarity_func': ['ensamble'],
        'fix_len_func': ['time_corr'],
        'window_len': [0],
        'y_col': ['Close_norm','close proc'],
    }

    experiment_params_LSTM = {
        'features_selection': [
            ('multivariate',
             [u'Close_proc', u'Close_norm',
              u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff', u'Volume_norm']
             ),
        ],
        'k': [25,50],
        'similarity_col': ['Close_norm','Close_proc'],
        'similarity_func': ['ensamble'],
        'fix_len_func': ['time_corr'],
        'window_len': [5],
        'weighted_sampleing': [True, False],
        'y_col': ['Close_norm','close proc'],
    }

    experiment_params = {
        'data_period': ['5yr'
                        ],
        # tech, finance, service, health, consumer, Industrial
        'stock_to_compare': ["GE", "JPM", "KO", "DIS", "JNJ", "MMM", "GOOG"],
        'n_folds': [6],

        'features_selection': [
            ('univariate', [u'Close_norm']),
            ('multivariate',
             [u'Close_proc', u'Close_norm',
              u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff', u'Volume_norm']
             ),
        ],
        'finance_features': [True],
        'normalization': ['Standard'],
        'transformation': ['None', 'SAX', 'PCA'], #TODO SAX for uni PCA for multi
        'k': [10,25,50],
        'select_k_func': ['get_top_k'],
        'similarity_col': ['Close_norm','Close_proc'],
        'similarity_func': similarity_funcs.keys(),
        'fix_len_func': fix_len_funcs.keys(),
        'window_len': [0,5,10],
        'slide': [1],
        'weighted_sampleing': [True, False],
        'y_col': ['Close_norm','close proc'],
        'weighted_sampleing': [True, False]
    }



def iterate_exp(experiment_params,experiment_static_params):
    experiments = get_index_product(experiment_params)
    for experiment in experiments:
        # not experiment
        if (experiment['k'] == 0 and experiment['weighted_sampleing'] is True) \
                or (experiment['window_len'] == 0 and experiment['weighted_sampleing'] is True) \
                or (experiment['transformation'] == 'SAX' and experiment['features_selection'][0] == 'multivariate') \
                or (experiment['transformation'] == 'PCA' and experiment['features_selection'][0] == 'univariate') \
                or (experiment['window_len'] > 0 and experiment['features_selection'][0] == 'multivariate'):
            continue

        print "run experiment: " + str(experiment)
        experiment.update(experiment_static_params)
        results_path = run_experiment(**experiment)

        pd.DataFrame().to_csv(os.path.join(results_path, 'models_values_evaluations.csv'), mode='a')
        # pd.DataFrame().to_csv(os.path.join(results_path, 'models_evaluations.csv'), mode='a')
        # pd.DataFrame().to_csv(os.path.join(results_path, 'similarity_evaluations.csv'), mode='a')


#calc_similarites('5yr')
main()
