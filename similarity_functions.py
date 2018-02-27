
#from data_preperation import *
import random

from dtw import dtw
import pickle

from sklearn.metrics import mean_squared_error
from data_preperation import prepare_train_test_points, prepare_stock_windows
from Constants import *
from data_preperation import *
#import pandas as pd
import numpy as np
import os

import pandas as pd


from utils.fastpip import fastpip

from utils.SAX_FILE import SAX


## length fixing
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


def pip_fix(stock1, stock2, factor=10, similarity_col=TARGET):
    stock1, stock2 = correlate_stock_len(stock1, stock2)
    min_len = min(*[len(stock1[similarity_col]), len(stock2[similarity_col])])
    pip_size = min_len / factor
    if pip_size < 25 and min_len > 25:
        pip_size = 25
    stock1_pairs = [(t, p) for t, p in zip(range(len(stock1[similarity_col])), stock1[similarity_col])]
    stock2_pairs = [(t, p) for t, p in zip(range(len(stock2[similarity_col])), stock2[similarity_col])]
    stock1_pairs_pip = fastpip(stock1_pairs, pip_size)
    stock2_pairs_pip = fastpip(stock2_pairs, pip_size)

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


def test():
    similarity_col = TARGET
    stock1 = pd.DataFrame()
    stock1[similarity_col] = np.random.rand(50)
    stock1[TIME] = range(5,55)
    stock1 = stock1.set_index(TIME)

    stock2 = pd.DataFrame()
    stock2[similarity_col] = np.random.rand(50)
    stock2[TIME] = range(10, 60)
    stock2 = stock2.set_index(TIME)

    for sim_func in [coinintegration,apply_euclidean,compare_sax,apply_dtw,apply_pearson]:
        print
        print sim_func.__name__
        for fix_func in [fix_stock_len,correlate_stock_len,correlate_stock_len_delay,pip_fix]:

            print fix_func.__name__
            sims = sim_func(stock1,stock2,fix_func,similarity_col)
            print sims

if __name__ == "__main__":
    test()