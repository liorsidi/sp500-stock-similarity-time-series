import random

from datetime import datetime
import pandas as pd
import numpy as np
import os
from dtw import dtw
import pickle

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

from Financial_FE import rsiFunc, computeMACD

home_path = 'C:\\Users\\Lior\\StockSimilarity'

TIME = 'Date'
ENTITY = 'Name'
TARGET = 'Close'
FEATURES = ['Close']
TARGET_PREP = 'traget_prep'

from SAX_FILE import SAX
sax_obj = SAX()

def get_data(data_period):
    file_path = os.path.join(home_path, 'sandp500','all_stocks_' + data_period + '.csv')
    all_stocks = pd.read_csv(file_path)
    all_stocks[TIME] = pd.to_datetime(all_stocks[TIME], format='%Y%m%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks= all_stocks.set_index(TIME,drop=False)
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
        add_values = [stock2[0]] * diff_len
        stock2 = add_values + stock2
    elif diff_len < 0:
        add_values = [stock1[0]] * abs(diff_len)
        stock1 = add_values + stock1
    return stock1, stock2


def apply_dtw(stock1, stock2):
    """
    apply DTW distance between 2 stocks
    :param stock1:
    :param stock2:
    :return:
    """
    stock1, stock2 = fix_stock_len(stock1, stock2)
    return dtw(stock1, stock2, dist=lambda x, y: abs(x - y))[0]


def apply_pearson(stock1, stock2):
    """
    apply pearson distance between 2 stocks
    :param stock1:
    :param stock2:
    :return:
    """
    stock1, stock2 = fix_stock_len(stock1, stock2)
    pearson = np.corrcoef(np.array(stock1), np.array(stock2))[0, 1]
    return abs(pearson - 1)


def apply_euclidean(stock1, stock2):
    """
    apply euclidean distance between 2 stocks
    :param stock1:
    :param stock2:
    :return:
    """

    stock1, stock2 = fix_stock_len(stock1, stock2)
    return np.linalg.norm(np.array(stock1) - np.array(stock2))


class model_bases_distance(object):
    def __init__(self, model):
        self.model = model

    def fit(self, stock_X):
        train_stock_to_compare_X, normalization_f, transformation_f, features_names = preprocess_stock_features(stock_X,
                                                                                                   stock_X[ENTITY].iloc[0],
                                                                                                   ('only_close', [u'Close']),
                                                                                                   finance_features= True,
                                                                                                   normalization = normalizations['Standard'],
                                                                                                   transformation = transformations['None'],y_col=TARGET)

        stock_train_windows_x, stock_train_windows_y, _ = prepare_stock_windows(train_stock_to_compare_X,
                                                                                list(features_names), 10, 1,
                                                                                [1],
                                                                                True, TARGET)

        features = [(feature_name, wl) for wl in range(10) for feature_name in list(features_names)]
        x = stock_train_windows_x[features]
        y = stock_train_windows_y[u'1'].values

        self.model.fit(x,y)

    def apply_distance(self, stock_X):
        """
        apply euclidean distance between 2 stocks
        :param stock1:
        :param stock2:
        :return:
        """
        train_stock_to_compare_X, normalization_f, transformation_f, features_names = preprocess_stock_features(stock_X,
                                                                                                                stock_X[
                                                                                                                    ENTITY].iloc[
                                                                                                                    0],
                                                                                                                [
                                                                                                                    TARGET],
                                                                                                                finance_features=True,
                                                                                                                normalization=
                                                                                                                normalizations[
                                                                                                                    'Standard'],
                                                                                                                transformation=
                                                                                                                transformations[
                                                                                                                    'None'],
                                                                                                                y_col=TARGET)
        stock_train_windows_x, stock_train_windows_y, _ = prepare_stock_windows(train_stock_to_compare_X,
                                                                                list(features_names), 10, 1,
                                                                                [1],
                                                                                True, TARGET)
        preds = self.model.predict(stock_train_windows_x)
        return mean_squared_error(stock_train_windows_y, preds)

    @property
    def __name__(self): return "model_bases_distance"


def compare_sax(stock1, stock2):
    stock1, stock2 = fix_stock_len(stock1, stock2)
    stock1_s = sax_obj.transform(stock1)
    stock2_s = sax_obj.transform(stock2)
    return sax_obj.compare_strings(stock1_s, stock2_s)



def calculate_similarity_all_stocks(df_stocks, stock_to_compare, stock_names, similarity_func, similarity_file_path, force = False, split_time ="", y_col ='Close'):
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
    print "calc similarities"
    if (not os.path.isfile(similarity_file_path)) or force:
        if isinstance(similarity_func, model_bases_distance):
            stock_X = df_stocks[df_stocks[ENTITY] == stock_to_compare]
            similarity_func.fit(stock_X)
            similarities = [
                similarity_func.apply_distance(df_stocks[df_stocks[ENTITY] == stock_name])
                for stock_name in stock_names
                ]
        else:
            similarities = [
                similarity_func(df_stocks[df_stocks[ENTITY] == stock_to_compare][y_col].tolist(),
                              df_stocks[df_stocks[ENTITY] == stock_name][y_col].tolist())
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
    k = k+1
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

    k-=1
    random.seed(0)
    for i in range(k-1):
        ind = random.choice(range(len(stock_names)))
        top_stocks[stock_names[ind]] = similarities[ind]
    return top_stocks


##################  Data preparation for Time Series #################
def prepare_stock_windows(stock_X, features_names, window_len, slide, next_t, to_pivot= True, y_col = 'Close'):
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
    if not to_pivot:
        stock_name = stock_X[ENTITY].unique()[0]
        X_stocks_windows = {}
    y = np.array(stock_X[TARGET_PREP].tolist()) #TODO chande to y_col for generic y

    while i < len(stock_X[ENTITY]) - window_len - max(next_t):
        y_ti = i + window_len  - 1  + np.asarray(next_t)
        stock_X_window = stock_X[i:i + window_len]
        stock_X_window.insert(0, 't', range(window_len))
        window_time = stock_X_window.index.values[-1]
        if to_pivot:
            stock_X_window_flat = stock_X_window[features_names + [ENTITY] + ['t']].pivot(index = ENTITY,columns = 't')#.iloc[0].to_dict()
            stock_X_window_flat = stock_X_window_flat.iloc[0].to_dict()
            stock_X_window_flat[TIME] = window_time
            X_stocks_windows.append(stock_X_window_flat)

        next_y = y[y_ti]
        y_vals = next_y.tolist()
        y_ = {}
        for c in range(len(y_column_names)):
            y_[str(y_column_names[c])] = y_vals[c]
        y_[TIME] = window_time
        Y_.append(y_)

        #the current window last price
        y_price = {}
        y_price[TARGET] = np.array(stock_X[TARGET].tolist())[i + window_len].tolist()
        y_price[TIME] = window_time
        Y_price.append(y_price)

        # next_price = np.array(stock_X[TARGET].tolist())[y_ti]
        # price_vals = next_price.tolist()
        # y_price = {}
        # for c in range(len(y_column_names)):
        #     y_price[str(y_column_names[c])] = price_vals[c]
        # y_price[TIME] = i
        # Y_price.append(y_price)

        i += slide
    if to_pivot:
        X = pd.DataFrame.from_records(X_stocks_windows)
    else:
        X = pd.concat(X_stocks_windows)
    return X.set_index(TIME) ,\
           pd.DataFrame(Y_).set_index(TIME),\
           pd.DataFrame(Y_price).set_index(TIME)


def combine_df(data, name, cols_names, idx_name,idx_col):
    cols_names_new = [s + name for s in cols_names]
    data_df = pd.DataFrame(data, columns=cols_names_new)
    data_df[idx_name] = idx_col
    data_df = data_df.set_index(TIME)
    return data_df


def preprocess_stock_features(stocks_df, stock_name,features_selection, finance_features, normalization, transformation,y_col, to_fit = True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if isinstance(features_selection,tuple):
        features = [fe for fe in features_selection[1]]
    else:
        features = [fe for fe in features_selection]
    selected_numeric_cols = stocks_df[features].select_dtypes(include=numerics).columns.tolist()

    stock_X_raw = stocks_df[stocks_df[ENTITY] == stock_name]

    if y_col in selected_numeric_cols:
        stock_X_raw_keep = stock_X_raw[[ENTITY]]
    else:
        stock_X_raw_keep = stock_X_raw[[ENTITY, y_col]]
    stock_X = stock_X_raw[selected_numeric_cols]  # TODO check id still index
    stock_X_prep = [stock_X]
    if finance_features:
        stock_X_finance = stock_X.pct_change()
        stock_X_finance.iloc[0] = 0
        stock_X_finance_df = combine_df(stock_X_finance.values, "_proc", stock_X.columns, TIME, stock_X.index)
        stock_X_finance_df['rsi'] = rsiFunc(stock_X[TARGET])
        stock_X_finance_df['MACD'] = computeMACD(stock_X[TARGET])[2]
        stock_X_prep.append(stock_X_finance_df)

    if to_fit:
        normalization.fit(stock_X)
    stock_X_norm = normalization.transform(stock_X)
    stock_X_norm_df = combine_df(stock_X_norm, "_norm", stock_X.columns, TIME, stock_X.index)
    if y_col in stock_X.columns:
        stock_X_norm_df = stock_X_norm_df.rename(columns={y_col + "_norm": TARGET_PREP})
    else:
        raise "target column not in data will error in evaluation"

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

    features_names = stock_X_prep_df.columns
    stock_X_prep_df = pd.merge(stock_X_raw_keep, stock_X_prep_df, left_index=True, right_index=True)
    return stock_X_prep_df, normalization, transformation, features_names


def calculate_features_all_stocks(path, features_selection, finance_features, force, normalization, prev_stocks_names,
                                  train_X, transformation, y_col):
    print "calc features all stocks"
    features_names_file_path = path + 'features_names.pkl'
    data_file_path = path + 'stocks_processed.csv'

    if (not os.path.isfile(data_file_path)) or force:
        all_stocks_processed = []
        features_names = []
        for stock in prev_stocks_names:
            train_stock_X, _, _, features_names = preprocess_stock_features(train_X, stock,features_selection, finance_features, normalization, transformation,y_col,)
            all_stocks_processed.append(train_stock_X)
        all_stocks_processed_df = pd.concat(all_stocks_processed)
        all_stocks_processed_df.to_csv(data_file_path)
        with open(features_names_file_path, 'wb') as f:
            pickle.dump(features_names.tolist(), f)
    with open(features_names_file_path, 'rb') as f:
        features_names = pickle.load(f)
    all_stocks_processed_df = pd.read_csv(data_file_path)
    return all_stocks_processed_df, features_names



def prepare_rolling_periods_for_top_stocks(data_period, stock_to_compare,
                                           start_period_train, end_period_train, start_period_test, end_period_test,
                                           features_selection, finance_features, normalization, transformation, to_pivot, \
                                           k, select_k_func, similarity_col, similarity_func, window_len, slide, weighted_sampleing, y_col, next_t,
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

    #select relevant time frames data
    train_X_all_prev_periods = data_period[(data_period[TIME] < end_period_train)]
    train_X = data_period[(start_period_train <= data_period[TIME]) & (data_period[TIME] < end_period_train)]
    test_X = data_period[(start_period_test <= data_period[TIME]) & (data_period[TIME] < end_period_test)]

    #calc similar stock on all previous data with atleast window len amount of recoeds
    count_stock = train_X_all_prev_periods.groupby([ENTITY]).count()[TIME].reset_index()
    prev_stocks_names = count_stock[count_stock[TIME] > window_len + max(next_t) + slide*15][ENTITY].tolist()

    file_name = "before_" + str(str(end_period_train))
    #calculate_features for all stocks
    data_file_path = os.path.join(data_path, file_name )
    all_stocks_processed_df, features_names = calculate_features_all_stocks(data_file_path, features_selection, finance_features, force, normalization, prev_stocks_names,
                                                            train_X, transformation, y_col)

    #calculate_similarity
    similarity_file_path = os.path.join(similarity_path, file_name + ".pkl")
    similarities = calculate_similarity_all_stocks(train_X_all_prev_periods, stock_to_compare, prev_stocks_names,
                                                   similarity_func, similarity_file_path, split_time = str(end_period_train), force=force, y_col = similarity_col)
    top_stocks = select_k_func(prev_stocks_names, similarities, k)
    #normalize similarity
    stocks_val = list(top_stocks.values())
    top_stock_w = {}
    for stock_k, v in top_stocks.items():
        if stock_k != stock_to_compare:
            top_stock_w[stock_k] = abs(float(v)-max(stocks_val)) / max(stocks_val)

    train_windows_x, train_windows_y, train_price, test_windows_x, test_windows_y, test_price = [], [], [], [], [], []

    # prepare train
    train_stock_to_compare_X, normalization_f, transformation_f,_ = preprocess_stock_features(train_X,stock_to_compare ,
                                                                                   features_selection, finance_features,
                                                                                   normalization, transformation, y_col)
    stock_train_windows_x, stock_train_windows_y, _ = prepare_stock_windows(train_stock_to_compare_X,features_names, window_len, slide, next_t,
                                                                            to_pivot, y_col)
    train_windows_x.append(stock_train_windows_x)
    train_windows_y.append(stock_train_windows_y)

    # prepare test
    test_stock_to_compare_X, _, _, _ = preprocess_stock_features(test_X, stock_to_compare, features_selection,
                                                     finance_features, normalization_f,
                                                     transformation_f, y_col,to_fit=False)

    stock_test_windows_x, stock_test_windows_y, stock_prices_test = prepare_stock_windows(test_stock_to_compare_X,features_names,
                                                                                          window_len, slide, next_t,
                                                                                          to_pivot, y_col)
    test_windows_x.append(stock_test_windows_x)
    test_windows_y.append(stock_test_windows_y)
    test_price.append(stock_prices_test)

    #prepare windows per stock
    for stock_name in top_stock_w.keys():
        stock_X = all_stocks_processed_df[all_stocks_processed_df[ENTITY] == stock_name]
        stock_train_windows_x, stock_train_windows_y, _ = prepare_stock_windows(stock_X,features_names, window_len, slide,
                                                                                next_t,
                                                                                to_pivot, y_col)
        if weighted_sampleing:
            np.random.seed(0)
            msk = np.random.rand(len(stock_train_windows_x)) < top_stock_w[stock_name]
            stock_train_windows_x = stock_train_windows_x[msk]
            stock_train_windows_y = stock_train_windows_y[msk]
        train_windows_x.append(stock_train_windows_x)
        train_windows_y.append(stock_train_windows_y)

    return pd.concat(train_windows_x), pd.concat(train_windows_y),\
           pd.concat(test_windows_x), pd.concat(test_windows_y), pd.concat(test_price),\
           top_stocks, features_names



def prepare_folds(data_period, stock_to_compare,n_folds,features_selection,finance_features,normalization,transformation,to_pivot,\
    k,select_k_func,similarity_col, similarity_func,window_len,slide,weighted_sampleing,y_col,next_t, data_path, similarity_path , force):
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
    period_len = int(abs(len(stock_times)/n_folds))
    folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test, folds_topk = [], [], [], [], [], []

    # for each fold
    for f in range(n_folds-1):
        # define rolling times by the tagets period
        start_period_train = stock_times[f*period_len]
        end_period_train = stock_times[f*period_len + period_len]
        start_period_test = stock_times[f*period_len + period_len]
        if f == n_folds-1 :
            end_period_test = stock_times[-1]
        else:
            end_period_test = stock_times[f*period_len + period_len + period_len-1]

        train_windows_x, train_windows_y,\
        test_windows_x, test_windows_y, test_price,\
        top_stocks, features_names = \
            prepare_rolling_periods_for_top_stocks(data_period, stock_to_compare,
                                                   start_period_train, end_period_train, start_period_test, end_period_test,
                                                   features_selection, finance_features, normalization, transformation, to_pivot, \
                                                   k, select_k_func, similarity_col, similarity_func, window_len, slide, weighted_sampleing, y_col, next_t
                                                   , data_path, similarity_path , force)

        folds_X_train.append(train_windows_x), folds_Y_train.append(train_windows_y), folds_X_test.append(test_windows_x)
        folds_Y_test.append(test_windows_y), folds_price_test.append(test_price), folds_topk.append(top_stocks)

    return [folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test], folds_topk, features_names


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
        #go long
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

        #go short
        if  predicted_price[i]<0:
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
                   features_names, model_class,model_args):
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
    #iterate folds
    for f in range(len(folds_X_train)):
        features = [(features_name, wl) for wl in range(window_len) for features_name in features_names]
        X_train = folds_X_train[f][features]
        X_test = folds_X_test[f][features]
        price_test = folds_price_test[f][TARGET]

        # iterate future value to predict
        for t in future_ys:
            y_train = folds_Y_train[f][t].values
            y_test = folds_Y_test[f][t].values

            model = model_class(**model_args)
            #todo - if pivot
            #prepare data for classification fit and evaluation
            # lb = LabelBinarizer()
            X_train_curr_price_prep = X_train[(TARGET_PREP, window_len - 1)].tolist()
            y_train_binary = np.sign(y_train - X_train_curr_price_prep)
            y_train_binary = [1 if x == 0 else x for x in y_train_binary]
            # lb.fit(y_train_binary)

            X_test_curr_price_prep = X_test[(TARGET_PREP, window_len - 1)].tolist()
            y_test_binary = np.sign(y_test - X_test_curr_price_prep)
            y_test_binary = [1 if x == 0 else x for x in y_test_binary]

            if isinstance(model, RegressorMixin):
                model.fit(X_train, y_train)
                y_preds_val = model.predict(X_test)
                y_preds_binary = np.sign(y_preds_val - X_test_curr_price_prep)
                y_preds_binary = [1 if x == 0 else x for x in y_preds_binary]
            else:
                model.fit(X_train, y_train_binary)
                y_preds_binary = model.predict(X_test)

            fold_eval = {}
            fold_eval["fold"] = f
            fold_eval["model"] = model_class.__name__
            fold_eval["next_t"] = t

            eval_values = pd.DataFrame()
            eval_values['curr_price'] = price_test
            eval_values['preds'] = y_preds_binary
            eval_values['y'] = y_test_binary
            # eval_values['curr_price2'] = folds_price_test[f][t].values
            for k1, v in fold_eval.items():
                eval_values[k1] = v

            evals = dict(fold_eval)
            evals['accuracy_score'] = accuracy_score(y_test_binary, y_preds_binary)
            evals['f1_score'] = f1_score(y_test_binary, y_preds_binary, average  = 'macro')
            evals['precision_score'] = precision_score(y_test_binary, y_preds_binary, average='macro')

            if not isinstance(model, RegressorMixin):
                #y_proba = model.predict_proba(X_test)
                evals['roc_auc_score'] = roc_auc_score(y_test_binary, y_preds_binary, average='macro')
            else:
                evals['roc_auc_score'] = 0

            evals["long_short_profit"], eval_values["long_short_profit"] = long_short_profit_evaluation(price_test, y_preds_binary)
            evals["sharp_ratio"] = np.mean(eval_values["long_short_profit"]) / (np.std(eval_values["long_short_profit"]) + 0.0001)

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


def save_evaluations(evaluations, results_path, folds_topk, processing_params,windowing_params):

    model_evals = []
    model_values_evals = []
    model_params = {}
    model_params.update(processing_params)
    model_params.update(windowing_params)


    for model_i in range(len(evaluations)):
        model_eval_df = evaluations[model_i][0]
        model_values_eval = evaluations[model_i][1]
        for k1,v1 in model_params.items():
            if isinstance(v1,tuple):
                v1 = v1[0]
            model_eval_df[k1] = v1
            model_values_eval[k1] = v1
        model_evals.append(model_eval_df)
        model_values_evals.append(model_values_eval)


    pd.concat(model_evals).to_csv(os.path.join(results_path, 'models_evaluations.csv'), mode='a')
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


def run_experiment(data_period, stock_to_compare,n_folds,features_selection,finance_features,normalization,transformation,to_pivot,\
    k,select_k_func,similarity_col, similarity_func,window_len,slide,weighted_sampleing,y_col,next_t, models, models_arg, force):
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
    data_path = os.path.join(experiment_path,stock_to_compare,'data', 'fs-' +features_selection[0] + '_finance_fe-' + str(finance_features) + "_norm-" + normalization
                                        + "_transform-" + transformation,'fold-')
    similarity_path = os.path.join(experiment_path,stock_to_compare,'similarity', 'func-' +similarity_func + '_col-' + similarity_col,'fold-' )
    if not os.path.exists(os.path.dirname(experiment_path)):
        os.makedirs(experiment_path)
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(data_path)
    if not os.path.exists(os.path.dirname(similarity_path)):
        os.makedirs(similarity_path)

    df_stocks = get_data(data_period)

    # preapare data with slide, time window as fold -> test + train X features + targets
    folds_loaded, folds_topk,features_names = prepare_folds(df_stocks, stock_to_compare,n_folds,features_selection,finance_features,
                                                            normalizations[normalization],transformations[transformation],to_pivot,k,select_k_funcs[select_k_func],similarity_col,
                                             similarity_funcs[similarity_func],window_len,slide,weighted_sampleing,y_col,next_t,
                                             data_path, similarity_path, force)

    folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test = \
        folds_loaded[0], folds_loaded[1], folds_loaded[2], folds_loaded[3], folds_loaded[4]

    evaluations = [evaluate_model(window_len, folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test,
                                  features_names, model,models_arg[model.__name__])
                   for model in models]

    processing_params = {'data_period' : data_period,
                         'stock_to_compare' : stock_to_compare,
                         'n_folds' : n_folds,
                         'features_selection' : features_selection,
                         'finance_features' : finance_features,
                         'normalization' : normalization,
                         'transformation' : transformation,
                         'k' : k,
                         'select_k_func' : select_k_func,
                         'similarity_col' : similarity_col,
                         'similarity_func' : similarity_func}

    windowing_params = {'window_len': window_len,
                         'slide': slide,
                         'weighted_sampleing' : weighted_sampleing,
                         'y_col' : y_col}
                         #'next_t' : next_t}

    save_evaluations(evaluations, experiment_path, folds_topk, processing_params,windowing_params)
    return experiment_path


transformations  = {'None' : None,
                    'SAX': SAX(),
                    'PCA' : PCA()}
normalizations  = {'Standard' : StandardScaler()}

select_k_funcs  = {'get_random_k' : get_random_k,
                    'get_top_k': get_top_k}

similarity_funcs = {'sax' : compare_sax,
                    'model_based_RFR': model_bases_distance(RandomForestRegressor(n_estimators = 50, random_state=0)),
                    'euclidean' : apply_euclidean,
                    'dtw' : apply_dtw,
                    'pearson' : apply_pearson
                    }
def main():

    #regular experiment
    experiment_params = {
        'data_period': ['5yr'],
        # tech, finance, service, health, consumer, Industrial
        'stock_to_compare': ["GOOGL","JPM", "DIS", "JNJ", "MMM", "KO", "GE"],
        'n_folds': [6],
        'features_selection': [ ('only_close', [u'Close'])],
        # ('full_features' ,[u'Open',u'High',u'Low',u'Close',u'Volume']),
       # ('open_close_volume', [u'Open', u'Close', u'Volume'])],
        'finance_features': [True],# False],
        'normalization': ['Standard'],
        'transformation': ['None'],# 'SAX', 'PCA'],
        'k': [10, 25],
        'select_k_func': ['get_top_k'],#, 'get_random_k'],
        'similarity_col': ['Close'],
        'similarity_func': ['model_based_RFR','pearson','sax','euclidean','dtw'],#],
        'window_len': [10],  # , 0, 20],
        'slide': [1],  # , 3, 5, 10],
        'weighted_sampleing': [False,  True],
        'y_col': ['Close'],
    }

    experiment_static_params = \
        {
            'next_t': [1, 3, 7],
            'to_pivot': True,
            'models': [RandomForestClassifier,RandomForestRegressor, GradientBoostingRegressor,GradientBoostingClassifier],
            'models_arg' : {RandomForestClassifier.__name__: {'n_estimators': 100, 'random_state' : 0},
                            RandomForestRegressor.__name__: {'n_estimators': 100, 'random_state' : 0},
                            GradientBoostingClassifier.__name__: {'learning_rate': 0.02, 'random_state': 0},
                            GradientBoostingRegressor.__name__: {'learning_rate': 0.02,'random_state' : 0}},
            'force' : False
        }

    experiments = get_index_product(experiment_params)
    for experiment in experiments:
        print "run experiment: " + str(experiment)
        experiment.update(experiment_static_params)
        results_path = run_experiment(**experiment)

    pd.DataFrame().to_csv(os.path.join(results_path, 'models_values_evaluations.csv'), mode='a')
    pd.DataFrame().to_csv(os.path.join(results_path, 'models_evaluations.csv'), mode='a')
    pd.DataFrame().to_csv(os.path.join(results_path, 'similarity_evaluations.csv'), mode='a')


    # experiment_static_params = \
    #     {
    #         'next_t': [1, 3, 7],
    #         'to_pivot': False
    #        # 'models': [LSTM_stock],
    #         #'models_arg': {LSTM_stock.__name__: {},
    #                        #}
    #     }
    #
    # experiments = get_index_product(experiment_params)
    # for experiment in experiments:
    #     print "run experiment: " + str(experiment)
    #     experiment.update(experiment_static_params)
    #
        #run_experiment(**experiment)

main()



