import random

import pandas as pd
import numpy as np
import os
from dtw import dtw
import pickle

from sklearn.base import RegressorMixin
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

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


home_path = 'C:\\Users\\Lior\\StockSimilarity'

TIME = 'Date'
ENTITY = 'Name'
TARGET = 'Close'
FEATURES = ['Close']

def get_data(data_period):
    file_path = os.path.join(home_path, 'sandp500','all_stocks_' + data_period + '.csv')
    all_stocks = pd.read_csv(file_path)
    all_stocks[TIME] = pd.to_datetime(all_stocks[TIME], format='%Y%m%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
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
        #TODO save static model
        self.model = model

    def stock_norm_prep(self,stock_X):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_cols = stock_X.select_dtypes(include=numerics).columns.tolist()
        numeric_cols_prep = [s + "_prep" for s in numeric_cols]

        mms = MinMaxScaler()

        train_stock_X_names_df = stock_X[[ENTITY, TARGET]].reset_index(drop=True)
        stock_X_norm = mms.fit_transform(stock_X[numeric_cols])
        train_stock_X_norm_df = pd.DataFrame(stock_X_norm, columns=numeric_cols_prep).reset_index(drop=True)
        train_stock_X_prep = pd.merge(train_stock_X_names_df, train_stock_X_norm_df, left_index=True, right_index=True)

        return train_stock_X_prep

    def fit(self, stock_X):
        stock_X_prep = self.stock_norm_prep(stock_X)
        x, y, _ = prepare_stock_windows(stock_X_prep, 10,1, [1], [TARGET+ '_prep'])
        self.model.fit(x,y)

    def apply_distance(self, stock_X):
        """
        apply euclidean distance between 2 stocks
        :param stock1:
        :param stock2:
        :return:
        """
        print len(stock_X)
        stock_X_prep = self.stock_norm_prep(stock_X)
        x, y, _ = prepare_stock_windows(stock_X_prep, 10, 1, [1], [TARGET + '_prep'])

        preds = self.model.predict(x)

        return mean_squared_error(y,preds)

    @property
    def __name__(self): return "model_bases_distance"


def get_similarity(df_stocks, stock_to_compare, stock_names, similarity_func, experiment_path, force = False, split_time = ""):
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
    file_name = "similarities_" + stock_to_compare + "_" + similarity_func.__name__ + "split_" + str(split_time) + ".pkl"
    similarities_path = os.path.join(experiment_path, file_name)

    if (not os.path.isfile(similarities_path)) or force:
        if isinstance(similarity_func, model_bases_distance):
            stock_X = df_stocks[df_stocks[ENTITY] == stock_to_compare]
            similarity_func.fit(stock_X)
            similarities = [
                similarity_func.apply_distance(df_stocks[df_stocks[ENTITY] == stock_name])
                for stock_name in stock_names
                ]
        else:
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
def prepare_stock_windows(stock_X, window_len, slide, next_t,column_to_window, to_pivot= True):
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
    while i < len(stock_X[ENTITY]) - window_len - max(next_t):
        y_ti = i + window_len + np.asarray(next_t)
        stock_X_window = stock_X[i:i + window_len]
        stock_X_window.insert(0, 't', range(window_len))
        if to_pivot:
            stock_X_window_flat = stock_X_window[column_to_window + [ENTITY] + ['t']].pivot(index = ENTITY,columns = 't').iloc[0].to_dict()
            stock_X_window_flat[TIME] = i
            X_stocks_windows.append(stock_X_window_flat)
        else:
            stock_X_window_df = stock_X_window[column_to_window]
            X_stocks_windows[stock_name + str(i)] = stock_X_window_df

        next_y = np.array(stock_X[TARGET + "_prep"].tolist())[y_ti]
        y_vals = next_y.tolist()
        y_ = {}
        for c in range(len(y_column_names)):
            y_[str(y_column_names[c])] = y_vals[c]
        Y_.append(y_)

        #the current window last price
        y_price = {}
        y_price[TARGET] = np.array(stock_X[TARGET].tolist())[i + window_len].tolist()
        y_price[TIME] = i
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
        X = pd.DataFrame.from_records(X_stocks_windows, index=[TIME])
    else:
        X = pd.concat(X_stocks_windows)
    return X ,\
           pd.DataFrame(Y_),\
           pd.DataFrame(Y_price)


def prepare_rolling_periods_for_top_stocks(df_stocks, stock_to_compare,
                                           start_period_train, end_period_train, start_period_test, end_period_test,
                                           similarity_func, select_k_func, k, preprocessing_pipeline, experiment_path,
                                           window_len, slide, next_t, weighted_sampleing, force = False, to_pivot = True):
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
    train_X_all_prev_periods = df_stocks[(df_stocks[TIME] < end_period_train)]
    train_X = df_stocks[(start_period_train <= df_stocks[TIME]) & (df_stocks[TIME] < end_period_train)]
    test_X = df_stocks[(start_period_test <= df_stocks[TIME]) & (df_stocks[TIME] < end_period_test)]

    del train_X[TIME]
    del test_X[TIME]

    train_windows_x, train_windows_y, train_price, test_windows_x, test_windows_y, test_price = [], [], [], [], [],[]

    #calc similar stock on all previous data with atleast window len amount of recoeds
    count_stock = train_X_all_prev_periods.groupby([ENTITY]).count()[TIME].reset_index()
    prev_stocks_names = count_stock[count_stock[TIME] > window_len][ENTITY].tolist()
    similarities = get_similarity(train_X_all_prev_periods, stock_to_compare, prev_stocks_names,
                                  similarity_func, experiment_path, split_time = str(end_period_train), force=force)
    top_stocks = select_k_func(prev_stocks_names, similarities, k)
    stocks_val = list(top_stocks.values())
    top_stock_w = {}
    for stock_k, v in top_stocks.items():
        if stock_k != stock_to_compare:
            top_stock_w[stock_k] = abs(float(v)-max(stocks_val)) / max(stocks_val)
        else:
            top_stock_w[stock_to_compare] = 1.0

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = train_X.select_dtypes(include=numerics).columns.tolist()
    numeric_cols_prep = [s + "_prep" for s in numeric_cols]

    ### prepare test only from the stock to compare
    test_stock_X = test_X[test_X[ENTITY] == stock_to_compare].reset_index(drop=True)
    test_stock_X_names_df = test_stock_X[[ENTITY, TARGET]]
    test_stock_X_norm = preprocessing_pipeline.fit_transform(test_stock_X[numeric_cols])
    test_stock_X_norm_df = pd.DataFrame(test_stock_X_norm, columns=numeric_cols_prep)
    test_stock_X_prep = pd.merge(test_stock_X_names_df, test_stock_X_norm_df, left_index=True, right_index=True)

    stock_test_windows_x, stock_test_windows_y, stock_prices_test = prepare_stock_windows(test_stock_X_prep,
                                                                                          window_len, slide, next_t,
                                                                                          numeric_cols_prep, to_pivot)
    test_windows_x.append(stock_test_windows_x)
    test_windows_y.append(stock_test_windows_y)
    test_price.append(stock_prices_test)

    #prepare windows per stock
    for stock_name in top_stocks.keys():
        train_stock_X = train_X[train_X[ENTITY] == stock_name].reset_index(drop = True)
        train_stock_X_names_df = train_stock_X[[ENTITY, TARGET]]
        train_stock_X_norm =  preprocessing_pipeline.fit_transform(train_stock_X[numeric_cols])
        train_stock_X_norm_df = pd.DataFrame(train_stock_X_norm, columns=numeric_cols_prep)
        train_stock_X_prep = pd.merge(train_stock_X_names_df, train_stock_X_norm_df, left_index=True, right_index=True)

        stock_train_windows_x, stock_train_windows_y, stock_prices_train = prepare_stock_windows(train_stock_X_prep,
                                                                                                 window_len, slide,
                                                                                                 next_t,
                                                                                                 numeric_cols_prep,to_pivot)

        stock_train_windows_x_s = stock_train_windows_x
        stock_train_windows_y_s = stock_train_windows_y
        if weighted_sampleing:
            #_, _, stock_train_windows_x_s, stock_train_windows_y_s = train_test_split(stock_train_windows_x, stock_train_windows_y, test_size = top_stock_w[stock_name], random_state = 42)
            np.random.seed(0)
            msk = np.random.rand(len(stock_train_windows_x)) < top_stock_w[stock_name]
            stock_train_windows_x_s = stock_train_windows_x[msk]
            stock_train_windows_y_s = stock_train_windows_y[msk]

        train_windows_x.append(stock_train_windows_x_s)
        train_windows_y.append(stock_train_windows_y_s)

    return pd.concat(train_windows_x), pd.concat(train_windows_y),\
           pd.concat(test_windows_x), pd.concat(test_windows_y), pd.concat(test_price),\
           top_stocks


def prepare_folds(df_stocks, stock_to_compare, window_len, slide,next_t, n_folds, experiment_path, force,
                  similarity_func, select_k_func, k, preprocessing_pipeline,weighted_sampleing,to_pivot= True):
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
    stock_times = df_stocks[df_stocks[ENTITY] == stock_to_compare][TIME].tolist()
    period_len = int(abs(len(stock_times)/n_folds))
    folds_X_train = []
    folds_Y_train = []

    folds_X_test = []
    folds_Y_test = []
    folds_price_test = []
    folds_topk = []
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
        top_stocks = \
            prepare_rolling_periods_for_top_stocks(df_stocks, stock_to_compare,
                                                   start_period_train, end_period_train, start_period_test, end_period_test,
                                                   similarity_func, select_k_func, k, preprocessing_pipeline, experiment_path,
                                                   window_len, slide, next_t, weighted_sampleing, force, to_pivot)

        folds_X_train.append(train_windows_x)
        folds_Y_train.append(train_windows_y)
        folds_X_test.append(test_windows_x)
        folds_Y_test.append(test_windows_y)
        folds_price_test.append(test_price)

        folds_topk.append(top_stocks)

    return [folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test], folds_topk


def statistical_targeting(curr_price, future_price):
    """
    prepare continues data for classification task - up = 1, down =-1 or same= 0
    :param curr_price:
    :param future_price:
    :return:
    """
    diff_prices = future_price - curr_price
    avg = np.mean(diff_prices)
    std = np.std(diff_prices)

    w = 2.0
    target = [0]
    while len(set(target)) != 3:
        target = [np.sign(diff_price) if abs(diff_price) > (avg + w*std) else 0 for diff_price in diff_prices]
        w = w * 0.5
    print w
    return target


def diffrence_targeting(curr_price, future_price):
    """
    prepare continues data for classification task - up = 1, down =-1 or same= 0
    :param curr_price:
    :param future_price:
    :return:
    """
    diff_prices = np.sign(future_price - curr_price)
    return diff_prices

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
                   features_selection, model_class,model_args, evaluation_methods, profit_methods, target_discretization):
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
        features = [(fe + '_prep', wl) for fe in features_selection[1] for wl in range(window_len)]
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
            lb = LabelBinarizer()
            X_train_curr_price_prep = X_train[(TARGET + "_prep", window_len - 1)].tolist()
            y_train_binary = np.sign(y_train - X_train_curr_price_prep)
            lb.fit(y_train_binary)

            X_test_curr_price_prep = X_test[(TARGET + "_prep", window_len - 1)].tolist()
            y_test_binary = np.sign(y_test - X_test_curr_price_prep)

            # lb = LabelBinarizer()
            # if classifier then descrtisize target value
            # if not isinstance(model, RegressorMixin):
            #     X_train_curr_price = X_train[(TARGET + "_prep", window_len - 1)].tolist()
            #     y_train = target_discretization(X_train_curr_price, y_train)
            #
            #     X_test_curr_price = X_test[(TARGET + "_prep", window_len - 1)].tolist()
            #     y_test = target_discretization(X_test_curr_price, y_test)
            #     lb.fit(y_train)
            #     y_test_multi = lb.transform(y_test)

            #if regression then train on original values and transform to binary

            if isinstance(model, RegressorMixin):
                model.fit(X_train, y_train)
                y_preds_val = model.predict(X_test)
                y_preds_binary = np.sign(y_preds_val - X_test_curr_price_prep)
            else:
                model.fit(X_train, y_train_binary)
                y_preds_binary = model.predict(X_test)

            fold_eval = {}
            fold_eval["fold"] = f
            fold_eval["model"] = model_class.__name__
            fold_eval["next_t"] = t

            # for evaluation_method in evaluation_methods:
            #     eval_error = dict(fold_eval)
            #     eval_error["method"] = evaluation_method.__name__
            #     eval_error["value"] = evaluation_method(y_test,y_preds)
            #     evaluations.append(eval_error)

            # if classifier then evaluate with relevant measures

            # for evaluation_method in [roc_auc_score]:
            #     y_proba = model.predict_proba(X_test)
            #     for target in range(len(lb.classes_)):
            #         eval_error = dict(fold_eval)
            #         eval_error["method"] = evaluation_method.__name__ + "_class_" + str(lb.classes_[target])
            #         eval_error["value"] = evaluation_method(y_test_binary[:, target], y_proba[:, target])
            #         evaluations.append(eval_error)
            for evaluation_method in [accuracy_score]: #,log_loss, f1_score]:
                eval_error = dict(fold_eval)
                eval_error["method"] = evaluation_method.__name__
                #eval_error["value"] = evaluation_method(lb.transform(y_test_binary), lb.transform(y_preds_binary))
                eval_error["value"] = evaluation_method(y_test_binary, y_preds_binary)
                evaluations.append(eval_error)

            eval_values = pd.DataFrame()

            eval_values['curr_price'] = price_test
            eval_values['preds'] = y_preds_binary
            eval_values['y'] = y_test_binary
            #eval_values['curr_price2'] = folds_price_test[f][t].values
            for k1, v in fold_eval.items():
                eval_values[k1] = v
            for profit_method in profit_methods:
                eval_error = dict(fold_eval)
                eval_error["method"] = profit_method.__name__
                eval_error["value"], eval_values[profit_method.__name__] = profit_method(price_test,y_preds_binary)
                eval_error["sharp_ratio"] = np.mean(eval_values[profit_method.__name__]) / (np.std(eval_values[profit_method.__name__]) + 0.0001)
                evaluations.append(eval_error)

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


def save_evaluations(data_period, evaluations, experiment_path, features_selection, folds_topk, k, n_folds,
                     preprocessing_pipeline, select_k_func, similarity_func, slide, stock_to_compare,
                     weighted_sampleing, window_len):

    evaluations_path = [os.path.join(experiment_path, 'models_evaluations.csv'),
                        os.path.join(experiment_path, 'models_values_evaluations.csv')]

    for model_i in range(len(evaluations)):
        #evaluations[model_i][0] = pd.concat(evaluations[model_i][0])
        for i in range(len(evaluations[model_i])):
            model_eval = evaluations[model_i][i]
            model_eval['data_period'] = data_period
            model_eval['stock_to_compare'] = stock_to_compare
            model_eval['preprocessing_pipeline'] = " ".join(list(preprocessing_pipeline.named_steps.keys()))
            model_eval['weighted_sampleing'] = weighted_sampleing
            model_eval['similarity_func'] = similarity_func.__name__
            model_eval['k'] = k
            model_eval['select_k_func'] = select_k_func.__name__
            model_eval['window_len'] = window_len
            model_eval['slide'] = slide
            model_eval['features_selection'] = features_selection[0]
            model_eval.to_csv(evaluations_path[i], mode='a')

    similar_stock_eval_folds = []
    for f in range(n_folds - 1):
        similar_stock_eval_fold = {}
        similar_stock_eval_fold['fold'] = f
        similar_stock_eval_fold['data_period'] = data_period
        similar_stock_eval_fold['stock_to_compare'] = stock_to_compare
        similar_stock_eval_fold['preprocessing_pipeline'] = preprocessing_pipeline.named_steps.keys
        similar_stock_eval_fold['similarity_func'] = similarity_func.__name__
        similar_stock_eval_fold['k'] = k
        similar_stock_eval_fold['select_k_func'] = select_k_func.__name__

        for name, distance in folds_topk[f].items():
            stock_f = dict(similar_stock_eval_fold)
            stock_f['name'] = name
            stock_f['distance'] = distance
            similar_stock_eval_folds.append(stock_f)

    sim_eval_path = os.path.join(experiment_path, 'similarity_evaluations.csv')
    pd.DataFrame(similar_stock_eval_folds).to_csv(sim_eval_path, mode='a')


def run_experiment(data_period, stock_to_compare, preprocessing_pipeline,features_selection,
                   similarity_func, k, select_k_func,
                   window_len, slide, next_t, n_folds,
                   models, models_arg, target_discretization,weighted_sampleing, to_pivot = True, force=False):
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
    experiment_path = os.path.join(home_path,'experiments', data_period)
    if not os.path.exists(os.path.dirname(experiment_path)):
        os.makedirs(experiment_path)

    df_stocks = get_data(data_period) #.head(8000)

    # preapare data with slide, time window as fold -> test + train X features + targets
    folds_loaded, folds_topk = prepare_folds(df_stocks, stock_to_compare,
                                 window_len, slide,next_t, n_folds,
                                 experiment_path, force,
                                 similarity_func, select_k_func, k,
                                 preprocessing_pipeline,weighted_sampleing,to_pivot)

    folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test = \
        folds_loaded[0], \
           folds_loaded[1], \
           folds_loaded[2], \
           folds_loaded[3], \
           folds_loaded[4]

    evaluations = [evaluate_model(window_len, folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, folds_price_test,
                                  features_selection, model,
                                  models_arg[model.__name__],
                                  [mean_squared_error], [simple_profit_evaluation, long_short_profit_evaluation],
                                  target_discretization) for model in models]

    save_evaluations(data_period, evaluations, experiment_path, features_selection, folds_topk, k, n_folds,
                     preprocessing_pipeline, select_k_func, similarity_func, slide, stock_to_compare,
                     weighted_sampleing, window_len)


def main():
    experiment_params = {
        'data_period': ['1yr'],
        'n_folds' : [6],
        # tech, finance, service, health, consumer, Industrial
        'stock_to_compare' : ["GOOGL"], #, "JPM", "DIS", "JNJ", "MMM", "KO", "GE"],
        'k' : [10, 1 , 25],
        'select_k_func' : [get_top_k, get_random_k],
        'window_len' : [10],#, 5, 20],
        'slide' : [1],#, 3, 5, 10],
        'preprocessing_pipeline' : [Pipeline([('minmax_normalize', MinMaxScaler())])],
        'similarity_func' : [model_bases_distance(RandomForestRegressor(100,random_state = 0)), apply_euclidean,apply_dtw, apply_pearson],
        'weighted_sampleing': [True],# False],
        'target_discretization' : [statistical_targeting],
        'features_selection': [#('full_features' ,[u'Open',u'High',u'Low',u'Close',u'Volume']),
                     ('only_close', [u'Close']),
                     ('open_close_volume', [u'Open', u'Close', u'Volume'])]
    }

    experiment_static_params = \
        {
            'next_t': [1, 3, 7],
            'to_pivot': True,
            'models': [RandomForestClassifier, RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier],
            'models_arg' : {RandomForestClassifier.__name__: {'n_estimators': 100, 'random_state' : 0, 'random_state' : 0},
                            RandomForestRegressor.__name__: {'n_estimators': 100, 'random_state' : 0, 'random_state' : 0},
                            GradientBoostingClassifier.__name__: {'learning_rate': 0.02, 'random_state': 0, 'random_state' : 0},
                            GradientBoostingRegressor.__name__: {'learning_rate': 0.02,'random_state' : 0, 'random_state' : 0}}
        }

    experiments = get_index_product(experiment_params)
    for experiment in experiments:
        print "run experiment: " + str(experiment)
        experiment.update(experiment_static_params)
        run_experiment(**experiment)

    experiment_static_params = \
        {
            'next_t': [1, 3, 7],
            'to_pivot': False
           # 'models': [LSTM_stock],
            #'models_arg': {LSTM_stock.__name__: {},
                           #}
        }

    experiments = get_index_product(experiment_params)
    for experiment in experiments:
        print "run experiment: " + str(experiment)
        experiment.update(experiment_static_params)

        #run_experiment(**experiment)

main()



