

import  pandas as pd
import  numpy as np
import os
from dtw import dtw
import pickle

from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import itertools

home_path = 'C:\\Users\\Lior\\StockSimilarity'

TIME = 'Date'
ENTITY = 'Name'
TARGET = 'Close'
FEATURES = ['Close']

def get_data(data_period):
    file_path = os.path.join(home_path, 'sandp500\\all_stocks_' + data_period + '.csv')
    all_stocks = pd.read_csv(file_path)
    all_stocks[TIME] = pd.to_datetime(all_stocks[TIME])
    all_stocks = all_stocks.dropna(axis=0)
    return all_stocks


##################  Similarities calculation ######################
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


def get_similarity(df_stocks, stock_to_compare, stock_names, similarity_func, experiment_path, force = False):
    file_name = "similarities_" + stock_to_compare + "_" + similarity_func.__name__ + ".pkl"
    similarities_path = os.path.join(experiment_path, file_name)
    if (not os.path.isfile(similarities_path)) or force:
        similarities = [
            similarity_func(df_stocks[df_stocks[ENTITY] == stock_to_compare][TARGET].tolist(),
                          df_stocks[df_stocks[ENTITY] == stock_name][TARGET].tolist())
                for stock_name in stock_names
                ]
        if not os.path.exists(os.path.dirname(similarities_path)):
            os.makedirs(similarities_path)
        with open(similarities_path, 'wb') as f:
            pickle.dump(similarities, f)
    with open(similarities_path, 'rb') as f:
        similarities = pickle.load(f)
    return similarities


def get_top_k(stock_names, similarities, k):
    s = np.array(similarities)
    idx = np.argpartition(s, k)
    names_top_k = np.array(stock_names)[idx[:k]]

    return names_top_k


##################  Data preparation for Time Series #################
def prepare_stock_windows(X, stock_name, window, slide, next_t):
    stock_X = X[X[ENTITY] == stock_name]
    i = 0
    y_column_names = [TIME] + next_t
    X_stocks_windows = []
    Y_ = []
    while i < len(stock_X[TIME]) - window - max(next_t) :
        stock_X_window = stock_X[i:i + window]
        stock_X_window.insert(0, 't', range(window))
        stock_X_window_flat = stock_X_window[FEATURES + [ENTITY] + ['t']].pivot(index = ENTITY,columns = 't').iloc[0].to_dict()
        # stock_X_windows = stock_X[i:i + window].as_matrix()
        X_stocks_windows.append(stock_X_window_flat)
        y_ti = i + window + np.asarray(next_t)
        curr_time = stock_X_window.loc[stock_X_window[TIME].idxmax()][TIME]
        next_targets = np.array(stock_X[TARGET].tolist())[y_ti]
        #next_times = np.array(stock_X[TIME].tolist())[y_ti]
        y_vals = [curr_time] + next_targets.tolist()
        y_ = {}
        for c in range(len(y_column_names)):
            y_[str(y_column_names[c])] =  y_vals[c]
        Y_.append(y_)
        i += slide
    return pd.DataFrame(X_stocks_windows), pd.DataFrame(Y_)


def prepare_rolling_periods(df_stocks, start_period_train, end_period_train, start_period_test,
                          end_period_test, window_len, slide, next_t):
    train_X = df_stocks[(start_period_train <= df_stocks[TIME]) & (df_stocks[TIME] <= end_period_train)]
    test_X = df_stocks[(start_period_test <= df_stocks[TIME]) & (df_stocks[TIME] <= end_period_test)]
    train_windows_x, train_windows_y, test_windows_x, test_windows_y = [], [], [], []
    stock_names = df_stocks[ENTITY].unique()
    #prepare windows per stock
    for stock_name in stock_names:
        stock_train_windows_x, stock_train_windows_y = prepare_stock_windows(train_X, stock_name,window_len, slide,next_t)
        stock_test_windows_x, stock_test_windows_y = prepare_stock_windows(test_X, stock_name, window_len, slide, next_t)

        train_windows_x.append(stock_train_windows_x)
        train_windows_y.append(stock_train_windows_y)
        test_windows_x.append(stock_test_windows_x)
        test_windows_y.append(stock_test_windows_y)

    return pd.concat(train_windows_x), pd.concat(train_windows_y), pd.concat(test_windows_x), pd.concat(test_windows_y)


def prepare_folds(df_stocks, stock_to_compare, window_len, slide,next_t, n_folds, experiment_path, force):
    folds_path = os.path.join(experiment_path, stock_to_compare + "_w-" + str(window_len) + "_slide-" + str(slide) + "_f-" + str(window_len) + ".pkl")
    if (not os.path.isfile(folds_path)) or force:
        stock_times = df_stocks[df_stocks[ENTITY] == stock_to_compare][TIME].tolist()
        period_len = abs(len(stock_times)/n_folds)
        folds_X_train = []
        folds_Y_train = []
        folds_X_test = []
        folds_Y_test = []
        for f in range(n_folds-1):
            start_period_train = stock_times[f*period_len]
            end_period_train = stock_times[f*period_len + period_len]
            start_period_test = stock_times[f*period_len + period_len + 1]
            if f == n_folds-1 :
                end_period_test = stock_times[-1]
            else:
                end_period_test = stock_times[f*period_len + period_len + 1 + period_len]

            train_windows_x, train_windows_y, test_windows_x, test_windows_y = \
                prepare_rolling_periods(df_stocks, start_period_train, end_period_train, start_period_test,
                          end_period_test, window_len, slide, next_t)

            folds_X_train.append(train_windows_x)
            folds_Y_train.append(train_windows_y)
            folds_X_test.append(test_windows_x)
            folds_Y_test.append(test_windows_y)

        if not os.path.exists(os.path.dirname(folds_path)):
            os.makedirs(folds_path)
        with open(folds_path, 'wb') as f:
            pickle.dump([folds_X_train, folds_Y_train, folds_X_test, folds_Y_test], f)
    with open(folds_path, 'rb') as f:
        folds_loaded = pickle.load(f)

    return folds_loaded


def statistical_targeting(curr_price, future_price, range = 1, avg = None, std = None):
    diff_prices = future_price - curr_price
    if avg is None:
        avg = np.mean(diff_prices)
        std = np.std(diff_prices)
    target = [np.sign(diff_price) if abs(diff_price) > (avg + range*std) else 0 for diff_price in diff_prices]
    return target, {'range' : 1, 'avg' : None, 'std' : None}


################# Evaluation methods ################################
def simple_profit_evaluation(curr_price, predicted_price):
    in_position = False
    profit = 0
    last_buy = 0
    for i in range(len(curr_price)):
        if curr_price[i] < predicted_price[i] and not in_position:
            in_position = True
            last_buy = curr_price[i]
        if curr_price[i] > predicted_price[i] and in_position:
            in_position = False
            profit = profit + (curr_price[i]-last_buy)
    return profit


def long_short_profit_evaluation(curr_price, predicted_price):
    is_long = None
    profit = 0
    last_buy = 0

    for i in range(len(curr_price)):
        #go long
        if curr_price[i] < predicted_price[i]:
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
        if curr_price[i] > predicted_price[i]:
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


################# Experiments Executions ############################
def evaluate_model(window_len, folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, model_class,model_args, evaluation_methods, profit_methods, to_classification):
    future_ys = folds_Y_train[0].columns.tolist()
    future_ys.remove(TIME)
    evaluations = []
    for f in range(len(folds_X_train)):
        X_train = folds_X_train[f]
        X_test = folds_X_test[f]

        X_test_curr_price = X_test[(TARGET, window_len - 1)].tolist()
        X_train_curr_price = X_train[(TARGET, window_len - 1)].tolist()
        for t in future_ys:
            y_train = folds_Y_train[f][t].values
            y_test = folds_Y_test[f][t].values
            if not isinstance(model_class, RegressorMixin):
                y_train, apply_args = to_classification(X_train_curr_price, y_train)
                y_test, _ = to_classification(X_test_curr_price, y_test, **apply_args)

            model = model_class(**model_args)
            model.fit(X_train,y_train)
            y_preds = model.predict(X_test)
            fold_eval = {}
            fold_eval["fold"] = f
            fold_eval["model"] = model_class.__name__
            fold_eval["next_t"] = t
            for evaluation_method in evaluation_methods:
                eval_error = dict(fold_eval)
                eval_error["method"] = evaluation_method.__name__
                eval_error["value"] = evaluation_method(y_test,y_preds)
                evaluations.append(eval_error)
            for profit_method in profit_methods:
                eval_error = dict(fold_eval)
                eval_error["method"] = profit_method.__name__

                eval_error["value"] = profit_method(X_test_curr_price,y_preds)
                evaluations.append(eval_error)
    return pd.DataFrame(evaluations)


def get_index_product(params):
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


def run_experiment(data_period, stock_to_compare,similarity_func, k, select_k_func, window_len, slide, next_t, n_folds, models, models_arg,to_classification, force=False):
    experiment_path = os.path.join(home_path, data_period)
    df_stocks = get_data(data_period).head(8000)
    stock_names = df_stocks[ENTITY].unique()

    folds_loaded = prepare_folds(df_stocks, stock_to_compare, window_len, slide,next_t, n_folds, experiment_path, force)

    similarities = get_similarity(df_stocks, stock_to_compare, stock_names, similarity_func, experiment_path, force)
    names_top_k = select_k_func(stock_names, similarities, k)

    folds_loaded_top_k = [fold_loaded[fold_loaded[ENTITY].isin(names_top_k)] for fold_loaded in folds_loaded]
    folds_X_train, folds_Y_train, folds_X_test, folds_Y_test = folds_loaded_top_k[0], folds_loaded_top_k[1], folds_loaded_top_k[2], folds_loaded_top_k[3]

    evaluations = [evaluate_model(window_len, folds_X_train, folds_Y_train, folds_X_test, folds_Y_test, model,
                                  models_arg[model.__name__],
                                  [mean_squared_error], [simple_profit_evaluation, long_short_profit_evaluation],to_classification) for model in models]

    eval_path = os.path.join(experiment_path, 'evaluations.csv')
    pd.concat(evaluations).to_csv(eval_path)


def main():

    experiment_params = {
        'data_period': ['1yr'],
        'n_folds' : [5],
        # tech, finance, service, health, consumer, Industrial
        'stock_to_compare' : ["GOOGL", "JPM", "DIS", "JNJ", "MMM", "KO", "GE"],
        'k' : [1, 5, 10, 100],
        'select_k_func' : [get_top_k],
        'window_len' : [5, 10, 20],
        'slide' : [1, 3, 5, 10],
        'to_classification' : [statistical_targeting],
        'similarity_func' : [apply_dtw]
    }

    experiment_static_params =  {
        'next_t': [1, 3, 7],
        'models': [RandomForestRegressor, GradientBoostingRegressor],
        'models_arg' : {RandomForestRegressor.__name__: {'n_estimators': 100},
                   GradientBoostingRegressor.__name__: {'learning_rate': 0.02}}
    }

    experiments = get_index_product(experiment_params)
    for experiment in experiments:
        experiment.update(experiment_static_params)
        run_experiment(**experiment)




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




