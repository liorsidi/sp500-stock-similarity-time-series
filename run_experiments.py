
from scipy.stats import rankdata
from sklearn.base import RegressorMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

from data_preperation import *
from similarity_functions import *
from Constants import *
import pandas as pd
import numpy as np
import os
import itertools
from support_funcs import *
from trade_strategies import long_short_profit_evaluation


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
        print folds_X_train[f].columns
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

    if transformation == 'SAX':
        features_names = [feature_name + "_transform" for feature_name in features_names ]

    if transformation == 'PCA':
        features_names = ['pc_' + str(i) for i in range(transformations[transformation].n_components)]

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
        similarity_col=[TARGET_PREP, 'Close_proc'],  # , 'Volume_norm'],
        y_col=[TARGET],
        similarity_func_name= similarity_funcs.keys(),
        fix_len_func_name=fix_len_funcs.keys(),
        stock_to_compare=stocks_to_compare
    )
    all_sim_path = os.path.join(home_path, 'experiments', 'similarities', data_name)
    similarity_params_product = get_index_product(similarity_params)
    sim_results = []
    for similarity_param in similarity_params_product:
        if (similarity_funcs[similarity_param['similarity_func_name']].__class__.__name__ == 'model_bases_distance'):
            continue
        sim_results.append(run_sim_analysis(all_sim_path, all_stocks_names, similarity_param, stock_X_prep_dfs))
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
experiment_params_1.update(experiment_params_base)
iterate_exp(experiment_params_1, experiment_predict_params)

