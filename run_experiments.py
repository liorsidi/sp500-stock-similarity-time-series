
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

calc_similarites('5yr')
experiment_params_1.update(experiment_params_base)
iterate_exp(experiment_params_1, experiment_predict_params)

