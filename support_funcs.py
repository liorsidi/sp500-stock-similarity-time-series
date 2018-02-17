
from sklearn.decomposition import PCA
from sklearn.ensemble import *
from sklearn.preprocessing import StandardScaler

from data_preperation import *
from similarity_functions import *
from utils.ANN_stock import ANN_stock
from utils.SAX_FILE import SAX



transformations = {'SAX': SAX(),
                  'None': None,
                    'PCA': PCA(n_components = 3,random_state = 0)
                   }

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
    'stock_to_compare': ['JPM', "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"],
    'n_folds': [6],
    'finance_features': [True],
    'normalization': ['Standard'],
    'select_k_func': ['get_top_k'],
    'slide': [1]
}


experiment_params_1 = {
    'features_selection': [

        ('multivariate',
         [u'Close_proc', u'Close_norm',
          u'rsi', u'MACD', u'Open_Close_diff', u'High_Low_diff', u'Volume_norm']
         ),
('univariate', [u'Close_norm']),
    ],
    'transformation': ['SAX', 'PCA','None'],
    'k': [0,10],
    'similarity_col': ['Close_norm'],
    'similarity_func': ['euclidean'],
    'fix_len_func': ['time_corr'],
    'window_len': [0,5,10],
    'weighted_sampleing': [True, False],
    'y_col': ['Close_proc','Close_norm'],
    'force' : [True]
}

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
    'y_col': ['Close_norm','close proc']
}

