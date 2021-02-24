import numpy as np
# import algorithm_type as at
from surprise import Dataset, Reader, similarities, accuracy
from enum import Enum

from surprise import KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering, BaselineOnly


from surprise.model_selection import GridSearchCV, KFold

RANDOM_STATE = 44
READER = Reader(rating_scale=(1, 5))
ALGO_TYPE = Enum('KNN', 'MODEL_BASED')


def find_best_parameters(data, algorithm, param_grid, measure=['rmse'], cv=5):
    ratings = Dataset.load_from_df(data[['user', 'item', 'rate']], READER)
    _cv = KFold(n_splits=cv, random_state=RANDOM_STATE)

    from time import time
    t = time()

    gs = GridSearchCV(algorithm, param_grid, measures=measure, cv=_cv)
    gs.fit(ratings)

    time = time() - t
    print("done in %0.3fs." % time)
    return gs


def run_kfold(algo_type, data, algorithm, params, cv=5):
    rmse = []
    ratings = Dataset.load_from_df(data[['user', 'item', 'rate']], READER)
    kf = KFold(n_splits=cv, random_state=RANDOM_STATE)
    for trainset, testset in kf.split(ratings):
        if algo_type == 'KNN':
            err = algorithm(k=params['k'], min_k=3, sim_options=params['sim_options'], verbose=False)
        elif algo_type == 'MODEL_BASED':
            err = algorithm(n_factors=params['n_factors'],
                            n_epochs=params['n_epochs'],
                            reg_all=params['reg_all'],
                            lr_all=params['lr_all']
                            )
        err.fit(trainset)
        predictions = err.test(testset)
        rmse.append(accuracy.rmse(predictions, verbose=False))

    rmse = np.array(rmse)
    return rmse


def batch_run(algo_type, data, algo_list, param_grid, cv=5):
    # if type(algo_type) != type(at.ALGO_TYPE):
    #     print('Wrong Algorithm Type!')
    #     return
    rmse = []
    for algo in algo_list:
        print(algo)
        result = find_best_parameters(data, algo, param_grid, ['RMSE'], 5)
        best_params = result.best_params['rmse']
        err = run_kfold(algo_type, data, algo, best_params, 5)
        print('Mean of the best RMSE across cross-validation: ', err.mean())
        rmse.append(err)
        print('--------------------------------------------------------------------------')
    return np.array(rmse)

