import numpy as np
from surprise import Dataset, Reader, similarities, accuracy
from surprise import KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering, BaselineOnly


from surprise.model_selection import GridSearchCV, KFold
RANDOM_STATE = 44
READER = Reader(rating_scale=(1, 5))


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


def run_kfold(data, algorithm, K, min_K, options, cv=5):
    rmse = []
    ratings = Dataset.load_from_df(data[['user', 'item', 'rate']], READER)
    kf = KFold(n_splits=cv, random_state=RANDOM_STATE)
    for trainset, testset in kf.split(ratings):
        err = algorithm(k=K, min_k=min_K, sim_options=options, verbose=False)
        err.fit(trainset)
        predictions = err.test(testset)
        rmse.append(accuracy.rmse(predictions, verbose=False))

    rmse = np.array(rmse)
    return rmse


def batch_run(data, algo_list, param_grid, cv=5):
    rmse = []
    for algo in algo_list:
        print(algo)
        result = find_best_parameters(data, algo, param_grid, ['RMSE'], 5)
        best_params = result.best_params['rmse']
        err = run_kfold(data, algo, best_params['k'], 3, best_params['sim_options'], 5)
        print('Mean of the best RMSE across cross-validation: ', err.mean())
        rmse.append(err)
        print('--------------------------------------------------------------------------')
    return np.array(rmse)
