import numpy as np
import pandas as pd
from surprise import Dataset, Reader, similarities, accuracy
from surprise.model_selection import train_test_split

from enum import Enum
from collections import defaultdict
from statistics import mean


from surprise import KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering, BaselineOnly


from surprise.model_selection import GridSearchCV, KFold

RANDOM_STATE = 44
READER = Reader(rating_scale=(1, 5))
ALGO_TYPE = Enum('KNN', 'MODEL_BASED')


# Return two types of testsets and trainsets given ratings data and test_ratio
def split_data(data, test_ratio):
    rateData = Dataset.load_from_df(data[['user', 'item', 'rate']], READER)
    raw_ratings = rateData.raw_ratings  # obtain raw ratings data from the Dataset

    cv_train = rateData  # initially make copies of the rateData
    cv_test = rateData

    # shuffle ratings
    np.random.shuffle(raw_ratings)

    threshold = int(test_ratio * len(raw_ratings))
    trainset_raw_ratings = raw_ratings[threshold:]
    test_raw_ratings = raw_ratings[:threshold]

    # create train set which is suitable for Surprise GridSearch function
    cv_train.raw_ratings = trainset_raw_ratings

    # create test set which is suitable for Surprise GridSearch function
    cv_test.raw_ratings = test_raw_ratings

    # create train set which is suitable for Surprise fit() function
    train = cv_train.build_full_trainset()

    # create test set which is suitable for Surprise test() function
    test = cv_test.construct_testset(test_raw_ratings)

    return train, test, cv_train, cv_test


# Find best parameters using Surprise GridSearch
# Get a Trianset type of object as data, so it does not need to convert to Surprise Dataset
def find_best_parameters(data, algorithm, param_grid, measure=['rmse'], cv=5):
    ratings = Dataset.load_from_df(data[['user', 'item', 'rate']], READER)
    # ratings = data
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
    # ratings = data
    kf = KFold(n_splits=cv, random_state=RANDOM_STATE)
    for trainset, testset in kf.split(ratings):
        if algo_type == 'KNN':
            algo = algorithm(k=params['k'], min_k=3, sim_options=params['sim_options'], verbose=False)
        elif algo_type == 'MODEL_BASED':
            algo = algorithm(n_factors=params['n_factors'],
                            n_epochs=params['n_epochs'],
                            reg_all=params['reg_all'],
                            lr_all=params['lr_all']
                            )
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse.append(accuracy.rmse(predictions, verbose=False))

    rmse = np.array(rmse)
    return rmse


# Perform evaluation
def evaluate(algo_type, algorithm, params, _trainset, _testset, _verbose=False):
    train = Dataset.load_from_df(_trainset[['user', 'item', 'rate']], READER)
    trainset = train.build_full_trainset()
    # test = Dataset.load_from_df(_testset[['user', 'item', 'rate']], READER)
    # testset = test.construct_testset(_testset)
    testset = _testset
    if algo_type == 'KNN':
        algo = algorithm(k=params['k'], min_k=3, sim_options=params['sim_options'], verbose=_verbose)
    elif algo_type == 'MODEL_BASED':
        algo = algorithm(n_factors=params['n_factors'],
                         n_epochs=params['n_epochs'],
                         reg_all=params['reg_all'],
                         lr_all=params['lr_all']
                         )

    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=_verbose)

    return rmse, predictions


# Predict rating for specific user_id on specific item_id
def predict(predictions, uid, iid):
    return


# Generate Top-N recommendation list
def get_top_n(predictions, N=10, user_id=None):
    top_n = defaultdict(list)

    # Map the predictions to each user
    for uid, iid, actual_rate, predicted_rate, _ in predictions:
        top_n[uid].append((iid, predicted_rate))

    # Sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:N]

    if user_id is None:
        return top_n
    else:
        return top_n[user_id]


# Return precision and recall at K metrics for each user
def precision_recall_at_k(predictions, K=10, threshold=3.5):

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, actual_rate, predicted_rate, _ in predictions:
        user_est_true[uid].append((predicted_rate, actual_rate))

    precisions = []  # dict()
    recalls = []  # dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value (predicted rate)
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((actual_rate >= threshold) for (_, actual_rate) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((predicted_rate >= threshold) for (predicted_rate, _) in user_ratings[:K])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((actual_rate >= threshold) and (predicted_rate >= threshold))
                              for (predicted_rate, actual_rate) in user_ratings[:K])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. So, it will set to 0.
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)


        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. So, it will set to 0.
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

        mean_pr = mean(precisions)
        mean_re = mean(recalls)
        f_measure = (2 * mean_pr * mean_re) / (mean_pr + mean_re)

    return mean_pr, mean_re, f_measure


# Perform all tasks(GridSearch, KFold, and Evaluation across a list of different algorithms) in a single batch run
def batch_search(algo_type, data, algo_list, param_grid, cv=5):
    rmse = []
    best_params = []
    for algo in algo_list:
        print(algo)
        result = find_best_parameters(data, algo, param_grid, ['RMSE'], 5)
        best_param = result.best_params['rmse']
        err = run_kfold(algo_type, data, algo, best_param, 5)
        print('Mean of the best RMSE across cross-validation: ', err.mean())
        rmse.append(err)
        best_params.append(best_param)
        print('--------------------------------------------------------------------------')
    return best_params, np.array(rmse)


# Perform Evaluation on prediction task for all given algorithms
def batch_evaluate(algo_type, algo_list, params, trainset, testset, verbose=True):
    rmses = []
    predictions = []
    i = 0
    for algo in algo_list:
        rmse, pred = evaluate(algo_type, algo, params[i], trainset, testset)
        if verbose:
            print(algo)
            print('RMSE: ', rmse)
            print('----------------------------------------------------------------')
        rmses.append(rmse)
        predictions.append(pred)
        i += 1
    return rmses, predictions


# return a list of Precision and Recall over diffrent value of K
def precision_recall_over_k(predictions, _threshold=3.5, _K=40):
    precision = []
    recalls = []
    for k in range(1, _K):
        avg_precision, avg_recall = precision_recall_at_k(predictions, K=k, threshold=_threshold)
        precision.append(avg_precision)
        recalls.append(avg_recall)

    return precision, recalls


# Return the number of items rated by given user
# def get_Iu(uid, trainset):
#     try:
#         return len(trainset.ur[trainset.to_inner_uid(uid)])
#     except ValueError:  # user was not part of the trainset
#         return 0
#
#
# # Return number of users that have rated given item
# def get_Ui(iid, trainset):
#     try:
#         return len(trainset.ir[trainset.to_inner_iid(iid)])
#     except ValueError:
#         return 0


# Return best predictions
# def best_predictions(predictions, trainset):
#     df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'actual_rate', 'predicted_rate', 'details'])
#     df['No Items rated by user'] = df.uid.apply(get_Iu, trainset)
#     df['No user have rated item'] = df.iid.apply(get_Ui, trainset)
#     df['err'] = abs(df.est - df.rui)
#     return df.sort_values(by='err')[:10]
#
#
# # worst_predictions = df.sort_values(by='err')[-10:]