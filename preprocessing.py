import pandas as pd
import random
from sklearn.utils import shuffle

RANDOM_STATE = 44


# Create a map between old ids to new numeric ids
def get_id_map(data):
    unique_user_ids = data.user.unique()
    unique_item_ids = data.item.unique()

    # Map old user ids to new numeric ids
    user_id_map = {}
    new_user_id = 0
    for old_id in unique_user_ids:
        user_id_map[old_id] = new_user_id
        new_user_id += 1

    # Map old item ids to new numeric ids
    item_id_map = {}
    new_item_id = 0
    for old_id in unique_item_ids:
        item_id_map[old_id] = new_item_id
        new_item_id += 1

    return user_id_map, item_id_map


# Convert user IDs and item IDs to numeric IDs
def convert_ids(data):
    user_map, item_map = get_id_map(data)
    new_data = []

    for index, row in data.iterrows():
        old_user_id = row[0]
        old_item_id = row[1]
        rate = row[2]
        new_user_id = user_map[old_user_id]  # get the new numeric user id
        new_item_id = item_map[old_item_id]  # get the new numeric item id
        new_data.append([new_user_id, new_item_id, int(rate)])

    new_df = pd.DataFrame(new_data, columns=['user', 'item', 'rate'])

    return new_df


# generate user profile
def get_user_profile(data):
    unique_users = data.user.unique()
    unique_items = data.item.unique()

    user_profile = {}

    for id in unique_users:
        user = data[data.user == id]
        rated_items = []
        for index, row in user.iterrows():
            item_id = row[1]
            rate = row[2]

            # Create a tuple of new numeric item_id and its related rate for the current user id and
            # append this tuple to a list of rated items for the current user
            rated_items.append((int(item_id), int(rate)))

        user_profile[id] = rated_items

    return user_profile


# Find list of items that each user likes based on a threshold for rates
def find_liked_items(user_profiles, rate_threshold):
    liked_items = {}
    for user_id in user_profiles:
        rated_items = user_profiles[user_id]
        user_liked = []
        for item in rated_items:
            rate = item[1]
            if rate >= rate_threshold:
                user_liked.append(item)
        liked_items[user_id] = user_liked

    return liked_items


# Split ratings data to test and train set based on test ratio.
def split_ratings_data(ratings, test_ratio):
    ratings = shuffle(ratings, random_state=RANDOM_STATE)
    num_tests = int(ratings.shape[0] * test_ratio)
    test = ratings[:num_tests]
    train = ratings[num_tests:]

    return train, test


# split each user profile to test and train set based on test ratio.
def split_user_profile(user_profiles, test_ratio):
    train = {}
    test = {}
    for user_id in user_profiles:
        rated_items = user_profiles[user_id]

        random.seed(RANDOM_STATE)
        random.shuffle(rated_items)  # Shuffle the rated items

        num_tests = int(len(rated_items) * test_ratio)  # Number of test cases
        test[user_id] = rated_items[:num_tests]
        train[user_id] = rated_items[num_tests:]

    return train, test
