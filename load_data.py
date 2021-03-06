import gzip
import json
import pandas as pd

PATH_RATING = 'data\\Software.csv'
PATH_REVIEW = 'data\\Software_5.json.gz'
PATH_METADATA = 'data\\meta_Software.json.gz'


# Parse json file
def parse(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield json.loads(line)


# Load review data json file
def get_reviews(path=PATH_REVIEW):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Load Meta Data json file for Content_Based Recommendations
def get_metadata(path=PATH_METADATA):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Extract ratings from review file
def get_ratings(data):
    data = data.filter(['reviewerID', 'asin', 'overall'], axis=1)
    return data.rename(columns={'reviewerID': 'user',
                                'asin': 'item',
                                'overall': 'rate'
                                })


# Load the ratings only csv file
def load_ratings(path=PATH_RATING):
    data = pd.read_csv(path, names=['item', 'user', 'rate', 'timestamp'], header=None)
    data.drop(columns=['timestamp'], inplace=True)
    return data.reindex(columns=['user', 'item', 'rate'])



