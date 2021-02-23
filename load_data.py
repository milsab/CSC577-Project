import gzip
import json
import pandas as pd

PATH_REVIEW = 'data\\Software_5.json.gz'
PATH_METADATA = 'data\\meta_Software.json.gz'


def parse(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield json.loads(line)


# Load review data
def get_reviews(path=PATH_REVIEW):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Load Meta Data for Content_Based Recommendations
def get_metadata(path=PATH_METADATA):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_ratings(data):
    data = data.filter(['reviewerID', 'asin', 'overall'], axis=1)
    return data.rename(columns={'reviewerID': 'user',
                                'asin': 'item',
                                'overall': 'rate'
                                })
