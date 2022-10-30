import string

import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from sklearn.metrics import accuracy_score, precision_score, recall_score


INPUTS = '../Datasets/cleared_train.csv'


def data_preparation(df):
    df['name_1'] = df['name_1'].str.lower()
    df['name_2'] = df['name_2'].str.lower()
    df = df.replace(r'[{}]'.format(string.punctuation), '', regex=True)
    return df


def run_fuzzywuzzy(df):
    df['WRatio'] = df.apply(lambda x: fuzz.WRatio(x['name_1'], x['name_2']), axis=1)
    df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x['name_1'], x['name_2']), axis=1)
    df['partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(x['name_1'], x['name_2']), axis=1)

    df['res_WRatio'] = np.where(df.WRatio >= 85, 1, 0)
    df['res_token_set_ratio'] = np.where(df.token_set_ratio >= 85, 1, 0)
    df['res_partial_token_set_ratio'] = np.where(df.partial_token_set_ratio >= 85, 1, 0)
    return df


def main():
    df = pd.read_csv(INPUTS)
    df = data_preparation(df)
    df = run_fuzzywuzzy(df)

    is_duplicate = df['is_duplicate'].to_numpy()

    WRatio = df['res_WRatio'].to_numpy()
    token_set_ratio = df['res_token_set_ratio'].to_numpy()
    partial_token_set_ratio = df['res_partial_token_set_ratio'].to_numpy()

    print('Fuzzywuzzy WRatio')
    print('Accuracy: %.3f' % (accuracy_score(is_duplicate, WRatio) * 100))
    print('Precision: %.3f' % (precision_score(is_duplicate, WRatio) * 100))
    print('Recall: %.3f' % (recall_score(is_duplicate, WRatio) * 100), '\n')

    print('Fuzzywuzzy token_set_ratio')
    print('Accuracy: %.3f' % (accuracy_score(is_duplicate, token_set_ratio) * 100))
    print('Precision: %.3f' % (precision_score(is_duplicate, token_set_ratio) * 100))
    print('Recall: %.3f' % (recall_score(is_duplicate, token_set_ratio) * 100), '\n')

    print('Fuzzywuzzy partial_token_set_ratio')
    print('Accuracy: %.3f' % (accuracy_score(is_duplicate, partial_token_set_ratio) * 100))
    print('Precision: %.3f' % (precision_score(is_duplicate, partial_token_set_ratio) * 100))
    print('Recall: %.3f' % (recall_score(is_duplicate, partial_token_set_ratio) * 100), '\n')


main()
