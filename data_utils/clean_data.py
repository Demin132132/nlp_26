import pandas as pd


DEFAULT_INPUT_FILE = '../Datasets/balance.csv'
DEFAULT_OUTPUT_FILE = '../Datasets/cleared_train.csv'


def clean_data(input_file=DEFAULT_INPUT_FILE, output_file=DEFAULT_OUTPUT_FILE):
    df = pd.read_csv(input_file, sep=',', header=0)
    df = df[(df.name_1.str.match(r'.*[^\x00-\x7f].*') == False) & (df.name_2.str.match(r'.*[^\x00-\x7f].*') == False)]
    df = df.drop('pair_id', axis=1)
    df.to_csv(output_file)
