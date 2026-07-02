import pandas as pd
from tqdm import tqdm

from .utils import OPTIONS


def expand_df(df, is_train=True):
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        for option in OPTIONS:
            sample = {
                'id': row['id'],
                'prompt': row['prompt'],
                'option': row[option],
                'option_label': option
            }

            if is_train:
                sample['target'] = int(option == row['answer'])

            rows.append(sample)

    return pd.DataFrame(rows)


def add_text_column(df):
    df['text'] = (
        df['prompt'].astype(str)
        + ' [SEP] '
        + df['option'].astype(str)
    )
    return df


def prepare_train(train_path):
    train = pd.read_csv(train_path)
    train_expanded = expand_df(train, is_train=True)
    return add_text_column(train_expanded)


def prepare_test(test_path):
    test = pd.read_csv(test_path)
    test_expanded = expand_df(test, is_train=False)
    return add_text_column(test_expanded)