import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class TFIDFPreprocessor:
    OPTIONS = ['A', 'B', 'C', 'D', 'E']

    def __init__(self):
        pass

    def expand_dataframe(self, df, is_train=True):
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding Dataset"):
            for option in self.OPTIONS:
                sample = {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "option": row[option],
                    "option_label": option,
                }
                if is_train:
                    sample["target"] = int(option == row["answer"])
                rows.append(sample)
        return pd.DataFrame(rows)

    def build_text(self, df):
        df = df.copy()
        df["text"] = (df["prompt"].astype(str)+ " [SEP] "+ df["option"].astype(str))
        return df

    def preprocess(self, df, is_train=True):
        df = self.expand_dataframe(df, is_train=is_train)
        df = self.build_text(df)
        return df

