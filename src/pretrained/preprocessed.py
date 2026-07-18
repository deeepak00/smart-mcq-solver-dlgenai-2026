from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

from .utils import LABEL_MAP, OPTION_COLS


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def load_train_df(path):
    df = pd.read_csv(path)
    df["label"] = df["answer"].map(LABEL_MAP)
    return df


def load_test_df(path):
    return pd.read_csv(path)


def make_preprocess_fn(tokenizer, max_len, option_order=OPTION_COLS):
    def preprocess_function(examples):
        first_sentences = [[context] * 5 for context in examples["prompt"]]
        second_sentences = []
        for i in range(len(examples["prompt"])):
            options = [str(examples[col][i]) for col in option_order]
            second_sentences.append(options)

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(
            first_sentences, second_sentences, truncation=True, max_length=max_len,
        )

        features = {k: [v[i: i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}
        return features

    return preprocess_function


def prepare_fold_datasets(train_data, val_data, tokenizer, max_len):
    preprocess_function = make_preprocess_fn(tokenizer, max_len)

    train_ds = Dataset.from_pandas(train_data)
    val_ds = Dataset.from_pandas(val_data)

    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)

    train_ds = train_ds.add_column("label", train_data["label"].tolist())
    val_ds = val_ds.add_column("label", val_data["label"].tolist())

    return train_ds, val_ds


def prepare_test_datasets(test_df, tokenizer, max_len):
    standard_fn = make_preprocess_fn(tokenizer, max_len, option_order=OPTION_COLS)
    reversed_fn = make_preprocess_fn(tokenizer, max_len, option_order=OPTION_COLS[::-1])

    test_ds = Dataset.from_pandas(test_df)

    test_ds_mapped = test_ds.map(
        standard_fn, batched=True,
        remove_columns=[c for c in test_ds.column_names if c != 'id']
    )
    test_ds_rev_mapped = test_ds.map(
        reversed_fn, batched=True,
        remove_columns=[c for c in test_ds.column_names if c != 'id']
    )
    return test_ds, test_ds_mapped, test_ds_rev_mapped


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features] if label_name in features[0].keys() else None
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [[{k: v[i] for k, v in feature.items() if k != 'id'} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch