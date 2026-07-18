import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from utils import INV_MAP, MAX_LEN, get_device, silence_warnings
from preprocessed import DataCollatorForMultipleChoice, load_test_df, prepare_test_datasets
from models.pretrained import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", default="/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv")
    p.add_argument("--saved_models_dir", default="/kaggle/input/saved-models/saved_models")
    p.add_argument("--output_csv", default="/kaggle/working/submission.csv")
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    return p.parse_args()


def predict_with_fold_model(model_dir, tokenizer, test_ds_mapped, test_ds_rev_mapped, device):
    model = build_model(model_dir, device)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp_inference", report_to="none"),
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    std_logits = trainer.predict(test_ds_mapped).predictions
    rev_logits = trainer.predict(test_ds_rev_mapped).predictions
    rev_logits = rev_logits[:, ::-1].copy()

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return std_logits, rev_logits


def main():
    args = parse_args()

    silence_warnings()
    device, _ = get_device()

    test_df = load_test_df(args.test_csv)

    fold_dirs = sorted(
        d for d in os.listdir(args.saved_models_dir)
        if d.startswith("fold_") and os.path.isdir(os.path.join(args.saved_models_dir, d))
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* model directories found under {args.saved_models_dir}")

    with open(os.path.join(args.saved_models_dir, "fold_weights.json")) as f:
        fold_weights = np.array(json.load(f)["fold_map3"])


    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.saved_models_dir, fold_dirs[0]))
    test_ds, test_ds_mapped, test_ds_rev_mapped = prepare_test_datasets(test_df, tokenizer, args.max_len)

    all_test_preds = []
    all_test_preds_rev = []

    for fold_dir in fold_dirs:
        print(f"Inferencing with {fold_dir} ...")
        model_path = os.path.join(args.saved_models_dir, fold_dir)
        std_logits, rev_logits = predict_with_fold_model(
            model_path, tokenizer, test_ds_mapped, test_ds_rev_mapped, device
        )
        all_test_preds.append(std_logits)
        all_test_preds_rev.append(rev_logits)

    # ================= ENSEMBLE & FINAL OUTPUT (identical to the original notebook) =================
    print("\n" + "=" * 40)
    print("Fold MAP@3 scores:", [f"{w:.4f}" for w in fold_weights])

    fold_weights = fold_weights / fold_weights.sum()

    std_probs = [torch.softmax(torch.tensor(p), dim=1).numpy() for p in all_test_preds]
    rev_probs = [torch.softmax(torch.tensor(p.copy()), dim=1).numpy() for p in all_test_preds_rev]

    per_fold_probs = [(s + r) / 2 for s, r in zip(std_probs, rev_probs)]
    final_probs = np.tensordot(fold_weights, np.stack(per_fold_probs), axes=(0, 0))

    final = []
    for i, p in enumerate(final_probs):
        top3_indices = np.argsort(p)[::-1][:3]
        top3_letters = [INV_MAP[idx] for idx in top3_indices]
        final.append({
            "id": test_ds["id"][i],
            "prediction": " ".join(top3_letters)
        })

    submission = pd.DataFrame(final)
    submission.to_csv(args.output_csv, index=False)
    print(submission.head())
    print(f"Ensemble submission ready -> {args.output_csv}")


if __name__ == "__main__":
    main()