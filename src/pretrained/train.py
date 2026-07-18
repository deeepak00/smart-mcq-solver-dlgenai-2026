import argparse
import gc
import json
import os
import shutil

import torch
import wandb
from sklearn.model_selection import StratifiedKFold

from utils import FOLDS, MAX_LEN, MODEL_NAME, SEED, get_device, set_seed, silence_warnings
from preprocessed import load_tokenizer, load_train_df, prepare_fold_datasets
from ...models.pretrained import build_model, build_training_args, build_trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")
    p.add_argument("--output_dir", default="/kaggle/working/pretrained_model")
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--folds", type=int, default=FOLDS)
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--wandb_project", default="23f3004133-t22026")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    silence_warnings()
    set_seed(args.seed)
    device, use_bf16 = get_device()

    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    train_df = load_train_df(args.train_csv)
    tokenizer = load_tokenizer(args.model_name)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_weights = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        checkpoint_dir = f"./tmp_checkpoints_fold_{fold}"                          # scratch dir, deleted after the fold
        final_model_dir = os.path.join(args.output_dir, f"fold_{fold}")            # persisted for inference.py
        print(f"\n{'='*20} FOLD {fold + 1}/{args.folds} {'='*20}")

        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)

        train_ds, val_ds = prepare_fold_datasets(train_data, val_data, tokenizer, args.max_len)

        model = build_model(args.model_name, device)
        training_args = build_training_args(checkpoint_dir, use_bf16, args.seed)
        trainer = build_trainer(model, training_args, train_ds, val_ds, tokenizer)

        trainer.train()

        # log every epoch's eval metrics for this fold (map3, accuracy, precision, f1)
        for entry in trainer.state.log_history:
            if "eval_map3" in entry:
                wandb.log({
                    "fold": fold,
                    "epoch": entry.get("epoch"),
                    "eval/map3": entry["eval_map3"],
                    "eval/accuracy": entry["eval_accuracy"],
                    "eval/precision": entry["eval_precision"],
                    "eval/f1": entry["eval_f1"],
                })

        val_metrics = trainer.evaluate()
        fold_map3 = val_metrics["eval_map3"]
        fold_weights.append(fold_map3)
        print(f"Fold {fold + 1} val MAP@3: {fold_map3:.4f} | val accuracy: {val_metrics['eval_accuracy']:.4f}")

        wandb.log({
            "fold": fold,
            "fold_summary/map3": val_metrics["eval_map3"],
            "fold_summary/accuracy": val_metrics["eval_accuracy"],
            "fold_summary/precision": val_metrics["eval_precision"],
            "fold_summary/f1": val_metrics["eval_f1"],
        })

        print(f"Saving fold {fold + 1}'s best model to {final_model_dir} ...")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        print(f"Cleaning up disk and memory for fold {fold + 1}...")
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

    weights_path = os.path.join(args.output_dir, "fold_weights.json")
    with open(weights_path, "w") as f:
        json.dump({"fold_map3": fold_weights}, f, indent=2)

    print("\n" + "=" * 40)
    print("Fold MAP@3 scores:", [f"{w:.4f}" for w in fold_weights])
    print(f"Saved {args.folds} fold models + fold_weights.json under {args.output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()