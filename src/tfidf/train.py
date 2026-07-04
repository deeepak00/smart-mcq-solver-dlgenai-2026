import argparse
import os
import warnings

import joblib
import numpy as np
import wandb
from sklearn.model_selection import GroupKFold

from models.tfidf import build_model, build_vectorizer
from .preprocessed import prepare_train
from .utils import mapk

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def run_cv(train_expanded, vectorizer, n_splits=5):
    X = vectorizer.fit_transform(train_expanded["text"])
    y = train_expanded["target"]
    groups = train_expanded["id"]

    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print("=" * 60)
        print(f"FOLD {fold + 1}")

        model = build_model(solver="saga")

        model.fit(
            X[train_idx],
            y.iloc[train_idx]
        )

        val_probs = model.predict_proba(
            X[val_idx]
        )[:, 1]

        val_df = train_expanded.iloc[val_idx].copy()
        val_df["prob"] = val_probs

        actuals = []
        predictions = []

        for _, group in val_df.groupby("id"):
            group = group.sort_values("prob", ascending=False)

            predictions.append(
                group["option_label"].head(3).tolist()
            )

            actuals.append(
                group.loc[group["target"] == 1, "option_label"].iloc[0]
            )

        fold_map3 = mapk(actuals, predictions)

        fold_scores.append(fold_map3)

        print(f"Fold MAP@3 : {fold_map3:.5f}")

        wandb.log(
            {
                "fold": fold + 1,
                "fold_map@3": fold_map3,
            }
        )

    mean_score = np.mean(fold_scores)

    print("\n")
    print("=" * 60)
    print(f"FINAL CV MAP@3 : {mean_score:.5f}")

    return mean_score


def train_final_model(train_expanded, vectorizer):

    X_train = vectorizer.fit_transform(
        train_expanded["text"]
    )

    y_train = train_expanded["target"]

    final_model = build_model(
        solver="lbfgs"
    )

    final_model.fit(
        X_train,
        y_train
    )

    return final_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path",default="/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")
    parser.add_argument("--output-dir",default="artifacts")
    parser.add_argument("--n-splits",type=int,default=5)

    args = parser.parse_args()

    wandb.init(
        project="23f3004133-t22026",
        name="tfidf-logistic-v2",
        job_type="train",
        tags=["tfidf", "logistic"],
        config={
            "max_features": 300000,
            "ngram_range": (1, 2),
            "min_df": 5,
            "max_df": 0.90,
            "sublinear_tf": True,
            "C": 2.0,
            "max_iter": 5000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "n_splits": args.n_splits,
        },
    )

    wandb.run.log_code(".")

    os.makedirs(args.output_dir,exist_ok=True,)

    print("Preparing training data...")

    train_expanded = prepare_train(args.train_path)

    wandb.log(
        {
            "num_rows": len(train_expanded),
            "num_questions": train_expanded["id"].nunique(),
            "num_positive_samples": int(train_expanded["target"].sum()),
        }
    )

    # -----------------------
    # Cross Validation
    # -----------------------
    print("\nRunning Cross Validation...")

    cv_vectorizer = build_vectorizer()

    cv_score = run_cv(
        train_expanded,
        cv_vectorizer,
        n_splits=args.n_splits,
    )

    wandb.log(
        {
            "cv_map@3": cv_score
        }
    )

    # -----------------------
    # Final Training
    # -----------------------
    print("\nTraining Final Model...")

    final_vectorizer = build_vectorizer()

    final_model = train_final_model(
        train_expanded,
        final_vectorizer,
    )

    wandb.log(
        {
            "vocabulary_size": len(final_vectorizer.vocabulary_)
        }
    )

    # -----------------------
    # Save model
    # -----------------------
   
    joblib.dump(final_vectorizer,os.path.join(args.output_dir,"vectorizer.joblib"))
    joblib.dump(final_model,os.path.join(args.output_dir,"model.joblib"))

    print(f"\nSaved model and vectorizer to {args.output_dir}/")

    # -----------------------
    # Upload artifacts
    # -----------------------
    artifact = wandb.Artifact(
        name="tfidf-logistic-model",
        type="model",
        description="TF-IDF Logistic Regression Model",
    )

    artifact.add_file(os.path.join(args.output_dir, "model.joblib"))
    artifact.add_file(os.path.join(args.output_dir, "vectorizer.joblib"))

    wandb.log_artifact(artifact)

    # -----------------------
    # Log summary metrics
    # -----------------------
    wandb.summary["Final CV MAP@3"] = cv_score
    wandb.summary["Vocabulary Size"] = len(final_vectorizer.vocabulary_)
    wandb.summary["Training Samples"] = len(train_expanded)
    wandb.summary["Questions"] = train_expanded["id"].nunique()

    # Finish run
    wandb.finish()

    print("\nTraining Completed Successfully.")


if __name__ == "__main__":
    main()