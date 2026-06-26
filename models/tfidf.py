import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from utils import mapk

class TFIDFModel:
    def __init__(self, max_features=1000, ngram_range=(1, 2), C=5):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        self.model = LogisticRegression(
            C=C, 
            max_iter=1000, 
            random_state=42
        )

    def fit(self, df, config=None):
        X_train = df["text"].values
        y_train = df["target"].values
        X_train_trans = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_trans, y_train)

    def predict(self, df, config=None):
        probs = self.predict_probab(df["text"])[:, 1]
        df = df.copy()
        df["prob"] = probs
        
        predictions = []
        for test_id in df["id"].unique():
            group = df[df["id"] == test_id]
            sorted_group = group.sort_values(by="prob", ascending=False)
            top_3 = sorted_group["option_label"].tolist()[:3]
            predictions.append(" ".join(top_3))
        return predictions

    def predict_probab(self, X):
        X = self.vectorizer.transform(X)
        return self.model.predict_proba(X)

    def cross_validate(self, df, config):
        n_splits = config.get("n_splits", 5)
        gkf = GroupKFold(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in gkf.split(df, groups=df["id"]):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            
            fold_model = TFIDFModel(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                C=self.C
            )
            fold_model.fit(train_fold)
            
            val_probs = fold_model.predict_probab(val_fold["text"])[:, 1]
            val_fold = val_fold.copy()
            val_fold["pred_prob"] = val_probs
            
            actuals = []
            predictions = []
            
            for _, group in val_fold.groupby("id"):
                actual_answer = group.loc[group["target"] == 1, "option_label"].values[0]
                actuals.append(actual_answer)
                
                sorted_group = group.sort_values(by="pred_prob", ascending=False)
                top_3 = sorted_group["option_label"].tolist()[:3]
                predictions.append(top_3)
                
            fold_score = mapk(actuals, predictions, k=3)
            scores.append(fold_score)
            
        return np.mean(scores)

    def save(self, path):
        joblib.dump(
            {
                'vectorizer': self.vectorizer,
                'model': self.model
            },
            path
        )

    def load(self, path):
        bundle = joblib.load(path)
        self.vectorizer = bundle['vectorizer']
        self.model = bundle['model']