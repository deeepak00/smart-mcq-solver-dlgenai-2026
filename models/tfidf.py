from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# def build_vectorizer():
#     return TfidfVectorizer(
#         max_features=200000,
#         ngram_range=(1, 3),
#         lowercase=True,
#         strip_accents='unicode',
#         sublinear_tf=True,
#         min_df=2,
#         max_df=0.95
#     )


# def build_model(solver='lbfgs'):
#     return LogisticRegression(
#         C=5.0,
#         max_iter=5000,
#         solver=solver,
#         class_weight='balanced',
#         n_jobs=-1
#     )


def build_vectorizer():
    return TfidfVectorizer(
        max_features=300000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )


def build_model(solver='lbfgs'):
    return LogisticRegression(
        C=10.0,
        max_iter=5000,
        solver=solver,
        class_weight='balanced',
        n_jobs=-1
    )