import argparse
import os

import joblib
import pandas as pd

from .preprocessed import prepare_test


def predict(test_expanded, vectorizer, model):
    X_test = vectorizer.transform(test_expanded['text'])
    test_expanded['prob'] = model.predict_proba(X_test)[:, 1]
    return test_expanded


def build_submission(test_expanded):
    submission_rows = []

    for question_id, group in test_expanded.groupby('id'):
        group = group.sort_values('prob', ascending=False)
        top3 = group['option_label'].head(3).tolist()
        submission_rows.append({'ID': question_id, 'Prediction': ' '.join(top3)})

    return pd.DataFrame(submission_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-path',
        default='/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv'
    )
    parser.add_argument('--artifacts-dir', default='artifacts')
    parser.add_argument('--output-path', default='submission.csv')
    args = parser.parse_args()

    test_expanded = prepare_test(args.test_path)

    vectorizer = joblib.load(os.path.join(args.artifacts_dir, 'vectorizer.joblib'))
    model = joblib.load(os.path.join(args.artifacts_dir, 'model.joblib'))

    test_expanded = predict(test_expanded, vectorizer, model)
    submission = build_submission(test_expanded)

    submission.to_csv(args.output_path, index=False)
    print(f'submission saved to {args.output_path}')


if __name__ == '__main__':
    main()