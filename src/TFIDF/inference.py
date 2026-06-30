import sys
import os
# Add relative paths for models/ and TFIDF/ folders
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import numpy as np
import pandas as pd

from utils import save_submission, seed_everything
from preprocess import TFIDFPreprocessor
from tfidf import TFIDFModel

# ============================================================
# ORIGINAL INFERENCER CLASS
# ============================================================
class Inferencer:
    def __init__(self, model, preprocessor, config):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config

    def inference(self):
        test_df = pd.read_csv(self.config["test_path"])
        test_df = self.preprocessor.preprocess(test_df, is_train=False)

        self.model.load(self.config["model_path"])

        predictions = self.model.predict(test_df, self.config)

        save_submission(
            test_df["id"].unique(),
            predictions,
            self.config["submission_path"]
        )
        print("Inference Completed Successfully.")

