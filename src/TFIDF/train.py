import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import time
import numpy as np
import pandas as pd
import wandb

from utils import seed_everything
from preprocess import TFIDFPreprocessor
from tfidf import TFIDFModel

class Trainer:
    def __init__(self, model, preprocessor, config):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config

    def train(self):
        seed_everything(self.config.get("seed", 42))
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config["project"],
                name=self.config["run_name"]
            )

        train_df = pd.read_csv(self.config["train_path"])
        train_df = self.preprocessor.preprocess(train_df, is_train=True)

        if self.config.get("do_cv", True):
            cv_score = self.model.cross_validate(
                train_df,
                self.config
            )

            print(f"CV Score : {cv_score:.5f}")

            if self.config.get("use_wandb", False):
                wandb.log({
                    "CV_SCORE": cv_score
                })

        self.model.fit(train_df, self.config)
        self.model.save(self.config["model_path"])

        if self.config.get("use_wandb", False):
            wandb.finish()

        print("Training Finished Successfully.")

