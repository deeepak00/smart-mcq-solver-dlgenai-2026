import pandas as pd
from utils import save_submission

class Inferencer:
    def __init__(self,model,preprocessor,config):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config

    def inference(self):
        test_df = pd.read_csv(self.config["test_path"])
        test_df = self.preprocessor.preprocess(test_df,is_train=False)

        self.model.load(self.config["model_path"])

        predictions = self.model.predict(test_df,self.config)

        save_submission(
            test_df["id"].unique(),
            predictions,
            self.config["submission_path"]
        )
        print("Inference Completed Successfully.")