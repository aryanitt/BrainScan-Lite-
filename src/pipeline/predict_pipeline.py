import sys
import pandas as pd
import numpy as np
from src.exception_config import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            label_encoder_path = 'artifacts/label_encoder.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            encoder = load_object(label_encoder_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            mbti_type = encoder.inverse_transform(preds.astype(int))

            return mbti_type

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10):
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = Q3
        self.Q4 = Q4
        self.Q5 = Q5
        self.Q6 = Q6
        self.Q7 = Q7
        self.Q8 = Q8
        self.Q9 = Q9
        self.Q10 = Q10

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Q1": [self.Q1],
                "Q2": [self.Q2],
                "Q3": [self.Q3],
                "Q4": [self.Q4],
                "Q5": [self.Q5],
                "Q6": [self.Q6],
                "Q7": [self.Q7],
                "Q8": [self.Q8],
                "Q9": [self.Q9],
                "Q10": [self.Q10]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
