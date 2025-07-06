import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from src.logger_config import logging
from src.exception_config import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_path: str = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_config(self):
        """
        Returns the preprocessing pipeline to be used in transformation.
        """
        try:
            numerical_features = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features)
                ]
            )

            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Applies transformations to training and testing data, saves preprocessing objects.
        """
        try:
            logging.info("Starting data transformation.")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test datasets read successfully.")

            target_column_name = "mbti_type"
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_features_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformation_config()

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(target_features_train_df)
            y_test = label_encoder.transform(target_features_test_df)

            if hasattr(input_features_train_arr, "toarray"):
                input_features_train_arr = input_features_train_arr.toarray()
                input_features_test_arr = input_features_test_arr.toarray()

            train_arr = np.c_[input_features_train_arr, y_train]
            test_arr = np.c_[input_features_test_arr, y_test]

            # Save transformation and encoding objects
            save_object(self.transformation_config.preprocessor_file_path, preprocessing_obj)
            save_object(self.transformation_config.label_encoder_path, label_encoder)

            logging.info("Data transformation completed and objects saved.")
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file_path,
                self.transformation_config.label_encoder_path
            )

        except Exception as e:
            raise CustomException(e, sys)

