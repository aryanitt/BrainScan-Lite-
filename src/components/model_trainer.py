import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception_config import CustomException
from src.logger_config import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boost": GradientBoostingClassifier()
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 7]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10]
                },
                "Gradient Boost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 150]
                }
            }

            model_report = evaluate_models(x_train=x_train, y_train=y_train,
                                           x_test=x_test, y_test=y_test,
                                           models=models, param=params)

            # Find the best model from the report
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info(f"Best model selected: {best_model_name} with accuracy: {model_report[best_model_name]}")

            # Train best model again on full training data
            best_model.fit(x_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
