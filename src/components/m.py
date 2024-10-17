import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_classification_model, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "classification_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "Support Vector Classifier": SVC(probability=True)
            }

            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                },
                "Decision Tree Classifier": {
                    'max_depth': [3, 5, 10],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest Classifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10]
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200]
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }

            # Evaluate models using GridSearchCV and collect their evaluation metrics
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get the best model based on ROC AUC score
            best_model_name = max(model_report, key=lambda x: model_report[x]["roc_auc"] if model_report[x]["roc_auc"] is not None else 0)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]["roc_auc"]

            if best_model_score is None or best_model_score < 0.6:
                raise CustomException("No suitable model found with ROC AUC > 0.6")

            logging.info(f"Best model: {best_model_name} with ROC AUC: {best_model_score}")

            # Fit the best model with the entire training set
            best_model.fit(X_train, y_train)

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions on the test set
            y_test_pred = best_model.predict(X_test)
            y_test_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

            # Evaluate the model on the test set
            accuracy, precision, recall, f1, roc_auc = evaluate_classification_model(y_test, y_test_pred, y_test_pred_proba)

            # Return the model's performance metrics
            return {
                "best_model": best_model_name,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                }
            }

        except Exception as e:
            raise CustomException(e, sys)
