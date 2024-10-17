import os
import sys

import numpy as np
import pandas as pd
import dill  # Importing dill for object serialization
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Using dill to dump the object instead of pickle
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classification_model(true, predicted, predicted_proba=None):
    accuracy = accuracy_score(true, predicted)
    precision = precision_score(true, predicted)
    recall = recall_score(true, predicted)
    f1 = f1_score(true, predicted)
    if predicted_proba is not None:
        roc_auc = roc_auc_score(true, predicted_proba)
    else:
        roc_auc = None  # SVC without probability estimation won't have ROC AUC
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            y_test_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

            # Evaluate on test set
            metrics = evaluate_classification_model(y_test, y_test_pred, y_test_pred_proba)
            report[model_name] = metrics

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        # Using dill to load the object instead of pickle
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
