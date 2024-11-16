import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_classification_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "classification_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
        
            param_grids = {
                # "Decision Tree Classifier": {
                #     'max_depth': [3, 5, 7, 10, None],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4],
                #     'criterion': ['gini', 'entropy']
                # },
               "Decision Tree Classifier": {

                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],  },
                
                "Random Forest Classifier": {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 10]
                },
                "Extra Trees Classifier": {
                    "n_estimators": [100, 200 ],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                },
                "XGBClassifier": {
                    "n_estimators": [100, 200 ],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly'],
                    'degree': [2, 3] ,
                    'probability': [True]
                }
            }
            

            models = {
                #"Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                #"Extra Trees Classifier": ExtraTreesClassifier(),
                "XGBClassifier": XGBClassifier(eval_metric='logloss'),
                #"Support Vector Classifier": SVC(probability=True)
            }

            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name} with GridSearchCV.")
                
                grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Cross-validation predictions
                y_cv_pred = cross_val_predict(best_model, X_train, y_train, cv=3)
                y_cv_pred_proba = cross_val_predict(best_model, X_train, y_train, cv=3, method="predict_proba")[:, 1] if hasattr(best_model, "predict_proba") else None

                # Full training and evaluation on test set
                best_model.fit(X_train, y_train)
                y_train_pred = best_model.predict(X_train)
                y_train_pred_proba = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None
                y_test_pred = best_model.predict(X_test)
                y_test_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

                # Evaluate training, cross-validation, and test performance
                train_metrics = evaluate_classification_model(y_train, y_train_pred, y_train_pred_proba)
                cv_metrics = evaluate_classification_model(y_train, y_cv_pred, y_cv_pred_proba)
                test_metrics = evaluate_classification_model(y_test, y_test_pred, y_test_pred_proba)

                model_report[model_name] = {
                    "best_params": grid_search.best_params_,
                    "train_metrics": train_metrics,
                    "cv_metrics": cv_metrics,
                    "test_metrics": test_metrics
                }

                logging.info(f"Metrics for {model_name}: Train: {train_metrics}, CV: {cv_metrics}, Test: {test_metrics}")

                        ## Selecting the best model based on ROC AUC score
            best_model_name, best_model_data = max(
                model_report.items(),
                key=lambda x: x[1]["test_metrics"].get("roc_auc", 0)
            )

            # Use the best estimator from GridSearchCV for the selected model
            best_model = models[best_model_name]
            best_model.set_params(**best_model_data["best_params"])
            best_model.fit(X_train, y_train)  # Train the model again on full training data
            best_model_score = best_model_data["test_metrics"].get("roc_auc", 0)

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with ROC AUC > 0.6")

            logging.info(f"Best model: {best_model_name} with ROC AUC: {best_model_score}")

            # Saving the fitted best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            return {
                "best_model": best_model_name,
                "best_params": best_model_data["best_params"],
                "metrics": best_model_data["test_metrics"]
            }

        except KeyError as e:
            raise CustomException(f"Model key error: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)
""" 
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_classification_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "classification_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "Extra Trees Classifier": ExtraTreesClassifier(),
                "XGBClassifier": XGBClassifier(eval_metric='logloss'),
                "Support Vector Classifier": SVC(probability=True)
                }
            param_grids={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,128,256]
                }, 
                "Extra Trees Classifier": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                },
                
                "Gradient Boosting":{
                    # 'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Support Vector Classifier": {
                        'C': [0.1, 1, 10],
                        'gamma': [0.01, 0.1],  # Kernel coefficient

                        'kernel': ['linear', 'poly'],
                        'degree': [2, 3],
                        'probability': [True]
                    },
                "XGBClassifier": {
                        "n_estimators": [100, 200],
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1, 0.2]
                },
            
                }


            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name} with GridSearchCV.")
                
                grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Cross-validation predictions
                y_cv_pred = cross_val_predict(best_model, X_train, y_train, cv=3)
                y_cv_pred_proba = cross_val_predict(best_model, X_train, y_train, cv=3, method="predict_proba")[:, 1] if hasattr(best_model, "predict_proba") else None

                # Full training and evaluation on test set
                best_model.fit(X_train, y_train)
                y_train_pred = best_model.predict(X_train)
                y_train_pred_proba = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None
                y_test_pred = best_model.predict(X_test)
                y_test_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

                # Evaluate training, cross-validation, and test performance
                train_metrics = evaluate_classification_model(y_train, y_train_pred, y_train_pred_proba)
                cv_metrics = evaluate_classification_model(y_train, y_cv_pred, y_cv_pred_proba)
                test_metrics = evaluate_classification_model(y_test, y_test_pred, y_test_pred_proba)

                model_report[model_name] = {
                    "best_params": grid_search.best_params_,
                    "train_metrics": train_metrics,
                    "cv_metrics": cv_metrics,
                    "test_metrics": test_metrics
                }

                logging.info(f"Metrics for {model_name}: Train: {train_metrics}, CV: {cv_metrics}, Test: {test_metrics}")

                        ## Selecting the best model based on ROC AUC score
            best_model_name, best_model_data = max(
                model_report.items(),
                key=lambda x: x[1]["test_metrics"].get("roc_auc", 0)
            )

            # Use the best estimator from GridSearchCV for the selected model
            best_model = models[best_model_name]
            best_model.set_params(**best_model_data["best_params"])
            best_model.fit(X_train, y_train)  # Train the model again on full training data
            best_model_score = best_model_data["test_metrics"].get("roc_auc", 0)

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with ROC AUC > 0.6")

            logging.info(f"Best model: {best_model_name} with ROC AUC: {best_model_score}")

            # Saving the fitted best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            return {
                "best_model": best_model_name,
                "best_params": best_model_data["best_params"],
                "metrics": best_model_data["test_metrics"]
            }

        except KeyError as e:
            raise CustomException(f"Model key error: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)


"""
