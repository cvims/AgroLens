import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def objective(trial,dtrain, dtest,Y_test, path_savemodel):
    """
    This function defines the optimization objective for the Optuna study. It:
    - Defines the hyperparameters to be tuned by Optuna.
    - Trains the XGBoost model with the given hyperparameters.
    - Calculates the RMSE of the model's predictions and saves the best model.
    """
    param = {
        'objective': 'reg:squarederror', 
        'eval_metric': 'rmse', 
        'tree_method': 'hist', 

        # Hyperparameters to tune:
        'max_depth': trial.suggest_int('max_depth', 3, 12), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),   # Column sampling per tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),   # Column sampling per tree
        'gamma': trial.suggest_float('gamma', 0, 1),  # Regularization parameter
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),   # L2 regularization
    }

    model = xgb.train(param,dtrain)

    Y_pred = model.predict(dtest)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    print(f'RMSE of Trial {trial.number}: {rmse}')

    if trial.number == 0 or mse < trial.study.best_value:
        model.save_model(path_savemodel)
        print(f'Model with RMSE {rmse} saved.')

    return rmse


def run_xgboost_train(X_train, X_test, Y_train, Y_test, path_savemodel):
    """
        This function initiates the XGBoost training and hyperparameter optimization process:
        - Converts the training and testing datasets into DMatrix format for XGBoost.
        - Creates an Optuna study to minimize the RMSE by optimizing the hyperparameters.
    """
    dtrain = xgb.DMatrix(X_train,label=Y_train)
    dtest = xgb.DMatrix(X_test,label=Y_test)
    
    study = optuna.create_study(direction='minimize')  #Minimize RMSE
    study.optimize(lambda trial: objective(trial, dtrain, dtest, Y_test, path_savemodel), n_trials=50)

    print("Best hyperparameters of the trial", study.best_params)
    print("RMSE error of the model with the best hyperparameters:", np.sqrt(study.best_value))
    