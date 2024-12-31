import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def objective(trial,dtrain, dtest,Y_test):
    param = {
        'objective': 'reg:squarederror',  # Regressionsziel
        'eval_metric': 'rmse',  # Metrik: Root Mean Squared Error (RMSE)
        'tree_method': 'hist',  # Benutze den 'hist' Baum-Algorithmus für Geschwindigkeit

         # Suche nach den besten Hyperparametern
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # Baumtiefe (zwischen 3 und 12)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  # Lernrate
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Subsampling-Rate
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Spalten-Sampling
        'gamma': trial.suggest_float('gamma', 0, 1),  # Gamma (Regularisierungsterm)
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),  # L1 Regularisierung
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),  # L2 Regularisierung
    }

    model = xgb.train(param,dtrain)

    Y_pred = model.predict(dtest)
    mse = mean_squared_error(Y_test, Y_pred)

    return mse


def run_xgboost_train(X_train, X_test, Y_train, Y_test):

    dtrain = xgb.DMatrix(X_train,label=Y_train)
    dtest = xgb.DMatrix(X_test,label=Y_test)
    
    study = optuna.create_study(direction='minimize')  #Minimize RMSE
    study.optimize(lambda trial: objective(trial, dtrain, dtest, Y_test), n_trials=100)

    print("Beste Hyperparameter:", study.best_params)
    print("Bester MSE-Wert:", study.best_value)