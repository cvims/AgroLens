import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def objective(trial, X_train, X_test, Y_train, Y_test):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # Amount of the trees
        'max_depth': trial.suggest_int('max_depth', 3, 30),  # Maximum tree depth
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # Minimum number of samples for split
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),  # Minimum number of samples in sheets
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)  # Maximum number of features for splits
    }

    model = RandomForestRegressor(**param, random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)

    return mse

def run_random_forest_train(X_train, X_test, Y_train, Y_test):
    
    study = optuna.create_study(direction='minimize')  # Minimize RMSE
    study.optimize(lambda trial: objective(trial, X_train, X_test, Y_train, Y_test), n_trials=100)

    print("Beste Hyperparameter:", study.best_params)
    print("Bester MSE-Wert:", study.best_value)