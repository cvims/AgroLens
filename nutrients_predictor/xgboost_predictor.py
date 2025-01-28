import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
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

    plot_data = optuna.visualization.plot_param_importances(study, evaluator=None, params=None, target=None, target_name='Objective Value')
    fig = go.Figure(plot_data)
    fig.update_layout(title="XGBoost Parameter Sensitivity",xaxis_title="Relative Sensitivity",yaxis_title="Parameter", font=dict(size=18))
    fig.show()

    print("Best hyperparameters of the trial", study.best_params)
    print("RMSE error of the model with the best hyperparameters:", np.sqrt(study.best_value))

def objective_with_scv(trial, folds, path_savemodel):
    """
        This function defines the optimization objective for the Optuna study. It:
        - Defines the hyperparameters to be tuned by Optuna.
        - Calculates the RMSE of the model's predictions and returns the average RMSE for the trial.

        Args:
            trial (optuna.Trial): The Optuna trial object, used to sample hyperparameters.
            folds (list): A list of cross-validation folds, where each fold is a tuple (X_train, y_train, X_test, y_test).
            path_savemodel (str): Path to save the model (optional, not used here).

        Returns:
            float: The average RMSE over all folds for this trial.
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

    fold_rmses = []
    
    for fold_idx,(X_train, y_train, X_test, y_test) in enumerate(folds):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(param, dtrain)

        prediction = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        fold_rmses.append(rmse)
    
    # Return the average RMSE over all folds for this trial
    avg_rmse = np.mean(fold_rmses)
    print(f'Average RMSE for trial {trial.number}: {avg_rmse}')
    return avg_rmse

def run_xgboost_scv_train(folds,X_val, y_val,path_savemodel, n_trials=50):
    """
        This function runs the XGBoost training with spatial cross-validation (SCV) and hyperparameter optimization.
        - Optimizes hyperparameters using Optuna to minimize RMSE.
        - Trains the final model with the best hyperparameters on the combined training data.
        - Evaluates the final model on a separate validation set and prints the validation RMSE.

        Args:
            folds (list): A list of cross-validation folds, where each fold is a tuple (X_train, y_train, X_test, y_test).
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
            path_savemodel (str): Path to save the trained model (optional, not used here).
            n_trials (int): Number of trials for Optuna optimization. Default is 50.
    """
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_with_scv(trial, folds, path_savemodel), n_trials=n_trials)
    
    print("Best hyperparameters:", study.best_params)
    print("Average RMSE of the model with the best hyperparameters:", study.best_value)

    # Train the final model with the best hyperparameters on the full training dataset
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'max_depth': study.best_params['max_depth'],
        'learning_rate': study.best_params['learning_rate'],
        'subsample': study.best_params['subsample'],
        'colsample_bytree': study.best_params['colsample_bytree'],
        'gamma': study.best_params['gamma'],
        'reg_alpha': study.best_params['reg_alpha'],
        'reg_lambda': study.best_params['reg_lambda'],
    }

    (X_train, y_train, X_test, y_test) = folds[0]
    X_train_full = np.vstack((X_train, X_test))
    y_train_full = np.hstack((y_train, y_test))
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    best_model = xgb.train(param, dtrain_full)
    
    best_model.save_model(path_savemodel)
    print(f"Final model saved to {path_savemodel}")

    dval = xgb.DMatrix(X_val, label=y_val)
    prediction = best_model.predict(dval)

    val_rmse = np.sqrt(mean_squared_error(y_val,prediction))
    print(f"Validation RMSE on unseen data: {val_rmse}")
    
    