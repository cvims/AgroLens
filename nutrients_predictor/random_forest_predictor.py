import joblib
import numpy as np
import optuna
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def objective(trial, X_train, X_test, Y_train, Y_test, save_path=None):
    """
    This function defines the optimization objective for the Optuna study. It:
    - Defines the hyperparameters to be tuned by Optuna.
    - Trains the random forest model with the given hyperparameters.
    - Calculates the RMSE of the model's predictions and saves the best model.
    """
    param = {
        # Amount of trees
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        # Maximum tree depth
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        # Minimum number of samples for split
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # Minimum number of samples in sheets
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        # Maximum number of features for splits
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }

    model = RandomForestRegressor(**param, random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE of Trial {trial.number}: {rmse}")

    # Save model with the best performance
    if save_path and (trial.number == 0 or rmse < trial.study.best_value):
        joblib.dump(model, save_path)
        print(f"Model with RMSE {rmse:.4f} saved to {save_path}.")

    return rmse


def run_random_forest_train(X_train, X_test, Y_train, Y_test, path_savemodel):
    """
    This function initiates the XGBoost training and hyperparameter optimization process:
    - Converts the training and testing datasets into DMatrix format for XGBoost.
    - Creates an Optuna study to minimize the RMSE by optimizing the hyperparameters.
    """
    study = optuna.create_study(direction="minimize")  # Minimize RMSE
    study.optimize(
        lambda trial: objective(
            trial, X_train, X_test, Y_train, Y_test, path_savemodel
        ),
        n_trials=10,
    )
    plot_data = optuna.visualization.plot_param_importances(
        study, evaluator=None, params=None, target=None, target_name="Objective Value"
    )
    fig = go.Figure(plot_data)
    fig.update_layout(
        title="Random Forest Parameter Sensitivity",
        xaxis_title="Relative Sensitivity",
        yaxis_title="Parameter",
        font=dict(size=18),
    )
    fig.show()

    print("Best hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)
