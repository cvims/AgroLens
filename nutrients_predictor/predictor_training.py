#!/usr/bin/env python
import json
import os
from pathlib import Path

import neural_network_predictor as nn_pred
import random_forest_predictor as rf_pred
import xgboost_predictor
from dataloader_creator import DataloaderCreator


def run_model(model_var, model_config, target, validation, include_optional_data=True):
    """
    This function trains the selected model (XGBoost, Neural Network, or Random Forest) based on the given
    model configuration and target nutrient. It also manages the loading of the required datasets and
    feature columns.

    - Loads the model configuration and feature columns.
    - Loads and prepares the training and test datasets.
    - Runs the training process for the selected model variant (XGBoost, NN, RF).
    - Saves the trained model with a specific path based on model configuration and target nutrient.
    """

    model_path = (
        Path(os.environ["MODEL_PATH"])
        / model_config
        / model_var
        / f"{model_config}_{model_var}_{target}"
    )

    config_path = (
        Path(os.environ["DATASET_PATH"]) / "Feature_Cols" / "model_settings.json"
    )

    # Load the configuration file and get feature columns for the model configuration
    with open(config_path, "r") as file:
        configfile = json.load(file)
    feature_columns = configfile[model_config]["feature_columns"]

    # If optional data is included, extend the feature columns
    if include_optional_data:
        feature_columns.extend(configfile[model_config]["optional_feature_columns"])
    input_size = len(feature_columns)

    file_path = Path(os.environ["DATASET_PATH"]) / f"{model_config}_norm.csv"
    dataloader_creator = DataloaderCreator(str(file_path), target, feature_columns)

    if model_var == "xgboost":
        if validation == "Single":
            print("-----Start model training: XGBoost-----")
            X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
            xgboost_predictor.run_xgboost_train(
                X_train, X_test, Y_train, Y_test, f"{model_path}.json"
            )
            print("-----End model training: XGBoost-----")

        if validation == "Spatial":
            print(
                "-----Start model training with Spatial Cross Validation: XGBoost-----"
            )
            folds, (X_val, y_val) = dataloader_creator.create_scv_data()
            xgboost_predictor.run_xgboost_scv_train(
                folds, X_val, y_val, f"{model_path}_SCV.json"
            )
            print("-----End model training with Spatial Cross Validation: XGBoost-----")

    elif model_var == "nn":
        print("-----Start model training: Neuronal Network-----")
        train_loader, test_loader = dataloader_creator.create_dataloaders()
        nn_pred.run_nn_train(input_size, train_loader, test_loader, f"{model_path}.pth")
        print("-----End model training: Neuronal Network-----")

    elif model_var == "rf":
        print("-----Start model training: Random Forest-----")
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        rf_pred.run_random_forest_train(
            X_train, X_test, Y_train, Y_test, f"{model_path}.joblib"
        )
        print("-----End model training: Neuronal Network-----")

    print(
        f"Model_config: {model_config}, Model variant: {model_var}, Selected target: ",
        target,
    )


def main():
    """
    The main function runs the model training process for all possible combinations of model variants,
    configurations, and target nutrients.

    It iterates over all model variants, configurations, and target nutrients, calling `run_model`
    to train and save models for each combination.
    """

    model_vars = ["xgboost", "nn", "rf"]
    # model_configs = ["Model_A", "Model_A+"]
    targets = ["pH_CaCl2", "pH_H2O", "P", "N", "K"]

    model_config = "Model_A+"  # Select model Model_A or Model_A+ (Differs in the used feature columns)
    model_var = "xgboost"  # Specify the model variant to be used: xgboost, nn, rf
    target = "pH_CaCl2"  # Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'
    include_optional_data = True  # Currently comprises Clay data
    validation = "Single"  # Select validation method 'Single' or 'Spatial' (Spatial Cross Validation is only available for XGBoost)

    # Loop over model variants and target nutrients
    for model_var in model_vars:
        # for model_config in model_configs:
        for target in targets:
            print("-" * 30)
            print(
                f"Training with Model Config: {model_config}, Model: {model_var}, Target: {target}, Optional Data: {include_optional_data}"
            )
            run_model(
                model_var, model_config, target, validation, include_optional_data
            )
            print()
    print("-" * 30)
    print("Done!")


if __name__ == "__main__":
    main()
