import json

import dataloader_predictor as DL
import nn_predictor as nn_pred
import rf_predictor as rf_pred
import xgboost_predictor


def main():
    
    """
        This script trains a selected model variant based on a specified model configuration and target nutrient. 
        It supports three model variants: XGBoost, Neural Network (NN), and Random Forest (RF). 

        Key functionalities:
        - Allows selection of a model configuration that defines the feature columns used for training.
        - Enables training for a specific target nutrient ('pH_CaCl2', 'pH_H2O', 'P', 'N', 'K').

        Usage:
        - Modify the `model_var`, `model_config`, and `target` variables to set the desired training parameters.
        - Ensure the dataset paths and feature columns are correctly defined for the selected model configuration.
    """

    model_var = 'xgboost'       # Specify the model variant to be used: xgboost, nn, rf
    model_config = 'Model_A+'    # Select model Model_A or Model_A+(Differs in the used feature columns)
    target = 'K'                # Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'

    path_savemodel = f"/media/data/Models/{model_config}/{model_var}/{model_config}_{model_var}_{target}"
    config_path = '/media/data/Datasets/Feature_Cols/model_settings.json'

    with open(config_path, 'r') as file:
        configfile = json.load(file)
    feature_columns = configfile[model_config]['feature_columns']

    # Define the feature columns used for model training
    if model_config == 'Model_A':
        file_path = '/media/data/Datasets/Model_A_norm.csv'
    elif model_config == 'Model_A+':
        file_path = '/media/data/Datasets/Model_A+_norm.csv'

    dataloader_creator = DL.DataloaderCreator(file_path,target,feature_columns)

    if model_var == 'xgboost':
        print('-----Start model training: XGBoost-----')
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        xgboost_predictor.run_xgboost_train(X_train, X_test, Y_train, Y_test, f'{path_savemodel}.json')
        print('-----End model training: XGBoost-----')

    elif model_var == 'nn':
        # Erstelle Dataloader für Training und Test
        print('-----Start model training: Neuronal Network-----')
        train_loader, test_loader = dataloader_creator.create_dataloaders()
        nn_pred.run_nn_train(train_loader, test_loader, f'{path_savemodel}.pth')
        print('-----End model training: Neuronal Network-----')

    elif model_var == 'rf': # Random forest
        print('-----Start model training: Random Forest-----')
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        rf_pred.run_random_forest_train(X_train, X_test, Y_train, Y_test, f'{path_savemodel}.joblib')
        print('-----End model training: Neuronal Network-----')

    print('Selected target: ', target)

if __name__ == "__main__":
    main()
