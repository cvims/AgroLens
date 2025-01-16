import dataloader_predictor as DL
import nn_predictor as nn_pred
import pandas as pd
import rf_predictor as rf_pred
import torch.nn as nn
import xgboost_predictor


def main():
    # Specify the model variant to be used: xgboost, nn, rf
    model_var = 'xgboost'
    model_config = 'Model_A'   # Select model (Differs in the used feature columns)

    # Define the feature columns used for model training
    if model_config == 'Model_A':
        feature_columns = ['norm_B01','norm_B02','norm_B03','norm_B04',
                        'norm_B05','norm_B06','norm_B07','norm_B08',
                        'norm_B8A','norm_B09','norm_B11','norm_B12']
    elif model_config == 'Model_A+':
        feature_columns = [] #tbd

    # Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'
    target = 'K'
    path_savemodel = f"/media/data/Models/{model_config}/{model_var}/{model_config}_{model_var}_{target}.json"

    # Path to the dataset
    file_path = '/media/data/Datasets/Model_A_Soil+Sentinel_norm.csv'

    dataloader_creator = DL.DataloaderCreator(file_path,target,feature_columns)

    if model_var == 'xgboost':
    # Call the function to train an XGBoost model
    # This function is assumed to be defined in the `xgboost_predictor` module
        print('-----Start model training: XGBoost-----')
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        xgboost_predictor.run_xgboost_train(X_train, X_test, Y_train, Y_test, path_savemodel)
        print('-----End model training: XGBoost-----')

    elif model_var == 'nn':
        # Erstelle Dataloader für Training und Test
        print('-----Start model training: Neuronal Network-----')
        train_loader, test_loader = dataloader_creator.create_dataloaders()
        nn_pred.run_nn_train(train_loader, test_loader)
        print('-----End model training: Neuronal Network-----')

    elif model_var == 'rf': # Random forest
        print('-----Start model training: Random Forest-----')
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        rf_pred.run_random_forest_train(X_train, X_test, Y_train, Y_test)
        print('-----End model training: Neuronal Network-----')

    print('Used features:', feature_columns)
    print('Selected target: ', target)

if __name__ == "__main__":
    main()
