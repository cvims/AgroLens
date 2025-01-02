import dataloader_predictor as DL
import nn_predictor as nn_pred
import pandas as pd
import torch.nn as nn
import xgboost_predictor


def main():
    # Specify the model variant to be used: xgboost, nn, rf
    model_var = 'nn'

    # Path to the dataset
    file_path = '/media/data/Datasets/Model_A_Dataset_v4_2024-12-30.csv'

    # Define the feature columns used for model training
    feature_columns = ['B01_normalized', 'B02_normalized', 'B03_normalized',
        'B04_normalized', 'B05_normalized', 'B06_normalized', 'B07_normalized',
        'B08_normalized', 'B8A_normalized', 'B09_normalized', 'B11_normalized',
        'B12_normalized']

    # Select target nutrient 'pH_CaCl2_normalized', 'pH_H2O_normalized', 'P_normalized', 'N_normalized', 'K_normalized'
    target = 'pH_CaCl2_normalized' 
    
    print('Used features:', feature_columns)
    print('Selected target: ', target)

    dataloader_creator = DL.DataloaderCreator(file_path,target,feature_columns)

    if model_var == 'xgboost':
    # Call the function to train an XGBoost model
    # This function is assumed to be defined in the `xgboost_predictor` module
        print('-----Start model training: XGBoost-----')
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        xgboost_predictor.run_xgboost_train(X_train, X_test, Y_train, Y_test)
        print('-----End model training: XGBoost-----')

    elif model_var == 'nn':
        # Erstelle Dataloader für Training und Test
        print('-----Start model training: Neuronal Network-----')
        train_loader, test_loader = dataloader_creator.create_dataloaders()
        nn_pred.run_nn_train(train_loader, test_loader)
        print('-----End model training: Neuronal Network-----')

    elif model_var == 'rf': # Random forest
        print('-----Start model training: Random Forest-----')

if __name__ == "__main__":
    main()
