import dataloader_predictor as DL
import nn_predictor as NN_Pred
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
    target = 'P_normalized'  

    dataloader_creator = DL.DataloaderCreator(file_path,target,feature_columns)

    if model_var == 'xgboost':
    # Call the function to train an XGBoost model
    # This function is assumed to be defined in the `xgboost_predictor` module
        X_train, X_test, Y_train, Y_test = dataloader_creator.create_xgboost_data()
        print('-----Start model training: XGBoost-----')
        xgboost_predictor.run_xgboost_train(X_train, X_test, Y_train, Y_test)

    elif model_var == 'nn':
        # Erstelle Dataloader für Training und Test
        train_loader, test_loader = dataloader_creator.create_dataloaders()

        # Initialize TrainingPipeline
        batch_size = 5
        pipeline = NN_Pred.TrainingPipeline(
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=0.001,
            optimizer_type="Adam",
            criterion=nn.MSELoss(), 
            batch_size=batch_size,
            num_epochs=5
        )
        print('-----Start model training: Neuronal network-----')
        pipeline.train()
        pipeline.evaluate()
        
    elif model_var == 'rf': # Random forest
        print('-----Start model training: Random Forest-----')

if __name__ == "__main__":
    main()
