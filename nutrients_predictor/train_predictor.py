import dataloader_predictor as DL
import nn_predictor as NN_Pred
import pandas as pd
import torch.nn as nn
import xgboost_predictor
from sklearn.model_selection import train_test_split

# Specify the model variant to be used:
# xgboost, nn, rf
model_var = 'nn'

# Path to the dataset
file_path = '/media/data/Datasets/Model_A_Dataset_v4_2024-12-30.csv'

# Define the feature columns used for model training
feature_columns = ['B01_normalized', 'B02_normalized', 'B03_normalized',
       'B04_normalized', 'B05_normalized', 'B06_normalized', 'B07_normalized',
       'B08_normalized', 'B8A_normalized', 'B09_normalized', 'B11_normalized',
       'B12_normalized']

# Select target nutrient 'pH_CaCl2_normalized', 'pH_H2O_normalized', 'P_normalized', 'N_normalized', 'K_normalized'
target = 'K_normalized'  

# Load data table
data = pd.read_csv(file_path)

# Extract the target and feature columns as seperate np arrays
targets = data[target].values
features = data[feature_columns].values

# Split the dataset into training and testing sets
# 80% of data will be used for training, and 20% for testing
X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2, random_state=42)


if model_var == 'xgboost':
# Call the function to train an XGBoost model with the training and testing sets
# This function is assumed to be defined in the `xgboost_predictor` module
    xgboost_predictor.run_xgboost_train(X_train, X_test, Y_train, Y_test)
elif model_var == 'nn':
    dataloader_creator = DL.DataloaderCreator(file_path,target)
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
    print('Training pipeline initalized, start training:')

    # Start training
    pipeline.train()

    # Start evaluation
    pipeline.evaluate()
    
# elif model_var = 'rf': # Random forest
