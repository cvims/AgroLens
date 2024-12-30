import matplotlib.pyplot as plt
import optuna
import pandas as pd
import xgboost as xgb
import xgboost_predictor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

model_variant = 1 # 1 NN, 2 xgboost, 3 RandomForest

file_path = '/media/data/Datasets/Model_A_Dataset_v4_2024-12-30.csv'

feature_columns = ['B01_normalized', 'B02_normalized', 'B03_normalized',
       'B04_normalized', 'B05_normalized', 'B06_normalized', 'B07_normalized',
       'B08_normalized', 'B8A_normalized', 'B09_normalized', 'B11_normalized',
       'B12_normalized']

# Select target 'P_normalized', 'K_normalized', 'N_normalized'
target = 'K_normalized'  


# Read CSV
data = pd.read_csv(file_path)

# Extract the target column
targets = data[target].values

features = data[feature_columns].values

# Split train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

xgboost_predictor.run_xgboost_train(X_train, X_test, Y_train, Y_test)
