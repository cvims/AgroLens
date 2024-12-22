import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Path to dataset
file_path = '/home/ubuntu/AgroLens/Example_Data_combined.csv'

# Select target 'P_normalized', 'K_normalized', 'N_normalized'
target = 'K_normalized'  

# Read CSV
data = pd.read_csv(file_path)

# Extract the target column
targets = data[target].values
        
# Select features columns
feature_columns = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']
data = data[feature_columns].values

# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

# Train XGBoost-Modell
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=100)
model.fit(X_train, y_train)

# Prediction on testset
y_pred = model.predict(X_test)

# Calculate MSE
MSE = mean_squared_error(y_test, y_pred)
print('Mean squared error:',MSE)