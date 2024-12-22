import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

file_path = '/home/ubuntu/AgroLens/Example_Data_combined.csv'
target = 'K_normalized'    # 'P_normalized', 'K_normalized', 'N_normalized'
# CSV
data = pd.read_csv(file_path)

# Extract the target column
targets = data[target].values
        
# Select features columns
feature_columns = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']
data = data[feature_columns].values

# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
print('Split train- and testdata')

# Train XGBoost-Modell
print('Modelltraining')
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=100)
model.fit(X_train, y_train)

# Vorhersagen für das Testset
y_pred = model.predict(X_test)

# Berechnung des RMSE und R²-Score
MSE = mean_squared_error(y_test, y_pred)

print('Mean squared error:',MSE)

# Feature-Importances visualisieren
# xgb.plot_importance(model, importance_type='weight', max_num_features=10)
# plt.show()

# Modell speichern (optional)
#joblib.dump(model, 'xgboost_model.pkl')