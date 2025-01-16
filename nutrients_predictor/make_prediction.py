import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def main():
    # Get input
    feature_columns = ['norm_B01','norm_B02','norm_B03','norm_B04',
                            'norm_B05','norm_B06','norm_B07','norm_B08',
                            'norm_B8A','norm_B09','norm_B11','norm_B12']

    # Load model
    model = xgb.Booster()
    model.load_model('/home/ubuntu/Data/Models/Model_A/xgboost_2025-01-16/Model_A_xgboost_N.json')

    # Path to dataset
    file_path = '/media/data/Datasets/Model_A_Soil+Sentinel_norm.csv'

    # Read CSV
    data = pd.read_csv(file_path)

    # Definiere die Eingabespalten und die Zielspalte (tatsächliche Werte)
    feature_columns = ['norm_B01','norm_B02','norm_B03','norm_B04',
                            'norm_B05','norm_B06','norm_B07','norm_B08',
                            'norm_B8A','norm_B09','norm_B11','norm_B12']
    # Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'
    target = 'K'

    # Wählen Sie die relevanten Spalten für den Input aus
    input_data = data[feature_columns]
    dinput_data = xgb.DMatrix(input_data)
    target_data = data[target]

    # Vorhersagen für den gesamten Datensatz machen
    predictions = model.predict(dinput_data)

    # RMSE-Fehler berechnen
    mse_error = mean_squared_error(target_data, predictions)
    rmse_error = np.sqrt(mse_error)

    # Vorhersagen und Fehler zum DataFrame hinzufügen
    data['prediction'] = predictions
    data['error'] = (data['prediction'] - data[target]) ** 2  # Fehler (quadratische Abweichung)

    # Ergebnisse anzeigen
    print(f"Root Mean Squared Error (RMSE): {rmse_error:.4f}")
    print(data[['norm_B01', 'norm_B02', 'norm_B03', 'norm_B04', 
                'norm_B05', 'norm_B06', 'norm_B07', 'norm_B08', 
                'norm_B8A', 'norm_B09', 'norm_B11', 'norm_B12', 
                'prediction', 'error']])
    
if __name__ == "__main__":
    main()