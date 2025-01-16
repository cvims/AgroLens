import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def main():
    # Specify the model variant to be used: xgboost, nn, rf
    model_var = 'xgboost'
    model_config = 'Model_A'   # Select model (Differs in the used feature columns)

    # Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'
    target = 'pH_H2O'
    path_loadmodel = f"/media/data/Models/{model_config}/{model_var}/{model_config}_{model_var}_{target}.json"

    # Feature columns
    if model_config == 'Model_A':
        feature_columns = ['norm_B01','norm_B02','norm_B03','norm_B04',
                        'norm_B05','norm_B06','norm_B07','norm_B08',
                        'norm_B8A','norm_B09','norm_B11','norm_B12']
    elif model_config == 'Model_A+':
        feature_columns = [] #tbd

    # Input data for predicition
    file_path = '/media/data/Datasets/Model_A_Soil+Sentinel_norm.csv'

    # Read CSV
    data = pd.read_csv(file_path)
    input_data = data[feature_columns]
    target_data = data[target]

    print(f'-----Prediction of {target} with {model_config} {model_var}-----')
    if model_var == 'xgboost':
        # Load xgboost and make prediciton
        model = xgb.Booster()
        model.load_model(path_loadmodel)

        # Datatype change
        dinput_data = xgb.DMatrix(input_data)

        # Vorhersagen für den gesamten Datensatz machen
        predictions = model.predict(dinput_data)

        # RMSE-Fehler berechnen
        mse_error = mean_squared_error(target_data, predictions)
        rmse_error = np.sqrt(mse_error)

    elif model_var == 'nn':
        print()

    elif model_var == 'rf':
        print()




    # Vorhersagen und Fehler zum DataFrame hinzufügen
    data['prediction'] = predictions
    data['RMSE'] = (data['prediction'] - data[target]) ** 2  # Fehler (quadratische Abweichung)

    # Ergebnisse anzeigen
    print(f"Root Mean Squared Error (RMSE): {rmse_error:.4f}")
    print(data[['norm_B01', 'norm_B02', 'norm_B03', 'norm_B04', 
                'norm_B05', 'norm_B06', 'norm_B07', 'norm_B08', 
                'norm_B8A', 'norm_B09', 'norm_B11', 'norm_B12', 
                'prediction', 'RMSE']])
    
if __name__ == "__main__":
    main()