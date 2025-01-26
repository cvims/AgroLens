import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # Specify the model variant to be used: xgboost, nn, rf
    model_var = 'xgboost'
    model_config = 'Model_A'   # Select model (Differs in the used feature columns)

    # Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'
    target = 'N'
    path_loadmodel = f"/media/data/Models/{model_config}/{model_var}/{model_config}_{model_var}_{target}"

    # Feature columns
    if model_config == 'Model_A':
        feature_columns = ['norm_B01','norm_B02','norm_B03','norm_B04',
                        'norm_B05','norm_B06','norm_B07','norm_B08',
                        'norm_B8A','norm_B09','norm_B11','norm_B12']
    elif model_config == 'Model_A+':
        feature_columns = [] #tbd

    # Input data for predicition
    file_path = '/media/data/Datasets/Model_A_norm.csv'

    # Input data for prediction
    data = pd.read_csv(file_path)
    input_data = data[feature_columns]
    target_data = data[target]

    print(f'-----Prediction of {target} with {model_config} {model_var}-----')

    if model_var == 'xgboost':
        model = xgb.Booster()
        model.load_model(f'{path_loadmodel}.json')

        # Datatype change
        dinput_data = xgb.DMatrix(input_data)

        # XGBoost predicitions
        predictions = model.predict(dinput_data)

    elif model_var == 'nn':
        try:
            # Load Neural Network model with dynamic architecture
            checkpoint = torch.load(f'{path_loadmodel}.pth')
            
            if "model_info" in checkpoint:
                model_info = checkpoint["model_info"]
    
                # Reconstruct the model
                model = RegressionNet(
                    input_size=model_info["input_size"],
                    hidden_sizes=model_info["hidden_sizes"],
                    output_size=model_info["output_size"],
                    dropout_rates=model_info["dropout_rates"]
                )
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
    
                # Extract loss function from model metadata
                loss_function = model_info.get("loss_function", "MSE")
                print(f"Loss Function used for training: {loss_function}")
            else:
                print("Warning: No model_info found. Loading state_dict only.")
                # Provide default architecture if model_info is not found
                model = RegressionNet(input_size=12, hidden_sizes=[64, 32, 16], output_size=1, dropout_rates=[0.3, 0.3, 0.3])
                model.load_state_dict(checkpoint)
                model.eval()
                loss_function = "MSE"  # Default to MSE
    
        except KeyError as e:
            print(f"Error loading model: {e}")
            raise
    
        # Convert input data (DataFrame) to a NumPy array, then to a tensor
        input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
    
        # Neural Network predictions
        with torch.no_grad():
            predictions = model(input_tensor).numpy()

    elif model_var == 'rf':
        # Load Random Forest model
        model = joblib.load(f'{path_loadmodel}.joblib')

        # Random Forest predictions
        predictions = model.predict(input_data)

    # Calculate errors based on loss function
    if loss_function == 'MAE':
        mae_error = mean_absolute_error(target_data, predictions)
        print(f"Mean Absolute Error (MAE): {mae_error}")
    elif loss_function == 'Huber':
        # Huber Loss calculation can be complex; here, we approximate it with MAE for simplicity
        mae_error = mean_absolute_error(target_data, predictions)  # Replace with Huber-specific logic if needed
        print(f"Huber Loss (approximated with MAE): {mae_error}")
    else:  # Default to MSE
        mse_error = mean_squared_error(target_data, predictions)
        rmse_error = np.sqrt(mse_error)
        print(f"Mean Squared Error (MSE): {mse_error}")
        print(f"Root Mean Squared Error (RMSE): {rmse_error}")

    # Add predictions and errors to DataFrame
    data['Target'] = target_data
    data['Prediction'] = predictions
    data['Error'] = (data['Prediction'] - data[target])  # Fehler (quadratische Abweichung)

    # Results
    print(f"Root Mean Squared Error (RMSE): {rmse_error:.4f}")
    print(data[['norm_B01', 'norm_B02', 'norm_B03', 'norm_B04', 
                'norm_B05', 'norm_B06', 'norm_B07', 'norm_B08', 
                'norm_B8A', 'norm_B09', 'norm_B11', 'norm_B12', 
                'Target','Prediction', 'Error']])
    
    # output_file = f"/home/ubuntu/AgroLens/output/{model_config}_{model_var}_{target}_predictions.csv"
    # data.to_csv(output_file, index=False)
    # print(f"Predictions saved to {output_file}")
    
if __name__ == "__main__":
    main()
