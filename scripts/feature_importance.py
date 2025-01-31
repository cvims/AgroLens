#!/usr/bin/env python
# Script to read a data table, calculate the feature importance for xgboost and plot it

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

model_var = "xgboost"
model_config = "Model_A"

# Select target nutrient 'pH_CaCl2', 'pH_H2O', 'P', 'N', 'K'
target = "N"

path_loadmodel = (
    f"/media/data/Models/{model_config}/{model_var}/{model_config}_{model_var}_{target}"
)

feature_columns = [
    "norm_B01",
    "norm_B02",
    "norm_B03",
    "norm_B04",
    "norm_B05",
    "norm_B06",
    "norm_B07",
    "norm_B08",
    "norm_B8A",
    "norm_B09",
    "norm_B11",
    "norm_B12",
]

# Input data for predicition
file_path = "/media/data/Datasets/Model_A_norm.csv"


data = pd.read_csv(file_path)
input_data = data[feature_columns]
target_data = data[target]

print(f"-----Prediction of {target} with {model_config} {model_var}-----")

if model_var == "xgboost":
    model = xgb.Booster()
    model.load_model(f"{path_loadmodel}.json")
    dinput_data = xgb.DMatrix(input_data)

importance = model.get_score(importance_type="gain")

feature_name_map = {f"f{i}": feature_columns[i] for i in range(len(feature_columns))}

importance_with_names = {
    feature_name_map[key]: value for key, value in importance.items()
}

importance_df = pd.DataFrame(
    importance_with_names.items(), columns=["Feature", "Importance"]
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

importance_df.plot(kind="bar", x="Feature", y="Importance", legend=False)

plt.ion()
plt.title("Feature Importance by Gain", fontsize=20)
plt.ylabel("Importance", fontsize=18)
plt.xlabel("Features", fontsize=18)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
plt.show()

plt.savefig("Feature-Importance-Gain_Model-A.norm.png", format="png")
