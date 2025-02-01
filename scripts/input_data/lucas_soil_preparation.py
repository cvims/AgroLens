#!/usr/bin/env python
import pandas as pd

# Specify LUCAS Soil path here
LUCAS_INPUT = "/home/ubuntu/Data/Datasets/Soil/raw/LUCAS/LUCAS-SOIL-2018.csv"
LUCAS_OUTPUT = "/home/ubuntu/Data/Datasets/Soil/processed/LUCAS/"
df = pd.read_csv(LUCAS_INPUT)

# Drop unnecessary chemical elements
df.drop(
    columns=[
        "EC",
        "OC",
        "CaCO3",
        "OC (20-30 cm)",
        "CaCO3 (20-30 cm)",
        "Ox_Al",
        "Ox_Fe",
    ],
    inplace=True,
)
# Drop depth column as it's values are '0-10 cm', '0-20 cm', or '10-20 cm'
df.drop(columns=["Depth"], inplace=True)
# Note: Values for pH exist in 2 columns as they have been tested by 2 separate test methods pH_CaCl2 and pH_H2O
# The patterns are very similar for both, however pH_H2O is about 0.4 higher than pH_CaCl2
# Hence assuming it doesn't matter which of the 2 we use for training

# Drop rows that match "< LOD" in 'N' and 'K' (Total number of deleted rows: 1)
df = df[df["N"].str.contains("< LOD") == False]
df = df[df["K"].str.contains("< LOD") == False]

# Impute 'P' where it is below 10 mg/kg (limit of detection) with 5 (Average between 0 and 10) -> total number of imputed records: 4945
df["P"] = df["P"].replace("< LOD", 5)
# Impute 'P' where it is <0.0 with 0 -> total number of imputed records: 1
df["P"] = df["P"].replace("<0.0", 0)

df.dropna(subset=["P"], inplace=True)
df["pH_CaCl2"] = df["pH_CaCl2"].astype(float)
df["pH_H2O"] = df["pH_H2O"].astype(float)
df["P"] = df["P"].astype(float)
df["N"] = df["N"].astype(float)
df["K"] = df["K"].astype(float)
df["SURVEY_DATE"] = pd.to_datetime(df["SURVEY_DATE"], format="%d-%m-%y")

df.to_csv(path_or_buf=LUCAS_OUTPUT + "LUCAS_Soil_2018.csv")
