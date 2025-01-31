#!/usr/bin/env python
# Preprocesses the WoSIS SOIL CSV

import pandas as pd

WOSIS_INPUT = "/home/ubuntu/Data/Datasets/Soil/raw/WoSIS/"
WOSIS_OUTPUT = "/home/ubuntu/Data/Datasets/Soil/processed/WoSIS/"
LUCAS_OUTPUT_FILE = (
    "/home/ubuntu/Data/Datasets/Soil/processed/LUCAS/LUCAS_Soil_2018.csv"
)

# Nitrogen: NITKJD
# Phosphorus: PHETB1, PHETM3, PHETOL, PHPRTN, PHPTOT, PHPWSL
# pH: PHAQ, PHCA, PHKC, PHNF
# Required elements
elements = [
    "NITKJD",
    "PHETB1",
    "PHETM3",
    "PHETOL",
    "PHPTOT",
    "PHPWSL",
    "PHAQ",
    "PHCA",
    "PHKC",
    "PHNF",
]

for element in elements:
    df = pd.read_csv(WOSIS_INPUT + "wosis_202312_" + element + ".tsv", sep="\t")

    # World without Europe to avoid overlap/bias with LUCAS Soil
    df = df.loc[df["continent"] != "Europe"]

    # Without Samples having depth > 25 cm as satellites cannot measure deeper then 25 cm into the ground
    df = df.loc[df["upper_depth"] <= 25]
    df = df.loc[df["lower_depth"] <= 25]

    # Regex to filter dates based on Landsat 8 availability
    pattern = r"\d{4}-\d{1,2}-(?:\W{1,2}|\d{2})$"
    date_column = df["date"]
    match = date_column.str.match(pattern)
    df = df[match]
    df = df.loc[df["date"] >= "2008-03-01"]

    df["element"] = element

    df.to_csv(WOSIS_OUTPUT + "tmp/" + element + "_new_.csv", index=False)

# Iterate through relevant files
for element in elements:
    df = pd.read_csv(WOSIS_OUTPUT + "tmp/" + element + "_new_.csv")

    # Group and calc average in case multiple values have been collected for the same sample and save to .csv
    df1 = (
        df.groupby(
            ["longitude", "latitude", "date", "continent", "region", "country_name"]
        )["value_avg"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "longitude": "TH_LONG",
                "latitude": "TH_LAT",
                "date": "SURVEY_DATE",
            }
        )
    )
    df1.to_csv(WOSIS_OUTPUT + "tmp/" + element + "_grouped_.csv", index=False)

# pH: PHAQ, PHCA, PHKC, PHNF
df1 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHAQ_grouped_.csv")
df2 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHCA_grouped_.csv")
df3 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHKC_grouped_.csv")
df4 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHNF_grouped_.csv")

# Set index per df to a combination of TH_LONG, TH_LAT, SURVEY_DATE
df1.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df2.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df3.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df4.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)

# Outer JOIN across PHAQ, PHCA, PHKC and PHNF
df_combined_pH = (
    df1.join(df2, how="outer", rsuffix="_PHCA")
    .join(df3, how="outer", rsuffix="_PHKC")
    .join(df4, how="outer", rsuffix="_PHNF")
)

# Calculate mean per row over PHCA, PHCA, PHNF and PHKC values
df_combined_pH["mean_pH"] = df_combined_pH[
    ["value_avg", "value_avg_PHCA", "value_avg_PHKC", "value_avg_PHNF"]
].mean(axis=1, skipna=True)
df_combined_pH.reset_index(inplace=True)

df_combined_pH.to_csv(path_or_buf=WOSIS_OUTPUT + "tmp/" + "WoSIS_pH.csv")

# N: NITKJD
df1 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "NITKJD_grouped_.csv")
df1.rename(columns={"value_avg": "N"}, inplace=True)
df1.to_csv(path_or_buf=WOSIS_OUTPUT + "tmp/" + "WoSIS_N.csv")

# P: PHETB1, PHETM3, PHETOL, PHPTOT, PHPWSL
df1 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHETB1_grouped_.csv")
df2 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHETM3_grouped_.csv")
df3 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHETOL_grouped_.csv")
df4 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHPTOT_grouped_.csv")
df5 = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "PHPWSL_grouped_.csv")

# Set index per df to a combination of TH_LONG, TH_LAT, SURVEY_DATE
df1.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df2.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df3.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df4.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df5.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)

# Outer JOIN across PHAQ, PHCA, PHKC and PHNF
df_combined_P = (
    df1.join(df2, how="outer", rsuffix="_PHETM3")
    .join(df3, how="outer", rsuffix="_PHETOL")
    .join(df4, how="outer", rsuffix="_PHPTOT")
    .join(df5, how="outer", rsuffix="_PHPWSL")
)

# Calculate mean per row over PHETB1, PHETM3, PHETOL, PHPTOT and PHPWSL values
df_combined_P["mean_P"] = df_combined_P[
    [
        "value_avg",
        "value_avg_PHETM3",
        "value_avg_PHETOL",
        "value_avg_PHPTOT",
        "value_avg_PHPWSL",
    ]
].mean(axis=1, skipna=True)
df_combined_P.reset_index(inplace=True)
df_combined_P.to_csv(path_or_buf=WOSIS_OUTPUT + "tmp/" + "WoSIS_P.csv")

# Merge pH, N and P together in single data frame
FILE_PATH = "/Users/lscheermann/THI/AgroLens/Data/WoSIS/WoSIS_pH.csv"
df_pH = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "WoSIS_pH.csv")
df_N = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "WoSIS_N.csv")
df_P = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "WoSIS_P.csv")

# Keep only TH_LONG, TH_LAT, SURVEY_DATE, Continent, Country and measured values for pH, N and P
df_pH.drop(columns=["Unnamed: 0"], inplace=True)
df_N.drop(columns=["Unnamed: 0"], inplace=True)
df_P.drop(columns=["Unnamed: 0"], inplace=True)

# Reset Index
df_pH.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df_N.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df_P.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)

# Outer Join across pH, N and P
df_final = df_pH.merge(
    df_N, on=["TH_LONG", "TH_LAT", "SURVEY_DATE"], how="outer"
).merge(df_P, on=["TH_LONG", "TH_LAT", "SURVEY_DATE"], how="outer")
df_final.rename(
    columns={
        "mean_pH": "pH_H2O",
        "mean_P": "P",
        "continent_x": "continent",
        "region_x": "region",
        "country_name_x": "country_name",
    },
    inplace=True,
)
df_final.to_csv(WOSIS_OUTPUT + "tmp/" + "WoSIS_2023.csv")

# Merge LUCAS Soil and WoSIS data in single file to get one file for whole world soil data
# 18909 LUCAS records
df_LUCAS = pd.read_csv(LUCAS_OUTPUT_FILE)
df_LUCAS.drop(columns="Unnamed: 0", inplace=True)
df_LUCAS.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)

# 3641 WoSIS records
df_WoSIS = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "WoSIS_2023.csv")
df_WoSIS.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)

# Union
df_combined_final = pd.concat([df_LUCAS, df_WoSIS])
df_combined_final.reset_index(inplace=True)

df_combined_final[
    [
        "TH_LONG",
        "TH_LAT",
        "SURVEY_DATE",
        "continent",
        "region",
        "country_name",
        "pH_CaCl2",
        "pH_H2O",
        "P",
        "N",
        "K",
    ]
].to_csv(WOSIS_OUTPUT + "WoSIS+LUCAS_Global.csv")

# WoSIS data in single file (WITHOUT Europe)
df_WoSIS = pd.read_csv(WOSIS_OUTPUT + "tmp/" + "WoSIS_2023.csv")
df_WoSIS.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)

# Union
df_combined_final = df_WoSIS
df_combined_final.reset_index(inplace=True)

df_combined_final[
    [
        "TH_LONG",
        "TH_LAT",
        "SURVEY_DATE",
        "continent",
        "region",
        "country_name",
        "pH_H2O",
        "P",
        "N",
    ]
].to_csv(WOSIS_OUTPUT + "WoSIS_Rest_of_World_without_Europe.csv")

# WoSIS data by continent
df_WoSIS = pd.read_csv(WOSIS_OUTPUT + "WoSIS_Rest_of_World_without_Europe.csv")
df_WoSIS.set_index(["TH_LONG", "TH_LAT", "SURVEY_DATE"], inplace=True)
df_WoSIS.drop(columns={"Unnamed: 0"}, inplace=True)

df_WoSIS[df_WoSIS["continent"] == "Africa"].to_csv(WOSIS_OUTPUT + "WoSIS_Africa.csv")
df_WoSIS[df_WoSIS["continent"] == "Asia"].to_csv(WOSIS_OUTPUT + "WoSIS_Asia.csv")
df_WoSIS[df_WoSIS["continent"] == "Northern America"].to_csv(
    WOSIS_OUTPUT + "WoSIS_Northern_America.csv"
)
df_WoSIS[df_WoSIS["continent"] == "Oceania"].to_csv(WOSIS_OUTPUT + "WoSIS_Oceania.csv")
df_WoSIS[df_WoSIS["continent"] == "South America"].to_csv(
    WOSIS_OUTPUT + "WoSIS_South America.csv"
)
