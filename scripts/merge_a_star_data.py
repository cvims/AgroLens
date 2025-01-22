import pandas as pd

# Define file paths
input_file1 = "/media/data/Datasets/Model_A+_Soil+Sentinel_v4_with_weather+yield_norm.csv"
input_file2 = "/media/data/Datasets/Model_A+_1024_Clay_Embeddings_znorm.csv"
output_file = "/media/data/Datasets/Model_A+_norm.csv"

# Load the CSV files into DataFrames
df1 = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)

# Merge the DataFrames on 'POINTID', keeping only matching rows
merged_df = pd.merge(df1, df2, on="POINTID", how="inner")

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved as {output_file}")
