import pandas as pd

# Define file paths
input_file1 = "/media/data/Datasets/Model_A+_weather.csv" # File 1, modify accordingly
input_file2 = "/media/data/Datasets/Model_A+_yield.csv"   # File 2, modify accordingly
output_file = "/media/data/Datasets/Model_A+.csv"

# Load the CSV files into DataFrames
df1 = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)

# Merge the DataFrames on 'POINTID', keeping only matching rows
merged_df = pd.merge(df1, df2, on="POINTID", how="inner")

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved as {output_file}")
