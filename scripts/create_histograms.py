import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # ----- CONFIGURATION -----
    input_csv = '/media/data/Datasets/Model_A+_Soil+Sentinel_v4_with_weather+yield.csv'  # CSV file to read from
    histogram_folder = '/media/data/Datasets/Histograms'
    
    # Create the histogram folder if it doesn't already exist
    if not os.path.exists(histogram_folder):
        os.makedirs(histogram_folder)

    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Generate and save histograms for numeric columns
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=30)  # Adjust bins to your preference
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Save the histogram
        output_path = os.path.join(histogram_folder, f"{col}.png")
        plt.savefig(output_path)
        plt.close()

if __name__ == '__main__':
    main()
