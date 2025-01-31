#!/usr/bin/env python
# Script to generate histograms for a given CSV data table

import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # ----- CONFIGURATION -----
    input_csv = f"{os.environ["DATASET_PATH"]}/Model_A+.csv"
    histogram_folder = "/media/data/Datasets/Histograms_A+"

    # Create the histogram folder if it doesn't already exist
    if not os.path.exists(histogram_folder):
        os.makedirs(histogram_folder)

    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Generate and save histograms for numeric columns
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=30)  # Adjust bins to your preference
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Save the histogram
        output_path = os.path.join(histogram_folder, f"{col}.png")
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    main()
