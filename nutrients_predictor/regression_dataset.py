import pandas as pd
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """
    A dataset class (PyTorch) to load data from a CSV file for training regression models.
    This class inherits from PyTorch's Dataset and is used to load and return features and targets for training.
    """

    def __init__(self, file_path, target_column, feature_columns, transform=None):
        """
        Initializes the dataset with data from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing data (Soil nutriens and corresponding multispectral values).
            target_column (str): Name of the target column.
            feature_columns (list): List of feature columns to be used for training.
            transform (optional): Optional, TBD
        """
        self.data = pd.read_csv(file_path)

        self.targets = self.data[target_column].values
        self.feature_columns = feature_columns
        self.data = self.data[self.feature_columns].values

        # Optional data transformations
        self.transform = transform

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        """
        This function returns a single sample (features and target) from the dataset at the specified index.

        Returns:
            tuple: Features and target as PyTorch tensors.
        """
        features = self.data[idx]
        target = self.targets[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )
