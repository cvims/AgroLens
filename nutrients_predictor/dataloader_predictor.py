import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


class RegressionDataset(Dataset):
    """
    A dataset class (pytorch) to load data from CSV file.
    
    """
    def __init__(self, file_path, target_column, feature_columns, transform=None):
        """
        Initializes the dataset with data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing data (Soil nutriens and corresponding multispectral values).
            target_column (str): Name of the target column.
            transform (optional): Optional, TBD
        """
        self.data = pd.read_csv(file_path)
        
        # Extract the target column
        self.targets = self.data[target_column].values
        
        # Select features columns
        self.feature_columns = feature_columns
        self.data = self.data[self.feature_columns].values
        
        # Optionally apply transformations (e.g., normalization)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        target = self.targets[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class DataloaderCreator:
    def __init__(self, file_path, target_column, feature_columns, batch_size=32, train_split=0.8, transform=None):
        """
        Initializes the DataloaderCreator class for splitting in training and test datasets based on pytorch

        Args:
            file_path (str): Path to the CSV file containing data (Soil nutriens and corresponding multispectral values).
            target_column (str): ame of the target column.
            batch_size (int): Batch size for the DataLoader. Default is 32.
            train_split (float): Proportion of data to use for training. Default is 0.8.
            transform (optional): TBD
        """
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = transform
        
        # Create the RegressionDataset
        self.dataset = RegressionDataset(self.file_path, self.target_column,self.feature_columns, transform=self.transform)

    def create_dataloaders(self):
        """
        Creates the training and testing DataLoader objects.
        
        Returns:
            tuple: The training and testing DataLoader objects.
        """
        # Get the dataset
        dataset = self.dataset

        # Split the dataset into train and test sets
        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        print(' Trainings dataset:',train_size,'samples')
        print(' Test dataset:', test_size, 'samples')

        # Create DataLoader objects
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, test_loader
    
    def create_xgboost_data(self):
        # Load data table
        data = pd.read_csv(self.file_path)

        # Extract the target and feature columns as seperate np arrays
        targets = data[self.target_column].values
        features = data[self.feature_columns].values

        # Split the dataset into training and testing sets
        # 80% of data will be used for training, and 20% for testing
        X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        print(' Trainings dataset:',len(X_train),'samples')
        print(' Test dataset:', len(X_test), 'samples')

        return X_train, X_test, Y_train, Y_test