import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


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
            transform (optional): none
        """
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = transform
        self.grid_size = 5
        
        self.dataset = RegressionDataset(self.file_path, self.target_column,self.feature_columns, transform=self.transform)

    def create_dataloaders(self):
        """
        Creates the training and testing DataLoader objects.
        
        Returns:
            tuple: The training and testing DataLoader objects.
        """

        dataset = self.dataset

        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        print(' Trainings dataset:',train_size,'samples')
        print(' Test dataset:', test_size, 'samples')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, test_loader
    
    def create_xgboost_data(self):
        """
            Prepares data for XGBoost training by splitting it into training and testing sets.
            This method loads the data from the CSV file and prepares it as NumPy arrays 
            for use with XGBoost or Random Forest.
        
            Returns:
                tuple: Training and testing data (features and targets) for XGBoost.
        """

        data = pd.read_csv(self.file_path)

        targets = data[self.target_column].values
        features = data[self.feature_columns].values

        X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        print(' Trainings dataset:',len(X_train),'samples')
        print(' Test dataset:', len(X_test), 'samples')

        return X_train, X_test, Y_train, Y_test

    def _create_grid_ids(self):
        """
        Assigns grid IDs to each data point based on geographic coordinates (TH_LAT, TH_LONG).

        Args:
            grid_size (float): Size of each grid cell in degrees. Default is 1.0.
        """

        data = pd.read_csv(self.file_path)
        min_long = data['TH_LONG'].min()
        max_long = data['TH_LONG'].max()
        min_lat = data['TH_LAT'].min()
        max_lat = data['TH_LAT'].max()

        print(f"Min Coordinates: TH_LONG = {min_long}, TH_LAT = {min_lat}")
        print(f"Max Coordinates: TH_LONG = {max_long}, TH_LAT = {max_lat}")

        data['grid_x'] = ((data['TH_LONG'] - min_long) // self.grid_size).astype(int)
        data['grid_y'] = ((data['TH_LAT'] - min_lat) // self.grid_size).astype(int)
        data['grid_id'] = data['grid_x'].astype(str) + "_" + data['grid_y'].astype(str)
        self.data_with_grid = data

    def create_scv_data(self, n_splits=5, validation_split = 0.1):
        """
            Create spatial cross-validation splits for XGBoost.

            Args:
                data_with_grid (pd.DataFrame): DataFrame containing the data with calculated grid IDs.
                feature_columns (list): List of feature column names.
                target_column (str): Target column name.
                n_splits (int): Number of folds for cross-validation. Default is 5.
                validation_split (float): Fraction of data to be used as validation set. Default is 0.1.
                random_state (int): Random seed for reproducibility.

            Returns:
                tuple: (folds, validation_data)
                    - folds: List of (X_train, y_train, X_test, y_test) for each fold.
                    - validation_data: Tuple (X_val, y_val) for the validation set.
        """

        random_state = 42
        self._create_grid_ids()

        # Separate validation set based on grids
        grid_ids = self.data_with_grid['grid_id'].unique()
        val_size = int(len(grid_ids) * validation_split)
        np.random.seed(random_state)
        np.random.shuffle(grid_ids)

        val_grids = grid_ids[:val_size]
        train_grids = grid_ids[val_size:]

        train_data = self.data_with_grid[self.data_with_grid['grid_id'].isin(train_grids)]
        val_data = self.data_with_grid[self.data_with_grid['grid_id'].isin(val_grids)]

        # Extract validation features and targets
        X_val = val_data[self.feature_columns].values
        y_val = val_data[self.target_column].values

        print(f'Validation samples: {len(val_data)}')

        # Create KFold splits for training data
        train_grid_ids = train_data['grid_id'].unique()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds= []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(train_grid_ids)):
            fold_train_grids = train_grid_ids[train_idx]
            fold_test_grids = train_grid_ids[test_idx]

            fold_train_data = train_data[train_data['grid_id'].isin(fold_train_grids)]
            fold_test_data = train_data[train_data['grid_id'].isin(fold_test_grids)]

            # Print sample counts
            print(f"Fold {fold_idx + 1}:")
            print(f"  Training samples: {len(fold_train_data)}")
            print(f"  Testing samples: {len(fold_test_data)}")

            X_train = fold_train_data[self.feature_columns].values
            y_train = fold_train_data[self.target_column].values
            X_test = fold_test_data[self.feature_columns].values
            y_test = fold_test_data[self.target_column].values

            folds.append((X_train, y_train, X_test, y_test))

        return folds, (X_val, y_val)