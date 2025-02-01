import pandas as pd
from regression_dataset import RegressionDataset
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, random_split


class DataloaderCreator:
    def __init__(
        self,
        file_path,
        target_column,
        feature_columns,
        batch_size=32,
        train_split=0.8,
    ):
        """
        Initializes the DataloaderCreator class for splitting in training and test datasets based on pytorch

        Args:
            file_path (str): Path to the CSV file containing data (Soil nutriens and corresponding multispectral values).
            target_column (str): name of the target column.
            batch_size (int): Batch size for the DataLoader. Default is 32.
            train_split (float): Proportion of data to use for training. Default is 0.8.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.train_split = train_split

        self.dataset = RegressionDataset(
            self.file_path, self.target_column, self.feature_columns
        )

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Creates the training and testing DataLoader objects.

        Returns:
            tuple[DataLoader, DataLoader]: The training and testing DataLoader objects.
        """

        dataset = self.dataset

        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        print(" Trainings dataset:", train_size, "samples")
        print(" Test dataset:", test_size, "samples")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, test_loader

    def create_xgboost_data(self) -> tuple:
        """
        Prepares data for XGBoost training by splitting it into training and testing sets.
        This method loads the data from the CSV file and prepares it as NumPy arrays
        for use with XGBoost or Random Forest.

        Returns:
            tuple: Training and testing data (features and targets) for XGBoost (X_train, X_test, Y_train, Y_test).
        """

        data = pd.read_csv(self.file_path)

        targets = data[self.target_column].values
        features = data[self.feature_columns].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        print(" Training dataset:", len(X_train), "samples")
        print(" Test dataset:", len(X_test), "samples")

        return X_train, X_test, Y_train, Y_test

    def _create_grid_ids(self) -> None:
        """
        Assigns grid IDs to each data point based on geographic coordinates (TH_LAT, TH_LONG).

        This method calculates grid IDs by dividing the longitude and latitude values into discrete grid cells.
        Each data point is assigned to a grid cell, and a unique grid ID is generated based on the grid cell
        coordinates (grid_x, grid_y).

        The data with grid IDs is stored in the class attribute `self.data_with_grid`.
        """
        data = pd.read_csv(self.file_path)
        min_long = data["TH_LONG"].min()
        min_lat = data["TH_LAT"].min()

        # Assign grid IDs based on geographic coordinates
        data["grid_x"] = ((data["TH_LONG"] - min_long) // self.grid_size).astype(int)
        data["grid_y"] = ((data["TH_LAT"] - min_lat) // self.grid_size).astype(int)
        data["grid_id"] = data["grid_x"].astype(str) + "_" + data["grid_y"].astype(str)

        self.data_with_grid = data

    def create_scv_data(self, n_splits=5) -> tuple[list, tuple]:
        """
        Create spatial cross-validation splits for XGBoost.

        This method splits the data into training, testing, and validation sets based on grid IDs.
        It uses K-fold cross-validation for the training set and reserves a specific set of grids for validation.

        Args:
            n_splits (int): Number of folds. Default is 5.

        Returns:
            tuple: (folds, validation_data)
                - folds: List of (X_train, y_train, X_test, y_test) for each fold.
                - validation_data: Tuple (X_val, y_val) for the validation set.
        """

        self.grid_size = 4
        # Define specific grid IDs to be used as validation data
        val_grids = ["0_0", "0_2", "0_1", "0_4", "0_5", "1_5"]
        self.random_state = 42

        self._create_grid_ids()

        # Separate the data into training and validation based on grid IDs
        val_data = self.data_with_grid[self.data_with_grid["grid_id"].isin(val_grids)]
        train_data = self.data_with_grid[
            ~self.data_with_grid["grid_id"].isin(val_grids)
        ]

        X_val = val_data[self.feature_columns].values
        y_val = val_data[self.target_column].values

        print(f"Validation samples: {len(val_data)}")

        # Create KFold splits for training data
        train_grid_ids = train_data["grid_id"].unique()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        folds = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(train_grid_ids)):
            fold_train_grids = train_grid_ids[train_idx]
            fold_test_grids = train_grid_ids[test_idx]

            fold_train_data = train_data[train_data["grid_id"].isin(fold_train_grids)]
            fold_test_data = train_data[train_data["grid_id"].isin(fold_test_grids)]

            print(f"Fold {fold_idx + 1}:")
            print(f"  Training samples: {len(fold_train_data)}")
            print(f"  Testing samples: {len(fold_test_data)}")

            X_train = fold_train_data[self.feature_columns].values
            y_train = fold_train_data[self.target_column].values
            X_test = fold_test_data[self.feature_columns].values
            y_test = fold_test_data[self.target_column].values

            folds.append((X_train, y_train, X_test, y_test))

        return folds, (X_val, y_val)
