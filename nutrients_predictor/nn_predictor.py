import dataloader_predictor as DL
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.trial import TrialState


class RegressionNet(nn.Module):
    """
    A fully connected neural network for regression tasks.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
        output (nn.Linear): The output layer producing a single value.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    
    Methods:
        forward(x):
            Defines the forward pass through the network.
    """  
    def __init__(self, input_size=12, hidden_sizes=[64, 32, 16], output_size=1, dropout_rates=[0.3, 0.3, 0.3]):
        """
        Initializes the network with specified input, hidden, and output sizes.

        Args:
            input_size (int): Number of input features. Default is 12.
            hidden_sizes (list): Sizes of the hidden layers. Default is [64, 32, 16].
            output_size (int): Number of output features. Default is 1 (regression output).
            dropout_rate (float): Dropout rate applied after each hidden layer. Default is [0.3, 0.3, 0.3].
        """
        super(RegressionNet, self).__init__()
        layers = []
        in_features = input_size

        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))  # Ausgabeschicht
        self.model = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.model(x)

class TrainingPipeline:
    """
    A training and evaluation pipeline for neural network models.

    Attributes:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing/validation dataset.
        model (nn.Module): The neural network model for regression.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Type of optimizer to use ('SGD' or 'Adam').
        criterion (nn.Module): Loss function used for training.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Number of training epochs.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Methods:
        train():
            Trains the model using the training dataset.
        evaluate():
            Evaluates the model using the testing dataset.
    """
    def __init__(self, train_loader, test_loader, model, learning_rate=0.001, optimizer_type="Adam", 
                 criterion=None, batch_size=32, num_epochs=10):
        """
        Initializes the training pipeline with specified parameters.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the testing/validation dataset.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            optimizer_type (str): Type of optimizer to use ('SGD' or 'Adam'). Default is 'Adam'.
            criterion (nn.Module, optional): Loss function. Default is nn.MSELoss().
            batch_size (int): Batch size for training and evaluation. Default is 32.
            num_epochs (int): Number of training epochs. Default is 10.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.criterion = criterion if criterion is not None else nn.MSELoss()        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def _get_optimizer(self):
        if self.optimizer_type == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def train(self):
        """
        Trains the model using the training dataset for the specified number of epochs.
        Prints the loss after each epoch.
        """
        self.model.train()
        optimizer = self._get_optimizer()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.view(-1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(self.train_loader):.4f}")

        print('Overview trained model:',self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}')
        print(f'Trainable parameters: {trainable_params}')

    def evaluate(self):
        """
        Evaluates the model using the testing dataset.
        Prints the average loss over the testing dataset.
        """
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.view(-1)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(self.test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss
    
    def save_model(self, file_path):
        """
        Saves the trained model to the specified file path. As well as the loss function as meta data.

        Args:
            file_path (str): The path where the model will be saved.
        """
        if isinstance(self.criterion, nn.MSELoss):
            save_loss = 'MSE'
        elif isinstance(self.criterion, nn.L1Loss):
            save_loss = 'MAE'
        elif isinstance(self.criterion, nn.SmoothL1Loss):
            save_loss = 'Huber'
        else:
            save_loss = 'Unknown'
        
        self.model.loss_function = save_loss
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

def objective(input_size, trial, train_loader, test_loader, path_savemodel):
    """
    Defines the objective function for hyperparameter optimization using Optuna.

    Args:
        input_size: The amount of model input features.
        trial (optuna.trial): An Optuna trial object that helps in suggesting hyperparameters.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing/validation dataset.

    Returns:
        float: The test loss after training the model, which Optuna will try to minimize.
    """
    
    # Dynamic amount of layer
    n_layers = trial.suggest_int("n_layers", 1, 5)
    
    # Dynamic adjustment of neurons and dropout rate
    hidden_sizes = [trial.suggest_int(f"n_units_l{i}", 8, 128, 4) for i in range(n_layers)]
    dropout_rates = [trial.suggest_float(f"dropout_l{i}", 0.1, 0.5) for i in range(n_layers)]
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer_type = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Define model
    model = RegressionNet(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rates=dropout_rates)

    pipeline = TrainingPipeline(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        criterion=nn.MSELoss(),
        batch_size=batch_size,
        num_epochs=10
    )

    # Train the model and evaluate its performance
    pipeline.train()
    test_loss = pipeline.evaluate()
    
    
    # Save model with the best performance
    if trial.number == 0 or test_loss < trial.study.best_value:
        torch.save(model.state_dict(), path_savemodel)
        print(f'Model with Loss {test_loss} saved.')

    return test_loss

    
def run_nn_train(input_size, train_loader, test_loader, path_savemodel):
    """
        Runs the training of a regression neuronal network using Optuna and manages model files.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the testing/validation dataset.
            path_savemodel (str): Path of the saved model file.

        """

    # Create optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(input_size, trial, train_loader, test_loader, path_savemodel), n_trials=20, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Optuna statistics
    print("-----Study statistics:-----")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Parameter: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
