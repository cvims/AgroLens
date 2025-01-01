import dataloader_predictor as DL
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    def __init__(self, input_size=12, hidden_sizes=[64, 32, 16], output_size=1, dropout_rate=0.3):
        """
        Initializes the network with specified input, hidden, and output sizes.

        Args:
            input_size (int): Number of input features. Default is 12.
            hidden_sizes (list): Sizes of the hidden layers. Default is [64, 32, 16].
            output_size (int): Number of output features. Default is 1 (regression output).
            dropout_rate (float): Dropout rate applied after each hidden layer. Default is 0.3.
        """
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output = nn.Linear(hidden_sizes[2], output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)  # No activation on the output layer for regression
        return x

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(diff)))

class TrainingPipeline:
    """
    A training and evaluation pipeline for neural network models.

    Attributes:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing/validation dataset.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Type of optimizer to use ('SGD' or 'Adam').
        criterion (nn.Module): Loss function used for training.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Number of training epochs.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
        model (nn.Module): The neural network model for regression.
    
    Methods:
        train():
            Trains the model using the training dataset.
        evaluate():
            Evaluates the model using the testing dataset.
    """
    def __init__(self, train_loader, test_loader, learning_rate=0.001, optimizer_type="Adam", 
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
        self.model = RegressionNet().to(self.device)

    def _get_optimizer(self):
        if self.optimizer_type == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def save_model(self, file_path):
        """
        Saves the trained model to the specified file path.

        Args:
            file_path (str): The path where the model will be saved.
        """
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

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
        print(f"Test Loss: {test_loss / len(self.test_loader):.4f}")


# def objective(trial):
#     model = RegressionNet
    
def run_nn_train(train_loader, test_loader):
    """
        Runs the training of a regression neuronal network using optuna

        Args:

        """

    # Initialize TrainingPipeline
    batch_size = 5
    pipeline = TrainingPipeline(
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        optimizer_type="Adam",
        criterion=nn.MSELoss(), 
        batch_size=batch_size,
        num_epochs=5
    )
    print('-----Start model training: Neuronal network-----')
    pipeline.train()
    pipeline.evaluate()
