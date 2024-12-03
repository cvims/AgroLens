
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

learning_rate = 0.001
 
# ----------------------------------------------------------------------- 
 
# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('Example.csv', delimiter=',')
X = dataset[:,0:12]
y = dataset[:,12] # needs to be adapted on the relevant value
 
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
# ----------------------------------------------------------------------- 
 
# define the model
model = nn.Sequential(
    nn.Linear(13, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.ReLU()
)
print(model)
 
# train the model
loss_fn   = nn.MSELoss()  # mean squared error
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
n_epochs = 100
batch_size = 10
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
 
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

 
# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))