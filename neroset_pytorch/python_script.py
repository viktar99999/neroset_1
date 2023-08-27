import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch import optim
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv('city.csv')
df.head()
print(df.shape)
Y = df['medv']
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]
X.head()
print(y[:5])
print(X.describe())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_tensor = torch.Tensor(np.array(X_train_scaled))
X_test_tensor = torch.Tensor(np.array(X_test_scaled))
y_train_tensor = torch.Tensor(np.array(y_train))
y_test_tensor = torch.Tensor(np.array(y_test))
n_data, n_features = X_train_tensor.shape
print(n_data)
print(n_features)
def loss(mean, target):
    """function loss"""
    return mean(F.l1_loss(input, target, reduction="none") / target) * 100
loss_func = F.mse_loss
metrics_func_1 = [loss_func, loss]
metrics_name = ["MSE", "LOSS"]
def evaluate(model, metrics_func, x_1, y_1):
    """function evaluate"""
    metrics_value = []
    with torch.no_grad():
        for metric_func in metrics_func:
            metric_value = metric_func(torch.Tensor(X).flatten(), torch.Tensor(y).flatten())
            metrics_value.append(metric_value)
    return metrics_value
def print_metrics(models, train_data, test_data, models_name):
    """function print_metrics"""
    results = np.ones(2 * len(models), len(metrics_func_1))
    models_name = []
    for model in models_name:
        models_name.extend([model + "Train", model + "Test"])
    for row, sample in enumerate([train_data, test_data]):
        results[row + sample * 2] = evaluate(models, metrics_func_1, sample[0], sample[1])
        results = pd.DataFrame(results, columns=metrics_name, index=models_name)
        return results
train_data_1 = (X_train_tensor, y_train_tensor)
test_data_1 = (X_test_tensor, y_test_tensor)
model_lr_sklearn = LinearRegression()
model_lr_sklearn.fit(X_train_scaled, y_train)
model_1 = [model_lr_sklearn.predict]
metrics_name = ["MSE", "LOSS"]
models_name_1 = ["LOSS"]
model_lr = nn.Sequential(
    nn.Linear(in_features=n_features,
    out_features=1))
print(model_lr)
opt_lr = optim.SGD(params=model_lr.parameters(), lr=0.001)
BATCH_SIZE_LR = 16
EPOCHS_LR = 1000
for epoch in tqdm.trange(EPOCHS_LR):
    for i in range((n_data - 1) // BATCH_SIZE_LR + 1):
        start_i = i * BATCH_SIZE_LR
        end_i = start_i + BATCH_SIZE_LR
        Xb = X_train_tensor[start_i: end_i]
        yb = y_train_tensor[start_i: end_i]
        pred = yb
        loss_1 = loss_func(pred, yb + 1)
print(X_test)
print(y_test)
def train(model_lr, loss_fn, optimizer):
    """function train"""
    size = len('city.csv')
    model_lr.train()
    for BATCH_SIZE_LR, (X, y) in tqdm.trange(EPOCHS_LR):
        X, y = X.to_device, y.to_device
        pred_1 = model_lr(X)
        loss = loss_fn(pred_1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if BATCH_SIZE_LR % 16 == 0:
            loss_2 = loss.iten(), BATCH_SIZE_LR * len(X)
            print(f'loss: {loss_2}')
with torch.no_grad():
    print(model_lr(X_test_tensor[-1:]))
x_t = torch.Tensor([0.5141, 0.0000, 0.6430, 0.0000, 0.6337, 0.1334, 1.0000, 0.0481, 1.0000,
        0.9141, 0.8085, 0.2218])
with torch.no_grad():
  print(model_lr(x_t))
torch.save(model_lr, "lr.pth")
model_lr_1 = nn.Sequential(
    nn.Linear(in_features=n_features, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=1))
print(model_lr_1)
opt_lr_1 = optim.SGD(params=model_lr_1.parameters(), lr=0.0001)
BATCH_SIZE_LR_1 = 16
EPOCHS_LR_1 = 1000
for epoch in tqdm.trange(EPOCHS_LR_1):
    for i in range((n_data - 1) // BATCH_SIZE_LR_1 + 1):
        start_i = i * BATCH_SIZE_LR_1
        end_i = start_i + BATCH_SIZE_LR_1
        Xb = X_train_tensor[start_i: end_i]
        yb = y_train_tensor[start_i: end_i]
        pred_2 = yb
        loss = loss_func(pred_2, yb + 1)
print(X_test)
print(y_test)
def train_1(model_lr_1, loss_fn, optimizer):
    """function_1 train_1"""
    size = len('city.csv')
    model_lr_1.train()
    for BATCH_SIZE_LR_1, (X, y) in tqdm.trange(EPOCHS_LR_1):
        X, y = X.to_device, y.to_device
        pred_3 = model_lr_1(X)
        loss = loss_fn(pred_3, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if BATCH_SIZE_LR_1 % 16 == 0:
            loss_3 = loss.iten(), BATCH_SIZE_LR * len(X)
            print(f'loss: {loss_3}')
with torch.no_grad():
    print(model_lr_1(X_test_tensor[-1:]))
torch.save(model_lr_1, "lr_1.pth")
