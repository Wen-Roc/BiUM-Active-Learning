import os
import math
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from gmlp1 import gMLP_eval_2
from ResNet import ResNet
from gMLP_BNN import MemoryBuffer
import torchbnn as bnn

# ---------------------- 超参数 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
epochs_model1 = 1000
epochs_model2 = 500
learning_rate = 1e-3
buffer_capacity = 50000
version = '0516_7'

# ---------------------- 初始化 ----------------------
model1 = gMLP_eval_2(in_features=5, out_features=48, mlp_features=256, num_blocks=3, scalar_sigma=0.001).to(device)
model2 = ResNet().to(device)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
buffer = MemoryBuffer(buffer_capacity)

# ---------------------- 数据加载 ----------------------
def load_data(cc_path, ss_path):
    x = torch.tensor(np.loadtxt(cc_path, delimiter=','), dtype=torch.float32).to(device)
    y = torch.tensor(np.loadtxt(ss_path, delimiter=','), dtype=torch.float32).to(device)
    y = y[:, :5]  # 只取前5列
    return x, y

x_TR0, y_TR0 = load_data('../data/ss_TR0.txt', '../data/cc_TR0.txt')
x_CA0, y_CA0 = load_data('../data/ss_CA0.txt', '../data/cc_CA0.txt')
x_VAL0, y_VAL0 = load_data('../data/ss_VAL0.txt', '../data/cc_VAL0.txt')

# ---------------------- 数据集合并 ----------------------
def create_dataloader(x_list, y_list, batch_size, shuffle=True):
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    dataset = TensorDataset(y, x)
    return list(enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)))

train1 = create_dataloader([x_TR0], [y_TR0], batch_size)
test1  = create_dataloader([x_VAL0], [y_VAL0], batch_size)

# ---------------------- 辅助函数 ----------------------
def train_one_epoch(model, optimizer, dataloader, dipole_fn=None):
    model.train()
    losses = []
    for _, (cc, ss) in dataloader:
        if dipole_fn is not None:
            ss_sim = dipole_fn(cc).to(device)
            ss_delta = ss - ss_sim
        else:
            ss_delta = ss
        pred, sigma = model(cc if dipole_fn is None else cc)
        loss = model.gaosiloss(ss_delta, sigma, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, dataloader, dipole_fn=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for _, (cc, ss) in dataloader:
            if dipole_fn is not None:
                ss_sim = dipole_fn(cc).to(device)
                ss_delta = ss - ss_sim
            else:
                ss_delta = ss
            pred, sigma = model(cc if dipole_fn is None else cc)
            loss = model.gaosiloss(ss_delta, sigma, pred)
            losses.append(loss.item())
    return np.mean(losses)

def plot_loss(train_losses, val_losses, title='Loss Curve'):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# ---------------------- Buffer 初始化 ----------------------
for _, (data, target) in train1:
    buffer.add(data, target)
torch.save(buffer.buffer, version + '_replay_buffer.pt')

# ---------------------- 模型1训练 (校准) ----------------------
best_loss_model1 = float('inf')
train_losses, val_losses = [], []

for epoch in tqdm(range(epochs_model1), desc='Model1 Training'):
    train_loss = train_one_epoch(model1, optimizer1, train1, dipole_fn=generate_dipole_data_torch)
    val_loss = validate(model1, test1, dipole_fn=generate_dipole_data_torch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss_model1:
        torch.save(model1.state_dict(), f'finl{version}_cal.pkl')
        best_loss_model1 = val_loss

    tqdm.write(f'Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_loss_model1:.6f}')

plot_loss(train_losses, val_losses, title='Model1 Loss Curve')

# ---------------------- 模型2训练 (位置) ----------------------
x_train_label = torch.tensor(generate_dipole_data("../data/cc_CA0.txt", "../para_sensor", N_sensors=16), dtype=torch.float32).to(device)
x_train_combined = torch.cat((x_train_label, x_TR0), dim=0)
y_train_combined = torch.cat((y_CA0, y_TR0), dim=0)
train2 = create_dataloader([x_train_combined], [y_train_combined], batch_size)
test2  = create_dataloader([x_VAL0], [y_VAL0], batch_size)

best_loss_model2 = float('inf')
train_losses2, val_losses2 = [], []

for epoch in tqdm(range(epochs_model2), desc='Model2 Training'):
    train_loss = train_one_epoch(model2, optimizer2, train2)
    val_loss = validate(model2, test2)
    train_losses2.append(train_loss)
    val_losses2.append(val_loss)

    if val_loss < best_loss_model2:
        torch.save(model2.state_dict(), f'finl{version}_pos_val.pkl')
        best_loss_model2 = val_loss

    tqdm.write(f'Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_loss_model2:.6f}')

plot_loss(train_losses2, val_losses2, title='Model2 Loss Curve')
