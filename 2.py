import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from find import find
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import wandb
run = wandb.init(project="mcm")
config = run.config

seed = 96
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
mm = StandardScaler()

class Args:
    def __init__(self) -> None:
        self.batch_size = 2
        self.lr = 0.0002
        self.epochs = 200
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.radio = 0.8

args = Args()

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_dim, 32)
        self.norm1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 16)
        self.norm2 = nn.LayerNorm(16)
        self.fc3 = nn.Linear(16, 16)
        self.norm3 = nn.LayerNorm(16)
        self.fc4 = nn.Linear(16, out_dim)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, x, test=False):
        x = self.fc1(x)
        if not test:
            x = self.dropout1(x)
        x = self.relu(self.norm1(x))
        x = self.fc2(x)
        if not test:
            x = self.dropout2(x)
        x = self.relu(self.norm2(x))
        x = self.fc3(x)
        if not test:
            x = self.dropout3(x)
        x = self.relu(self.norm3(x))
        x = self.fc4(x)
        return x

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list
    sublist_2 = full_list[offset:,:]
    return sublist_1, sublist_2

class Dataset(Dataset):
    def __init__(self, data, flag='train') -> None:
        label = np.reshape(data[:, -2:], (len(data), 2))

        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'
        self.label = label
        train_data, val_data = data_split(data[:, :-2], ratio=args.radio, shuffle=True)
        if self.flag == 'train':
            self.data = torch.tensor(train_data, dtype=torch.float32)
            self.len = len(train_data)
        else:
            self.data = torch.tensor(val_data, dtype=torch.float32)
            self.len = len(val_data)

    def __getitem__(self, index: int):
        data = self.data[index]
        label = self.label[index]
        return torch.tensor(label, dtype=torch.float), torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return self.len

def train():
    df = pd.read_excel('./df.xlsx')
    tries = df.iloc[1:, 12:20]
    train_data = tries.values[:,:-2]
    train_data = mm.fit_transform(train_data)
    data = tries.values
    data[:,:-2] = train_data
    train_dataset = Dataset(data=data, flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = Dataset(data=data, flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    model = Net(6, 2).to(args.device)
    run.watch(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  

    train_epochs_loss = []
    valid_epochs_loss = []
    RMSEs = []
    MAEs = []

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        nums = 0
        for idx, (label, inputs) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(args.device)
            label = label.to(args.device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.cpu().item())
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        print('train loss = {}'.format(np.average(train_epoch_loss)))

        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            nums = 0

            for idx, (label, inputs) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = model(inputs, test=True)
                loss = criterion(outputs, label)
                if epoch == 199:
                    RMSE = np.sqrt(metrics.mean_squared_error(outputs, label))
                    MAE = metrics.mean_absolute_error(outputs, label)
                    RMSEs.append(RMSE.item())
                    MAEs.append(MAE.item())
                val_epoch_loss.append(loss.item())
                nums += label.size()[0]
            valid_epochs_loss.append(np.average(val_epoch_loss))
            print("epoch = {}, loss = {}".format(epoch, np.average(val_epoch_loss)))
        run.log({"train_epochs_loss": train_epochs_loss,
                 "valid_epochs_loss": valid_epochs_loss,
                 "epoch": epoch
                 })

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_epochs_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss, '-o', label="train_loss")
    plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()
    plt.grid()
    plt.plot(range(len(RMSEs)), MAEs, label='MAE')
    plt.plot(range(len(RMSEs)), RMSEs, label='RMSE')
    plt.rcParams.update({"font.size": 15})
    plt.title('model evaluation')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'model.pth')

def normal_distribution(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def pred(val):
    df = pd.read_excel('./df.xlsx')
    tries = df.iloc[1:, 12:20]
    train_data = tries.values[:,:-2]
    train_data = mm.fit_transform(train_data)
    data = tries.values
    data[:,:-2] = train_data
    val = np.array(val)
    print(val)
    val = mm.transform(val.reshape(1, -1))
    model = Net(6,2)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    val = torch.tensor(val).reshape(1, -1).float()
    res = model(val, test=True)
    res = res.cpu().detach().numpy()
    print("date:")
    print(val)
    print("predicted:")
    print(res)
    fit = normal_distribution(np.linspace(0,7,100), res[0][0], res[0][1])
    preds = []
    for i in range(7):
        preds.append((fit[round(i * (100/7))]))
    plt.plot(range(0,7),preds)
    for a, b in zip(range(0,7),preds):
        plt.text(a+0.4, b, round(b,2), ha='center', va='bottom', fontsize=14)
    plt.plot(np.linspace(0, 7, 100), fit, '--', label='pred')
    plt.title('word: eerie mean:%.2f std:%.2f'%(round(res[0][0],2), round(res[0][1],2)))
    plt.show()

if __name__ == '__main__':
    train()
    pred(find("eerie","2023-03-01"))
