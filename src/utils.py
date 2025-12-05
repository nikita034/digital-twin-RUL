# src/utils.py
import torch
from torch.utils.data import Dataset, DataLoader


class RULDataset(Dataset):
def __init__(self, X, y):
self.X = torch.from_numpy(X) # (N, L, D)
self.y = torch.from_numpy(y) # (N,)
def __len__(self):
return len(self.y)
def __getitem__(self, idx):
return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size=64, shuffle=True, num_workers=4):
ds = RULDataset(X, y)
return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
