# src/model.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
def __init__(self, d_model, max_len=512):
super().__init__()
pe = torch.zeros(max_len, d_model)
pos = torch.arange(0, max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(pos * div_term)
pe[:, 1::2] = torch.cos(pos * div_term)
self.register_buffer('pe', pe.unsqueeze(0))
def forward(self, x):
return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
super().__init__()
self.input_proj = nn.Linear(input_dim, d_model)
self.pos = PositionalEncoding(d_model)
encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
self.pool = nn.AdaptiveAvgPool1d(1)
self.head = nn.Sequential(
nn.Linear(d_model, d_model//2),
nn.ReLU(),
nn.Linear(d_model//2, 1)
)
def forward(self, x):
# x: (B, L, D)
h = self.input_proj(x)
h = self.pos(h)
h = self.transformer(h)
h = h.transpose(1,2)
h = self.pool(h).squeeze(-1)
out = self.head(h).squeeze(-1)
return out
