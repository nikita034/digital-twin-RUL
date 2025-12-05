# src/data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os


# Read a CMAPSS whitespace-delimited file and name columns dynamically.
def read_cmapss(path):
df = pd.read_csv(path, sep='\s+', header=None)
ncols = df.shape[1]
# default pattern: engine, cycle, op1..op3, s1..sN
cols = ['engine', 'cycle'] + [f'op{i+1}' for i in range(3)] + [f's{i+1}' for i in range(ncols - 5)]
cols = cols[:ncols]
df.columns = cols
return df


# Add RUL column to training dataframe
def add_rul(df):
df = df.copy()
max_cycle = df.groupby('engine')['cycle'].transform('max')
df['RUL'] = max_cycle - df['cycle']
return df


# Fit and save StandardScaler for feature columns
def fit_scaler(df, feature_cols, scaler_path):
scaler = StandardScaler()
scaler.fit(df[feature_cols].values)
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
with open(scaler_path, 'wb') as f:
pickle.dump(scaler, f)
return scaler


# Load scaler
def load_scaler(scaler_path):
with open(scaler_path, 'rb') as f:
return pickle.load(f)


# Create fixed-length windows from engine data
def create_windows(df, feature_cols, window_size=64, step=1, max_rul_clip=130):
X, y = [], []
engines = df['engine'].unique()
for e in engines:
edf = df[df['engine'] == e].reset_index(drop=True)
n = len(edf)
if n < window_size:
pad_count = window_size - n
pad = np.repeat(edf.loc[[0], feature_cols].values, pad_count, axis=0)
seq0 = np.vstack([pad, edf[feature_cols].values])
X.append(seq0.astype(np.float32))
y.append(min(edf.loc[n-1, 'RUL'], max_rul_clip))
else:
for end in range(window_size, n+1, step):
seq = edf.loc[end-window_size:end-1, feature_cols].values
label = edf.loc[end-1, 'RUL']
X.append(seq.astype(np.float32))
y.append(min(label, max_rul_clip))
return np.array(X), np.array(y, dtype=np.float32)
