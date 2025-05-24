import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RiskSequenceDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, pred_len=3):
        df = pd.read_csv(csv_file)
        X = df[['VSV', 'VLV', 'Headway']].values
        y = df['Risk'].values

        self.inputs = []
        self.targets = []

        for i in range(len(df) - seq_len - pred_len):
            x_seq = X[i:i+seq_len]
            y_seq = y[i+seq_len:i+seq_len+pred_len]
            self.inputs.append(x_seq)
            self.targets.append(y_seq)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
