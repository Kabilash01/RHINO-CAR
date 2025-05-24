import torch
import pandas as pd
from torch.utils.data import Dataset

class CollisionDataset(Dataset):
    def __init__(self, csv_file, x_cols=["VSV", "Headway", "VLV"], y_col="Risk", normalize=True):
        self.df = pd.read_csv(csv_file)
        self.x_cols = x_cols
        self.y_col = y_col

        if normalize:
            from sklearn.preprocessing import MinMaxScaler
            self.x_scaler = MinMaxScaler()
            self.df[self.x_cols] = self.x_scaler.fit_transform(self.df[self.x_cols])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.df.loc[idx, self.x_cols].values.astype("float32"))
        y = torch.tensor([self.df.loc[idx, self.y_col]], dtype=torch.float32)
        return x, y
