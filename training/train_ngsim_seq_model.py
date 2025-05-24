import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from risk_sequence_dataset import RiskSequenceDataset

# Config
SEQ_LEN = 5
PRED_LEN = 3
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CSV_FILE = "C:/RHINO-CAR/training/real_gps_speed_data_ngsim.csv"

# === Dataset Loader ===
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

# === Model ===
class RiskSequenceModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=PRED_LEN):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last timestep
        return self.sigmoid(out)

# === Training Script ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = RiskSequenceDataset(CSV_FILE, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = RiskSequenceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Batch {i} Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "risk_seq_model.pth")
    print("[✔] Model saved as risk_seq_model.pth")
