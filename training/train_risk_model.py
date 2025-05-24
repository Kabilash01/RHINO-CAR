import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from risk_sequence_dataset import RiskSequenceDataset


# === CONFIG ===
SEQ_LEN = 5
PRED_LEN = 3
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
CSV_PATH = "real_gps_speed_data_ngsim.csv"

# === Dataset Class ===
class NGSIMSequenceDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df[['VSV', 'VLV', 'Headway']].values
        y = df['Risk'].values

        self.inputs = []
        self.targets = []

        for i in range(len(df) - SEQ_LEN - PRED_LEN):
            seq_x = X[i:i + SEQ_LEN]
            seq_y = y[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN]
            self.inputs.append(seq_x)
            self.targets.append(seq_y)

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
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# === Prepare Data ===
dataset = NGSIMSequenceDataset(CSV_PATH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RiskSequenceModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

print("[INFO] Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
    avg_val = val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val: {avg_val:.4f}")

# === Save Model ===
torch.save(model.state_dict(), "risk_seq_model.pth")
print("[✔] Model saved as risk_seq_model.pth")
