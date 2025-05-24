import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from collision_dataset import CollisionDataset

# ✅ Config
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CSV_FILE = "C:/ViCoWS-CAR/real_gps_speed_data_ngsim.csv"

# ✅ Model definition
class RiskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        return self.net(x)

# ✅ Main Training Block
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = CollisionDataset(CSV_FILE, normalize=True)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = RiskPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss()

    best_val = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 200 == 0:
                print(f"[Epoch {epoch+1}] Batch {i} Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "collision_risk_model.pth")
            print("[💾] Best model saved")

    print("[✔] Final model training complete.")
