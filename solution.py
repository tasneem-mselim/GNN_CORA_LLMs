import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -------------------- config --------------------
SEED = 42
DATA_DIR = "data"
EDGE_PATH = os.path.join(DATA_DIR, "edge_index.csv")
X_PATH = os.path.join(DATA_DIR, "x.csv")
YTR_PATH = os.path.join(DATA_DIR, "y_train.csv")
YVA_PATH = os.path.join(DATA_DIR, "y_val.csv")
TESTID_PATH = os.path.join(DATA_DIR, "test_ID.csv")
SUB_PATH = "submission.csv"

DEVICE = torch.device("cpu")
HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 400
PATIENCE = 50

# -------------------- seeding --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- load data --------------------
X = pd.read_csv(X_PATH).to_numpy(dtype=np.float32)          # (N, F)
N, num_features = X.shape

edge_df = pd.read_csv(EDGE_PATH)
src = edge_df["source"].to_numpy(dtype=np.int64)
dst = edge_df["target"].to_numpy(dtype=np.int64)

# make undirected, add self-loops, remove duplicates
edges = set()
for s, d in zip(src, dst):
    edges.add((s, d))
    edges.add((d, s))
for i in range(N):
    edges.add((i, i))

edge_index = torch.tensor([[s, d] for s, d in edges], dtype=torch.long).t().contiguous()

tr = pd.read_csv(YTR_PATH)
va = pd.read_csv(YVA_PATH)
te = pd.read_csv(TESTID_PATH)

tr_idx = tr["index"].to_numpy(dtype=np.int64)
tr_y   = tr["label"].to_numpy(dtype=np.int64)
va_idx = va["index"].to_numpy(dtype=np.int64)
va_y   = va["label"].to_numpy(dtype=np.int64)
te_idx = te["id"].to_numpy(dtype=np.int64)

# row-normalize features (L2 norm) – robust to Gaussian noise
row_norm = np.linalg.norm(X, axis=1, keepdims=True)
row_norm[row_norm == 0] = 1.0
X = X / row_norm

x = torch.from_numpy(X)
y = torch.full((N,), -1, dtype=torch.long)
y[tr_idx] = torch.from_numpy(tr_y.copy())
y[va_idx] = torch.from_numpy(va_y.copy())

train_mask = torch.zeros(N, dtype=torch.bool)
val_mask   = torch.zeros(N, dtype=torch.bool)
test_mask  = torch.zeros(N, dtype=torch.bool)
train_mask[tr_idx] = True
val_mask[va_idx]   = True
test_mask[te_idx]  = True

data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# -------------------- model --------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(num_features, HIDDEN, 7).to(DEVICE)
data = data.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------- training --------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc

best_val_acc = 0.0
best_state = None
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    loss = train()
    val_acc = evaluate(data.val_mask)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
        break

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

# -------------------- predict --------------------
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[data.test_mask].argmax(dim=1).cpu().numpy()

sub = pd.DataFrame({"id": te_idx, "target": pred})
sub.to_csv(SUB_PATH, index=False)
print(f"Saved submission to {SUB_PATH} (best val acc: {best_val_acc:.4f})")
