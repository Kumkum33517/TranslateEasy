import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KP = Path("data/include50/keypoints")
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)


def normalize(seq):
    """Center on the nose and scale by shoulder width, so position/size don't matter."""
    out = []
    for f in seq:
        p = f.reshape(75, 3).copy()
        mask = (p != 0).any(axis=1)
        if mask[0]:
            p[mask] -= p[0]
        if mask[11] and mask[12]:
            d = np.linalg.norm(p[11, :2] - p[12, :2])
            if d > 1e-6:
                p /= d
        out.append(p.flatten())
    return np.stack(out)


df = pd.read_csv(KP / "labels.csv")
classes = sorted(df.label.unique())
X = np.stack([normalize(np.load(KP / f)) for f in df.file]).astype(np.float32)
y = np.array([classes.index(l) for l in df.label])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)


class BiLSTM(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.lstm = nn.LSTM(225, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(256, n_cls)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(o.mean(dim=1)))


model = BiLSTM(len(classes))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 81):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 16):
        idx = perm[i:i + 16]
        xb, yb = Xtr[idx], ytr[idx]
        xb = xb * (0.9 + 0.2 * torch.rand(1)) + 0.01 * torch.randn_like(xb)  # augmentation
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {acc:.2f}")

model.eval()
with torch.no_grad():
    pred = model(Xte).argmax(1).numpy()
print("Final test accuracy:", round(accuracy_score(yte.numpy(), pred), 3))

cm = confusion_matrix(yte.numpy(), pred, labels=range(len(classes)))
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=60, ha="right")
ax.set_yticklabels(classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("results/word_confusion_matrix.png", dpi=150)

torch.save({"state": model.state_dict(), "classes": classes}, "models/word_model.pt")
json.dump(classes, open("models/word_classes.json", "w"))
print("Saved models/word_model.pt and results/word_confusion_matrix.png")