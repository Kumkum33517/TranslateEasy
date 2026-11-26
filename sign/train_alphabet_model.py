import random
import cv2
import joblib
import numpy as np
from pathlib import Path
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("data/alphabet/Indian")
CACHE = Path("data/alphabet/landmarks.npz")
PER_CLASS = 100
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


def hand_vec(lm):
    a = np.array([[p.x, p.y, p.z] for p in lm.landmark])
    wrist = a[0, :2].copy()
    a = a - a[0]
    s = np.abs(a).max()
    return (a / s if s > 0 else a).flatten(), wrist


def features(res):
    if not res.multi_hand_landmarks:
        return None
    hands = [hand_vec(h) for h in res.multi_hand_landmarks][:2]
    hands.sort(key=lambda h: h[1][0])  # left-most hand first
    vec = np.zeros(126)
    for i, (v, _) in enumerate(hands):
        vec[i * 63:(i + 1) * 63] = v
    offset = (hands[1][1] - hands[0][1]) if len(hands) == 2 else np.zeros(2)
    return np.concatenate([vec, offset])  # 128 values


if CACHE.exists():
    d = np.load(CACHE, allow_pickle=False)
    X, y, classes = d["X"], d["y"], list(d["classes"])
    print("Loaded cached landmarks:", X.shape)
else:
    classes = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    rng = random.Random(0)
    X, y, found = [], [], {}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=0.5) as hands:
        for ci, c in enumerate(classes):
            imgs = sorted(p for p in (ROOT / c).iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            imgs = rng.sample(imgs, min(PER_CLASS, len(imgs)))
            ok = 0
            for p in imgs:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                if max(h, w) > 640:
                    k = 640 / max(h, w)
                    img = cv2.resize(img, (int(w * k), int(h * k)))
                f = features(hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if f is not None:
                    X.append(f)
                    y.append(ci)
                    ok += 1
            found[c] = ok
            print(f"{c}: {ok}/{len(imgs)} detected")
    X, y = np.array(X), np.array(y)
    np.savez(CACHE, X=X, y=y, classes=np.array(classes))
    print("Total usable samples:", len(X))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print("Alphabet test accuracy:", round(accuracy_score(yte, pred), 3))

cm = confusion_matrix(yte, pred, labels=range(len(classes)))
fig, ax = plt.subplots(figsize=(11, 10))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, fontsize=7)
ax.set_yticklabels(classes, fontsize=7)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("results/alphabet_confusion_matrix.png", dpi=150)

joblib.dump({"model": clf, "classes": classes}, "models/alphabet_model.joblib")
print("Saved models/alphabet_model.joblib and results/alphabet_confusion_matrix.png")