import cv2
import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 30


def pts(lms, n):
    if lms is None:
        return np.zeros(n * 3)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark]).flatten()


def frame_features(res):
    return np.concatenate([
        pts(res.pose_landmarks, 33),
        pts(res.left_hand_landmarks, 21),
        pts(res.right_hand_landmarks, 21),
    ])


def video_to_sequence(path, holistic):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = set(np.linspace(0, max(n - 1, 0), SEQ_LEN).astype(int).tolist())
    feats, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in keep:
            frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feats.append(frame_features(res))
        i += 1
    cap.release()
    if not feats:
        return None
    while len(feats) < SEQ_LEN:
        feats.append(feats[-1])
    return np.stack(feats[:SEQ_LEN])


def normalize(seq):
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


def load_word_model(path="models/word_model.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = BiLSTM(len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["classes"]