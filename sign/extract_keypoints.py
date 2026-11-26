import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp

RAW = Path("data/include50/raw")
OUT = Path("data/include50/keypoints")
OUT.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 30
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def pts(lms, n):
    if lms is None:
        return np.zeros(n * 3)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark]).flatten()


def frame_features(res):
    return np.concatenate([
        pts(res.pose_landmarks, 33),
        pts(res.left_hand_landmarks, 21),
        pts(res.right_hand_landmarks, 21),
    ])  # 225 values per frame


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


def clean_label(folder_name):
    return re.sub(r"^\d+\.\s*", "", folder_name).strip().lower().replace(" ", "_")


rows = []
with mp.solutions.holistic.Holistic(model_complexity=0) as holistic:
    videos = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    print(f"Found {len(videos)} videos")
    for k, v in enumerate(videos):
        label = clean_label(v.parent.name)
        out = OUT / f"{label}__{v.stem}.npy"
        if not out.exists():
            seq = video_to_sequence(v, holistic)
            if seq is None:
                print("skipped (unreadable):", v)
                continue
            np.save(out, seq)
        rows.append({"file": out.name, "label": label})
        if k % 10 == 0:
            print(f"{k + 1}/{len(videos)}")

pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)
print("Done:", len(rows), "sequences")
print(pd.DataFrame(rows)["label"].value_counts())