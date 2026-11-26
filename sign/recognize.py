import sys
import cv2
import joblib
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
from sign.common import video_to_sequence, normalize, load_word_model
from sign.lexicon import EN, HI

IMG_EXT = {".jpg", ".jpeg", ".png"}


def recognize_word(video_path):
    """Video -> (word_label, confidence)."""
    model, classes = load_word_model()
    with mp.solutions.holistic.Holistic(model_complexity=0) as h:
        seq = video_to_sequence(video_path, h)
    if seq is None:
        return None, 0.0
    x = torch.tensor(normalize(seq)[None], dtype=torch.float32)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    i = int(p.argmax())
    return classes[i], float(p[i])


def _hand_vec(lm):
    a = np.array([[p.x, p.y, p.z] for p in lm.landmark])
    wrist = a[0, :2].copy()
    a = a - a[0]
    s = np.abs(a).max()
    return (a / s if s > 0 else a).flatten(), wrist


def _alpha_features(res):
    if not res.multi_hand_landmarks:
        return None
    hands = [_hand_vec(h) for h in res.multi_hand_landmarks][:2]
    hands.sort(key=lambda h: h[1][0])
    vec = np.zeros(126)
    for i, (v, _) in enumerate(hands):
        vec[i * 63:(i + 1) * 63] = v
    offset = (hands[1][1] - hands[0][1]) if len(hands) == 2 else np.zeros(2)
    return np.concatenate([vec, offset])


def recognize_letter(image_path):
    """Image -> (letter_or_digit, confidence)."""
    d = joblib.load("models/alphabet_model.joblib")
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
    h, w = img.shape[:2]
    if max(h, w) > 640:
        k = 640 / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)))
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        f = _alpha_features(hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if f is None:
        return None, 0.0
    proba = d["model"].predict_proba([f])[0]
    i = int(proba.argmax())
    return d["classes"][i], float(proba[i])


def build_sentence(words, lang="en"):
    table = EN if lang == "en" else HI
    return " ".join(table[w] for w in words if w in table)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    if path.suffix.lower() in IMG_EXT:
        print("letter:", recognize_letter(path))
    else:
        word, conf = recognize_word(path)
        print("word:", word, f"(confidence {conf:.2f})")
        if word:
            print("English:", build_sentence([word], "en"))
            print("Hindi:  ", build_sentence([word], "hi"))