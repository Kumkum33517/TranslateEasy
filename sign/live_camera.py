import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont, features

from sign.common import frame_features, normalize, load_word_model, SEQ_LEN
from sign.lexicon import EN, HI
from sign.recognize import _alpha_features

MIN_FRAMES = 12
WORD_CONF = 0.8        # word must be at least this confident
WORD_MARGIN = 0.3      # and this far ahead of the runner-up
MIN_HAND_RATIO = 0.4   # share of recorded frames that must contain a hand
LETTER_THR = 0.25      # averaged letter probability needed to accept a letter
SESSION = Path("results/live_session.json")
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
RAQM = features.check("raqm")   # Hindi text shaping available?

MODES = ["signs", "words"]
MODE_NAME = {"signs": "Gestures + Alphabet", "words": "Words (beta)"}
HINTS = {"signs": "Show a gesture or letter, hold it steady, press SPACE to add",
         "words": "Press SPACE, sign one word, press SPACE again"}

GEST_EN = {"hi": "Hi", "bye": "Bye", "ok": "OK", "love": "I love you", "thanks": "Thank you"}
GEST_HI = {"hi": "नमस्ते", "bye": "अलविदा", "ok": "ठीक है",
           "love": "मुझे तुमसे प्यार है", "thanks": "धन्यवाद"}
GEST_HI_ROMAN = {"hi": "Namaste", "bye": "Alvida", "ok": "Theek hai",
                 "love": "Mujhe tumse pyaar hai", "thanks": "Dhanyavad"}
HI_ROMAN = {"hello": "Namaste", "how_are_you": "Aap kaise hain?", "alright": "Theek hai",
            "good_morning": "Suprabhat", "good_afternoon": "Shubh dopahar",
            "good_evening": "Shubh sandhya", "good_night": "Shubh ratri",
            "thank_you": "Dhanyavad", "pleased": "Aapse milkar khushi hui"}

# ---------- fonts (OpenCV's own text cannot draw Hindi, so we use Pillow) ----------
LATIN_FONTS = ["/System/Library/Fonts/Helvetica.ttc",
               "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/Library/Fonts/Arial.ttf"]
HINDI_FONTS = ["/System/Library/Fonts/Kohinoor.ttc",
               "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
               "/System/Library/Fonts/Supplemental/ITFDevanagari.ttc",
               "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
               "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
               "/Library/Fonts/Arial Unicode.ttf"]
DEVANAGARI = "आपकसहैंनमतेअलविदाठीकमुझेतुमसप्यारधन्"
FONTS = {}


def _covers(font, chars):
    blank = font.getmask("\uffff").tobytes()   # the "missing glyph" box
    return all(font.getmask(c).tobytes() != blank for c in chars)


def find_font(paths, chars, size):
    for p in paths:
        if not Path(p).exists():
            continue
        for idx in range(8):                    # .ttc files hold several faces
            try:
                f = ImageFont.truetype(p, size, index=idx)
            except Exception:
                break
            try:
                if _covers(f, chars):
                    return f
            except Exception:
                continue
    return None


def make_fonts():
    for name, size in (("big", 34), ("small", 22)):
        en = find_font(LATIN_FONTS, "AbMode:%", size) or ImageFont.load_default()
        hi = (find_font(HINDI_FONTS, DEVANAGARI + "0123456789()%?,", size)
              or find_font(HINDI_FONTS, DEVANAGARI, size))
        FONTS.setdefault("en", {})[name] = en
        FONTS.setdefault("hi", {})[name] = hi


def draw_texts(img, items):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, size, color in items:
        if not text:
            continue
        key = "hi" if any("\u0900" <= ch <= "\u097f" for ch in text) else "en"
        font = FONTS[key][size] or FONTS["en"][size]
        d.text(xy, text, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------- rule-based gesture recognition (no training data) ----------
def _pt(lm, i):
    p = lm.landmark[i]
    return np.array([p.x, p.y])


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def finger_states(lm):
    wrist = _pt(lm, 0)
    ext = {}
    for name, (pip_i, tip_i) in (("index", (6, 8)), ("middle", (10, 12)),
                                 ("ring", (14, 16)), ("pinky", (18, 20))):
        ext[name] = _dist(_pt(lm, tip_i), wrist) > _dist(_pt(lm, pip_i), wrist) * 1.1
    pinky_mcp = _pt(lm, 17)
    ext["thumb"] = _dist(_pt(lm, 4), pinky_mcp) > _dist(_pt(lm, 3), pinky_mcp) * 1.1
    return ext


def hand_size(lm):
    return _dist(_pt(lm, 0), _pt(lm, 9)) + 1e-6


def n_fingers(e):
    return sum(e[k] for k in ("index", "middle", "ring", "pinky"))


def classify_hand(lm):
    e = finger_states(lm)
    size = hand_size(lm)
    n4 = n_fingers(e)
    if e["thumb"] and e["index"] and e["pinky"] and not e["middle"] and not e["ring"]:
        return "love"
    if e["thumb"] and n4 == 0 and lm.landmark[4].y < lm.landmark[0].y - 0.5 * size:
        return "ok"                              # thumbs up
    if _dist(_pt(lm, 4), _pt(lm, 8)) < 0.35 * size and e["middle"] and e["ring"] and e["pinky"]:
        return "ok"                              # thumb-index circle
    if n4 == 4:
        return "open"
    return None


def is_namaste(hl):
    a, b = hl[0], hl[1]
    size = (hand_size(a) + hand_size(b)) / 2
    close = _dist(_pt(a, 9), _pt(b, 9)) < 1.0 * size
    return close and n_fingers(finger_states(a)) >= 3 and n_fingers(finger_states(b)) >= 3


# ---------- helpers ----------
def resample(feats):
    idx = np.linspace(0, len(feats) - 1, SEQ_LEN).astype(int)
    return np.stack([feats[i] for i in idx])


def hand_ratio(frames):
    arr = np.stack(frames)
    return float((np.abs(arr[:, 99:]).sum(axis=1) > 0).mean())


def stable_of(dq, need=5):
    c = Counter(x for x in dq if x)
    if not c:
        return None
    k, n = c.most_common(1)[0]
    return k if n >= need else None


def letter_from_history(recent_p, classes, min_frames=5, thr=LETTER_THR):
    """Average the last frames' probabilities. Returns (letter or None, confidence, top3)."""
    ps = [p for p in recent_p if p is not None]
    if len(ps) < min_frames:
        return None, 0.0, []
    mean = np.mean(ps, axis=0)
    order = np.argsort(mean)[::-1][:3]
    top = [(classes[i], float(mean[i])) for i in order]
    letter = top[0][0] if top[0][1] >= thr else None
    return letter, top[0][1], top


def save_session(words, letters, phrases):
    SESSION.parent.mkdir(exist_ok=True)
    SESSION.write_text(json.dumps({"words": words, "letters": letters,
                                   "phrases": phrases, "updated": time.time()}))


def get_tables(lang, devanagari_ok):
    if lang == "hi":
        return (HI, GEST_HI) if devanagari_ok else (HI_ROMAN, GEST_HI_ROMAN)
    return EN, GEST_EN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "hi"])
    ap.add_argument("--mode", default="signs", choices=MODES)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    lang, mode = args.lang, args.mode

    make_fonts()
    devanagari_ok = RAQM and FONTS["hi"]["big"] is not None
    if not devanagari_ok:
        print("Hindi text shaping is not available here (install fribidi: brew install fribidi). "
              "This window shows Hindi in Roman letters; the Streamlit page shows real Devanagari.")
    big, small_f = "big", "small"

    word_model, classes = load_word_model()
    alpha = joblib.load("models/alphabet_model.joblib")
    a_model, a_classes = alpha["model"], alpha["classes"]

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Cannot open camera. Allow camera access for Terminal in System Settings > "
              "Privacy & Security > Camera, or try --cam 1.")
        return

    holistic = mp_holistic.Holistic(model_complexity=0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    hands_flip = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    words, letters, phrases = [], "", []
    frames, recording = [], False
    recent_g, recent_p, wave = deque(maxlen=8), deque(maxlen=10), deque(maxlen=20)
    status = HINTS[mode]
    save_session(words, letters, phrases)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        live_gesture = None

        if mode == "words":
            res = holistic.process(rgb)
            if recording:
                frames.append(frame_features(res))
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        else:
            res = hands.process(rgb)
            res2 = hands_flip.process(cv2.flip(rgb, 1))   # mirrored view for the letter model
            hl = res.multi_hand_landmarks
            if hl:
                for h in hl:
                    mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                if len(hl) == 2 and is_namaste(hl):
                    live_gesture = "thanks"
                    wave.clear()
                else:
                    live_gesture = classify_hand(hl[0])
                    if live_gesture == "open":
                        wave.append(hl[0].landmark[0].x)
                        waving = len(wave) >= 10 and (max(wave) - min(wave)) > 0.15
                        live_gesture = "bye" if waving else "hi"
                    else:
                        wave.clear()
            else:
                wave.clear()
            cands = []
            if hl:
                cands.append(a_model.predict_proba([_alpha_features(res)])[0])
            if res2.multi_hand_landmarks:
                cands.append(a_model.predict_proba([_alpha_features(res2)])[0])
            recent_p.append(max(cands, key=lambda p: p.max()) if cands else None)
            recent_g.append(live_gesture)

        stable_g = stable_of(recent_g)
        stable_l, letter_conf, top3 = letter_from_history(recent_p, a_classes)

        disp = cv2.flip(frame, 1)   # mirror for display only
        h_img = disp.shape[0]
        if recording:
            cv2.circle(disp, (disp.shape[1] - 40, 35), 14, (0, 0, 255), -1)

        table, g_table = get_tables(lang, devanagari_ok)
        out_name = "English" if lang == "en" else ("Hindi" if devanagari_ok else "Hindi (Roman)")
        white = (255, 255, 255)
        items = [(f"Mode: {MODE_NAME[mode]}   |   Output: {out_name}", (15, 10), small_f, (255, 255, 0))]
        if mode == "words":
            items.append((" ".join(table[w] for w in words) or "...", (15, 45), big, white))
        else:
            items.append((g_table[live_gesture] if live_gesture else "-", (15, 45), big, white))
            items.append((f"Letter: {stable_l} ({letter_conf:.0%})" if stable_l else "Letter: -",
                          (15, 95), big, white))
            if top3:
                items.append(("Guesses: " + " | ".join(f"{c} {p:.0%}" for c, p in top3),
                              (15, 140), small_f, (200, 200, 200)))
            items.append((", ".join(g_table[p] for p in phrases), (15, 170), big, white))
            items.append(("Spelling: " + letters, (15, 220), big, white))
        items.append((status, (15, h_img - 70), small_f, (0, 255, 255)))
        items.append(("SPACE add | A add letter | M mode | L language | BACKSPACE | C clear | Q quit",
                      (15, h_img - 35), small_f, white))
        cv2.imshow("Sign recognition", draw_texts(disp, items))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("c"):
            words, letters, phrases, status = [], "", [], "Cleared"
            save_session(words, letters, phrases)
        elif k == ord("m"):
            mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            recording, frames = False, []
            recent_g.clear()
            recent_p.clear()
            wave.clear()
            status = HINTS[mode]
        elif k == ord("l"):
            lang = "hi" if lang == "en" else "en"
        elif k in (8, 127):
            if mode == "words" and words:
                words.pop()
            elif mode == "signs":
                if letters:
                    letters = letters[:-1]
                elif phrases:
                    phrases.pop()
            save_session(words, letters, phrases)
        elif k == ord("a") and mode == "signs":
            if stable_l:
                letters += stable_l
                save_session(words, letters, phrases)
                status = f"Added letter {stable_l}"
                recent_p.clear()
            else:
                status = "No steady letter yet, hold the sign still"
        elif k == ord(" "):
            if mode == "signs":
                if stable_g:
                    phrases.append(stable_g)
                    save_session(words, letters, phrases)
                    status = "Added gesture"
                    recent_g.clear()
                elif stable_l:
                    letters += stable_l
                    save_session(words, letters, phrases)
                    status = f"Added letter {stable_l}"
                    recent_p.clear()
                else:
                    status = "Hold the sign steady, then press SPACE"
            elif not recording:
                recording, frames, status = True, [], "Recording... sign now, SPACE when done"
            else:
                recording = False
                if len(frames) < MIN_FRAMES:
                    status = "Too short, try again"
                elif hand_ratio(frames) < MIN_HAND_RATIO:
                    status = "No hands seen, try again"
                else:
                    x = torch.tensor(normalize(resample(frames))[None], dtype=torch.float32)
                    with torch.no_grad():
                        p = torch.softmax(word_model(x), dim=1)[0]
                    top = torch.topk(p, 2)
                    c1, c2, i = float(top.values[0]), float(top.values[1]), int(top.indices[0])
                    if c1 >= WORD_CONF and c1 - c2 >= WORD_MARGIN:
                        words.append(classes[i])
                        save_session(words, letters, phrases)
                        status = f"{table[classes[i]]} ({c1:.0%})"
                    else:
                        status = "Not sure, try again"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()