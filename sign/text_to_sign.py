import sys
from pathlib import Path
from sign.lexicon import EN, HI

CLIPS = {p.stem: p for p in Path("data/sign_clips").glob("*.*")}
LETTERS = {p.stem.upper(): p for p in Path("data/alphabet/samples").glob("*.*")}

PHRASES = {}
for k, v in EN.items():
    PHRASES[v.lower().strip("?!.")] = k
for k, v in HI.items():
    PHRASES[v.strip("?!.")] = k
PHRASES.update({"hi": "hello", "thanks": "thank_you", "pleased to meet you": "pleased"})

PUNCT = ".,!?;:\"'।"


def text_to_sign(text):
    """Text (English or Hindi) -> list of steps: clip, letter, or unsupported."""
    tokens = [t.strip(PUNCT) for t in text.lower().split()]
    tokens = [t for t in tokens if t]
    steps, i = [], 0
    while i < len(tokens):
        for n in range(4, 0, -1):               # longest phrase first
            key = " ".join(tokens[i:i + n])
            if key in PHRASES and PHRASES[key] in CLIPS:
                w = PHRASES[key]
                steps.append({"type": "clip", "label": w, "path": str(CLIPS[w])})
                i += n
                break
        else:
            tok = tokens[i]
            if tok.isascii():                    # fingerspell unknown English words
                for ch in tok.upper():
                    if ch in LETTERS:
                        steps.append({"type": "letter", "label": ch, "path": str(LETTERS[ch])})
            else:
                steps.append({"type": "unsupported", "label": tok})
            i += 1
    return steps


if __name__ == "__main__":
    for s in text_to_sign(" ".join(sys.argv[1:])):
        print(s)