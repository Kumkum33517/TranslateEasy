from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from translators.translate_pa_bn import translate

DATA = Path("data/flores/flores200_dataset/devtest")
OUT_CSV = Path("results/translation_scores_indictrans2.csv")
MODEL_TAG = "indictrans2-dist-200M"
N = 100
PAIRS = [("eng_Latn", "hin_Deva"), ("hin_Deva", "eng_Latn"),
         ("eng_Latn", "pan_Guru"), ("pan_Guru", "eng_Latn"),
         ("eng_Latn", "ben_Beng"), ("ben_Beng", "eng_Latn"),
         ("pan_Guru", "ben_Beng"), ("ben_Beng", "pan_Guru")]


def read_lang(lang):
    return (DATA / f"{lang}.devtest").read_text(encoding="utf-8").splitlines()[:N]


try:
    bleu = BLEU(tokenize="flores200")
    bleu.corpus_score(["a"], [["a"]])
except Exception:
    print("flores200 tokenizer unavailable, using default BLEU tokenizer")
    bleu = BLEU()
chrf = CHRF(word_order=2)

rows = []
Path("results/hyps").mkdir(parents=True, exist_ok=True)
for src, tgt in PAIRS:
    refs_src, refs_tgt = read_lang(src), read_lang(tgt)
    hyps = []
    for i, s in enumerate(refs_src):
        hyps.append(translate(s, src, tgt))
        if i % 25 == 0:
            print(f"  {src}->{tgt}: {i}/{N}", flush=True)
    Path(f"results/hyps/{MODEL_TAG}_{src}_{tgt}.txt").write_text("\n".join(hyps), encoding="utf-8")
    row = {"model": MODEL_TAG, "src": src, "tgt": tgt, "n": len(hyps),
           "bleu": round(bleu.corpus_score(hyps, [refs_tgt]).score, 2),
           "chrf++": round(chrf.corpus_score(hyps, [refs_tgt]).score, 2)}
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print("Saved", OUT_CSV)