from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA = Path("data/flores/flores200_dataset/devtest")
OUT_CSV = Path("results/translation_scores_pivot.csv")
MODEL_NAME = "facebook/nllb-200-distilled-600M"
MODEL_TAG = "nllb-600M-pivot-en"
N, BATCH = 100, 8
PAIRS = [("pan_Guru", "ben_Beng"), ("ben_Beng", "pan_Guru")]


def read_lang(lang):
    return (DATA / f"{lang}.devtest").read_text(encoding="utf-8").splitlines()[:N]


def translate_batch(tok, model, sents, src, tgt):
    tok.src_lang = src
    outs = []
    for i in range(0, len(sents), BATCH):
        inputs = tok(sents[i:i + BATCH], return_tensors="pt", padding=True,
                     truncation=True, max_length=256)
        with torch.no_grad():
            out = model.generate(**inputs,
                                 forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                 num_beams=4, max_length=256)
        outs += tok.batch_decode(out, skip_special_tokens=True)
    return outs


try:
    bleu = BLEU(tokenize="flores200")
    bleu.corpus_score(["a"], [["a"]])
except Exception:
    print("flores200 tokenizer unavailable, using default BLEU tokenizer")
    bleu = BLEU()
chrf = CHRF(word_order=2)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

rows = []
for src, tgt in PAIRS:
    refs_src, refs_tgt = read_lang(src), read_lang(tgt)
    mid = translate_batch(tok, model, refs_src, src, "eng_Latn")   # step 1: -> English
    hyps = translate_batch(tok, model, mid, "eng_Latn", tgt)       # step 2: English ->
    Path(f"results/hyps/{MODEL_TAG}_{src}_{tgt}.txt").write_text("\n".join(hyps), encoding="utf-8")
    row = {"model": MODEL_TAG, "src": src, "tgt": tgt, "n": len(hyps),
           "bleu": round(bleu.corpus_score(hyps, [refs_tgt]).score, 2),
           "chrf++": round(chrf.corpus_score(hyps, [refs_tgt]).score, 2)}
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print("Saved", OUT_CSV)