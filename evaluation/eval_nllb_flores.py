import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
DATA = Path("data/flores")
OUT_CSV = Path("results/translation_scores.csv")
MODEL_NAME = "facebook/nllb-200-distilled-600M"
MODEL_TAG = "nllb-600M"
N = 100          # first N devtest sentences (state this in the report)
BATCH = 8
LANGS = ["eng_Latn", "hin_Deva", "pan_Guru", "ben_Beng"]
PAIRS = [("eng_Latn", "hin_Deva"), ("hin_Deva", "eng_Latn"),
         ("eng_Latn", "pan_Guru"), ("pan_Guru", "eng_Latn"),
         ("eng_Latn", "ben_Beng"), ("ben_Beng", "eng_Latn"),
         ("pan_Guru", "ben_Beng"), ("ben_Beng", "pan_Guru")]


def get_flores():
    marker = DATA / "flores200_dataset" / "devtest" / "eng_Latn.devtest"
    if marker.exists():
        return
    tar_path = DATA / "flores200.tar.gz"
    if not tar_path.exists():
        print("Downloading FLORES-200 ...")
        urllib.request.urlretrieve(URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        want = [m for m in tar.getmembers()
                if any(m.name.endswith(f"/devtest/{l}.devtest") for l in LANGS)]
        tar.extractall(DATA, members=want, filter="data")
    tar_path.unlink()


def read_lang(lang):
    p = DATA / "flores200_dataset" / "devtest" / f"{lang}.devtest"
    return p.read_text(encoding="utf-8").splitlines()[:N]


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


get_flores()
try:
    bleu = BLEU(tokenize="flores200")
    bleu.corpus_score(["a"], [["a"]])
except Exception:
    print("flores200 tokenizer unavailable, using default BLEU tokenizer")
    bleu = BLEU()
chrf = CHRF(word_order=2)  # chrF++

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

rows = []
for src, tgt in PAIRS:
    refs_src, refs_tgt = read_lang(src), read_lang(tgt)
    hyps = translate_batch(tok, model, refs_src, src, tgt)
    Path(f"results/hyps/{MODEL_TAG}_{src}_{tgt}.txt").write_text("\n".join(hyps), encoding="utf-8")
    row = {"model": MODEL_TAG, "src": src, "tgt": tgt, "n": len(hyps),
           "bleu": round(bleu.corpus_score(hyps, [refs_tgt]).score, 2),
           "chrf++": round(chrf.corpus_score(hyps, [refs_tgt]).score, 2)}
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print("Saved", OUT_CSV) 