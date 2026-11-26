# Marathi + Gujarati translation part

Scope: {English, Hindi} <-> {Marathi, Gujarati}, plus Marathi <-> Gujarati via English pivot.
Models: NLLB-200 (600M distilled) baseline, IndicTrans2 distilled zero-shot, IndicTrans2 + LoRA.
Test set: FLORES-200 devtest (1012 sentences). IN22-Gen was loaded but **not evaluated**.

## Files

| File | What it is |
|---|---|
| `marathi_gujarati_clean.ipynb` | Clean experiment notebook, run in stages (see its first cell) |
| `translate_gu_mr.py` | Module for the app: `translate(text, src, tgt)` |
| `flores100_gu_mr.csv` | Scores on the first 100 FLORES devtest sentences, columns `model,src,tgt,n,bleu,chrf++` |
| `flores_full_1012.csv` | Scores on all 1012 FLORES devtest sentences, same columns |
| `adapters.zip` (Drive link, not in Git) | LoRA adapters `lora_en-indic`, `lora_indic-en` |

The two CSVs were typed in from the notebook outputs. If your own `results/*.csv` files exist, those are the source of truth.

## Metrics

sacrebleu defaults: `corpus_bleu` and `corpus_chrf(word_order=2)` (chrF++), beam size 5.
Please check these match your settings before merging tables.

## Results (FLORES devtest, 1012 sentences; BLEU / chrF++)

| Direction | NLLB | IT2 zero-shot | IT2 + LoRA |
|---|---|---|---|
| eng->mar | 14.23 / 44.80 | 19.44 / 50.73 | 17.61 / 48.26 |
| mar->eng | 33.87 / 58.86 | 38.27 / 62.52 | 36.98 / 61.26 |
| eng->guj | 22.36 / 50.27 | 27.33 / 55.60 | 25.96 / 54.27 |
| guj->eng | 39.35 / 62.64 | 42.82 / 65.38 | 41.21 / 64.23 |

Marathi <-> Gujarati:

| Direction | NLLB | IT2 direct | IT2 Hindi pivot | IT2 English pivot | IT2+LoRA English pivot |
|---|---|---|---|---|---|
| mar->guj | 16.83 / 44.66 | 17.37 / 45.80 | 16.84 / 45.02 | 19.17 / 47.05 | 18.01 / 45.62 |
| guj->mar | 13.80 / 42.86 | 14.16 / 44.58 | 13.56 / 43.88 | 15.96 / 45.80 | 14.47 / 43.70 |

Findings:
- IndicTrans2 zero-shot beats NLLB in every direction.
- LoRA fine-tuning did **not** beat zero-shot (1.2 to 1.8 BLEU lower on all four English directions).
  We did not test why. Possible reasons only: noisy Samanantar pairs, IndicTrans2 already trained on similar data, learning rate 2e-4.
- English pivot beat direct and Hindi pivot for Marathi <-> Gujarati.
- Single runs, no significance test. Treat small differences carefully.

## Routing used in `translate_gu_mr.py`

`USE_LORA = False`. English pairs use zero-shot, Marathi <-> Gujarati uses English pivot, Hindi pairs use zero-shot
(the LoRA adapters were trained on English <-> Marathi/Gujarati only, and Hindi pairs were only evaluated on the 100-sentence set).

## Setup

```
pip install torch transformers==4.46.3 "datasets<4" peft accelerate sacrebleu sentencepiece pandas tqdm gTTS cython setuptools
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

Tested on a Mac (Apple M-series, MPS), Python 3.11. IndicTrans2 remote code is version-sensitive, so keep `transformers==4.46.3`.
`peft` is only imported when `USE_LORA = True`.

Hugging Face: IN22-Gen is gated, and IndicTrans2 repos gave a `GatedRepoError` before terms were accepted.
Log in (`huggingface-cli login`) and accept the terms on those pages. Never commit a token.

## Known issues and lessons

- LoRA training targets must go through `IndicProcessor` (script conversion). Raw Gujarati targets gave loss about 10.1, processed gave about 2.3. The notebook does this correctly.
- ASR: Whisper `small` was poor on Marathi/Gujarati (average WER 0.75 and 1.0 on 3 synthetic gTTS sentences each, a tiny sample).
  The app uses Sarvam Saaras v3 (cloud API, `mr-IN`, `gu-IN`). We did not measure its WER.
- Not done: IN22 evaluation, LoRA on Hindi pairs, significance tests.
- On a 16GB-class Mac, run NLLB, IndicTrans2 eval and LoRA training in separate kernels (swap makes MPS very slow).
