"""Gujarati/Marathi translation: IndicTrans2 (+ optional LoRA).

translate(text, src, tgt) with codes: eng_Latn, hin_Deva, guj_Gujr, mar_Deva

Routing (decided from the FLORES evaluation, see README.md):
  - eng <-> mar/guj : IndicTrans2 zero-shot (LoRA did not beat zero-shot)
  - mar <-> guj     : English pivot (best of direct / Hindi pivot / English pivot)
  - pairs with hin  : IndicTrans2 zero-shot
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

ENG, HIN, MAR, GUJ = "eng_Latn", "hin_Deva", "mar_Deva", "guj_Gujr"
MODELS = {
    "en-indic":    "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en":    "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
}
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "adapters")   # lora_en-indic, lora_indic-en live here
USE_LORA = False   # LoRA was below zero-shot in our FLORES results
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

_ip = IndicProcessor(inference=True)
_cache = {}


def _direction(src, tgt):
    if src == ENG:
        return "en-indic"
    if tgt == ENG:
        return "indic-en"
    return "indic-indic"


def _load(direction, lora):
    key = (direction, lora)
    if key not in _cache:
        name = MODELS[direction]
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        m = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True)
        if lora:
            from peft import PeftModel   # only needed when USE_LORA=True
            m = PeftModel.from_pretrained(m, os.path.join(ADAPTER_DIR, f"lora_{direction}"))
            m = m.merge_and_unload()
        _cache[key] = (tok, m.to(DEVICE).eval())
    return _cache[key]


@torch.no_grad()
def _hop(text, src, tgt, lora):
    d = _direction(src, tgt)
    tok, mdl = _load(d, lora and d != "indic-indic")
    batch = _ip.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
    inp = tok(batch, truncation=True, padding="longest", max_length=256,
              return_tensors="pt", return_attention_mask=True).to(DEVICE)
    out = mdl.generate(**inp, num_beams=5, max_length=256)
    dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return _ip.postprocess_batch(dec, lang=tgt)[0]


def translate(text, src, tgt):
    valid = {ENG, HIN, MAR, GUJ}
    if src not in valid or tgt not in valid:
        raise ValueError(f"supported codes: {sorted(valid)}")
    if src == tgt:
        return text
    if HIN in (src, tgt):                      # Hindi pairs: zero-shot (LoRA never saw Hindi)
        return _hop(text, src, tgt, lora=False)
    if ENG in (src, tgt):                      # eng <-> mar/guj
        return _hop(text, src, tgt, lora=USE_LORA)
    mid = _hop(text, src, ENG, lora=USE_LORA)  # mar <-> guj: English pivot
    return _hop(mid, ENG, tgt, lora=USE_LORA)


if __name__ == "__main__":
    print(translate("Where is the railway station?", ENG, MAR))
    print(translate("तुम्ही कुठे जात आहात?", MAR, GUJ))
    print(translate("તમે ક્યાં જઈ રહ્યા છો?", GUJ, HIN))
