import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

SUPPORTED = {"eng_Latn", "hin_Deva", "pan_Guru", "ben_Beng"}
EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"

_cache = {}
_ip = IndicProcessor(inference=True)


def _load(name):
    if name not in _cache:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).eval()
        _cache[name] = (tok, model)
    return _cache[name]


def _translate_direct(text, src, tgt):
    name = EN_INDIC if src == "eng_Latn" else INDIC_EN
    tok, model = _load(name)
    batch = _ip.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
    inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, use_cache=False, num_beams=5, max_length=256)
    decoded = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return _ip.postprocess_batch(decoded, lang=tgt)[0]


def translate(text, src, tgt):
    """src/tgt use IndicTrans2 codes: eng_Latn, hin_Deva, pan_Guru, ben_Beng."""
    if src not in SUPPORTED or tgt not in SUPPORTED:
        raise ValueError(f"Unsupported pair: {src} -> {tgt}")
    if src == tgt:
        return text
    if src == "eng_Latn" or tgt == "eng_Latn":
        return _translate_direct(text, src, tgt)
    mid = _translate_direct(text, src, "eng_Latn")   # pivot through English
    return _translate_direct(mid, "eng_Latn", tgt)


if __name__ == "__main__":
    s = "How are you? I hope you are doing well."
    for t in ["hin_Deva", "pan_Guru", "ben_Beng"]:
        print(t, "->", translate(s, "eng_Latn", t))
    print("pa -> bn:", translate(translate(s, "eng_Latn", "pan_Guru"), "pan_Guru", "ben_Beng"))