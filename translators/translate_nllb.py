import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NAME = "facebook/nllb-200-distilled-600M"
SUPPORTED = {"eng_Latn", "hin_Deva", "pan_Guru", "ben_Beng", "guj_Gujr", "mar_Deva"}
_tok = _model = None


def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NAME).eval()
    return _tok, _model


def translate(text, src, tgt):
    """NLLB-200 baseline. Codes: eng_Latn, hin_Deva, pan_Guru, ben_Beng."""
    if src not in SUPPORTED or tgt not in SUPPORTED:
        raise ValueError(f"Unsupported pair: {src} -> {tgt}")
    if src == tgt:
        return text
    tok, model = _load()
    tok.src_lang = src
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
            num_beams=4,
            max_length=256,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


if __name__ == "__main__":
    s = "How are you? I hope you are doing well."
    for t in ["hin_Deva", "pan_Guru", "ben_Beng"]:
        print(t, "->", translate(s, "eng_Latn", t))
    pa = translate(s, "eng_Latn", "pan_Guru")
    print("pa -> bn:", translate(pa, "pan_Guru", "ben_Beng"))