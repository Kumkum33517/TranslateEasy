from translators import translate_nllb, translate_gu_mr

ENG, HIN, PAN, BEN, GUJ, MAR = ("eng_Latn", "hin_Deva", "pan_Guru",
                                "ben_Beng", "guj_Gujr", "mar_Deva")
GM = {GUJ, MAR}   # Gujarati/Marathi are handled by the IndicTrans2 module


def _owner(lang):
    return translate_gu_mr if lang in GM else translate_nllb


def translate(text, src, tgt):
    if src == tgt:
        return text
    # no Gujarati/Marathi involved: NLLB (English, Hindi, Punjabi, Bengali)
    if src not in GM and tgt not in GM:
        return translate_nllb.translate(text, src, tgt)
    other = tgt if src in GM else src
    # English <-> mar/guj and mar <-> guj: IndicTrans2 module
    if other in GM or other == ENG:
        return translate_gu_mr.translate(text, src, tgt)
    # Hindi <-> mar/guj needs the gated indic-indic model; if unavailable, pivot below
    if other == HIN:
        try:
            return translate_gu_mr.translate(text, src, tgt)
        except Exception:
            pass
    # Punjabi/Bengali <-> mar/guj (or Hindi without access): pivot through English
    mid = text if src == ENG else _owner(src).translate(text, src, ENG)
    return mid if tgt == ENG else _owner(tgt).translate(mid, ENG, tgt)