import base64
import os
import tempfile

from sarvamai import SarvamAI

CODES = {"eng_Latn": "en-IN", "hin_Deva": "hi-IN", "pan_Guru": "pa-IN",
         "ben_Beng": "bn-IN", "guj_Gujr": "gu-IN", "mar_Deva": "mr-IN"}
STT_MODEL = "saaras:v3"   # if the API rejects this name, see the note below
TTS_MODEL = "bulbul:v3"


def _client():
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set in this terminal")
    return SarvamAI(api_subscription_key=key)


def transcribe(audio_bytes, lang):
    """WAV bytes + IndicTrans-style code (e.g. 'hin_Deva') -> text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    with open(path, "rb") as fh:
        r = _client().speech_to_text.transcribe(file=fh, model=STT_MODEL,
                                                language_code=CODES[lang])
    return r.transcript


def speak(text, lang):
    """Text -> WAV bytes."""
    r = _client().text_to_speech.convert(text=text, language_code=CODES[lang],
                                         model=TTS_MODEL)
    return base64.b64decode(r.audios[0])