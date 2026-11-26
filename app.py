import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# from translators.translate_nllb import translate
from translators.router import translate
from sign.recognize import recognize_word, build_sentence
from sign.text_to_sign import text_to_sign
from sign.live_camera import GEST_EN, GEST_HI
from voice.sarvam_voice import transcribe, speak

ROOT = Path(__file__).parent.resolve()

st.set_page_config(page_title="Indian Language + ISL Assistant", layout="wide")
st.title("Multilingual Communication Assistant")
st.caption("Voice & text translation (NLLB-200 + Sarvam) · Sign recognition (MediaPipe)")

LANGS = {"English": "eng_Latn", "Hindi": "hin_Deva", "Punjabi": "pan_Guru",
         "Bengali": "ben_Beng", "Gujarati": "guj_Gujr", "Marathi": "mar_Deva"}

# Hardcoded demo sentences (clearly labelled in the UI)
DEMO = ["How are you? I hope you are doing well.",
        "Good morning, thank you for coming.",
        "Where is the nearest hospital?",
        "I would like some water, please.",
        "The meeting will start at five o'clock."]

SIGN_MODES = {"Gestures + Alphabet": "signs", "Words (experimental)": "words"}


def _load_demo():
    d = st.session_state.demo
    if d != "(type your own)":
        st.session_state.t_text = d


def launch_camera(lang_arg, mode_arg, direct):
    """Returns (started_ok, error_text)."""
    cmd = [sys.executable, "-m", "sign.live_camera", "--lang", lang_arg, "--mode", mode_arg]
    if direct:
        log_path = ROOT / "results" / "camera_error.log"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log)
            time.sleep(4)
        if proc.poll() is not None:
            return False, log_path.read_text()[-1500:] or "(no output)"
        return True, ""
    script = ROOT / "run_camera.command"
    script.write_text(
        "#!/bin/bash\n"
        f'cd "{ROOT}"\n'
        'export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"\n'
        f'"{sys.executable}" -m sign.live_camera --lang {lang_arg} --mode {mode_arg}\n'
        "echo\n"
        'echo "Camera closed. Press any key to close this window."\n'
        "read -n 1\n"
    )
    script.chmod(0o755)
    subprocess.Popen(["open", "-a", "Terminal", str(script)])
    return True, ""


@st.fragment(run_every=1)
def live_results(lang_arg):
    path = ROOT / "results" / "live_session.json"
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return
    g_table = GEST_HI if lang_arg == "hi" else GEST_EN
    shown = False
    if data.get("phrases"):
        st.success("Gestures: " + ", ".join(g_table[p] for p in data["phrases"]))
        shown = True
    if data.get("letters"):
        st.success("Spelled: " + data["letters"])
        shown = True
    if data.get("words"):
        st.success("Words: " + build_sentence(data["words"], lang_arg))
        shown = True
    if not shown:
        st.caption("Nothing recognized yet. Results appear here live.")


tab1, tab2, tab3, tab4 = st.tabs(["Translate", "Sign → Text", "Text → Sign", "Results"])

# ---------- Translate ----------
with tab1:
    c1, c2 = st.columns(2)
    src = c1.selectbox("From", list(LANGS), index=0, key="t_src")
    tgt = c2.selectbox("To", list(LANGS), index=1, key="t_tgt")
    st.selectbox("Demo examples (pre-written sentences)", ["(type your own)"] + DEMO,
                 key="demo", on_change=_load_demo)

    audio = st.audio_input("Or speak in the source language")
    if audio is not None and st.button("Transcribe speech"):
        try:
            with st.spinner("Transcribing..."):
                st.session_state.t_text = transcribe(audio.getvalue(), LANGS[src])
        except Exception as e:
            st.error(f"Voice input failed: {e}. You can still type the text below.")

    text = st.text_area("Text", key="t_text", height=120)
    if st.button("Translate"):
        if text.strip():
            with st.spinner("Translating..."):
                st.session_state.result = translate(text, LANGS[src], LANGS[tgt])
        else:
            st.warning("Enter some text.")
    if st.session_state.get("result"):
        st.success(st.session_state.result)
        if st.button("Speak translation"):
            try:
                st.audio(speak(st.session_state.result, LANGS[tgt]), format="audio/wav")
            except Exception as e:
                st.error(f"Voice output failed: {e}")

# ---------- Sign -> Text ----------
with tab2:
    st.subheader("Live sign recognition")
    c1, c2, c3 = st.columns(3)
    out_lang = c1.radio("Show result in", ["English", "Hindi"], key="sign_lang")
    start_label = c2.radio("Start in mode", list(SIGN_MODES), key="sign_mode")
    direct = c3.checkbox("Open without extra Terminal window",
                         help="Works only if Streamlit was started from the macOS Terminal app.")
    st.write("In the camera window: **SPACE** adds the gesture or letter (or records a word), **A** adds "
             "the letter, **M** switches mode, **L** switches English/Hindi, **BACKSPACE** deletes, "
             "**C** clears, **Q** closes.")
    st.caption("Gestures: open palm = Hi, waving palm = Bye, thumbs up or OK circle = OK, "
               "thumb + index + pinky = I love you, palms together = Thank you. Letters: A-Z and 1-9.")

    lang_arg = "hi" if out_lang == "Hindi" else "en"
    mode_arg = SIGN_MODES[start_label]
    if st.button("Open camera window"):
        started, err = launch_camera(lang_arg, mode_arg, direct)
        if started:
            st.info("The camera is starting (a few seconds). Results appear below as you sign.")
        else:
            st.error("The camera closed right after starting. Untick the box above, or see the details:")
            st.code(err)
    with st.expander("If no window opens, run this in the Terminal app instead"):
        st.code(f'cd "{ROOT}" && source .venv/bin/activate && '
                f"python -m sign.live_camera --lang {lang_arg} --mode {mode_arg}", language="bash")

    live_results(lang_arg)

    with st.expander("Backup: test the word model on a sample video file"):
        clips = sorted((ROOT / "data" / "sign_clips").glob("*.*"))
        sample = st.selectbox("Sample clip", [c.name for c in clips])
        if st.button("Recognize sample clip"):
            word, conf = recognize_word(str(ROOT / "data" / "sign_clips" / sample))
            st.write(f"{build_sentence([word], 'en')} / {build_sentence([word], 'hi')} ({conf:.0%})")

# ---------- Text -> Sign ----------
with tab3:
    t = st.text_input("English or Hindi text", "hello how are you")
    if st.button("Show signs"):
        steps = text_to_sign(t)
        if not steps:
            st.warning("Nothing to show.")
        for s in steps:
            if s["type"] == "clip":
                st.write(f"Sign: **{s['label'].replace('_', ' ')}**")
                st.video(s["path"])
            elif s["type"] == "letter":
                st.image(s["path"], caption=f"Fingerspell: {s['label']}", width=120)
            else:
                st.write(f"(no sign available for '{s['label']}')")

# ---------- Results ----------
with tab4:
    frames = [pd.read_csv(p) for p in sorted((ROOT / "results").glob("translation_scores*.csv"))]
    if frames:
        df = pd.concat(frames)
        st.subheader("Translation quality on FLORES-200 devtest (first 100 sentences)")
        st.dataframe(df.pivot_table(index=["src", "tgt"], columns="model",
                                    values=["bleu", "chrf++"]).round(2))
    cm = ROOT / "results" / "word_confusion_matrix.png"
    if cm.exists():
        st.subheader("Sign word classifier: confusion matrix (test accuracy 84.2%)")
        st.image(str(cm), width=500)
    cm2 = ROOT / "results" / "alphabet_confusion_matrix.png"
    if cm2.exists():
        st.subheader("Alphabet classifier (99.9%, likely inflated by near-duplicate images)")
        st.image(str(cm2), width=600)