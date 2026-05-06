# Live Inference — Approach 1: Fine-tuned Whisper
# Requires: run approach1_whisper_finetune.py first to create approach1_whisper_finetuned/
# pip install transformers torch librosa sounddevice pynput numpy
#
# macOS: System Settings → Privacy & Security → Accessibility → add Terminal / iTerm2
#
# Class mapping:  0-9 = Hindi digits,  10 = UNKNOWN (non-digit / out-of-vocabulary)
# Detection:  Whisper transcribes → keyword match → no match ⇒ class 10 (unknown)

import os
import sys
import threading
import numpy as np
import sounddevice as sd
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as hf_logging
from pynput import keyboard

hf_logging.set_verbosity_error()  # silence HF_TOKEN notices and load reports

MODEL_DIR   = "approach1_whisper_finetuned"
SAMPLE_RATE = 16000

NUM_CLASSES = 11   # 0-9 digits + 10 = unknown

DIGIT_KEYWORDS = {
    0: ["शून्य", "zero", "0", "shunya", "sunya"],
    1: ["एक",   "one",  "1", "ek"],
    2: ["दो",   "two",  "2", "do"],
    3: ["तीन",  "three","3", "teen", "tin"],
    4: ["चार",  "four", "4", "char", "chaar"],
    5: ["पाँच", "five", "5", "panch", "paanch", "पांच"],
    6: ["छह",   "six",  "6", "chhe", "chhah", "छः"],
    7: ["सात",  "seven","7", "saat", "sat"],
    8: ["आठ",   "eight","8", "aath", "ath"],
    9: ["नौ",   "nine", "9", "nau",  "no"],
    # class 10 = unknown:  fine-tuned model outputs "अज्ञात" for non-digit speech;
    # that word won't match any digit keyword, so it falls through to class 10 below.
}


def match_transcription(text: str) -> int:
    """Return digit class 0-9, or 10 (unknown) if no keyword matches."""
    t = text.lower().strip()
    for digit, kws in DIGIT_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in t:
                return digit
    return 10   # class 10 = unknown


# ── shared state ──────────────────────────────────────────────
_is_held   = False
_recording = False
_buf: list  = []
_lock      = threading.Lock()
_model     = None
_processor = None
_device    = None


def _audio_cb(indata, frames, time, status):
    if _recording:
        with _lock:
            _buf.append(indata[:, 0].copy())


def _infer():
    with _lock:
        chunks = list(_buf)
    if not chunks:
        print("  [!] No audio captured.\n"); _prompt(); return

    audio = np.concatenate(chunks).astype(np.float32)
    dur   = len(audio) / SAMPLE_RATE
    print(f"  Captured {dur:.2f}s — transcribing …", end="", flush=True)

    enc = _processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        gen_ids = _model.generate(
            enc.input_features.to(_device),
            attention_mask=enc.attention_mask.to(_device),
            language="hi", task="transcribe", max_new_tokens=15)
    text = _processor.tokenizer.batch_decode(
        gen_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    pred = match_transcription(text)

    print(f"\r  Whisper : \"{text}\"")
    if pred == 10:
        print("  Class   : *** UNKNOWN (10) ***  [no digit keyword matched]\n")
    else:
        print(f"  Class   : *** {pred} ***\n")
    _prompt()


def _prompt():
    print("Hold [Enter] to record  |  Esc / Ctrl-C to quit")


def on_press(key):
    global _is_held, _recording
    if key == keyboard.Key.enter and not _is_held:
        _is_held = True; _recording = True
        with _lock: _buf.clear()
        print("\n  [● REC] Recording … (hold Enter)", end="", flush=True)


def on_release(key):
    global _is_held, _recording
    if key == keyboard.Key.enter:
        _is_held = False; _recording = False
        print(" ■ stopped")
        threading.Thread(target=_infer, daemon=True).start()
    if key == keyboard.Key.esc:
        return False


# ── startup ───────────────────────────────────────────────────
if not os.path.exists(MODEL_DIR):
    print(f"ERROR: Fine-tuned model not found at '{MODEL_DIR}/'")
    print("Run approach1_whisper_finetune.py first.")
    sys.exit(1)

_device = torch.device("mps"  if torch.backends.mps.is_available()  else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_device}")
print(f"Loading fine-tuned Whisper from {MODEL_DIR} …")
# Processor loaded from hub (not MODEL_DIR) — avoids generation_config.json pollution
_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Hindi", task="transcribe")
_processor.tokenizer.clean_up_tokenization_spaces = False
_model     = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(_device)
cfg = _model.generation_config
cfg.max_length = cfg.forced_decoder_ids = cfg.suppress_tokens = cfg.begin_suppress_tokens = None
_model.eval()
print("Ready.\n")
print("=" * 48)
print("  Whisper Fine-tuned  |  Hindi Digit 0-9 + Unknown (10)")
print("=" * 48)
_prompt()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                    dtype="float32", callback=_audio_cb, blocksize=1024):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as lst:
        try:
            lst.join()
        except KeyboardInterrupt:
            pass

print("\nBye.")
