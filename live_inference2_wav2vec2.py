# Live Inference — Approach 2: Wav2Vec2 (cached embeddings → fast training on first run)
# Run augment_dataset.py FIRST, then optionally approach2_wav2vec2.py to pre-train.
# pip install transformers torch torchaudio librosa sounddevice pynput scikit-learn numpy
#
# macOS: System Settings → Privacy & Security → Accessibility → add Terminal / iTerm2

import os
import json
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, logging as hf_logging
import librosa
import sounddevice as sd
from sklearn.metrics import accuracy_score
from pynput import keyboard

hf_logging.set_verbosity_error()

ORIGINAL_PATH = "SMAI Dataset (wav)"
AUG_PATH      = "SMAI Dataset (wav) - Augmented"
SPLIT_FILE    = "dataset_split.json"
MODEL_SAVE    = "approach2_model.pt"       # head weights only
TRAIN_CACHE   = "approach2_train_cache.pt"
TEST_CACHE    = "approach2_test_cache.pt"
MODEL_ID      = "facebook/wav2vec2-base"
SAMPLE_RATE   = 16000
MAX_AUDIO_LEN = 32000
BATCH_SIZE    = 64
MAX_EPOCHS    = 150
LR            = 1e-3
PATIENCE      = 20
NUM_CLASSES   = 11      # 0-9 digits + 10 = unknown


# ── classifier head ───────────────────────────────────────────

class DigitHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.net(x)


# ── embedding helpers ─────────────────────────────────────────

def _embed_file(fpath, processor, encoder, device):
    audio, _ = librosa.load(fpath, sr=SAMPLE_RATE)
    if len(audio) < MAX_AUDIO_LEN:
        audio = np.pad(audio, (0, MAX_AUDIO_LEN - len(audio)))
    else:
        audio = audio[:MAX_AUDIO_LEN]
    inp = processor(audio, sampling_rate=SAMPLE_RATE,
                    return_tensors="pt", padding=False)
    with torch.no_grad():
        return encoder(inp.input_values.to(device)).last_hidden_state.mean(dim=1).cpu()


def extract_embeddings(files, labels, processor, encoder, device, cache_path):
    if os.path.exists(cache_path):
        print(f"  Cache hit  → {cache_path}")
        d = torch.load(cache_path, map_location="cpu")
        return d["emb"], d["labels"]
    print(f"  Extracting {len(files)} embeddings (one-time) …")
    encoder.eval()
    embs = [_embed_file(f, processor, encoder, device) for f in files]
    emb_t = torch.cat(embs, dim=0)
    lbl_t = torch.LongTensor([int(l) for l in labels])
    torch.save({"emb": emb_t, "labels": lbl_t}, cache_path)
    return emb_t, lbl_t


# ── training (runs once if no saved model) ────────────────────

def train_and_save(device, processor, encoder):
    print("No saved model — training head on cached embeddings …")
    assert os.path.exists(SPLIT_FILE), \
        f"Run augment_dataset.py first to create {SPLIT_FILE}"
    with open(SPLIT_FILE) as f:
        split = json.load(f)

    aug_files, aug_labels = [], []
    for d in range(NUM_CLASSES):
        folder = os.path.join(AUG_PATH, str(d))
        if not os.path.exists(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.endswith(".wav"):
                aug_files.append(os.path.join(folder, f))
                aug_labels.append(d)

    train_files  = split["train"]        + aug_files
    train_labels = split["train_labels"] + aug_labels
    test_files   = split["test"]
    test_labels  = split["test_labels"]

    print(f"  Train: {len(train_files)} | Test: {len(test_files)} (original only)")
    print("  Train embeddings:")
    train_emb, train_lbl = extract_embeddings(
        train_files, train_labels, processor, encoder, device, TRAIN_CACHE)
    print("  Test embeddings:")
    test_emb, test_lbl = extract_embeddings(
        test_files, test_labels, processor, encoder, device, TEST_CACHE)

    tr_dl = DataLoader(TensorDataset(train_emb, train_lbl), BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(TensorDataset(test_emb,  test_lbl),  BATCH_SIZE)

    head  = DigitHead(num_classes=NUM_CLASSES).to(device)
    opt   = torch.optim.Adam(head.parameters(), lr=LR)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit  = nn.CrossEntropyLoss()
    best_acc, no_improve, best_epoch = 0.0, 0, 0

    for ep in range(1, MAX_EPOCHS + 1):
        head.train()
        total = 0.0
        for X, y in tr_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(head(X), y); loss.backward(); opt.step()
            total += loss.item()
        sch.step()

        head.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in te_dl:
                preds += head(X.to(device)).argmax(1).cpu().tolist()
                trues += y.tolist()
        acc = accuracy_score(trues, preds)
        print(f"  Epoch {ep:3d} | loss={total/len(tr_dl):.4f} | acc={acc:.4f}", end="")

        if acc > best_acc:
            best_acc, best_epoch, no_improve = acc, ep, 0
            torch.save(head.state_dict(), MODEL_SAVE)
            print("  ✓ saved")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"  Early stop @ epoch {ep}. Best: epoch {best_epoch}, acc {best_acc:.4f}")
                break

    head.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    print(f"  Best model loaded.\n")
    return head


# ── shared state ──────────────────────────────────────────────
_is_held   = False
_recording = False
_buf: list  = []
_lock      = threading.Lock()
_head      = None
_encoder   = None
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
    print(f"  Captured {dur:.2f}s — inferring …", end="", flush=True)

    # Pad / truncate then embed → head
    if len(audio) < MAX_AUDIO_LEN:
        audio = np.pad(audio, (0, MAX_AUDIO_LEN - len(audio)))
    else:
        audio = audio[:MAX_AUDIO_LEN]
    inp = _processor(audio, sampling_rate=SAMPLE_RATE,
                     return_tensors="pt", padding=False)
    with torch.no_grad():
        emb    = _encoder(inp.input_values.to(_device)).last_hidden_state.mean(dim=1)
        logits = _head(emb)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred  = int(probs.argmax())

    label_str = "UNKNOWN" if pred == 10 else str(pred)
    print(f"\r  Digit   : *** {label_str} ***  (conf {probs[pred]:.1%})")
    CLASS_NAMES = [str(i) for i in range(10)] + ["UNK"]
    for d in probs.argsort()[::-1][:3]:
        bar = "█" * int(probs[d] * 30)
        print(f"    {CLASS_NAMES[d]}: {bar:<30} {probs[d]:.1%}")
    print()
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
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {_device}")
print(f"Loading encoder from {MODEL_ID} …")
_processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
_encoder   = Wav2Vec2Model.from_pretrained(MODEL_ID).to(_device)
_encoder.eval()
for p in _encoder.parameters():
    p.requires_grad = False

if os.path.exists(MODEL_SAVE):
    print(f"Loading head from {MODEL_SAVE} …")
    _head = DigitHead(num_classes=NUM_CLASSES).to(_device)
    _head.load_state_dict(torch.load(MODEL_SAVE, map_location=_device))
else:
    _head = train_and_save(_device, _processor, _encoder)

_head.eval()
print("Ready.\n")
print("=" * 48)
print("  Wav2Vec2 Classifier  |  Hindi Digit 0-9")
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
