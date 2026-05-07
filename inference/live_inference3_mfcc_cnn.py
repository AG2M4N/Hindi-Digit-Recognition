# Live Inference — Approach 3: MFCC + 2D CNN
# Run augment_dataset.py FIRST, then optionally approach3_mfcc_cnn.py to pre-train.
# First run without saved model: trains with early stopping on augmented data.
# pip install torch librosa sounddevice pynput scikit-learn numpy
#
# macOS: System Settings → Privacy & Security → Accessibility → add Terminal / iTerm2

import os
import json
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import sounddevice as sd
from sklearn.metrics import accuracy_score
from pynput import keyboard

ORIGINAL_PATH = "SMAI Dataset (wav)"
AUG_PATH      = "SMAI Dataset (wav) - Augmented"
SPLIT_FILE    = "../dataset_split.json"
MODEL_SAVE    = "../approach3_model.pt"
SAMPLE_RATE   = 16000
N_MFCC        = 40
MAX_FRAMES    = 100
BATCH_SIZE    = 16
MAX_EPOCHS    = 100
LR            = 1e-3
PATIENCE      = 15
NUM_CLASSES   = 11      # 0-9 digits + 10 = unknown


# ── feature extraction ────────────────────────────────────────

def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.stack([mfcc, delta, delta2], axis=0)

    if feat.shape[2] < MAX_FRAMES:
        feat = np.pad(feat, ((0,0),(0,0),(0, MAX_FRAMES - feat.shape[2])))
    else:
        feat = feat[:, :, :MAX_FRAMES]

    for c in range(feat.shape[0]):
        m, s = feat[c].mean(), feat[c].std() + 1e-8
        feat[c] = (feat[c] - m) / s

    return feat.astype(np.float32)


# ── model ─────────────────────────────────────────────────────

class MFCCCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── dataset ───────────────────────────────────────────────────

class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── training ──────────────────────────────────────────────────

def _load_features_from_split():
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

    all_train = split["train"] + aug_files
    all_labels_tr = split["train_labels"] + aug_labels

    print(f"  Train: {len(all_train)} | Test: {len(split['test'])} (original only)")
    print("  Extracting MFCCs …")

    def feats(files, labels):
        X, y = [], []
        for fp, lb in zip(files, labels):
            audio, _ = librosa.load(fp, sr=SAMPLE_RATE)
            X.append(extract_mfcc(audio)); y.append(int(lb))
        return np.stack(X), np.array(y)

    X_tr, y_tr = feats(all_train, all_labels_tr)
    X_te, y_te = feats(split["test"], split["test_labels"])
    return X_tr, y_tr, X_te, y_te


def train_and_save(device):
    print("No saved model — training with early stopping …")
    X_tr, y_tr, X_te, y_te = _load_features_from_split()

    tr_dl = DataLoader(MFCCDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(MFCCDataset(X_te, y_te), BATCH_SIZE)

    mdl  = MFCCCNN(num_classes=NUM_CLASSES).to(device)
    opt  = torch.optim.Adam(mdl.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best_acc, no_improve, best_epoch = 0.0, 0, 0

    for ep in range(1, MAX_EPOCHS + 1):
        mdl.train()
        total = 0.0
        for X_b, y_b in tr_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(mdl(X_b), y_b); loss.backward(); opt.step()
            total += loss.item()
        sch.step()

        mdl.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_b, y_b in te_dl:
                preds += mdl(X_b.to(device)).argmax(1).cpu().tolist()
                trues += y_b.tolist()
        acc = accuracy_score(trues, preds)
        print(f"  Epoch {ep:3d} | loss={total/len(tr_dl):.4f} | acc={acc:.4f}", end="")

        if acc > best_acc:
            best_acc, best_epoch, no_improve = acc, ep, 0
            torch.save(mdl.state_dict(), MODEL_SAVE)
            print("  ✓ saved")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"  Early stop @ epoch {ep}. Best: epoch {best_epoch}, acc {best_acc:.4f}")
                break

    mdl.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    print(f"  Best model loaded.\n")
    return mdl


# ── shared state ──────────────────────────────────────────────
_is_held   = False
_recording = False
_buf: list  = []
_lock      = threading.Lock()
_model     = None
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

    feat  = extract_mfcc(audio)
    x     = torch.from_numpy(feat).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)
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
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_device}")

if os.path.exists(MODEL_SAVE):
    print(f"Loading saved model from {MODEL_SAVE} …")
    _model = MFCCCNN(num_classes=NUM_CLASSES).to(_device)
    _model.load_state_dict(torch.load(MODEL_SAVE, map_location=_device))
else:
    _model = train_and_save(_device)

_model.eval()
print("Ready.\n")
print("=" * 48)
print("  MFCC + CNN Classifier  |  Hindi Digit 0-9 + Unknown")
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
