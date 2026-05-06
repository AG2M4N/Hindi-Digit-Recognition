
# pip install librosa soundfile numpy scikit-learn

import os
import json
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

ORIGINAL_PATH = "SMAI Dataset (wav)"
AUG_PATH      = "SMAI Dataset (wav) - Augmented"
SPLIT_FILE    = "dataset_split.json"
SAMPLE_RATE   = 16000
N_AUG         = 5   # augmented variants per training file → 160 × 5 = 800 extra clips
NUM_CLASSES   = 11  # 0-9 digits + 10 = unknown 


# ── augmentation strategies ───────────────────────────────────

def _augment(audio: np.ndarray, sr: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    aug = audio.copy().astype(np.float32)

    # time stretch  (0.85 – 1.15×)
    if rng.rand() < 0.8:
        aug = librosa.effects.time_stretch(aug, rate=rng.uniform(0.85, 1.15))

    # pitch shift  ±3 semitones
    if rng.rand() < 0.8:
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=rng.uniform(-3, 3))

    # additive gaussian noise
    if rng.rand() < 0.7:
        aug = aug + rng.uniform(0.001, 0.010) * rng.randn(len(aug)).astype(np.float32)

    # volume perturbation  (0.5 – 1.5×)
    if rng.rand() < 0.7:
        aug = aug * rng.uniform(0.5, 1.5)

    # speed perturbation without pitch change
    if rng.rand() < 0.5:
        speed  = rng.uniform(0.9, 1.1)
        aug    = librosa.resample(aug, orig_sr=sr, target_sr=int(sr * speed))
        aug    = librosa.resample(aug, orig_sr=int(sr * speed), target_sr=sr)

    return np.clip(aug, -1.0, 1.0).astype(np.float32)


# ── file loading ──────────────────────────────────────────────

def collect_files():
    files, labels = [], []
    for d in range(NUM_CLASSES):          # 0-9 digits + 10 = unknown
        folder = os.path.join(ORIGINAL_PATH, str(d))
        if not os.path.isdir(folder):
            continue                       # skip if unknown class not yet downloaded
        for f in sorted(os.listdir(folder)):
            if f.endswith(".wav"):
                files.append(os.path.join(folder, f))
                labels.append(d)
    return files, labels


# ── main ──────────────────────────────────────────────────────

def main():
    files, labels = collect_files()
    n_classes = len(set(labels))
    class_desc = "digits" if n_classes <= 10 else f"classes (0-9 digits + unknown)"
    print(f"Found {len(files)} original files across {n_classes} {class_desc}.\n")

    # ── create / load split ───────────────────────────────────
    if os.path.exists(SPLIT_FILE):
        print(f"Split already exists → loading {SPLIT_FILE}")
        with open(SPLIT_FILE) as f:
            split = json.load(f)
    else:
        tr_f, te_f, tr_l, te_l = train_test_split(
            files, labels, test_size=0.2, random_state=42, stratify=labels)
        split = {
            "train":        tr_f,
            "test":         te_f,
            "train_labels": [int(l) for l in tr_l],
            "test_labels":  [int(l) for l in te_l],
        }
        with open(SPLIT_FILE, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Split saved → {SPLIT_FILE}")

    print(f"  Train (original): {len(split['train'])} files")
    print(f"  Test  (original): {len(split['test'])} files  ← never augmented\n")

    # ── create augmented folder structure ─────────────────────
    for d in range(NUM_CLASSES):
        os.makedirs(os.path.join(AUG_PATH, str(d)), exist_ok=True)

    # ── augmenting training files only ───────────────────────────
    total = 0
    for idx, (fpath, label) in enumerate(zip(split["train"], split["train_labels"])):
        audio, _ = librosa.load(fpath, sr=SAMPLE_RATE)
        stem     = os.path.splitext(os.path.basename(fpath))[0]
        digit    = str(label)

        for i in range(N_AUG):
            aug_audio = _augment(audio, SAMPLE_RATE, seed=idx * 100 + i)
            out_path  = os.path.join(AUG_PATH, digit, f"{stem}_aug_{i}.wav")
            sf.write(out_path, aug_audio, SAMPLE_RATE)
            total += 1

        print(f"  [{label}] {os.path.basename(fpath)} → {N_AUG} variants")

    print(f"\nDone.")
    print(f"  Augmented files : {total}  (saved to '{AUG_PATH}/')")
    print(f"  Total train pool: {len(split['train'])} original + {total} augmented = {len(split['train']) + total}")
    print(f"  Test set        : {len(split['test'])} original files (untouched)")


if __name__ == "__main__":
    main()
