# Hindi Digit Recognition — SMAI Assignment 3

Spoken Hindi digit recognition (0–9 + Unknown) using three different deep learning approaches, plus a live Gradio demo deployable on HuggingFace Spaces.

---

## Dataset

**SMAI Dataset (wav)** — WAV recordings of Hindi digits organized by class:

```
SMAI Dataset (wav)/
├── 0/   ← शून्य (zero)
├── 1/   ← एक   (one)
├── ...
└── 10/  ← unknown / non-digit speech
```

**Augmentation** — run `augment_dataset.py` once before training. It applies time-stretch, pitch-shift, additive noise, and volume perturbation to generate 5× more training data, saving results to `SMAI Dataset (wav) - Augmented/` and creating `dataset_split.json` (80/20 train-test split on original files).

```bash
python augment_dataset.py
```

---

## Approaches

### Approach 1 — Fine-tuned Whisper (tiny)
- **Model:** `openai/whisper-tiny` with encoder frozen, decoder fine-tuned
- **Strategy:** Seq2seq transcription → keyword matching against Hindi/English digit vocabulary
- **Unknown class:** Any transcription that doesn't match a known digit keyword → class 10
- **Script:** `Main/approach1_whisper_finetune.py`
- **Output:** `approach1_whisper_finetuned/` (saved via `save_pretrained`)

### Approach 2 — Wav2Vec2 + MLP Classifier
- **Model:** Frozen `facebook/wav2vec2-base` encoder + trainable 2-layer MLP head
- **Strategy:** Encoder embeddings (768-dim, mean-pooled) are extracted once and cached; only the head is trained
- **Script:** `Main/approach2_wav2vec2.py`
- **Output:** `approach2_model.pt` (head weights only, ~784 KB)
- **Cache:** `approach2_train_cache.pt`, `approach2_test_cache.pt` (not needed for inference)

### Approach 3 — MFCC + 2D CNN
- **Features:** MFCC + Δ + ΔΔ stacked into a 3×40×100 tensor
- **Model:** Lightweight 2D ConvNet (3 conv blocks + MLP head) trained from scratch
- **Script:** `Main/approach3_mfcc_cnn.py`
- **Output:** `approach3_model.pt` (~2.6 MB)

---

## Project Structure

```
SMAI A3/
├── augment_dataset.py              # Step 1: augment data + create split
├── dataset_split.json              # Auto-generated train/test split
│
├── Main/
│   ├── approach1_whisper_finetune.py
│   ├── approach1_whisper_zeroshot.py
│   ├── approach2_wav2vec2.py
│   ├── approach3_mfcc_cnn.py
│   ├── live_inference1_whisper.py  # Terminal live demo (hold Enter to record)
│   ├── live_inference2_wav2vec2.py
│   ├── live_inference3_mfcc_cnn.py
│   ├── approach1_whisper_finetuned/
│   ├── approach2_model.pt
│   └── approach3_model.pt
│
├── all_approaches.ipynb            # Single notebook: all 3 training pipelines
│
├── demo/
│   ├── app.py                      # Gradio web demo (all 3 approaches)
│   ├── requirements.txt
│   └── .gitattributes              # git-lfs config for large model files
│
├── approach1_results.png
├── approach2_results.png
└── approach3_results.png
```

---

## Training
Run use the unified notebook:
```bash
jupyter notebook all_approaches.ipynb
```

Each script saves the best model (by validation accuracy, with early stopping) and outputs a results PNG with training loss, validation accuracy, confusion matrix and saves the model.

---

## Terminal Live Inference

Hold **Enter** to record, release to get a prediction. Press **Esc** to quit.

```bash
python Main/live_inference1_whisper.py
python Main/live_inference2_wav2vec2.py
python Main/live_inference3_mfcc_cnn.py
```

Requires microphone access. On macOS: System Settings → Privacy & Security → Microphone → allow Terminal.

---

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list. Key dependencies:

| Package | Purpose |
|---|---|
| `torch` / `torchaudio` | Model training and inference |
| `transformers` | Whisper, Wav2Vec2 |
| `librosa` | Audio loading, MFCC extraction, augmentation |
| `soundfile` | WAV I/O |
| `sounddevice` + `pynput` | Terminal live inference (mic capture + keypress) |
| `scikit-learn` | Metrics, train/test split |
| `gradio` | Web demo |
