# 🎭 Face Mask AI - Production-Grade Classification & Evaluation System

A deep transfer learning pipeline on **MobileNetV2** engineered to classify face mask compliance into 4 classes (`with_mask`, `without_mask`, `mask_worn_incorrectly`, `not_related`), audit training convergence curves, and achieve **99.55% multi-class test accuracy** with edge inference latency.

---

## 📊 Key Highlights & Metrics
- **Test Accuracy / F1-Score:** `99.55%`
- **Out-of-Distribution (OOD) ROC AUC:** `1.000 / 0.999` (Zero false alarms on background clutter / random objects)
- **Dataset Scale:** 23,291 images merged across 4 multi-source repositories
- **Inference Latency:** `< 14 ms` (>70 FPS on edge GPU streams)
- **Model Size:** `3.8 MB` (TensorFlow Lite INT8 quantized)

---

## 🖥️ Interactive Presentation
Open [`face-mask-ai-system.html`](./face-mask-ai-system.html) in any modern web browser for an interactive slide deck showcasing:
- Problem formulation & 4-class taxonomy
- Parallel data ETL pipeline & 3-tier MD5 deduplication
- 3-split validation protocol
- Neural architecture trade-offs & fine-tuning dynamics
- Confusion matrices, ROC/PR curves & academic diagnostic suites
- Production deployment specifications

> **Navigation:** Use arrow keys (`←` / `→`) or **mouse scroll** (`wheel`) to seamlessly transition between slides.

---

## 📁 Repository Structure
```
├── face-mask-ai-system.html      # Interactive slide deck
├── assets/                       # Academic charts, training curves, confusion matrices
├── 01_split1_training.ipynb      # Split 1 Baseline model training
├── 02_split2_training.ipynb      # Split 2 Balanced & augmented training
├── 03_split3_training.ipynb      # Split 3 Carved validation training
├── last.ipynb                    # Evaluation & inference benchmarks
└── helper.py                     # Training & preprocessing utilities
```
