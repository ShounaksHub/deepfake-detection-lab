---
title: Multimedia Authenticity Lab
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: deepfake_detectorV3.py
pinned: false
---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/License-Proprietary-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">🔍 MULTIMEDIA AUTHENTICITY LAB</h1>

<p align="center">
  <em>AI-powered multimedia forensics platform for detecting manipulated and synthetic media</em><br>
  <strong>Built by TheSweetDuo</strong>
</p>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Model Registry](#-model-registry--11-models)
6. [Analysis Modes](#-analysis-modes)
7. [Digital Forensics Engine](#-digital-forensics-engine)
8. [Security Framework](#-security-framework)
9. [Tech Stack](#-tech-stack)
10. [Installation](#-installation)
11. [Usage](#-usage)
12. [Project Structure](#-project-structure)
13. [API & Configuration](#-api--configuration)
14. [Performance & Caching](#-performance--caching)
15. [Screenshots](#-screenshots)
16. [Roadmap](#-roadmap)
17. [License](#-license)

---

## 🎯 Overview

**MULTIMEDIA AUTHENTICITY LAB** is a comprehensive, production-grade web application that detects manipulated or AI-generated multimedia content. It combines **11 independent deep learning models**, **digital forensics analysis**, **metadata inspection**, and **content verification** techniques into a single unified platform.

The application analyzes **images**, **videos**, and **audio** files, providing detailed confidence scores, forensic evidence, and human-readable explanations to help users determine whether content is authentic or synthetically generated — **before or after sharing it online**.

---

## 🧩 Problem Statement

> *"Develop a solution capable of detecting manipulated or AI-generated multimedia by leveraging AI/ML models, digital forensics, metadata analysis, and content verification techniques. The platform should verify images and videos, provide a confidence score, and help users check whether content is real or AI-generated before or after sharing it online."*

### How We Address Each Requirement

| Requirement | Our Solution |
|:---|:---|
| **AI/ML Models** | 11-model weighted ensemble spanning ViT, EfficientNet, Xception, Swin Transformer, ConvNeXt V2 architectures |
| **Digital Forensics** | SHA-256 hashing, JPEG compression analysis, quantization-table inspection, EXIF metadata extraction |
| **Metadata Analysis** | Full EXIF tag parsing (camera, GPS, software, timestamps), color mode & dimension profiling |
| **Content Verification** | Google Lens reverse image search integration for web provenance checking |
| **Confidence Score** | Per-model percentage scores + weighted ensemble aggregate score with visual gauges |
| **Images & Videos** | Dedicated analysis tabs with frame-by-frame video timeline and per-frame verdicts |
| **Audio** | Dedicated audio deepfake detection using a fine-tuned audio classification model |
| **Before/After Sharing** | URL Scanner tab to paste links from Twitter, YouTube, or news sites and analyze media directly |

---

## ✨ Key Features

### 🎬 Netflix-Inspired Premium UI
- Cinematic splash screen with animated logo on launch
- Custom floating navigation bar with scroll-based transparency
- Dark-theme design system with glassmorphism cards and micro-animations
- Plotly-powered circular gauge charts for confidence visualization
- Glitch-text animation on "FAKE" verdicts for dramatic emphasis

### 🖼️ Multi-Modal Media Analysis
- **Image Analysis** — Upload or paste URL, supports JPG/PNG/WEBP
- **Video Analysis** — Frame extraction with temporal confidence timeline
- **Audio Analysis** — WAV/MP3 voice authenticity detection
- **URL Scanner** — Paste any media URL for direct analysis

### 🧠 11-Model Ensemble Engine
- Accuracy-weighted voting across architecturally diverse models
- Individual per-model verdicts shown alongside ensemble consensus
- Automatic face detection and cropping for face-focused models (via OpenCV)

### 🔬 Digital Forensics Panel
- SHA-256 file integrity hash
- JPEG compression quality estimation (quantization table analysis)
- Full EXIF metadata extraction and display
- Image dimension, color mode, and format profiling

### 🌐 Web Verification
- Google Lens reverse image search integration
- Finds visually similar instances across the web to corroborate authenticity

### 📊 History & Audit Trail
- SQLite-backed local database storing last 200 analyses
- Thumbnails, verdicts, confidence scores, model used, and timestamps
- CSV export for offline reporting
- One-click history clear

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MULTIMEDIA AUTHENTICITY LAB                  │
│                         (Streamlit Frontend)                        │
├─────────┬──────────┬──────────┬──────────────┬─────────────────────┤
│  Image  │  Video   │  Audio   │ URL Scanner  │     History         │
│   Tab   │   Tab    │   Tab    │     Tab      │      Tab            │
├─────────┴──────────┴──────────┴──────────────┴─────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SECURITY LAYER                            │   │
│  │  • MIME validation  • Magic-byte verification               │   │
│  │  • File size limits • Dimension caps • Rate limiting        │   │
│  │  • Filename sanitization • Safe error messages              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  INFERENCE ENGINE │  │ FORENSICS ENGINE │  │  HISTORY DB     │  │
│  │                   │  │                  │  │                 │  │
│  │  11 HF Models     │  │  SHA-256 Hash    │  │  SQLite         │  │
│  │  (cached)         │  │  EXIF Parsing    │  │  200-row cap    │  │
│  │                   │  │  JPEG Q-Tables   │  │  CSV Export     │  │
│  │  Weighted Voting  │  │  Compression Est │  │  Thumbnails     │  │
│  │  Face Detection   │  │  Google Lens     │  │                 │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  HUGGING FACE HUB                           │   │
│  │  Model weights downloaded & cached via @st.cache_resource   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Registry — 11 Models

The platform integrates **11 independent deepfake detection models** spanning 6 distinct neural network architectures, ensuring genuine diversity in the ensemble:

| # | Model | Architecture | Accuracy | Loader |
|:-:|:------|:------------|:---------|:-------|
| 1 | **Exp-02-21** (prithivMLmods) | ViT-base-patch16-224 | **98.84%** | HF Pipeline |
| 2 | **AdityaManojShinde** | EfficientNet-B4 + Xception (SRM) | **98.6%** | Custom PyTorch |
| 3 | **dima806** | EfficientNet | ~95% | HF Pipeline |
| 4 | **prithivMLmods** (Deep-Fake-Detector) | ViT (fine-tuned) | ~93% | HF Pipeline |
| 5 | **Wvolf** | ViT | ~92% | HF Pipeline |
| 6 | **fc63** | EfficientNetB0 (Transfer Learning) | 80% / AUC 0.88 | Keras (.keras) |
| 7 | **MaanVad3r** | Custom Pure CNN | 71% | Keras (.h5) |
| 8 | **dataflow** | EfficientNetB4 | N/A | Keras (.h5) |
| 9 | **Purnachander-Konda** | Swin Transformer | N/A | HF Pipeline |
| 10 | **computervisionpro** | ConvNeXt V2 | N/A | HF Pipeline |
| 11 | **umm-maybe** (AI Image Detector) | ViT-Large/16 (300MB+) | ~98% | HF Pipeline |

### Audio Model

| Model | Architecture | Task |
|:------|:------------|:-----|
| **MelodyMachine** (Deepfake-audio-detection-V2) | Audio Classification | Voice authenticity detection |

### Architecture Diversity

```
Architectures Used:
  ├── Vision Transformer (ViT)        — 4 models
  ├── EfficientNet (B0, B4)           — 3 models
  ├── Xception + SRM Filters          — 1 model (hybrid)
  ├── Swin Transformer                — 1 model
  ├── ConvNeXt V2                     — 1 model
  └── Custom CNN                      — 1 model
```

### Weighted Ensemble Logic

Models are weighted by their reported accuracy. The ensemble computes a **weighted fake score**:

```
weighted_fake_score = Σ (confidence_i × weight_i) / Σ weight_i

Final Verdict:
  • weighted_fake_score > 50% → FAKE
  • weighted_fake_score ≤ 50% → REAL
```

This ensures higher-accuracy models exert more influence on the final consensus while still incorporating signals from all available classifiers.

---

## 🔎 Analysis Modes

### 1. Single Model Mode
Upload an image and select any one model from the registry. View the individual model's verdict, confidence score, and a human-readable explanation.

### 2. Compare Mode
Run **two models side-by-side** on the same image. Useful for comparing architecturally different models (e.g., ViT vs. EfficientNet) to see if they agree.

### 3. 4-Panel Mode
Run the **top 4 models** simultaneously in a 2×2 grid layout. Each panel shows its own verdict and confidence gauge. A consensus summary is computed at the bottom.

### 4. Full Ensemble Mode
Run **all 11 models** sequentially with a real-time progress bar. Produces:
- Per-model verdict cards
- Weighted aggregate confidence score
- Final ensemble verdict with explanation

### 5. Video Analysis
Upload a video file (MP4/AVI/MOV). The system:
1. Extracts **6 evenly-spaced frames** using OpenCV
2. Analyzes each frame independently
3. Displays a **temporal confidence timeline** showing how fakeness varies across the video

### 6. Audio Analysis
Upload a WAV or MP3 file. The dedicated audio classification model determines whether the voice is authentic or synthetically generated.

### 7. URL Scanner
Paste a URL from Twitter, YouTube, or any news site. The system extracts the media content and runs the full ensemble analysis on it.

---

## 🔬 Digital Forensics Engine

The forensics panel is available as an expandable section under every image analysis. It provides:

| Feature | Details |
|:--------|:--------|
| **SHA-256 Hash** | Cryptographic file integrity fingerprint |
| **JPEG Compression Estimate** | Analyzes quantization tables to estimate compression level (High/Medium/Low) |
| **File Metadata** | Dimensions, color mode, file size, format |
| **EXIF Data** | Full metadata extraction — camera model, software, GPS coordinates, timestamps, orientation |
| **AI Generation Flags** | Checks EXIF software field for known AI tool signatures |

---

## 🔒 Security Framework

The application implements a defense-in-depth security strategy:

| Layer | Implementation |
|:------|:---------------|
| **MIME Type Validation** | Whitelist of allowed MIME types per media category |
| **Magic Byte Verification** | Binary header inspection to prevent extension spoofing |
| **File Size Limits** | 10 MB images, 100 MB videos, 50 MB audio (configurable) |
| **Dimension Caps** | Max 8000×8000 px to prevent memory exhaustion |
| **Rate Limiting** | Sliding window — max 10 analyses per 60-second window per session |
| **Filename Sanitization** | Path-traversal prevention, control-character stripping, length limiting |
| **Safe Error Messages** | Full stack traces logged to console only; generic messages shown to user |
| **PIL Bomb Protection** | Full image decode forced (`pil_img.load()`) to catch truncated/corrupt files |
| **Temp File Cleanup** | Video processing temp files always deleted in `finally` blocks |

---

## 🛠️ Tech Stack

### Core

| Technology | Purpose |
|:-----------|:--------|
| **Python 3.9+** | Runtime |
| **Streamlit** | Web application framework |
| **PyTorch** | Deep learning inference engine |
| **Transformers** (Hugging Face) | Model loading and pipeline inference |
| **Hugging Face Hub** | Model weight downloading |

### Analysis

| Technology | Purpose |
|:-----------|:--------|
| **Pillow (PIL)** | Image processing, EXIF extraction |
| **OpenCV** | Video frame extraction, face detection (Haar cascades) |
| **librosa** | Audio waveform processing (optional) |
| **NumPy** | Numerical operations |

### Visualization & Storage

| Technology | Purpose |
|:-----------|:--------|
| **Plotly** | Interactive circular gauge charts |
| **SQLite** | Local history database |
| **CSV module** | History export |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ShounaksHub/deepfake-detection-lab.git
cd deepfake-detection-lab
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install streamlit torch torchvision transformers huggingface_hub plotly pillow numpy opencv-python
```

> **Note:** For GPU acceleration, install the appropriate CUDA-compatible PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 4 — (Optional) Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token if using gated models:

```env
HF_TOKEN=hf_your_token_here
```

### Step 5 — Launch the Application

```bash
streamlit run deepfake_detectorV3.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🚀 Usage

### Image Analysis

1. Navigate to the **Image** tab
2. Upload an image (JPG/PNG/WEBP, max 10 MB) or paste a URL
3. Choose an analysis mode:
   - **Single** — Pick one model
   - **Compare** — Two models side-by-side
   - **4-Panel** — Top 4 models in a grid
   - **Ensemble** — All 11 models with weighted voting
4. Click **Analyze** to run detection
5. Expand **Digital Forensics** for hash, EXIF, and compression data
6. Expand **Web Verification** to run a reverse image search

### Video Analysis

1. Navigate to the **Video** tab
2. Upload a video (MP4/AVI/MOV, max 100 MB)
3. The system extracts 6 frames and analyzes each one
4. View the temporal confidence timeline

### Audio Analysis

1. Navigate to the **Audio** tab
2. Upload an audio file (WAV/MP3, max 50 MB)
3. View the authenticity verdict and confidence score

### URL Scanner

1. Navigate to the **URL Scanner** tab
2. Paste a media URL (e.g., from Twitter, YouTube, or a news article)
3. Click **Scan URL** — the system fetches and analyzes the media

### History

1. Navigate to the **History** tab
2. Browse past analyses with thumbnails and verdicts
3. **Export CSV** for offline reporting
4. **Clear History** to reset the database

---

## 📁 Project Structure

```
deepfake-detection-lab/
│
├── deepfake_detectorV3.py      # Main application (single-file architecture)
│                                #   ├── Security Utilities (lines 148-419)
│                                #   ├── Page Config & Splash Screen (lines 421-498)
│                                #   ├── Navigation Bar (lines 500-830)
│                                #   ├── CSS Design System (lines 830-1470)
│                                #   ├── Label Mapping Functions (lines 1471-1515)
│                                #   ├── Model Registry (lines 1718-1883)
│                                #   ├── Hybrid CNN Architecture (lines 1885-1960)
│                                #   ├── Model Loaders (lines 1960-2250)
│                                #   ├── Inference & Forensics (lines 2250-2640)
│                                #   ├── Audio Pipeline (lines 2640-2670)
│                                #   ├── UI Tabs (lines 2670-4833)
│                                #   └── History Tab & Footer (lines 4670-4833)
│
├── README.md                   # This documentation file
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
└── .git/                       # Git repository data
```

### Runtime Data (auto-created)

```
~/.deepfake_lab/
└── history.db                  # SQLite database for analysis history
```

---

## ⚙️ API & Configuration

### Environment Variables

| Variable | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `HF_TOKEN` | Optional | — | Hugging Face access token for gated models |
| `MAX_IMAGE_SIZE_MB` | Optional | `10` | Maximum image upload size in MB |
| `MAX_VIDEO_SIZE_MB` | Optional | `100` | Maximum video upload size in MB |
| `MAX_AUDIO_SIZE_MB` | Optional | `50` | Maximum audio upload size in MB |

### Configurable Constants (in source)

| Constant | Default | Description |
|:---------|:--------|:------------|
| `_MAX_IMAGE_DIM` | `8000` | Maximum pixels per side |
| `_RATE_LIMIT_MAX` | `10` | Max analyses per rate-limit window |
| `_RATE_LIMIT_WINDOW` | `60` | Rate-limit window in seconds |
| `_MAX_HISTORY` | `200` | Maximum history records retained |

---

## ⚡ Performance & Caching

- **Model Caching**: All model pipelines use `@st.cache_resource` — models are downloaded and loaded **once**, then reused across reruns and sessions.
- **Lazy Loading**: Models are only loaded when first requested, not at startup.
- **Face Detection**: OpenCV Haar cascade is used for optional face cropping to improve accuracy on face-focused models.
- **History Pruning**: SQLite database is automatically pruned to the most recent 200 records.

---

## 🗺️ Roadmap

- [ ] Batch analysis mode (multiple files at once)
- [ ] PDF report generation with forensic evidence
- [ ] Browser extension for one-click analysis
- [ ] Real-time webcam/video stream analysis
- [ ] Model fine-tuning interface
- [ ] API endpoint for programmatic access

---

## 📄 License

Copyright &copy; 2026 **TheSweetDuo**. All rights reserved.

---

<p align="center">
  <strong>MULTIMEDIA AUTHENTICITY LAB</strong><br>
  <em>Detecting synthetic media, one frame at a time.</em>
</p>
