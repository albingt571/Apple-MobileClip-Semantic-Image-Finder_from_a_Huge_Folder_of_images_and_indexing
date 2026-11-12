# Apple-MobileCLIP-Semantic-Image-Finder 🔍🖼️

> **Fast, 100% offline natural-language semantic image search for huge image folders on Windows, powered by Apple MobileCLIP2-S2, FAISS vector indexing, and SQLite.**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Device](https://img.shields.io/badge/Hardware-GPU%20(CUDA)%20%2F%20CPU-orange.svg)](#-hardware--performance)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-brightgreen.svg)](#-privacy--offline-guarantee)

---

## 🌟 Overview

Have thousands of photos across large, nested directories, but can't find that specific picture of a *"red sports car in rain"* or *"cat sleeping on keyboard"*?

**MobileCLIP-Semantic-Image-Finder** lets you point to any large image folder on your PC and search for visual content using everyday natural language queries. 

Unlike traditional file name search or cloud-dependent photo tools:
- **Zero Cloud / 100% Local**: No images are ever uploaded; no API keys, cloud subscriptions, or external network requests during indexing or search.
- **Cutting-Edge Efficiency**: Uses Apple's official **MobileCLIP2-S2** vision-language model—delivering high zero-shot retrieval accuracy with ultra-low latency and tiny memory footprint.
- **Large-Folder Scale**: Uses a local **FAISS** vector index (`IndexFlatIP`) paired with an **SQLite** database for lightning-fast similarity search across tens of thousands of images.
- **Real-Time Camera & File Watchdog**: Background observer continuously monitors the folder; incoming photos saved from cameras, capture cards, or downloads are auto-detected, embedded, and added to the index within seconds.
- **Incremental & Resilient**: Detects modified/added images without re-embedding old ones, purges deleted entries, and safely ignores corrupt or unreadable image formats.
- **Responsive Desktop GUI**: Native Windows Tkinter UI with multithreaded indexing, dynamic progress reporting, and an interactive thumbnail result gallery with one-click opening.

---

## 📸 Interface Preview

- **Folder Selection**: Pick any root folder; scans subdirectories recursively.
- **Natural Language Search Bar**: Type concepts, colors, objects, or scenery (e.g., `"dog jumping into water"`).
- **Interactive Result Gallery**: Displays ranked images with semantic similarity scores and click-to-open integration with Windows Explorer.

---

## 🧠 Architecture & Model Details

```
+-------------------------------------------------------------+
|                      Tkinter Desktop UI                     |
|  - Modern Windows native interface                          |
|  - Real-time progress bar & thread-safe status updates       |
|  - Responsive thumbnail gallery with similarity scores      |
|  - Single-click image opening in Windows Photo Viewer       |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     SearchEngine Core                       |
|  - Model: MobileCLIP2-S2 (dfndr2b) via open-clip-torch      |
|  - Local Vector Index: FAISS IndexFlatIP (Cosine Similarity)|
|  - Metadata Store: SQLite (path, folder, file size, mtime)  |
|  - Incremental sync (indexes only new/changed images)       |
|  - Corrupt file resilience & dead-link auto cleanup         |
|  - 100% On-Device (GPU CUDA-enabled or CPU fallback)        |
+-------------------------------------------------------------+
```

### Model & Citation
- **Model Architecture**: `MobileCLIP2-S2` (Small variant 2, trained on DFNDR-2B).
- **Research Citation**: *"MobileCLIP2: Improving Multi-Modal Reinforced Training"* (Apple AIML Research, TMLR 2025).
- **Official Repository**: [apple-aiml-research/ml-mobileclip](https://github.com/apple-aiml-research/ml-mobileclip)
- **Hugging Face Checkpoint**: [`timm/MobileCLIP2-S2-OpenCLIP`](https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP) / [`apple/MobileCLIP2-S2`](https://huggingface.co/apple/MobileCLIP2-S2)
- **License**: Checkpoint released under Apple Machine Learning Research License Agreement (`apple-amlr`). Code is licensed under MIT.

---

## ⚡ Hardware & Performance

| Mode | Minimum Spec | Recommended |
|---|---|---|
| **GPU (CUDA)** | NVIDIA GPU with 4GB VRAM (e.g., GTX 1650) | NVIDIA GPU with 6GB+ VRAM |
| **CPU** | Any modern quad-core x86-64 processor | 8+ thread processor |

*Note: The app automatically detects CUDA availability upon launch and defaults to GPU acceleration, falling back to CPU if no compatible GPU is present.*

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/apple-mobileclip-semantic-image-finder.git
cd apple-mobileclip-semantic-image-finder
```

### 2. Install Dependencies

#### Option A: GPU Accelerated (Recommended if you have NVIDIA GPU)
```powershell
pip install open-clip-torch faiss-cpu pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Option B: CPU Only
```powershell
pip install open-clip-torch faiss-cpu pillow torch torchvision
```

---

## 💻 Running the Application

### Launch the GUI
```powershell
python app.py
```

### Usage Steps:
1. Click **Browse...** to pick any folder containing images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`).
2. Click **Start Indexing**. 
   - A background thread will scan and embed the photos without freezing the UI.
   - The first run downloads the lightweight model checkpoint (~380 MB) to your local cache once. Subsequent runs are completely offline.
3. Type any visual description in the **Search** box (e.g., *"black car"*, *"sunset over ocean"*, *"person wearing glasses"*) and hit **Enter**.
4. Click any result thumbnail or the **Open File** button to launch the photo in your default Windows viewer.

---

## 🧪 Testing the Engine Offline

A standalone pipeline test is included to verify end-to-end embedding, FAISS indexing, metadata tracking, and ranking on sample synthetic images:

```powershell
python test_pipeline.py
```

Expected output:
```text
--- Initializing Search Engine ---
Device: cuda (or cpu)

--- Running Indexing on Sample Images ---
[4/4] Processed 4 of 4 images...
Indexing Stats: {'total_files_found': 4, 'new_or_updated_indexed': 3, 'corrupt_or_skipped': 1, 'total_in_index': 3}

--- Running Natural Language Searches ---
Query: 'black car'
  Rank 1: black_car.jpg | Score: 0.2819
Query: 'red apple'
  Rank 1: red_apple.png | Score: 0.3212
Query: 'green forest tree'
  Rank 1: green_tree.png | Score: 0.2753
```

---

## 📁 Repository Structure

```
.
├── app.py              # Desktop GUI interface (Tkinter + Canvas)
├── search_engine.py    # Core engine: MobileCLIP2 embedding, FAISS indexing, SQLite sync
├── test_pipeline.py    # Automated test suite with synthetic image generation
├── sample_images/      # Test images created during verification
├── sensie_data/        # Local database (metadata.db) and FAISS index (faiss.index)
├── requirements.txt    # Python package dependencies
├── .gitignore          # Ignores local databases, virtual environments, and caches
└── README.md           # Documentation
```

---

## 🔒 Privacy & Offline Guarantee

- **Zero Remote Calls**: Search queries and images never leave your local machine.
- **Model Caching**: After the one-time initial weight download via Hugging Face/OpenCLIP, the app can run on an air-gapped machine with no internet connection.

---

## 📄 License

- Software Code: [MIT License](LICENSE)
- Model Weights: [Apple Machine Learning Research License Agreement](https://github.com/apple-aiml-research/ml-mobileclip)
