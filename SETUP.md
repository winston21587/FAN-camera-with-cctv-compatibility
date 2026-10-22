# 👁️ CCTV Multi-Face Detection & FAN Alignment — Setup Guide

A real-time, surveillance-ready multi-face detection, tracking, and alignment pipeline built with **OpenCV** and the **Face Alignment Network (FAN)**. Handles CCTV footage, RTSP/HTTP network streams, recorded video files, and local webcams with GPU acceleration.

---

## 📋 Requirements

| Requirement | Minimum |
| :--- | :--- |
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.10 or newer |
| **GPU** | NVIDIA GPU with CUDA support (recommended for FAN alignment) |
| **CUDA** | 12.1 (for the default PyTorch wheel) or 11.8 |
| **RAM** | 4 GB minimum, 8 GB recommended |

> [!NOTE]
> FAN alignment runs on CPU if no NVIDIA GPU is available, but will be slower. Use `--no-fan` for lightweight CPU-only mode.

---

## 🚀 Setup Instructions

### Option A — Automated Setup (Recommended for Windows)

> **This is the fastest method.** The script handles everything automatically and uses the **Aliyun mirror** (fastest CDN for Asia-Pacific) with the **official PyTorch CUDA index**.

**1. Clone or download this repository:**
```bash
git clone https://github.com/your-username/simple-face-detection.git
cd simple-face-detection
```

**2. Run the setup script (double-click or from terminal):**
```bash
setup.bat
```

The script will automatically:
- Create organized output folders (`snapshots/`, `aligned_faces/`, `output_restoration/`)
- Upgrade `pip` via the Aliyun mirror
- Install **PyTorch + torchvision** with CUDA 12.1 support from the official PyTorch index
- Install **opencv-python**, **numpy**, and **face-alignment** via the Aliyun mirror
- Run a verification check to confirm everything is working

---

### Option B — Manual Setup (Step-by-Step with Fast Mirrors)

**1. Upgrade pip using the Aliyun mirror:**
```bash
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**2. Install PyTorch with CUDA support** (from the official PyTorch index):
```bash
# For CUDA 12.1 (RTX 30xx / 40xx series — most common):
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs):
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu118

# CPU-only (no GPU):
python -m pip install torch torchvision -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**3. Install core packages via Aliyun mirror:**
```bash
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**4. Verify the installation:**
```bash
python -c "import cv2, numpy, face_alignment, torch; print('cv2:', cv2.__version__); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output:
```
cv2: 4.10.x
torch: 2.x.x+cu121
CUDA: True
```

---

### Alternative Mirrors

If Aliyun is slow in your region, switch to Tsinghua:

| Mirror | URL | Best For |
| :--- | :--- | :--- |
| **Aliyun** *(default)* | `https://mirrors.aliyun.com/pypi/simple/` | Asia-Pacific / Philippines |
| **Tsinghua** | `https://pypi.tuna.tsinghua.edu.cn/simple` | China mainland / Backup |
| **PyPI** *(official)* | `https://pypi.org/simple/` | Global |

To use Tsinghua:
```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

---

## ▶️ Running the Application

### 1. Live Surveillance Viewer
```bash
python simple_face_detection.py
```
- Press <kbd>C</kbd> at any time to record a 5-second snippet on the fly. The live view never stops, and the top 5 aligned crops are automatically extracted in the background!
- Press <kbd>S</kbd> to save a high-resolution snapshot to `snapshots/`.

### 2. Direct Snippet Recording & Auto-Processing
Record directly from your camera and extract the 5 best-aligned faces without manual video editing:
```bash
python simple_face_detection.py --record-snippet 5
```
*Or using the dedicated batch pipeline:*
```bash
python clip_face_pipeline.py --record-snippet 5
```

### 3. Process an Existing CCTV Video Clip
```bash
python clip_face_pipeline.py --clip path/to/cctv_5s.mp4
```

---

## ⚙️ Configuration (`config.json`)

All runtime options can be adjusted in [`config.json`](config.json):

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **`source`** | `video_source` | `"0"` | `0` = webcam, `"video.mp4"` = file, RTSP/HTTP URL |
| **`recording`** | `record_duration_sec` | `5.0` | **Length in seconds** to record when pressing <kbd>C</kbd> or `--record-snippet` |
| **`detection`** | `max_faces` | `5` | Maximum faces tracked simultaneously per frame |
| | `scale_factor` | `1.08` | Detector granularity (lower = finer, more CPU) |
| | `min_neighbors` | `5` | Threshold (higher = fewer false positives) |
| | `min_face_size` | `[20, 20]` | Smallest detectable face (px) for distant CCTV views |
| | `enable_clahe` | `true` | Contrast boost for low-light / night CCTV footage |
| **`clip_processing`**| `top_k_faces` | `5` | Number of best-aligned crops to extract per identity |
| | `min_frame_interval` | `5` | Minimum frame gap between chosen images (anti-redundancy) |
| | `output_dir` | `"output_restoration"` | Destination for extracted top-5 crops & metadata |
| **`multiprocessing`**| `enabled` | `true` | Runs face alignment in isolated processes |
| | `max_workers` | `2` | Number of worker processes (protects CPU from 100% lockup) |
| **`snapshots`** | `output_dir` | `"snapshots"` | Directory for full CCTV frame snapshots (<kbd>S</kbd>) |
| **`fan_alignment`** | `enable_fan` | `true` | Enable 68-point Face Alignment Network |
| | `device` | `"cuda"` | Compute device: `"cuda"` or `"cpu"` |
| | `crop_size` | `[512, 512]` | Output crop size (512×512 standard for restoration models) |
| | `crop_output_dir` | `"aligned_faces"` | Output folder for snapshot face crops |
| **`display`** | `target_fps` | `30` | Loop FPS throttle — keeps CPU usage between ~5-10% |

---

## ⌨️ Hotkeys (Live Viewer)

| Key | Action |
| :---: | :--- |
| <kbd>SPACE</kbd> | Pause / resume playback |
| <kbd>C</kbd> | **Record 5s Snippet** & automatically extract top-5 aligned faces in background |
| <kbd>F</kbd> | Toggle FAN alignment ON / OFF dynamically |
| <kbd>L</kbd> | Toggle 68-point landmark dots rendering |
| <kbd>S</kbd> | Save full frame to `snapshots/` + export crops to `aligned_faces/` |
| <kbd>R</kbd> | Replay / rewind video from beginning (video files only) |
| <kbd>Q</kbd> | Quit |

---

## 📁 Directory Structure

```text
simple-face-detection/
├── clip_face_pipeline.py      # Batch snippet alignment & Top-5 selection tool
├── simple_face_detection.py   # Live CCTV viewer with background snippet recorder
├── core/                      # Modular pipeline engine
│   ├── tracker.py             # Inter-frame face tracker (IoU + centroid proximity)
│   ├── alignment.py           # 68-point ArcFace/FFHQ affine warper & square fallback
│   ├── quality_scorer.py      # Frontality, Laplacian sharpness & top-5 selector
│   └── worker_pool.py         # Multi-process pool for CPU protection
├── config.json                # Runtime configuration (recording length, workers, source)
├── requirements.txt           # Core packages (opencv-python, numpy, face-alignment)
├── requirements-torch.txt     # PyTorch + CUDA wheels
├── setup.bat                  # Automated Windows installer (Aliyun / Tsinghua mirrors)
├── SETUP.md                   # Full setup & installation guide
├── README.md                  # Project overview & architecture documentation
├── .gitignore                 # Excludes caches, videos, snapshots, and output dirs
│
├── snapshots/                 # [ORGANIZED] Full frame snapshots taken with [S]
├── aligned_faces/             # [ORGANIZED] Individual face crops from live snapshots
└── output_restoration/        # [ORGANIZED] 5s snippet videos, Top-5 crops & metadata.json
```

---

## 🔧 Troubleshooting

**`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`**
> You have a conflicting OpenCV headless build. Fix it:
> ```bash
> pip uninstall opencv-python-headless -y
> pip install opencv-python==4.10.0.84 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
> ```

**High CPU usage (>20%)**
> Check `config.json`: Ensure `target_fps` is set to `30` or `15`, and `multiprocessing.max_workers` is set to `2`.

**Camera not opening**
> Change `video_source` in `config.json` to `1` or `2` if you have multiple webcams.
