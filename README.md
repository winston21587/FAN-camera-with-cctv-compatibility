# 👁️ CCTV Multi-Face Detection, FAN Alignment & Restoration Pipeline

A surveillance-ready multi-face detection, tracking, and alignment pipeline built with **OpenCV** and the **Face Alignment Network (FAN)**. Designed for real-time monitoring on live CCTV/webcams and batch processing of short **video snippets (configurable duration, default 5s)** to extract the **top 5 best-aligned face crops** for downstream **degraded face restoration models** (e.g., CodeFormer, GFPGAN, RestoreFormer, GPEN).

---

## ✨ Features

- **Live CCTV & Stream Monitoring:**
  - Real-time multi-face detection with ID badges (`Face #1`, `Face #2`, etc.).
  - PiP (Picture-in-Picture) thumbnails of aligned faces on the display edge.
  - OSD HUD showing active faces, source, and real-time FPS with automatic CPU sleep throttling.
- **On-the-Fly Live Snippet Recording (<kbd>C</kbd>):**
  - Press <kbd>C</kbd> while viewing live footage to record a 5-second snippet without pausing or freezing the stream.
  - An asynchronous background worker immediately analyzes the snippet and extracts the top 5 aligned crops.
- **Configurable Recording Length:**
  - Adjust snippet length (`record_duration_sec`) in `config.json` (e.g., 3s, 5s, 10s). The HUD dynamically updates its countdown and hotkey labels.
- **Inter-Frame Face Tracking:**
  - Tracks multiple faces across all frames of a clip using IoU and centroid proximity so each person has a distinct identity track.
- **CPU-Guarded Multiprocessing:**
  - Isolates heavy FAN inference and scoring into separate worker processes so your CPU cores are never choked or frozen.
- **Best Aligned Selection:**
  - Evaluates pose frontality (Yaw/Pitch/Roll symmetry), landmark confidence, Laplacian focus/sharpness, and temporal diversity ($\ge 5$ frame gap) to pick the **top 5 highest-quality images** per face.
- **Toggleable FAN (ON / OFF):**
  - Can operate in 68-point affine alignment mode or fallback square aspect-ratio mode.
- **Restoration Model Ready:**
  - Standardized crops (512×512 FFHQ / ArcFace coordinate standard with eyes level at 38% height).
  - Exports a `metadata.json` for each face identity including frame timestamps, quality scores, and the 2×3 affine transformation matrix $M$ (allowing seamless inverse-warping of restored faces back into the CCTV frame).

---

## 📁 Project Structure

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
├── setup.bat                  # Automated installer (Aliyun / Tsinghua mirrors)
├── SETUP.md                   # Full setup & installation guide
├── README.md                  # Project overview & architecture documentation
├── .gitignore                 # Clean repository exclusions
│
├── snapshots/                 # [ORGANIZED] Full frame snapshots taken with [S]
├── aligned_faces/             # [ORGANIZED] Individual face crops from live snapshots
└── output_restoration/        # [ORGANIZED] Snippet videos, Top-5 crops & metadata.json
```

---

## 🚀 Quick Start

### 1. Installation
Run the automated Windows batch installer:
```cmd
setup.bat
```
*(Or see [SETUP.md](SETUP.md) for step-by-step pip commands with mirror sources).*

### 2. Option A: Live Surveillance Viewer & On-the-Fly Snippet Recording
Run the real-time interactive CCTV HUD:
```bash
python simple_face_detection.py
```
- **Press <kbd>C</kbd>** at any time to record a snippet. The HUD shows a red `● REC 5.0s` countdown. The live stream keeps playing while the top 5 aligned crops are automatically extracted in the background!
- **Press <kbd>S</kbd>** to save a full-resolution snapshot into `snapshots/`.

### 3. Option B: Record & Process Directly from CLI
Record directly from your camera/RTSP feed without opening the GUI viewer:
```bash
# Uses configured recording length from config.json (default 5s):
python simple_face_detection.py --record-snippet 5

# Or via the dedicated batch pipeline:
python clip_face_pipeline.py --record-snippet 5
```

### 4. Option C: Process an Existing Video Clip
Extract the 5 best-aligned crops for each person from an existing CCTV video file:
```bash
python clip_face_pipeline.py --clip path/to/cctv_snippet.mp4
```

---

## 🧠 Downstream Context: Face Restoration

Surveillance cameras produce severe facial degradation due to low resolution, digital zoom, compression artifacts, and motion blur.

```
+--------------------+       +-----------------------+       +------------------------+
| CCTV Video Snippet |  -->  | clip_face_pipeline.py |  -->  | Top 5 Aligned Crops    |
| (5s @ ~30 FPS)     |       | (Tracking + Workers)  |       | (512x512 FFHQ standard)|
+--------------------+       +-----------------------+       +-----------+------------+
                                                                         |
                                                                         v
                                                             +------------------------+
                                                             | Face Restoration Model |
                                                             | (CodeFormer / GFPGAN)  |
                                                             +------------------------+
```

### Why Canonical Alignment Matters
Models like **CodeFormer**, **GFPGAN**, and **RestoreFormer** use learned facial codebooks pre-trained on high-resolution FFHQ datasets. Feeding unaligned or severely angled faces produces hallucinated facial contours and eye displacement.

By selecting the **top 5 sharpest, most frontal frames** per identity:
1. Downstream restoration networks achieve optimal feature reconstruction.
2. Operators can select the best restored result or apply multi-frame super-resolution.
3. The stored `affine_matrix` in `metadata.json` can be used with `cv2.warpAffine(restored_face, cv2.invertAffineTransform(M))` to paste the restored face back into the CCTV frame!

---

## ⚙️ Configuration (`config.json`)

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **`source`** | `video_source` | `"0"` | Camera index (`0`), video path (`"cctv.mp4"`), or RTSP URL |
| **`recording`** | `record_duration_sec` | `5.0` | **Length in seconds** for snippet recordings (<kbd>C</kbd> and `--record-snippet`) |
| **`detection`** | `max_faces` | `5` | Maximum faces tracked simultaneously per frame |
| | `scale_factor` | `1.08` | Detection multi-scale step (lower = higher sensitivity) |
| | `enable_clahe` | `true` | Contrast equalization for low-light CCTV footage |
| **`clip_processing`**| `top_k_faces` | `5` | Number of best-aligned crops to extract per identity |
| | `min_frame_interval` | `5` | Minimum frame distance between chosen images (anti-redundancy) |
| | `output_dir` | `"output_restoration"` | Destination directory for snippet crops and metadata |
| **`multiprocessing`**| `enabled` | `true` | Isolates face alignment into separate worker processes |
| | `max_workers` | `2` | Number of parallel worker processes (prevents 100% CPU lockup) |
| **`snapshots`** | `output_dir` | `"snapshots"` | Directory for full CCTV frame snapshots (<kbd>S</kbd>) |
| **`fan_alignment`** | `enable_fan` | `true` | Toggle 68-point Face Alignment Network (FAN) |
| | `device` | `"cuda"` | Compute device: `"cuda"` or `"cpu"` |
| | `crop_size` | `[512, 512]` | Target dimensions for restoration models `[W, H]` |
| **`display`** | `target_fps` | `30` | Loop FPS cap — maintains smooth viewing at ~5-10% CPU |

---

## ⌨️ Live Viewer Hotkeys (`simple_face_detection.py`)

| Key | Action |
| :---: | :--- |
| <kbd>SPACE</kbd> | Pause or resume video playback |
| <kbd>C</kbd> | **Record Snippet** (configured length) & extract top-5 aligned faces in background |
| <kbd>F</kbd> | Toggle FAN Face Alignment ON / OFF dynamically |
| <kbd>L</kbd> | Toggle 68-point landmark dots rendering |
| <kbd>S</kbd> | Save full CCTV snapshot to `snapshots/` + export crops to `aligned_faces/` |
| <kbd>R</kbd> | Replay / rewind video from the beginning (for video files) |
| <kbd>Q</kbd> | Quit application |
