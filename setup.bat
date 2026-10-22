@echo off
REM ============================================================
REM  setup.bat — Automated installer for CCTV Face Detection
REM  Mirrors: Aliyun (PyPI) + PyTorch official CUDA index
REM  Run once: double-click OR execute from terminal
REM ============================================================

title CCTV Face Detection — Setup

echo.
echo ========================================================
echo   CCTV Multi-Face Detection ^& FAN Alignment — SETUP
echo ========================================================
echo.

REM ── Step 0: Check Python availability ────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set PY_VER=%%V
echo [OK] Python %PY_VER% detected.
echo.

REM ── Create directory structure ───────────────────────────
if not exist "snapshots" mkdir snapshots
if not exist "aligned_faces" mkdir aligned_faces
if not exist "output_restoration" mkdir output_restoration

REM ── Step 1: Upgrade pip (Aliyun mirror) ──────────────────
echo [1/4] Upgrading pip via Aliyun mirror...
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
echo.

REM ── Step 2: Install PyTorch + torchvision (CUDA 12.1) ────
REM    Uses the official PyTorch index — only source for CUDA wheels.
REM    Change cu121 to cu118 if your CUDA version is 11.8.
echo [2/4] Installing PyTorch 2.x with CUDA 12.1 support...
echo       (This is a large download ~2.5 GB — please be patient)
echo.
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo [WARN] PyTorch CUDA install failed. Trying CPU-only fallback...
    python -m pip install torch torchvision -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
)
echo.

REM ── Step 3: Install core packages (Aliyun mirror) ────────
echo [3/4] Installing core packages via Aliyun mirror...
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
if errorlevel 1 (
    echo [WARN] Aliyun mirror failed. Trying Tsinghua mirror...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
)
echo.

REM ── Step 4: Verify installation ──────────────────────────
echo [4/4] Verifying installation...
python -c "import cv2, numpy, face_alignment, torch; print('[OK] cv2:', cv2.__version__); print('[OK] numpy:', numpy.__version__); print('[OK] torch:', torch.__version__); print('[OK] CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo.
    echo [ERROR] Verification failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Setup complete! Launch with:
echo.
echo   1. Live CCTV Viewer (Press [C] to record snippet):
echo      python simple_face_detection.py
echo.
echo   2. Record 5s Snippet ^& Extract Top-5 Aligned Faces:
echo      python clip_face_pipeline.py --record-snippet 5
echo.
echo   3. Process Existing Video Clip:
echo      python clip_face_pipeline.py --clip path\to\clip.mp4
echo ========================================================
echo.
pause
