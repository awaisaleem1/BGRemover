# 🎯 VisionClean

**AI-powered background & text removal API** — remove backgrounds and text/watermarks from images with a single API call or web interface.

Built with **BiRefNet** (state-of-the-art background removal) and **CRAFT** (text detection), running entirely on **CPU** — no GPU required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Technology | Status |
|---|---|---|
| 🖼 Background Removal | BiRefNet via rembg | ✅ Production Ready |
| 📝 Text/Watermark Removal | CRAFT + OpenCV Inpainting | ✅ Production Ready |
| 🛞 Wheel/Tire Preservation | HoughCircles Detection | ✅ Auto-detected |
| 🧹 Edge Cleanup | Halo removal, ground/sky cleanup | ✅ Automatic |
| ✂ Auto-Crop | Tight crop to subject | ✅ Automatic |
| 🌐 Web UI | HTML upload page | ✅ Built-in |
| 📡 REST API | FastAPI with Swagger docs | ✅ Built-in |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/VisionClean.git
cd VisionClean
python -m venv venv
```

**Activate virtual environment:**

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

Run these commands **one by one** (important for Windows):

```bash
python -m pip install --upgrade pip
```

```bash
pip install numpy Pillow fastapi uvicorn python-multipart opencv-python-headless
```

```bash
pip install rembg onnxruntime gdown scipy
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install craft-text-detector --no-deps
```

> ⚠️ **Important:** Install `craft-text-detector` with `--no-deps` to avoid an old OpenCV conflict.

### 3. Run

```bash
python visionclean_fixed_upload.py
```

First run will download the BiRefNet model (~400MB) automatically. After that, it starts instantly.

### 4. Use

- **Web UI:** http://127.0.0.1:8000/upload
- **API Docs:** http://127.0.0.1:8000/docs
- **Status:** http://127.0.0.1:8000/status

---

## 📡 API Usage

### Process an Image

```bash
curl -X POST "http://127.0.0.1:8000/process" \
  -F "file=@your_image.jpg" \
  --output result.png
```

### With Options

```bash
# Background removal only (no text removal)
curl -X POST "http://127.0.0.1:8000/process?remove_text=false" \
  -F "file=@car.jpg" --output result.png

# Text removal only (keep background)
curl -X POST "http://127.0.0.1:8000/process?remove_bg=false" \
  -F "file=@document.jpg" --output result.png

# No auto-crop
curl -X POST "http://127.0.0.1:8000/process?auto_crop=false" \
  -F "file=@photo.jpg" --output result.png
```

### Python Client

```python
import requests

url = "http://127.0.0.1:8000/process"

with open("car.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        params={"remove_bg": True, "remove_text": True, "auto_crop": True}
    )

data = response.json()
print(data)

# Download result
if data["success"]:
    result = requests.get(data["download_url"])
    with open("result.png", "wb") as f:
        f.write(result.content)
    print("Saved: result.png")
```

### API Response

```json
{
  "success": true,
  "filename": "visionclean_car_1709312400.png",
  "download_url": "http://127.0.0.1:8000/output/visionclean_car_1709312400.png",
  "time": "6.2s",
  "model": "birefnet-general-lite",
  "text_regions_removed": 3,
  "steps": ["text_removal", "background_removal"],
  "format": "PNG (RGBA transparent)"
}
```

---

## 🏗 Project Structure

```
VisionClean/
├── visionclean_fixed_upload.py    ← Main app (run this)
├── requirements.txt               ← Dependencies
├── README.md                      ← This file
│
├── app/
│   ├── config.py                  ← Settings & paths
│   └── utils/
│       ├── image_io.py            ← Image load/save
│       ├── mask_utils.py          ← Mask utilities
│       └── resize.py              ← Resize/padding
│
├── models/
│   └── craft/
│       └── craft_mlt_25k.pth     ← CRAFT text detection model
│
├── output/                        ← Processed images
└── venv/                          ← Python environment
```

---

## ⚙️ How It Works

```
Input Image
     │
     ▼
┌─────────────────────────────────────┐
│  1. TEXT REMOVAL (Pass 1)           │
│     CRAFT detects text regions      │
│     OpenCV inpaints/fills them      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. SMART PREPROCESSING            │
│     Detect small subjects           │
│     Crop close for better results   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. BACKGROUND REMOVAL             │
│     BiRefNet AI model (via rembg)   │
│     Outputs RGBA with transparency  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. POST-PROCESSING                │
│     🛞 Detect wheels (protect them) │
│     🧹 Remove edge halo            │
│     🧹 Clean mask noise            │
│     🧹 Remove ground shadows       │
│     🧹 Remove sky remnants         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. TEXT REMOVAL (Pass 2)           │
│     Catch remaining text on result  │
│     Only in opaque areas            │
│     Alpha channel preserved         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. AUTO-CROP & OUTPUT              │
│     Tight crop with 3% padding      │
│     Save as transparent PNG         │
└─────────────────────────────────────┘
```

---

## 🤖 Models

| Model | Purpose | Size | Auto-Download |
|---|---|---|---|
| **BiRefNet-general-lite** | Background removal | ~400MB | ✅ Yes (via rembg, cached at `~/.u2net/`) |
| **CRAFT** | Text detection | ~90MB | Manual (place in `models/craft/`) |

### Model Quality Comparison

| Model | Quality | CPU Speed | Recommendation |
|---|---|---|---|
| `birefnet-general-lite` | ★★★★☆ | ~4-6s | **✅ Best for CPU** |
| `birefnet-general` | ★★★★★ | ~8-12s | Better quality, slower |
| `birefnet-massive` | ★★★★★ | ~20-30s | Needs GPU |
| `isnet-general-use` | ★★★★☆ | ~3-5s | Good fallback |

---

## 🔧 Configuration

Edit settings in `app/config.py`:

```python
# Change background removal model
BG_REMOVAL_MODEL = "birefnet-general-lite"  # or "birefnet-general"

# Output settings
OUTPUT_FORMAT = "png"
OUTPUT_BACKGROUND = "transparent"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
```

---

## 🐛 Troubleshooting

### `craft-text-detector` install fails

```bash
# Install with --no-deps to skip old opencv dependency
pip install craft-text-detector --no-deps
pip install gdown scipy
```

### First run is slow

BiRefNet model (~400MB) downloads on first run. After that it's cached at `~/.u2net/` and starts instantly.

### `torch` install is huge

Use the CPU-only version (much smaller):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Tires/wheels getting cut off

The pipeline auto-detects wheels using HoughCircles and protects them. If detection fails, it protects the bottom 30% of the subject as a fallback.

### Text not being detected

If CRAFT is not installed, a fallback text detector (adaptive threshold + contour filtering) is used. It works for most cases but CRAFT is more accurate.

---

## 📋 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/status` | API status & loaded models |
| `POST` | `/process` | Process an image |
| `GET` | `/output/{filename}` | Download processed image |
| `GET` | `/upload` | Web UI |
| `GET` | `/docs` | Swagger API documentation |

### POST /process Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image file (JPG, PNG, WebP) |
| `remove_bg` | bool | `true` | Remove background |
| `remove_text` | bool | `true` | Remove text/watermarks |
| `auto_crop` | bool | `true` | Crop tightly to subject |

---

## 🚀 Deployment

### Local Development

```bash
python visionclean_fixed_upload.py
```

### Production (with multiple workers)

```bash
uvicorn visionclean_fixed_upload:app --host 0.0.0.0 --port 8000 --workers 2
```

### Docker (optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install craft-text-detector --no-deps && pip install gdown scipy

COPY . .

EXPOSE 8000
CMD ["uvicorn", "visionclean_fixed_upload:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📄 License

MIT License — free for personal and commercial use.

---

## 🙏 Credits

- [rembg](https://github.com/danielgatis/rembg) — Background removal wrapper
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — State-of-the-art segmentation model
- [CRAFT](https://github.com/clovaai/CRAFT-pytorch) — Text detection model
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [OpenCV](https://opencv.org/) — Image processing & inpainting