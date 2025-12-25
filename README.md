# PatchCore Flask Service

**Package Damage Detection using PatchCore Anomaly Detection**

> ✅ EXACT copy of notebook logic - ZERO modifications  
> ✅ Production-ready Flask service  
> ✅ Simple setup & deployment  

---

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Environment

**Using PowerShell:**
```powershell
.\setup.ps1
```

**Using Command Prompt:**
```cmd
setup.bat
```

**Or manually:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run Service

```powershell
python app.py
```

Output:
```
======================================================================
PatchCore Flask Service - Ready for Hackathon Demo
======================================================================
Endpoints:
  GET  /              - Service info
  GET  /health        - Health check
  POST /load_reference - Upload normal image
  POST /infer         - Get overlay PNG
  POST /infer_json    - Get JSON result
======================================================================

Starting Flask server on http://127.0.0.1:5000
======================================================================
```

### Step 3: Test It

```powershell
python test_service.py
```

---

## 📁 Project Structure

```
patchcore_service/
│
├── app.py                  ← Flask service (EXACT notebook logic)
├── requirements.txt        ← Dependencies (notebook-compatible)
├── setup.ps1              ← Setup script (PowerShell)
├── setup.bat              ← Setup script (CMD)
├── test_service.py        ← Automated test suite
├── README.md              ← This file
│
├── models/                ← Memory bank storage (auto-created)
├── static/
│   └── outputs/           ← Saved result images
│
└── venv/                  ← Virtual environment (after setup)
```

---

## 🔌 API Usage

### 1️⃣ Health Check

```bash
curl http://127.0.0.1:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "PatchCore with WideResNet-50-2",
  "memory_bank_loaded": false,
  "device": "cpu"
}
```

### 2️⃣ Load Normal Reference

Upload a pristine package image (one-time setup):

```bash
curl -X POST http://127.0.0.1:5000/load_reference \
  -F "image=@normal_package.jpg"
```

**Response:**
```json
{
  "status": "success",
  "message": "Memory bank built successfully",
  "patches": 784,
  "feature_dim": 1536
}
```

### 3️⃣ Inference - Get Overlay PNG

```bash
curl -X POST http://127.0.0.1:5000/infer \
  -F "image=@test_package.jpg" \
  -o result.png
```

Returns: PNG image with anomaly heatmap overlay

### 4️⃣ Inference - Get JSON Result

```bash
curl -X POST http://127.0.0.1:5000/infer_json \
  -F "image=@test_package.jpg"
```

**Response:**
```json
{
  "status": "success",
  "damage_percentage": 12.34,
  "anomaly_detected": true,
  "output_image": "result_12345.png",
  "threshold": 0.5
}
```

---

## 🐍 Python Client Example

```python
import requests

# 1. Load reference image
with open('normal_package.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/load_reference',
        files={'image': f}
    )
print(response.json())

# 2. Run inference (PNG output)
with open('test_package.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/infer',
        files={'image': f}
    )

# Save result
with open('result.png', 'wb') as f:
    f.write(response.content)

# 3. Run inference (JSON output)
with open('test_package.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/infer_json',
        files={'image': f}
    )
data = response.json()
print(f"Damage: {data['damage_percentage']:.2f}%")
```

---

## 🔬 Notebook Fidelity

### Architecture (EXACT)
- **Backbone:** WideResNet-50-2 (pretrained ImageNet)
- **Feature Layers:** Layer 2 (512 channels) + Layer 3 (1024 channels)
- **Feature Extraction:** Concatenate layer2 + upsampled layer3
- **Patches:** 28×28 = 784 patches per image
- **Feature Dim:** 1536 (512 + 1024)

### Parameters (EXACT)
- **Input Size:** 224×224
- **Normalization:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **k-NN Neighbors:** 9
- **Distance Metric:** Euclidean
- **Anomaly Threshold:** 0.5
- **Heatmap Colormap:** COLORMAP_JET
- **Overlay Blend:** 0.5 weight for original + 0.5 for heatmap

### Pipeline (EXACT)
```
Input → RGB Conversion → Resize 224×224 → ToTensor → Normalize
  ↓
WideResNet-50-2 (layer2: 28×28, layer3: 14×14)
  ↓
Upsample layer3 to 28×28 (bilinear)
  ↓
Concatenate → 784 patches × 1536 dims
  ↓
k-NN distance to memory bank (k=9)
  ↓
Mean distance → Anomaly scores
  ↓
Reshape 28×28 → Upsample 224×224
  ↓
Normalize [0,1] → Apply COLORMAP_JET → Blend with original
  ↓
Output: Overlay PNG
```

---

## ⚙️ Configuration

### GPU Support

Change line in `app.py`:

```python
# CPU (default)
device = torch.device('cpu')

# GPU
device = torch.device('cuda')
```

### Port Change

```python
app.run(host='127.0.0.1', port=8080, debug=False)
```

### Max File Size

```python
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

---

## 🐛 Troubleshooting

### "Memory bank not loaded"
**Solution:** Call `/load_reference` before `/infer`

### Slow inference
**Cause:** CPU processing is slow for WideResNet-50-2  
**Solution:** Use GPU (change `device` to `'cuda'`)

### Import errors
```powershell
pip install flask torch torchvision opencv-python pillow scikit-learn
```

### Port already in use
**Solution:** Change port in `app.py` or kill existing process

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| flask | 3.0.0 | Web framework |
| torch | 2.1.0 | Deep learning |
| torchvision | 0.16.0 | Pretrained models |
| opencv-python | 4.8.1.78 | Image processing |
| pillow | 10.1.0 | Image I/O |
| scikit-learn | 1.3.2 | k-NN index |
| numpy | 1.24.3 | Array operations |

---

## 🚢 Production Deployment (Optional)

### Using Gunicorn

```powershell
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app --timeout 120
```

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY models/ models/
COPY static/ static/

EXPOSE 5000
CMD ["python", "app.py"]
```

Build & run:
```bash
docker build -t patchcore-service .
docker run -p 5000:5000 patchcore-service
```

---

## ✅ Hackathon Demo Checklist

- ✅ Model loads successfully
- ✅ Reference image builds memory bank
- ✅ Inference returns overlay image
- ✅ All logic matches notebook exactly
- ✅ No architectural changes
- ✅ No parameter modifications
- ✅ CPU-compatible
- ✅ Single-file service
- ✅ Simple REST API
- ✅ Test suite included
- ✅ Easy setup (3 commands)

---

## 📝 Notes

- **No modifications:** All notebook logic preserved exactly
- **CPU-only default:** Works on any machine (use GPU for speed)
- **Memory bank:** Loaded once, persists for all requests
- **Image formats:** Auto-converts RGBA→RGB
- **Threshold:** Fixed at 0.5 (same as notebook)

---

## 🎯 Next Steps

1. **Test locally:** Run `python test_service.py`
2. **Use real images:** Replace test images with actual package photos
3. **Deploy:** Use gunicorn/Docker for production
4. **Integrate:** Call from frontend/mobile app
5. **Monitor:** Add logging/metrics if needed

---

## 📄 License

Same as original notebook

---

**Questions?** Check the test script or API documentation above.
