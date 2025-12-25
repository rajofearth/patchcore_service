# PatchCore Damage Detection Service

Minimal Flask service for package damage detection using PatchCore anomaly detection. Features a modern web interface and RESTful API with CORS support for external integrations.

![image.png](static/image.png)
## Quick Start

1. Create virtual environment:
```bash
uv venv
```

2. Activate virtual environment:
```bash
uv activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Run service:
```bash
uv run app.py
```

5. Open browser:
```
http://127.0.0.1:5000
```

## API Endpoints

### `GET /`
Web interface for interactive damage detection.

### `POST /load_reference`
Upload a normal (undamaged) reference image to build the memory bank.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "status": "success",
  "message": "Memory bank built successfully",
  "patches": 784,
  "feature_dim": 1536
}
```

### `POST /infer`
Upload a test image and receive the overlay PNG with anomaly heatmap.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
- Content-Type: `image/png`
- Returns PNG image at original input resolution

### `POST /infer_json`
Upload a test image and receive JSON with damage statistics and base64-encoded image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "status": "success",
  "damage_percentage": 12.34,
  "anomaly_detected": true,
  "threshold": 0.5,
  "image": "data:image/png;base64,..."
}
```

## External API Usage

### cURL Example
```bash
# Load reference image
curl -X POST http://127.0.0.1:5000/load_reference \
  -F "image=@normal_image.jpg"

# Get JSON response with stats
curl -X POST http://127.0.0.1:5000/infer_json \
  -F "image=@test_image.jpg"

# Get PNG image directly
curl -X POST http://127.0.0.1:5000/infer \
  -F "image=@test_image.jpg" \
  -o result.png
```

### Python Example
```python
import requests

# Load reference
with open('normal.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/load_reference',
        files={'image': f}
    )
print(response.json())

# Get damage detection results
with open('test.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/infer_json',
        files={'image': f}
    )
data = response.json()
print(f"Damage: {data['damage_percentage']}%")
print(f"Detected: {data['anomaly_detected']}")
```

### Next.js/React Example
```javascript
const formData = new FormData();
formData.append('image', file);

const response = await fetch('http://127.0.0.1:5000/infer_json', {
  method: 'POST',
  body: formData
});

const data = await response.json();
// Use data.image (base64), data.damage_percentage, data.anomaly_detected
```

## Features

- **Original Resolution**: Output images match input image dimensions
- **CORS Enabled**: Ready for cross-origin requests from web applications
- **Dual Output Formats**: PNG image endpoint and JSON endpoint with statistics
- **Real-time Detection**: Fast inference with WideResNet-50-2 backbone
- **Modern UI**: Sleek web interface with drag-and-drop support

## Configuration

Change device in `app.py`:
```python
device = torch.device('cuda')  # For GPU
```

Change port in `app.py`:
```python
app.run(host='127.0.0.1', port=8080, debug=False)
```

## Notes

- Memory bank must be loaded via `/load_reference` before running inference
- Detection threshold is fixed at 0.5
- Anomaly is detected when damage percentage > 5%
- Output images are returned at original input resolution
