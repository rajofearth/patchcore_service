# PatchCore Damage Detection Service

Minimal Flask service for package damage detection using PatchCore anomaly detection.

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
UV pip install -r requirements.txt
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

- `GET /` - Web interface
- `POST /load_reference` - Upload normal reference image
- `POST /infer` - Upload test image, returns overlay PNG

## Usage

1. Upload a normal reference image
2. Upload a test image to detect damage
3. View the result with anomaly heatmap overlay

## Configuration

Change device in `app.py`:
```python
device = torch.device('cuda')  # For GPU
```
