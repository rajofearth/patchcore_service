"""
PatchCore Package Damage Detection - Flask Service
===================================================
EXACT COPY of notebook logic - NO MODIFICATIONS
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import cv2
import io
import os
from flask import Flask, request, jsonify, send_file, render_template

# ============================================================================
# EXACT COPY FROM NOTEBOOK - Feature Extractor
# ============================================================================

class FeatureExtractor(torch.nn.Module):
    """
    EXACT COPY FROM NOTEBOOK
    Extract features from WideResNet-50-2 layers 2 and 3
    """
    def __init__(self):
        super().__init__()
        # Load pretrained WideResNet-50-2
        backbone = models.wide_resnet50_2(pretrained=True)
        
        # Extract specific layers (same as notebook)
        self.layer1 = torch.nn.Sequential(*list(backbone.children())[:5])
        self.layer2 = torch.nn.Sequential(*list(backbone.children())[5:6])
        self.layer3 = torch.nn.Sequential(*list(backbone.children())[6:7])
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Pass through layers sequentially
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        return feat2, feat3


# ============================================================================
# EXACT COPY FROM NOTEBOOK - PatchCore Implementation
# ============================================================================

class PatchCore:
    """
    EXACT COPY FROM NOTEBOOK
    PatchCore anomaly detection with k-NN memory bank
    """
    def __init__(self, feature_extractor, device, num_neighbors=9):
        self.feature_extractor = feature_extractor
        self.device = device
        self.num_neighbors = num_neighbors
        self.memory_bank = None
        self.nn_index = None
        
        # EXACT preprocessing from notebook (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _extract_patch_features(self, image_tensor):
        """
        EXACT COPY FROM NOTEBOOK
        Extract and concatenate features from layer2 and layer3
        """
        with torch.no_grad():
            feat2, feat3 = self.feature_extractor(image_tensor)
            
            # Get spatial dimensions
            b, c2, h2, w2 = feat2.shape
            b, c3, h3, w3 = feat3.shape
            
            # Upsample feat3 to match feat2 spatial dimensions (EXACT from notebook)
            feat3_upsampled = F.interpolate(
                feat3, 
                size=(h2, w2), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Concatenate along channel dimension
            features = torch.cat([feat2, feat3_upsampled], dim=1)
            
            # Reshape to (batch, num_patches, feature_dim)
            features = features.permute(0, 2, 3, 1)
            features = features.reshape(b, h2 * w2, -1)
            
            return features, (h2, w2)
    
    def fit_normal_reference(self, normal_image):
        """
        EXACT COPY FROM NOTEBOOK
        Build memory bank from normal (pristine) reference image
        """
        # Convert RGBA to RGB if needed (EXACT from notebook)
        if normal_image.mode != 'RGB':
            normal_image = normal_image.convert('RGB')
        
        # Preprocess
        image_tensor = self.transform(normal_image).unsqueeze(0).to(self.device)
        
        # Extract features
        features, spatial_size = self._extract_patch_features(image_tensor)
        
        # Build memory bank (EXACT from notebook)
        self.memory_bank = features.squeeze(0).cpu().numpy()
        self.spatial_size = spatial_size
        
        # Build k-NN index (EXACT parameters from notebook)
        self.nn_index = NearestNeighbors(
            n_neighbors=self.num_neighbors,
            metric='euclidean',
            algorithm='auto'
        )
        self.nn_index.fit(self.memory_bank)
        
        return self.memory_bank.shape
    
    def predict(self, test_image):
        """
        EXACT COPY FROM NOTEBOOK
        Compute anomaly scores for test image
        """
        if self.memory_bank is None:
            raise ValueError("Memory bank not initialized. Call fit_normal_reference first.")
        
        # Convert RGBA to RGB if needed (EXACT from notebook)
        if test_image.mode != 'RGB':
            test_image = test_image.convert('RGB')
        
        # Preprocess
        image_tensor = self.transform(test_image).unsqueeze(0).to(self.device)
        
        # Extract features
        features, _ = self._extract_patch_features(image_tensor)
        test_features = features.squeeze(0).cpu().numpy()
        
        # Compute k-NN distances (EXACT from notebook)
        distances, _ = self.nn_index.kneighbors(test_features)
        
        # Mean distance across k neighbors (EXACT from notebook)
        anomaly_scores = distances.mean(axis=1)
        
        # Reshape to spatial map (EXACT from notebook)
        h, w = self.spatial_size
        anomaly_map = anomaly_scores.reshape(h, w)
        
        # Upsample to 224x224 (EXACT from notebook)
        anomaly_map_upsampled = cv2.resize(
            anomaly_map, 
            (224, 224), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to [0, 1] (EXACT from notebook)
        anomaly_map_normalized = (anomaly_map_upsampled - anomaly_map_upsampled.min()) / \
                                 (anomaly_map_upsampled.max() - anomaly_map_upsampled.min() + 1e-8)
        
        return anomaly_map_normalized


# ============================================================================
# EXACT COPY FROM NOTEBOOK - Visualization
# ============================================================================

def create_overlay_image(original_image, anomaly_map, threshold=0.5):
    """
    EXACT COPY FROM NOTEBOOK
    Create heatmap overlay visualization
    """
    # Convert RGBA to RGB if needed (EXACT from notebook)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Resize original to 224x224 (EXACT from notebook)
    original_resized = original_image.resize((224, 224))
    original_np = np.array(original_resized)
    
    # Create heatmap (EXACT from notebook)
    heatmap = (anomaly_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original (EXACT weights from notebook)
    overlay = cv2.addWeighted(original_np, 0.5, heatmap_colored, 0.5, 0)
    
    # Create damage mask (EXACT threshold from notebook)
    damage_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    
    # Calculate damage percentage (EXACT from notebook)
    damage_percentage = (anomaly_map > threshold).sum() / anomaly_map.size * 100
    
    return {
        'overlay': Image.fromarray(overlay),
        'heatmap': Image.fromarray(heatmap_colored),
        'mask': Image.fromarray(damage_mask),
        'damage_percentage': damage_percentage
    }


# ============================================================================
# Flask Application
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model instance (loaded once)
device = torch.device('cpu')  # Change to 'cuda' for GPU
feature_extractor = None
patchcore = None

print("\n" + "="*70)
print("Initializing PatchCore Flask Service")
print("="*70)
print(f"Device: {device}")

# Initialize model at startup (EXACT from notebook)
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()
print("✓ WideResNet-50-2 backbone loaded")

patchcore = PatchCore(feature_extractor, device, num_neighbors=9)
print("✓ PatchCore initialized (waiting for normal reference)")
print("="*70)


@app.route('/', methods=['GET'])
def home():
    """Web interface"""
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_info():
    """API info endpoint"""
    return jsonify({
        'service': 'PatchCore Package Damage Detection',
        'status': 'running',
        'model': 'WideResNet-50-2',
        'memory_bank_loaded': patchcore.memory_bank is not None,
        'endpoints': {
            'GET /': 'Web interface',
            'GET /api': 'This API info page',
            'GET /health': 'Service status',
            'POST /load_reference': 'Upload normal reference image',
            'POST /infer': 'Upload test image, returns overlay PNG',
            'POST /infer_json': 'Upload test image, returns JSON with damage info'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Service health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'PatchCore with WideResNet-50-2',
        'memory_bank_loaded': patchcore.memory_bank is not None,
        'device': str(device)
    })


@app.route('/load_reference', methods=['POST'])
def load_reference():
    """
    Load normal reference image and build memory bank
    EXACT LOGIC FROM NOTEBOOK
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Load image (EXACT from notebook)
        image = Image.open(io.BytesIO(file.read()))
        
        # Build memory bank (EXACT from notebook)
        memory_shape = patchcore.fit_normal_reference(image)
        
        return jsonify({
            'status': 'success',
            'message': 'Memory bank built successfully',
            'patches': int(memory_shape[0]),
            'feature_dim': int(memory_shape[1])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/infer', methods=['POST'])
def infer():
    """
    Run inference on test image - returns overlay PNG
    EXACT LOGIC FROM NOTEBOOK
    """
    if patchcore.memory_bank is None:
        return jsonify({'error': 'Memory bank not loaded. Call /load_reference first'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Load image (EXACT from notebook)
        test_image = Image.open(io.BytesIO(file.read()))
        
        # Run inference (EXACT from notebook)
        anomaly_map = patchcore.predict(test_image)
        
        # Create visualization (EXACT from notebook)
        result = create_overlay_image(test_image, anomaly_map, threshold=0.5)
        
        # Save overlay to bytes
        img_io = io.BytesIO()
        result['overlay'].save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/infer_json', methods=['POST'])
def infer_json():
    """
    Run inference on test image - returns JSON with damage info
    EXACT LOGIC FROM NOTEBOOK
    """
    if patchcore.memory_bank is None:
        return jsonify({'error': 'Memory bank not loaded. Call /load_reference first'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Load image (EXACT from notebook)
        test_image = Image.open(io.BytesIO(file.read()))
        
        # Run inference (EXACT from notebook)
        anomaly_map = patchcore.predict(test_image)
        
        # Create visualization (EXACT from notebook)
        result = create_overlay_image(test_image, anomaly_map, threshold=0.5)
        
        # Save to static folder
        output_filename = f"result_{hash(file.filename)}.png"
        output_path = os.path.join('static', 'outputs', output_filename)
        result['overlay'].save(output_path)
        
        return jsonify({
            'status': 'success',
            'damage_percentage': float(result['damage_percentage']),
            'anomaly_detected': bool(result['damage_percentage'] > 5.0),
            'output_image': output_filename,
            'threshold': 0.5
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PatchCore Flask Service - Ready for Hackathon Demo")
    print("="*70)
    print("Endpoints:")
    print("  GET  /              - Service info")
    print("  GET  /health        - Health check")
    print("  POST /load_reference - Upload normal image")
    print("  POST /infer         - Get overlay PNG")
    print("  POST /infer_json    - Get JSON result")
    print("="*70)
    print("\nStarting Flask server on http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False)
