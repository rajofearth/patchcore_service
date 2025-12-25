import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import cv2
import io
from flask import Flask, request, jsonify, send_file, render_template

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.wide_resnet50_2(pretrained=True)
        self.layer1 = torch.nn.Sequential(*list(backbone.children())[:5])
        self.layer2 = torch.nn.Sequential(*list(backbone.children())[5:6])
        self.layer3 = torch.nn.Sequential(*list(backbone.children())[6:7])
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        return feat2, feat3

class PatchCore:
    def __init__(self, feature_extractor, device, num_neighbors=9):
        self.feature_extractor = feature_extractor
        self.device = device
        self.num_neighbors = num_neighbors
        self.memory_bank = None
        self.nn_index = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _extract_patch_features(self, image_tensor):
        with torch.no_grad():
            feat2, feat3 = self.feature_extractor(image_tensor)
            b, c2, h2, w2 = feat2.shape
            b, c3, h3, w3 = feat3.shape
            feat3_upsampled = F.interpolate(
                feat3, 
                size=(h2, w2), 
                mode='bilinear', 
                align_corners=False
            )
            features = torch.cat([feat2, feat3_upsampled], dim=1)
            features = features.permute(0, 2, 3, 1)
            features = features.reshape(b, h2 * w2, -1)
            return features, (h2, w2)
    
    def fit_normal_reference(self, normal_image):
        if normal_image.mode != 'RGB':
            normal_image = normal_image.convert('RGB')
        image_tensor = self.transform(normal_image).unsqueeze(0).to(self.device)
        features, spatial_size = self._extract_patch_features(image_tensor)
        self.memory_bank = features.squeeze(0).cpu().numpy()
        self.spatial_size = spatial_size
        self.nn_index = NearestNeighbors(
            n_neighbors=self.num_neighbors,
            metric='euclidean',
            algorithm='auto'
        )
        self.nn_index.fit(self.memory_bank)
        return self.memory_bank.shape
    
    def predict(self, test_image):
        if self.memory_bank is None:
            raise ValueError("Memory bank not initialized. Call fit_normal_reference first.")
        if test_image.mode != 'RGB':
            test_image = test_image.convert('RGB')
        image_tensor = self.transform(test_image).unsqueeze(0).to(self.device)
        features, _ = self._extract_patch_features(image_tensor)
        test_features = features.squeeze(0).cpu().numpy()
        distances, _ = self.nn_index.kneighbors(test_features)
        anomaly_scores = distances.mean(axis=1)
        h, w = self.spatial_size
        anomaly_map = anomaly_scores.reshape(h, w)
        anomaly_map_upsampled = cv2.resize(
            anomaly_map, 
            (224, 224), 
            interpolation=cv2.INTER_LINEAR
        )
        anomaly_map_normalized = (anomaly_map_upsampled - anomaly_map_upsampled.min()) / \
                                 (anomaly_map_upsampled.max() - anomaly_map_upsampled.min() + 1e-8)
        return anomaly_map_normalized

def create_overlay_image(original_image, anomaly_map, threshold=0.5):
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    original_resized = original_image.resize((224, 224))
    original_np = np.array(original_resized)
    heatmap = (anomaly_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_np, 0.5, heatmap_colored, 0.5, 0)
    return Image.fromarray(overlay)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

device = torch.device('cpu')
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()
patchcore = PatchCore(feature_extractor, device, num_neighbors=9)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/load_reference', methods=['POST'])
def load_reference():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
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
    if patchcore.memory_bank is None:
        return jsonify({'error': 'Memory bank not loaded. Call /load_reference first'}), 400
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        test_image = Image.open(io.BytesIO(file.read()))
        anomaly_map = patchcore.predict(test_image)
        result = create_overlay_image(test_image, anomaly_map, threshold=0.5)
        img_io = io.BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
