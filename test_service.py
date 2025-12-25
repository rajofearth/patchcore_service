"""
Test Script for PatchCore Flask Service
========================================
Tests all endpoints with sample images
"""

import requests
import os
import time
from PIL import Image
import numpy as np

BASE_URL = "http://127.0.0.1:5000"

def create_test_image(filename, has_damage=False):
    """Create a simple test image (for demo purposes)"""
    # Create 224x224 RGB image
    if has_damage:
        # Damaged: with dark spots
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        # Add "damage" spots
        img[50:100, 50:100] = [50, 0, 0]  # Dark red spot
        img[150:180, 150:180] = [0, 0, 50]  # Dark blue spot
    else:
        # Normal: uniform color
        img = np.full((224, 224, 3), 180, dtype=np.uint8)
    
    pil_img = Image.fromarray(img)
    pil_img.save(filename)
    print(f"  ✓ Created test image: {filename}")


def test_health():
    """Test 1: Health check"""
    print("\n[Test 1/5] Health Check")
    print("-" * 50)
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"  Status Code: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_home():
    """Test 2: Home endpoint"""
    print("\n[Test 2/5] Home Endpoint")
    print("-" * 50)
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"  Status Code: {response.status_code}")
        data = response.json()
        print(f"  Service: {data.get('service')}")
        print(f"  Status: {data.get('status')}")
        print(f"  Memory Bank Loaded: {data.get('memory_bank_loaded')}")
        return response.status_code == 200
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_load_reference():
    """Test 3: Load normal reference image"""
    print("\n[Test 3/5] Load Reference Image")
    print("-" * 50)
    
    # Create test normal image
    create_test_image("test_normal.jpg", has_damage=False)
    
    try:
        with open("test_normal.jpg", "rb") as f:
            files = {"image": f}
            response = requests.post(f"{BASE_URL}/load_reference", files=files)
        
        print(f"  Status Code: {response.status_code}")
        data = response.json()
        print(f"  Message: {data.get('message')}")
        print(f"  Patches: {data.get('patches')}")
        print(f"  Feature Dim: {data.get('feature_dim')}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_infer_png():
    """Test 4: Inference with PNG output"""
    print("\n[Test 4/5] Inference - PNG Output")
    print("-" * 50)
    
    # Create test damaged image
    create_test_image("test_damaged.jpg", has_damage=True)
    
    try:
        with open("test_damaged.jpg", "rb") as f:
            files = {"image": f}
            response = requests.post(f"{BASE_URL}/infer", files=files)
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Save result
            output_path = "result_overlay.png"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"  ✓ Saved result to: {output_path}")
            return True
        else:
            print(f"  ✗ Error: {response.json()}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_infer_json():
    """Test 5: Inference with JSON output"""
    print("\n[Test 5/5] Inference - JSON Output")
    print("-" * 50)
    
    try:
        with open("test_damaged.jpg", "rb") as f:
            files = {"image": f}
            response = requests.post(f"{BASE_URL}/infer_json", files=files)
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Damage %: {data.get('damage_percentage'):.2f}%")
            print(f"  Anomaly Detected: {data.get('anomaly_detected')}")
            print(f"  Output Image: {data.get('output_image')}")
            print(f"  Threshold: {data.get('threshold')}")
            return True
        else:
            print(f"  ✗ Error: {response.json()}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def cleanup():
    """Clean up test files"""
    print("\n[Cleanup] Removing test files...")
    for filename in ["test_normal.jpg", "test_damaged.jpg"]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"  ✓ Removed: {filename}")


def main():
    print("="*70)
    print("PatchCore Flask Service - Test Suite")
    print("="*70)
    print("\nMake sure the Flask service is running at http://127.0.0.1:5000")
    print("Start it with: python app.py")
    print("\nPress Enter to continue...")
    input()
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Home Endpoint", test_home()))
    results.append(("Load Reference", test_load_reference()))
    
    # Wait a bit for memory bank to be ready
    time.sleep(1)
    
    results.append(("Inference PNG", test_infer_png()))
    results.append(("Inference JSON", test_infer_json()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        color = "green" if result else "red"
        print(f"  {status:8} {name}")
    
    print("-"*70)
    print(f"  Total: {passed}/{total} tests passed")
    print("="*70)
    
    # Cleanup
    cleanup()
    
    # Final notes
    if passed == total:
        print("\n✓ All tests passed! Service is working correctly.")
        print("\nYou can now use the service with your own images:")
        print("  curl -X POST http://127.0.0.1:5000/infer \\")
        print("    -F \"image=@your_image.jpg\" \\")
        print("    -o result.png")
    else:
        print("\n✗ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
