import argparse
import requests
from PIL import Image
import numpy as np
import io
import json
import torch
from torchvision import transforms


def preprocess_image(image_path):
    """Preprocess image as per ImageNet stats"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0) 
    return tensor.numpy()  
def predict_from_api(image_tensor, api_url, api_key):
    """Send POST request to Cerebrium ONNX endpoint"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": image_tensor.tolist()
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()

        # Assume output format: {"output": [[...]]}
        output_probs = output["output"][0]
        predicted_class = int(np.argmax(output_probs))
        print(f"[✓] Predicted Class ID: {predicted_class}")
    except Exception as e:
        print(f"[X] Error during API call: {e}")


def run_tests(api_url, api_key):
    """Run built-in test cases"""
    print("[TEST] Running preset tests...")
    test_images = [
        ("n01440764_tench.JPEG", 0),
        ("n01667114_mud_turtle.JPEG", 35)
    ]

    for img_path, expected_class in test_images:
        print(f"\nTesting image: {img_path}")
        img_tensor = preprocess_image(img_path)
        predict_from_api(img_tensor, api_url, api_key)


def main():
    parser = argparse.ArgumentParser(description="Test ONNX Cerebrium Deployment")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--api-url", type=str, required=True, help="Cerebrium API URL")
    parser.add_argument("--api-key", type=str, required=True, help="Cerebrium API Key")
    parser.add_argument("--run-tests", action="store_true", help="Run preset image classification tests")

    args = parser.parse_args()

    if args.run_tests:
        run_tests(args.api_url, args.api_key)
    elif args.image:
        image_tensor = preprocess_image(args.image)
        predict_from_api(image_tensor, args.api_url, args.api_key)
    else:
        print("Error: Please provide either --image or --run-tests")


if __name__ == "__main__":
    main()
