from model import ImagePreprocessor, ONNXModel
import argparse
import numpy as np

def main(image_path: str, model_path: str):
    preprocessor = ImagePreprocessor()
    input_tensor = preprocessor.preprocess(image_path)

    model = ONNXModel(model_path)
    
    output = model.predict(input_tensor)

    predicted_class = int(np.argmax(output))
    print(f"Predicted class ID: {predicted_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX Image Classifier")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    
    args = parser.parse_args()
    main(args.image, args.model)