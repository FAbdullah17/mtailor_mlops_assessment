# model.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  
        return tensor.numpy()


class ONNXModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]
