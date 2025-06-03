# Mtailor ONNX Classifier Deployment
This repository contains a complete project for deploying an image classification neural network to the serverless GPU platform [Cerebrium](https://www.cerebrium.ai), using ONNX format and Docker-based deployment.
## 🚀 Project Overview
This project: * Converts a PyTorch classification model to ONNX. * Wraps the model in a Python interface for Cerebrium. * Dockerizes the deployment. * Provides testing scripts for both local and remote (Cerebrium) inference. * Includes a fully configured `cerebrium.toml` for deployment.
## 🧠 Model Details
* Base Architecture: ResNet18 (customized) * Dataset: [ImageNet](https://www.image-net.org/) * Input: RGB image of shape `(224, 224)` * Output: Array of probabilities for 1000 ImageNet classes * Sample Images: * `n01440764_tench.JPEG` → Class ID 0 * `n01667114_mud_turtle.JPEG` → Class ID 35
## 🛠️ Preprocessing Details
Preprocessing included during inference: * Convert image to RGB * Resize to 224x224 (bilinear) * Divide by 255 * Normalize with mean `[0.485, 0.456, `0.406]` and std `[0.229, 0.224, 0.225]` These steps are embedded into the ONNX model or handled pre-inference.
## 🔄 Conversion to ONNX
Use `convert_to_onnx.py` to convert the PyTorch model to ONNX: `python convert_to_onnx.py` This will generate `model.onnx`.
## 🧪 Local Testing
Run local tests (ONNX inference + preprocessing) using: `python test.py`
## 🌐 Remote Testing (Deployed Model)
Use `test_server.py` to test the deployed Cerebrium model: `python test_server.py --image n01440764_tench.JPEG --api-url <DEPLOYMENT_URL> --api-key <YOUR_API_KEY>` This will return the predicted class ID from the live endpoint.
## 🐳 Docker Deployment
A custom Dockerfile is provided for Cerebrium-compatible deployment. It installs all dependencies, loads the ONNX model, and exposes the `predict()` method in `main.py`.
## 📦 Install Requirements
Install dependencies locally using: `pip install -r requirements.txt`
## 🧪 Files Overview
### `main.py` Defines the `predict(input)` function used by Cerebrium during inference. ### `model.py` * `OnnxModel`: loads the ONNX model and runs inference. * `ImagePreprocessor`: prepares raw image data for model input. ### `test.py` Tests model inference locally using sample inputs. ### `test_server.py` Tests model inference remotely using your deployed Cerebrium endpoint. ### `convert_to_onnx.py` Converts the PyTorch model to ONNX format with preprocessing embedded. ### `pytorch_model.py` Contains the original PyTorch model and preprocessing logic (`preprocess_numpy`).
## 📤 Deployment Steps
1. Clone this repository 2. Convert PyTorch model to ONNX 3. Push code to GitHub 4. Link the repo on [Cerebrium](https://www.cerebrium.ai) 5. Cerebrium detects `Dockerfile` and `main.py` 6. Deploy and monitor logs 7. Use `test_server.py` to verify deployment
## 🔑 API and Inference Notes
* Make sure to pass: * `--api-url`: Cerebrium deployment URL * `--api-key`: Your API key from Cerebrium dashboard * Both are required for `test_server.py`
## 📹 Loom Video
Please refer to the submitted Loom video(s) for a full walkthrough, including: * Code explanation * Git usage * Cerebrium deployment * Local + remote testing
## ✅ Deliverables Summary
| Deliverable          | Included | Description                            |
| -------------------- | -------- | -------------------------------------- |
| `convert_to_onnx.py` | ✅        | PyTorch → ONNX conversion              |
| `model.py`           | ✅        | ONNX inference + preprocessing classes |
| `test.py`            | ✅        | Local tests                            |
| `test_server.py`     | ✅        | Tests against deployed endpoint        |
| `main.py`            | ✅        | Required entrypoint for Cerebrium      |
| `requirements.txt`   | ✅        | All required Python packages           |
| `cerebrium.toml`     | ✅        | Cerebrium deployment configuration     |
| `Dockerfile`         | ✅        | Custom deployment base                 |
| `.dockerignore`      | ✅        | Optimizes Docker build                 |
| `README.md`          | ✅        | This file                              |
## 📌 Notes
* Do **not** deploy PyTorch model directly. Only use the ONNX model. * Cerebrium **requires Docker-based deployment**. * Preprocessing should be part of ONNX or handled inside `main.py`.
