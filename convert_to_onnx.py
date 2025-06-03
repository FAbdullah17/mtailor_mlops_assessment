import torch
import torch.nn as nn
from pytorch_model import Classifier, BasicBlock

model = Classifier(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load("resnet18-f37072fd.pth", map_location=torch.device('cpu')))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

print("model.onnx created successfully.")