import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(input_array):
    return input_array

def predict(input):
    input_tensor = preprocess(input["input"])
    outputs = session.run([output_name], {input_name: input_tensor})
    return {"prediction": outputs[0].tolist()}
