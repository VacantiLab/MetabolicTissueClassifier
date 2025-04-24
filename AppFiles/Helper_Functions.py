import torch
import torch.nn.functional as F
from AppFiles.models import FullyConnectedNN
import numpy as np
import json

with open("AppFiles/label_map.json", "r") as f:
    int_to_label = json.load(f)

def ComputeAppOutput(model_input_tensor):
    input_size = 153
    hidden_size1 = 256
    hidden_size2 = 128
    output_size = 54

    trained_model = FullyConnectedNN(input_size, hidden_size1, hidden_size2, output_size)
    trained_model.load_state_dict(torch.load("/ContainerWD/trained_model.pth", weights_only=True))
    trained_model.eval()

    with torch.no_grad():
        model_outputs = trained_model(model_input_tensor)  # shape: (n_samples, output_size)
        softmax_probs = F.softmax(model_outputs, dim=1)     # shape: (n_samples, output_size)
        predicted_classes = torch.argmax(softmax_probs, dim=1)

        predicted_class_names = [int_to_label[str(int(cls))] for cls in predicted_classes]
        softmax_scores = softmax_probs.tolist()  # Convert to Python list for easier processing

    return predicted_class_names, softmax_scores, int_to_label
