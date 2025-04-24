import torch
import torch.nn.functional as F
from AppFiles.models import FullyConnectedNN
import numpy as np
import json

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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

def generate_tsne_image(softmax_scores, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(softmax_scores)

    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab20', s=10)
    ax.set_title("t-SNE on Softmax Outputs")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"
