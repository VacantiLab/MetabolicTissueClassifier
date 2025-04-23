import torch
from AppFiles.models import FullyConnectedNN  # Relative import
import numpy as np

def ComputeAppOutput(model_input_tensor):

    # Model logic as before
    #Pathway_Genes = np.loadtxt("AppFiles/pathway_list.txt",dtype=str)

    input_size = 30
    hidden_size1 = 16
    hidden_size2 = 8
    output_size = 2
    trained_model = FullyConnectedNN(input_size, hidden_size1, hidden_size2, output_size)
    trained_model.load_state_dict(torch.load("/ContainerWD/trained_model.pth", weights_only=True))
    trained_model.eval()
    with torch.no_grad():
        model_outputs = trained_model(model_input_tensor)
        int_to_label = {0: 'Subcutaneous Adipose Tissue', 1: 'Visceral Adipose Tissue'}
        predicted_classes = torch.argmax(model_outputs, dim=1)  # shape: (n_samples,)
        predicted_class_names = [int_to_label[int(cls)] for cls in predicted_classes]
    return(predicted_class_names)