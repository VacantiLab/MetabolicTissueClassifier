import torch
from AppFiles.models import FullyConnectedNN  # Relative import
import numpy as np

def ComputeAppOutput(input_size, hidden_size1, hidden_size2, output_size, number_list):

    # Model logic as before
    Pathway_Genes = np.loadtxt("AppFiles/pathway_list.txt",dtype=str)

    input_size = 30
    hidden_size1 = 16
    hidden_size2 = 8
    output_size = 2
    trained_model = FullyConnectedNN(input_size, hidden_size1, hidden_size2, output_size)
    trained_model.load_state_dict(torch.load("/ContainerWD/trained_model.pth", weights_only=True))
    trained_model.eval()
    Test_Array = torch.tensor(number_list, dtype=torch.float32)
    model_outputs = trained_model(Test_Array)
    value, predicted_class = torch.max(model_outputs, 0)
    int_to_label = {0: 'Subcutaneous Adipose Tissue', 1: 'Visceral Adipose Tissue'}
    predicted_class_name = int_to_label[predicted_class.item()]
    return(predicted_class_name)