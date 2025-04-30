def test_addition():
    assert 2 + 2 == 4

def test_addition():
    assert 2 * 3 == 6

def test_model_output():
    from AppFiles.models import FullyConnectedNN  # Relative import
    import torch

    numbers = '0.737062 -0.297366 0.918709 1.195301 0.212010 0.881349 -0.116011 0.256502 0.148948 0.306263 -2.290483 0.296994 0.818196 0.792278 0.118434 0.384367 0.707052 0.643651 0.688815 0.005301 0.472231 0.264970 0.468226 1.798377 0.595257 0.423784 0.831144 -0.923132 0.466556 1.230604'
    number_list = [float(num) for num in numbers.split()]
    input_size = 30
    hidden_size1 = 16
    hidden_size2 = 8
    output_size = 2
    trained_model = FullyConnectedNN(input_size, hidden_size1, hidden_size2, output_size)
    trained_model.load_state_dict(torch.load("/ContainerWD/trained_model.pth", weights_only=True))
    trained_model.eval()
    Test_Array = torch.tensor(number_list, dtype=torch.float32)
    model_outputs = trained_model(Test_Array)
    value, predicted_class = torch.max(model_outputs, 0)  # Use dimension 0 for a single data point
    int_to_label = {0: 'Subcutaneous Adipose Tissue', 1: 'Visceral Adipose Tissue'}
    predicted_class_name = int_to_label[predicted_class.item()]  # Convert to the original label
    calculation = predicted_class_name
    assert predicted_class_name == 'Visceral Adipose Tissue'