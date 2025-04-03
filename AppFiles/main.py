from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import torch

from .models import FullyConnectedNN  # Relative import

FastAPI_Object = FastAPI()
templates = Jinja2Templates(directory="/ContainerWD/AppFiles/templates")

@FastAPI_Object.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        #"result": None,
        "numbers": "",
        "calculation": ""
    })

@FastAPI_Object.post("/add", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    numbers: str = Form(...)
):
    try:
        # Split by spaces and convert to floats
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

    except ValueError:
        total = "Invalid input"
        calculation = "Please enter numbers separated by spaces"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        #"result": total,
        "numbers": numbers,  # Preserve original input
        "calculation": calculation
    })