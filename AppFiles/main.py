from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import os
import uvicorn

import torch

from AppFiles.models import FullyConnectedNN  # Relative import

from pathlib import Path

# Needed to return request counts for scaling
from prometheus_client import generate_latest
from AppFiles.metrics import request_count

from fastapi.responses import PlainTextResponse

FastAPI_Object = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@FastAPI_Object.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        #"result": None,
        "numbers": "",
        "calculation": ""
    })

@FastAPI_Object.get("/health")
def health_check():
    print("✅ /health was hit")
    return {"status": "ok"}


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

# Increments counter
@FastAPI_Object.middleware("http")
async def count_requests_middleware(request: Request, call_next):
    request_count.inc()
    response = await call_next(request)
    return response

@FastAPI_Object.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # 8080 default for local
    uvicorn.run("AppFiles.main:FastAPI_Object", host="0.0.0.0", port=port)