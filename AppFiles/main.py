from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import uvicorn
import torch
from AppFiles.Helper_Functions import ComputeAppOutput  # Relative import
from pathlib import Path
import traceback

# Needed to return request counts for scaling
from prometheus_client import generate_latest
from AppFiles.metrics import request_count

from fastapi.responses import PlainTextResponse

import pandas as pd
import io

# Import what is needed to create the relational database
from AppFiles.db import engine
from AppFiles.db_models import Base
Base.metadata.create_all(bind=engine)
from sqlalchemy.orm import Session
from AppFiles.db import SessionLocal
from AppFiles.db_models import UploadedData

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


FastAPI_Object = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@FastAPI_Object.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        #"result": None,
        "calculation": ""
    })

@FastAPI_Object.get("/health")
def health_check():
    print("✅ /health was hit")
    return {"status": "ok"}

@FastAPI_Object.post("/add", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    datafile: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        contents = await datafile.read()
        text = contents.decode("utf-8")

        # Save to DB
        db_entry = UploadedData(content=text)
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        # Now fetch the data back and use for prediction
        text = db_entry.content
        df = pd.read_csv(io.StringIO(text), header=None, delim_whitespace=True)
        number_list = df.values.flatten().tolist()

        # Model logic as before
        input_size = 30
        hidden_size1 = 16
        hidden_size2 = 8
        output_size = 2
        calculation = ComputeAppOutput(input_size, hidden_size1, hidden_size2, output_size, number_list)

    except Exception as e:
        traceback.print_exc()
        calculation = f"Error: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
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

# run only if this file is executed directly by the python interpreter
#   (not when imported as a module)
if __name__ == "__main__":
    # the port is set to be equal to the environment variable PORT
    #   if it is not set, it defaults to 8080
    port = int(os.environ.get("PORT", 8080))  # 8080 default for local
    uvicorn.run("AppFiles.main:FastAPI_Object", host="0.0.0.0", port=port)