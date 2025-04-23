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
from AppFiles.db_models import GeneExpression

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

from fastapi import UploadFile, File, Form, Request
from sqlalchemy.orm import Session
import pandas as pd
import io

@FastAPI_Object.post("/add", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    datafile: UploadFile = File(...),
    groupfile: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Read and decode both uploaded files
        expr_text = (await datafile.read()).decode("utf-8")
        group_text = (await groupfile.read()).decode("utf-8")

        # Load expression matrix (genes = rows, samples = columns)
        expr_df = pd.read_csv(io.StringIO(expr_text), sep="\t", index_col=0)

        # Load group assignments
        group_df = pd.read_csv(io.StringIO(group_text), sep="\t")
        group_map = group_df.set_index("sample")["group"].to_dict()


        # Convert to long format
        expr_long = expr_df.reset_index().melt(id_vars="Gene", var_name="sample", value_name="expression")
        #expr_long.rename(columns={"index": "gene"}, inplace=True)

        # Map group assignments
        expr_long["group"] = expr_long["sample"].map(group_map)

        # Drop any rows with missing group info or expression
        expr_long.dropna(subset=["group", "expression"], inplace=True)

        # Clear existing data (optional but recommended for test/demo)
        db.query(GeneExpression).delete()
        db.commit()

        # Insert into the database
        rows = [
            GeneExpression(
                gene=row['Gene'],
                sample=row['sample'],
                group=row['group'],
                expression=row['expression'],
            )
            for _, row in expr_long.iterrows()
        ]

        db.bulk_save_objects(rows)
        db.commit()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "calculation": f"Uploaded {len(rows)} gene expression records."
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "calculation": f"Error: {str(e)}"
        })


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


@FastAPI_Object.post("/predict", response_class=HTMLResponse)
async def run_model(request: Request, db: Session = Depends(get_db)):
    try:
        # Load all expression data
        data = db.query(GeneExpression).all()

        # Convert to DataFrame
        df = pd.DataFrame([{
            "gene": row.gene,
            "sample": row.sample,
            "expression": row.expression
        } for row in data])

        # Pivot to wide format: rows = samples, columns = genes
        wide_df = df.pivot(index="sample", columns="gene", values="expression").fillna(0)

        # OPTIONAL: align columns with model input
        # wide_df = wide_df[model_expected_columns]

        # Run your model (example assumes a PyTorch model)
        input_tensor = torch.tensor(wide_df.values, dtype=torch.float32)
        
        # Run your custom function
        predicted_class_names = ComputeAppOutput(input_tensor)

        # Map predictions to sample names
        sample_names = wide_df.index.tolist()
        result = dict(zip(sample_names, predicted_class_names))


        return templates.TemplateResponse("index.html", {
            "request": request,
            "predictions": result
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "calculation": f"Error running model: {str(e)}"
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