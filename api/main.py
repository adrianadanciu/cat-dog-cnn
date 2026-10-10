#contextlib = context library
#asynccontextmanager is a decorator
from contextlib import asynccontextmanager
from datetime import datetime
import io
#pathlib builds file paths that work on any operating system
from pathlib import Path
#typing from type, which represents a data type
from typing import Literal
#http exception deals with web errors
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
#staticfiles sends the files of the web page (html, css, js) to the browser
from fastapi.staticfiles import StaticFiles
#pil = python imaging library
from PIL import Image, UnidentifiedImageError
#pydantic checks the data
from pydantic import BaseModel
#the server doesn't run the model anymore, the workers do (api/worker.py)
from api import db
#these classes describe what the API sends back
class Prediction(BaseModel):
    id: int
    filename: str
    status: Literal["pending", "processing", "done", "failed"]
    #None until a worker classifies the photo
    label: Literal["cat", "dog"] | None
    confidence: float | None
    worker: str | None
    created_at: datetime
class LabelStats(BaseModel):
    label: str
    count: int
    avg_confidence: float
#parts the function lifespan into two parts: one which happens when the server starts
#and one when it ends
@asynccontextmanager
async def lifespan(app: FastAPI):
    #runs once, when the server starts
    db.pool.open(wait=True, timeout=10)
    db.create_schema()
    #stops the function at half so it can be continued later
    yield
    #runs once, when the server stops
    db.pool.close()
#the app is created
#lifespan is the function that starts/stops the app
app = FastAPI(title="Cat vs. Dog API", lifespan=lifespan)
#connects the function health with the health adress
@app.get("/health")
def health():
    return {"status": "ok"}
#the api takes what the function returns and runs it through the Prediction class
#the server only checks that the file is a photo and puts it in the queue
#202 = accepted
@app.post("/predict", response_model=Prediction, status_code=202)
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    try:
        Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The file is not an image.")
    return db.add_to_queue(file.filename or "upload", image_bytes)
@app.get("/predictions", response_model=list[Prediction])
def list_predictions(limit: int = Query(20, ge=1, le=100)):
    return db.list_predictions(limit)
@app.get("/predictions/{prediction_id}", response_model=Prediction)
def get_prediction(prediction_id: int):
    row = db.get_prediction(prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return row
@app.get("/stats", response_model=list[LabelStats])
def stats():
    return db.get_stats()
#the folder with the web page
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
#the page is mounted last, so the api routes above are checked first
#html=True means that the address "/" returns index.html
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
