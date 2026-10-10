#contextlib = context library
#asynccontextmanager is a decorator
from contextlib import asynccontextmanager
from datetime import datetime
#pathlib builds file paths that work on any operating system
from pathlib import Path
#typing from type, which represents a data type
from typing import Literal
#http exception deals with web errors
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
#staticfiles sends the files of the web page (html, css, js) to the browser
from fastapi.staticfiles import StaticFiles
#pil = python imaging library
from PIL import UnidentifiedImageError
#pydantic checks the data
from pydantic import BaseModel
from api import db, ml
#these classes describe what the API sends back
class Prediction(BaseModel):
    id: int
    filename: str
    label: Literal["cat", "dog"]
    confidence: float
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
    ml.load()
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
@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    try:
        label, confidence = ml.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The file is not an image.")
    return db.insert_prediction(file.filename or "upload", label, confidence)
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
