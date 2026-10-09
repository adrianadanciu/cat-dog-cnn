"""Loads the trained CNN once and runs it on the photos sent to the API."""
#io = input/output
import io
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
#src/ is added
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
from constants import classes, image_size, norm_mean, norm_std 
from predict import load_model
#creates the complete path to the trained model
CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "checkpoints" / "best_model.pt"
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
model = None
def load():
    global model
    model = load_model(CHECKPOINT_PATH, device)
def predict(image_bytes: bytes) -> tuple[str, float]:
    """Returns the predicted class and its probability."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)
    confidence, index = probs.max(dim=0)
    return classes[index.item()], confidence.item()
