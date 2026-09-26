import argparse
from pathlib import Path
import torch
#contains the same operations as torch.nn, but as simple functions
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from constants import classes, norm_mean, norm_std, image_size
from model import SimpleCNN
def load_model(checkpoint_path, device):
    model = SimpleCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
#preprocessing means preparing the data before it gets to the model
def preprocess(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    image = Image.open(image_path).convert("RGB")
    #add batch dimension
    return transform(image).unsqueeze(0)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--top", type=int, default=len(classes))
    args = parser.parse_args()
    if not Path(args.image_path).exists():
        raise SystemExit(f"Image not found: {args.image_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    image_tensor = preprocess(args.image_path).to(device)
    with torch.no_grad():
        #logits are the brute scores of the network
        logits = model(image_tensor)
        #probs = probability
        #softmax transforms the scores in probabilities
        probs = F.softmax(logits, dim=1).squeeze(0)
    #top creates a ranking in probabilities and indices
    top_probs, top_indices = probs.topk(min(args.top, len(classes)))
    print(f"Predictions for {args.image_path}:")
    for prob, idx in zip(top_probs, top_indices):
        print(f"  {classes[idx.item()]:<12s} {prob.item() * 100:5.1f}%")
if __name__ == "__main__":
    main()
