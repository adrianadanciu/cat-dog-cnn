import argparse
#the library is used to save the historical outputs
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ImageFolderDataset, get_transforms
from model import SimpleCNN
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
#criterion = loss function
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                #puts 0 on the saved gradients for each weight
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                #calculates the weights for the current batch
                loss.backward()
                #adjusts the gradients
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    train_ds = ImageFolderDataset(args.data_dir, "train", transform=get_transforms(train=True))
    val_ds = ImageFolderDataset(args.data_dir, "val", transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    model = SimpleCNN().to(device)
    #CrossEntropyLoss gives out a number: if it's smaller, the prediction is more accurate
    criterion = nn.CrossEntropyLoss()
    #Adam is the optimizer
    #Adam = Adaptive Moment Estimation
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #schedulere represents the rate of learning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = output_dir / "checkpoints" / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc}, best_path)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              #flush empties the buffer memory
              f"best={best_val_acc:.4f} | {elapsed:.1f}s", flush=True)
        if is_best:
            print("New best model saved")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, output_dir / "figures" / "training_curves.png")
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
def plot_training_curves(history: dict, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
if __name__ == "__main__":
    main()
