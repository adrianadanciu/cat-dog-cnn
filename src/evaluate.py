#helps change the settings and the number of epochs without changing the code
import argparse
import random
#works with folders' paths
from pathlib import Path
#for charts and figures
import matplotlib
#saves the figures into a image in storage
matplotlib.use("Agg")
#creates the figures and the axes
import matplotlib.pyplot as plt
import torch
from PIL import Image
#measures how good is a model
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from constants import classes
from dataset import ImageFolderDataset, get_transforms
from model import SimpleCNN
#devices decides where the calculations are made: CPU or GPU
#checkpoint represents the trained state of the model, with the trained weights
def load_model(checkpoint_path, device):
    #it creates the network and moves in onto the device
    #CNN = Convutional Neural Network
    model = SimpleCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #checkpoint has three keys: model_state_dict, epoch and val_acc
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
#decorator that stops the calculation of the gradients
@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        #argmax gives the position of the highest value in the pair
        #.cpu() moves the result on the processor
        preds = outputs.argmax(dim=1).cpu()
        #it adds the new data at the end of the list
        all_preds.append(preds)
        all_labels.append(labels)
        #return predictions and labels as numpy array
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()
#it shows what classes the model confuses with one another
def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    #cm is a matrix with the real values
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    #the size is in inches
    fig, ax = plt.subplots(figsize=(7, 6))
    #imshow = image show
    #0 is the lightest color and 1 the darkest
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    #puts the names cat and dog instead of 0 and 1 on the ticks
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            #saves the number on the line i and collum j in the confusion matrix
            value = cm[i, j]
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            #writes the number of photos in the center of the box
            #str(value) transforms the number into text
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)
    #set the chart next to ax
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #rearranges the charts in the figure, so nothing is overlapped
    fig.tight_layout()
    #draws the figure in pixels
    #dpi = dots per inch
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
#n is how many photos end up in the grid
#seed decides which set is plotted as sample
def plot_sample_predictions(model, dataset, device, save_path: Path, n=16, seed=0):
    #random.Random() is used so the code uses its own package
    rng = random.Random(seed)
    #sample chooses a few elements from the list, so they are not repeated
    indices = rng.sample(range(len(dataset)), n)
    #the first two parameters represent the lines and columns
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    #zip goes through two lists at the same time and matches them item by item
    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        #it stops the model from training, so only predictions are given
        with torch.no_grad():
            #unsqueeze so the photo goes in batch form
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
        #show the original photo from disk, not the normalized tensor
        path = dataset.samples[idx][0]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
        is_correct = pred == label
        color = "seagreen" if is_correct else "firebrick"
        ax.set_title(f"true: {classes[label]}\npred: {classes[pred]}",
                      fontsize=9, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
def main():
    #parsing = breaking down text into data
    #parsing rules
    parser = argparse.ArgumentParser(description=__doc__) #creates the reader
    #add fields in the form
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    #here is where the text is actually parsed
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints" / "best_model.pt"
    #cuda = video boards
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    #get_transforms resizes, cuts photo etc
    test_ds = ImageFolderDataset(args.data_dir, "test", transform=get_transforms(train=False))
    #num_workes is how many helping processed help data loader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    y_pred, y_true = collect_predictions(model, test_loader, device)
    accuracy = (y_pred == y_true).mean()
    print(f"Test accuracy: {accuracy:.4f}")
    #digits = number of decimals
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print("\nPer-class report:")
    print(report)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    plot_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png")
    plot_sample_predictions(model, test_ds, device, figures_dir / "sample_predictions.png")
#so the programmer doesn't have to run the test by hand
if __name__ == "__main__":
    main()