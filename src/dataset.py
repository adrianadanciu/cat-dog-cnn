from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from constants import classes, norm_mean, norm_std, image_size
class ImageFolderDataset(Dataset):
    #dunder methods
    def __init__(self, root, split, transform=None):
        #creates the pathway to a split folder
        self.root = Path(root) / split
        #it transforms the raw data, as the network accepts only tensors of number
        self.transform = transform
        self.samples = []
        for label_idx, class_name in enumerate(classes):
            class_dir = self.root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected folder {class_dir}"
                )
            #class_dir.glob("*.jpg") searches for all the .jpg files
            for path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((path, label_idx))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        #self.samples[idx] is a pair
        path, label = self.samples[idx]
        #it opens the photo and converts is to rgb
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
#train: bool
#if train = true, Resize, RandomCrop, flip, ToTensor, normalize
#else Resize, ToTensor, normalize
#toTensor transform the image into a tensor, 
def get_transforms(train: bool):
    normalize = transforms.Normalize(norm_mean, norm_std)
    if train:
        #Compose gets a line of steps and turns it into a single one
        return transforms.Compose([
            #so the model learns to figure out cats or dogs whichout seeing
            #the entire animal
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
