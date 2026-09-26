import argparse
import random
#shutil = shell utilities
import shutil
#starts other programs inside this programs
import subprocess
#sys = system
import sys
#used for temporary files
import tempfile
from pathlib import Path
from constants import classes
#25000 photos of cats and dogs
source_repo = "https://github.com/laxmimerit/dog-cat-full-dataset.git"
#the source repo uses plural folder names ("cats", "dogs") under data/train
#and data/test
source_folder = {"cat": "cats", "dog": "dogs"}
#a tuple is an organized set of values
def get_source_dir() -> Path:
    #cloning = downloading a version of a repo on my computer
    tmp_dir = Path(tempfile.mkdtemp(prefix="dogcat_source_"))
    subprocess.run(
        ["git", "clone", "--depth", "1", source_repo, str(tmp_dir)],
        check=True,
    )
    return tmp_dir
def list_images(class_dir: Path) -> list[Path]:
    return list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
def sample_and_copy(files: list[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for src_file in files:
        #"/" puts together pieces of a path
        shutil.copy(src_file, destination / src_file.name)
    return len(files)
#seed is used for random numbers
def build_test_split(source_dir: Path, output_dir: Path, per_class: int, seed: int):
    rng = random.Random(seed)
    for class_name in classes:
        class_dir = source_dir / "data" / "test" / source_folder[class_name]
        available = list_images(class_dir)
        if len(available) < per_class:
            sys.exit(f"Only {len(available)} test images for '{class_name}', but {per_class} requested.")
        rng.shuffle(available)
        chosen = available[:per_class]
        n_copied = sample_and_copy(chosen, output_dir / "test" / class_name)
        print(f"test/{class_name}: {n_copied} images")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"Removing existing {output_dir} so re-runs start from a clean, consistent state")
        shutil.rmtree(output_dir)
    source_dir = get_source_dir()
    try:
        rng = random.Random(args.seed)
        pools = {}
        for class_name in classes:
            class_dir = source_dir / "data" / "train" / source_folder[class_name]
            available = list_images(class_dir)
            rng.shuffle(available)
            #pool is used as a reserve
            pools[class_name] = available
            #needed is how many photos are needed per class for training
            needed = args.train_per_class + args.val_per_class
            if len(available) < needed:
                sys.exit(f"Only {len(available)} train images for '{class_name}', but {needed} needed.")
            train_pool = available[:args.train_per_class]
            n_copied = sample_and_copy(train_pool, output_dir / "train" / class_name)
            print(f"train/{class_name}: {n_copied} images")
        for class_name in classes:
            #val exists because the source only has two values: test and train
            #so the values for validation are taken from the training photos
            val_pool = pools[class_name][args.train_per_class: args.train_per_class + args.val_per_class]
            n_copied = sample_and_copy(val_pool, output_dir / "val" / class_name)
            print(f"val/{class_name}: {n_copied} images")
        build_test_split(source_dir, output_dir, args.test_per_class, args.seed)
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)
if __name__ == "__main__":
    main()
