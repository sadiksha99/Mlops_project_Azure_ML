import os
import argparse
import logging
from glob import glob
import math
import random
from shutil import copyfile
import re

def copy_with_label_folder(images, destination_root, label_folder):
    label_dir = os.path.join(destination_root, label_folder)
    os.makedirs(label_dir, exist_ok=True)
    for img_path in images:
        dest = os.path.join(label_dir, os.path.basename(img_path))
        copyfile(img_path, dest)

def main():
    SEED = 42

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", help="List of dataset folder paths")
    parser.add_argument("--training_data_output", type=str, help="Path to output training images")
    parser.add_argument("--testing_data_output", type=str, help="Path to output testing images")
    parser.add_argument("--split_size", type=int, help="Test percentage (e.g. 20)")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    os.makedirs(args.training_data_output, exist_ok=True)
    os.makedirs(args.testing_data_output, exist_ok=True)

    train_test_split_factor = args.split_size / 100
    total_train, total_test = 0, 0

    for dataset in args.datasets:
        images = glob(os.path.join(dataset, "*.jpg"))
        print(f"Found {len(images)} images in {dataset}")

        random.seed(SEED)
        random.shuffle(images)

        # Extract digit from folder name like 'mnist_2_jpg' -> '2' -> 'mnist-2'
        match = re.search(r'\d+', os.path.basename(dataset))
        label_folder = f"mnist-{match.group()}" if match else "unknown"
        print(f"Using label folder: {label_folder}")

        num_test = math.ceil(len(images) * train_test_split_factor)
        test_images = images[:num_test]
        train_images = images[num_test:]

        copy_with_label_folder(train_images, args.training_data_output, label_folder)
        copy_with_label_folder(test_images, args.testing_data_output, label_folder)

        total_train += len(train_images)
        total_test += len(test_images)

    print(f"✅ Done! Wrote {total_train} training and {total_test} testing images into label folders.")

if __name__ == "__main__":
    main()
