import os
import zipfile
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ShanghaiTechDataset(Dataset):
    """
    A PyTorch Dataset for the ShanghaiTech crowd counting dataset.
    Assumes the following folder structure:
      <root>/part_A/test_data/images/ and <root>/part_B/test_data/images/
    """

    def __init__(self, root, transform=None):
        self.transform = transform
        self.image_paths = []

        print("\n\nBuilding dataset...\n\n")
        for part in ["part_A", "part_B"]:
            images_dir = os.path.join(root, part, "test_data/images")
            print("Looking in:", images_dir)
            if os.path.isdir(images_dir):
                for fname in os.listdir(images_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(
                            os.path.join(images_dir, fname))
                        print("Found image:", fname)
        print(
            f"Found {len(self.image_paths)} images in the ShanghaiTech dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
