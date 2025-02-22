import os
import zipfile
import scipy.io
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms


class CrowdDataset(Dataset):
    """
    A generic PyTorch Dataset for crowd counting.

    Assumes that the root directory has two subdirectories:
      - 'images': containing the input images (e.g. .jpg).
      - 'ground-truth': containing corresponding ground truth annotations as .mat files.

    Ground truth .mat files are expected to be named "GT_<base>.mat" and store head coordinates 
    under a key (e.g., "image_info" or "annPoints").
    The loader converts these coordinates into a binary point map where pixels with a head are 1,
    and 0 otherwise.
    """

    def __init__(self, root, transform=None, gt_transform=None,
                 image_extensions=('.jpg', '.jpeg', '.png'),
                 gt_extensions=('.mat',)):
        """
        Args:
            root (str): Root directory of the dataset.
            transform (callable, optional): Transform to apply to the images.
            gt_transform (callable, optional): Transform to apply to ground truth.
            image_extensions (tuple): Allowed image file extensions.
            gt_extensions (tuple): Allowed ground truth file extensions.
        """
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.image_paths = []
        self.gt_paths = []

        images_dir = os.path.join(root, "images")
        gt_dir = os.path.join(root, "ground-truth")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        if not os.path.exists(gt_dir):
            raise ValueError(f"Ground truth directory not found: {gt_dir}")

        for fname in os.listdir(images_dir):
            if fname.lower().endswith(image_extensions):
                image_path = os.path.join(images_dir, fname)
                base, _ = os.path.splitext(fname)

                gt_path = None
                for ext in gt_extensions:
                    candidate = os.path.join(gt_dir, "GT_" + base + ext)
                    print(candidate)
                    if os.path.exists(candidate):
                        gt_path = candidate
                        break
                if gt_path is None:
                    print(
                        f"Warning: Ground truth for {fname} not found. Skipping.")
                    continue
                self.image_paths.append(image_path)
                self.gt_paths.append(gt_path)

        print(
            f"Found {len(self.image_paths)} images with ground truth in {root}")

    def __len__(self):
        return len(self.image_paths)

    def points_to_point_map(self, points, shape):
        point_map = np.zeros(shape, dtype=np.float32)
        H, W = shape

        for point in points[0]:
            x, y = int(round(point[0])), int(round(point[1]))
            if x < W and y < H:
                point_map[y, x] = 1.0

        return point_map

    def load_gt(self, gt_path, original_size, target_size=(224, 224)):
        mat = scipy.io.loadmat(gt_path)
        # Adjust extraction based on .mat structure.
        if "image_info" in mat:
            points = mat["image_info"][0, 0][0, 0]
        elif "annPoints" in mat:
            points = mat["annPoints"]
        else:
            raise ValueError(
                "Ground truth format not recognized in " + gt_path)

        orig_shape = (original_size[1], original_size[0])
        point_map_np = self.points_to_point_map(points, orig_shape)

        # Convert numpy array to torch tensor with shape [1, H, W]
        point_map_tensor = torch.from_numpy(point_map_np).unsqueeze(0)

        # Add a batch dimension so we can interpolate: shape [1, 1, H, W]
        point_map_tensor = point_map_tensor.unsqueeze(0)

        point_map_resized = F.interpolate(
            point_map_tensor, size=target_size, mode='nearest')

        point_map_resized = point_map_resized.squeeze(0)
        return point_map_resized

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        if self.transform:
            image = self.transform(image)

        gt_path = self.gt_paths[idx]
        gt = self.load_gt(gt_path, original_size, target_size=(224, 224))
        if self.gt_transform:
            gt = self.gt_transform(gt)

        return image, gt
