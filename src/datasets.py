import os
import zipfile
import scipy.io
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

"""
For verification.
"""
def plot_sample(image, gt_map):
    image = image.permute(1, 2, 0).cpu().numpy()
   
    # Squeeze maps to [H, W]
    gt_np = gt_map.squeeze().cpu().numpy()
    
    # Extract point coordinates: (row, col)
    gt_points = np.argwhere(gt_np > 0.5)
    
    gt_count = int(gt_np.sum())
    # Plot image with ground truth points
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    if gt_points.size > 0:
        plt.scatter(gt_points[:, 1], gt_points[:, 0], c='green', marker='o', label='GT Points')
    plt.title(f"Ground Truth (Count = {gt_count})")
    plt.axis("off")
    plt.legend()
    
    # Plot image with predicted points
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.axis("off")
    plt.legend()
    
    plt.show()


def points_to_point_map(points, shape):
    """
    Converts an array of head coordinates into a binary point map.
    
    Args:
        points (np.array): Array of shape [N, 2] containing (x, y) coordinates.
        shape (tuple): (height, width) of the output map.
    Returns:
        point_map (np.array): Binary map with 1's at annotated head positions.
    """
    point_map = np.zeros(shape, dtype=np.float32)

    H, W = shape
    for point in points[0]:
        x, y = int(round(point[0])), int(round(point[1]))
        if x < W and y < H:
            point_map[y, x] = 1.0  # note: indexing is [row, col] = [y, x]
    return point_map

def load_gt_from_mat(gt_path, original_size):
    """
    Loads the .mat file and extracts the raw head coordinates,
    then converts them into a binary point map at full resolution.
    
    Args:
        gt_path (str): Path to the .mat file.
        original_size (tuple): (width, height) of the original image.
    Returns:
        point_map (np.array): Binary point map of shape (height, width).
    """
    mat = scipy.io.loadmat(gt_path)
    if "image_info" in mat:
        points = mat["image_info"][0, 0][0, 0]
    elif "annPoints" in mat:
        points = mat["annPoints"]
    else:
        raise ValueError("Ground truth format not recognized in " + gt_path)
    
    # If stored as a 3D array [1, N, 2], use the first element.
    if points.ndim == 3:
        points = points[0]
    # original_size is (width, height); convert to (height, width)
    orig_shape = (original_size[1], original_size[0])
    point_map = points_to_point_map(points, orig_shape)
    return point_map

def split_into_patches(arr, patch_size):
    """
    Splits a NumPy array (an image or ground truth) into non-overlapping patches.
    
    Args:
        arr (np.array): Array of shape (H, W, C) for images or (H, W) for GT.
        patch_size (tuple): Desired patch size (patch_h, patch_w).
        
    Returns:
        patches (list): List of patches as NumPy arrays.
    """
    patches = []
    H, W = arr.shape[:2]
    ph, pw = patch_size
    for i in range(0, H, ph):
        for j in range(0, W, pw):
            if i + ph <= H and j + pw <= W:
                patch = arr[i:i+ph, j:j+pw]
                patches.append(patch)
    return patches

def preprocess(root, processed_dir, patch_size=(224,224),
               image_extensions=('.jpg', '.jpeg', '.png'),
               gt_extensions=('.mat',)):
    """
    Preprocesses the dataset at 'root' by splitting high-resolution images and their
    ground truth annotations into patches of size patch_size, then saving them in
    a processed directory.
    
    The images are saved as JPEG files and the ground truth patches as .npy files.
    """
    images_dir = os.path.join(root, "images")
    gt_dir = os.path.join(root, "ground-truth")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(gt_dir):
        raise ValueError(f"Ground truth directory not found: {gt_dir}")
    
    # Create processed directories for images and ground truth.
    proc_images_dir = os.path.join(processed_dir, "images")
    proc_gt_dir = os.path.join(processed_dir, "ground-truth")
    os.makedirs(proc_images_dir, exist_ok=True)
    os.makedirs(proc_gt_dir, exist_ok=True)
    
    # List image files.
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
    print(f"Found {len(image_files)} images.")
    
    for fname in tqdm(image_files, desc="Preprocessing"):
        image_path = os.path.join(images_dir, fname)
        base, _ = os.path.splitext(fname)
        
        # Find corresponding ground truth .mat file.
        gt_path = None
        for ext_gt in gt_extensions:
            candidate = os.path.join(gt_dir, "GT_" + base + ext_gt)
            if os.path.exists(candidate):
                gt_path = candidate
                break
        if gt_path is None:
            print(f"Warning: GT for {fname} not found. Skipping.")
            continue
        
        # Load the image.
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        image_np = np.array(image)
        
        # Load the ground truth and create a binary point map.
        gt_map = load_gt_from_mat(gt_path, original_size)  # shape: (H, W)
        
        # Split the image and ground truth into patches.
        image_patches = split_into_patches(image_np, patch_size)
        gt_patches = split_into_patches(gt_map, patch_size)
        
        # Save each patch.
        for idx, (img_patch, gt_patch) in enumerate(zip(image_patches, gt_patches)):
            # Save image patch as JPEG.
            patch_img = Image.fromarray(img_patch)
            img_patch_filename = f"{base}_patch_{idx}.jpg"
            img_patch_filepath = os.path.join(proc_images_dir, img_patch_filename)
            patch_img.save(img_patch_filepath, quality=95)
            
            # Save ground truth patch as .npy (matrix).
            gt_patch_filename = f"{base}_patch_{idx}.npy"
            gt_patch_filepath = os.path.join(proc_gt_dir, gt_patch_filename)
            np.save(gt_patch_filepath, gt_patch)
            
    print("Preprocessing complete. Processed data saved to", processed_dir)

class CrowdDataset(Dataset):
    """
    A generic PyTorch Dataset for crowd counting.

    Assumes that the root directory has two subdirectories:
      - 'images': containing the input images (e.g. .jpg).
      - 'ground-truth': containing corresponding ground truth annotations as .mat files.

    """

    def __init__(self, root, transform=None, gt_transform=None,
                 image_extensions=('.jpg', '.jpeg', '.png'),
                 gt_extensions=('.npy',)):
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
                    candidate = os.path.join(gt_dir,base + ext)
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
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        gt_np = np.load(gt_path)  # This should be a 2D array of shape [H, W] with binary values.
        gt = torch.from_numpy(gt_np).float()

        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, gt
