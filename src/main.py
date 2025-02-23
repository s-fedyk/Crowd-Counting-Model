
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from CLIPGCC import CLIPGCC, CrowdCountingLoss
from CLIP.tokenizer import tokenize
from CLIP.factory import create_model_and_transforms, create_model_from_pretrained
from datasets import CrowdDataset, preprocess

def plot_sample(image, gt_map, pred_map):
    """
    Plots the original image with overlayed ground truth and predicted points.
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W].
        gt_map (torch.Tensor): Ground truth binary point map of shape [1, H, W].
        pred_map (torch.Tensor): Predicted binary point map of shape [1, H, W].
    """
    # Convert image to numpy, shape [H, W, C]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Squeeze maps to [H, W]
    gt_np = gt_map.squeeze().cpu().numpy()
    pred_np = pred_map.squeeze().cpu().numpy()
    
    # Extract point coordinates: (row, col)
    gt_points = np.argwhere(gt_np > 0.5)
    pred_points = np.argwhere(pred_np > 0.5)
    
    gt_count = int(gt_np.sum())
    pred_count = int(pred_np.sum())
    # Plot image with ground truth points
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    if gt_points.size > 0:
        plt.scatter(gt_points[:, 1], gt_points[:, 0], c='green', marker='o', label='GT Points')
    plt.title(f"Ground Truth (Count = {gt_count})")
    plt.axis("off")
    plt.legend()
    
    # Plot image with predicted points
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    if pred_points.size > 0:
        plt.scatter(pred_points[:, 1], pred_points[:, 0], c='red', marker='x', label='Predicted Points')
    plt.title(f"Prediction (Count = {pred_count})")
    plt.axis("off")
    plt.legend()
    
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CLIP model and associated transforms.
    clip_model, img_transforms = create_model_from_pretrained("ViT-B/32", pretrained="openai")
    clip_model.to(device)
    clip_model.eval()  # Freeze the CLIP backbone for now

    # Create the CLIP-guided crowd counting model.
    clipgcc_model = CLIPGCC(clip_model).to(device)
    clipgcc_model.train()


    #preprocess("./data/ShanghaiTech/part_B/train_data", "./processed/train_dataset")
    # Training dataset
    train_dataset_root = "./processed/train_dataset"  # Adjust as needed
    train_dataset = CrowdDataset(root=train_dataset_root, transform=img_transforms)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    #preprocess("./data/ShanghaiTech/part_B/train_data"  , "./processed/eval_dataset")
    # Evaluation dataset
    eval_dataset_root = "./processed/eval_dataset"  # Adjust as needed
    eval_dataset = CrowdDataset(root=eval_dataset_root, transform=img_transforms)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Loss and optimizer.
    loss_fn = CrowdCountingLoss()
    optimizer = optim.Adam(clipgcc_model.parameters(), lr=1e-4)

    num_epochs = 900
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, gt_maps in dataloader:
            images = images.to(device)    # [B, 3, 224, 224]
            gt_maps = gt_maps.to(device)    # [B, 1, 224, 224]

            optimizer.zero_grad()
            pred_map = clipgcc_model(images)  # [B, 1, 224, 224]
            loss = loss_fn(pred_map, gt_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # --- Evaluation on Eval Dataset: Compute MAE ---
        clipgcc_model.eval()
        total_abs_error = 0.0
        total_images = 0
        with torch.no_grad():
            for images, gt_maps in eval_dataloader:
                images = images.to(device)
                gt_maps = gt_maps.to(device)
                pred_map = clipgcc_model(images)  # [B, 1, 224, 224]
                pred_count = pred_map.sum(dim=[1,2,3])
                gt_count = gt_maps.sum(dim=[1,2,3])
                total_abs_error += torch.sum(torch.abs(pred_count - gt_count)).item()
                total_images += images.size(0)
        mae = total_abs_error / total_images
        print(f"Epoch [{epoch+1}/{num_epochs}] Evaluation MAE: {mae:.2f}")

        # --- Every 5 Epochs: Display a couple evaluation samples ---
        if True: 
            clipgcc_model.eval()
            with torch.no_grad():
                # Let's pick the first 2 samples from the eval dataset.
                for i in range(10):
                    image, gt_map = eval_dataset[i]
                    image_tensor = image.unsqueeze(0).to(device)  # [1, 3, 224, 224]
                    pred_map = clipgcc_model(image_tensor)  # [1, 1, 224, 224]
                    print(f"Epoch {epoch+1}: Sample {i+1} predicted count: {pred_map.sum().item():.2f}, real count: {gt_map.sum().item():.2f}")
            clipgcc_model.train()
        # Switch back to train mode.
        clipgcc_model.train()

