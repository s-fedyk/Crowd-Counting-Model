
import logging
import os
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.summary import image
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from CLIP import transform
from CLIPGCC import CLIPGCC
from losses import CrowdCountingLoss

from CLIP.factory import create_model_from_pretrained
from datasets import CrowdDataset, preprocess

def plot_sample(image: torch.Tensor, gt_map: torch.Tensor, pred_map: torch.Tensor):
    """
    Plots the original image along with the ground truth and predicted density maps.
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W]. Assumed to be a float tensor
                              (e.g. normalized to [0,1] or [0,255]) already on the CPU or moved via .cpu().
        gt_map (torch.Tensor): Ground truth density map of shape [1, H, W].
        pred_map (torch.Tensor): Predicted density map of shape [1, H, W].
    
    Returns:
        matplotlib.figure.Figure: Figure containing the three subplots.
    """
    # Convert tensors to NumPy arrays for plotting.
    # Permute the image tensor from [C, H, W] to [H, W, C].
    image_np = image.cpu().detach().permute(1, 2, 0).numpy()
    gt_density = gt_map.cpu().detach().squeeze().numpy()   # Shape [H, W]
    pred_density = pred_map.cpu().detach().squeeze().numpy()  # Shape [H, W]

    # Calculate counts by summing density values.
    gt_count = gt_density.sum()
    pred_count = pred_density.sum()

    # Create a figure with three subplots.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the original image.
    axs[0].imshow(image_np.astype('uint8') if image_np.max() > 1 else image_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Plot the ground truth density map.
    im1 = axs[1].imshow(gt_density, cmap='jet')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title(f"Ground Truth Density Map\nCount: {gt_count:.1f}")
    axs[1].axis("off")

    # Plot the predicted density map.
    im2 = axs[2].imshow(pred_density, cmap='jet')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set_title(f"Predicted Density Map\nCount: {pred_count:.1f}")
    axs[2].axis("off")

    plt.tight_layout()
    return fig

def load_model_from_checkpoint(checkpoint_path, clip_model_type='ViT-B/32', device='cuda'):
    """
    Load a trained CLIPGCC model from checkpoint for evaluation.
    
    Args:
        checkpoint_path (str): Path to the saved checkpoint file
        clip_model_type (str): CLIP model variant used during training (default: ViT-B/32)
        device (str): Device to load the model onto (default: cuda)
    
    Returns:
        CLIPGCC: Loaded model in evaluation mode
    """
    # Load CLIP model with the same configuration used during training
    clip_model, transforms = create_model_from_pretrained(clip_model_type, pretrained="openai", force_quick_gelu=True)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Initialize CLIPGCC model
    model = CLIPGCC(clip_model).to(device)
    
    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, transforms

def load_from_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """Load both model and optimizer states from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state if available
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-Guided Crowd Counting Training')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                      choices=['ViT-B/32', 'ViT-B/16'],
                      help='CLIP model variant to use')
    parser.add_argument('--eval-path', type=str, default='ShanghaiTech/part_B/test_data',
                      help='Input evaluation directory')
    parser.add_argument('--checkpoint-path', type=str, default='experiments/save.pth.tar',
                      help='Path of model to evaluate')


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, transforms = create_model_from_pretrained(args.clip_model, pretrained="openai", force_quick_gelu=True)
    clip_model = clip_model.to(device)
    clip_model.eval()

    model = CLIPGCC(clip_model).to(device)

    load_from_checkpoint(args.checkpoint_path, model)

    input_eval_path = f"./data/{args.eval_path}"
    processed_eval_path = f"./processed/eval_{args.eval_path}"
    if not os.path.exists(processed_eval_path):
        preprocess(input_eval_path, processed_eval_path)

    eval_dataset = CrowdDataset(root=processed_eval_path, patch_transform=transforms)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)

    total_abs_error = 0
    total_mape = 0
    total_images = 0
    epsilon = 1e-6  # small constant to avoid division by zero
    model.eval()
    model = model.to(device)

    # each iteration is for 1 image. An image has patches 
    # [image, image patches, gt_patches, gt_blur_patches]
    for batched_full_image, batched_image_patches, batched_gt_patches, _ in eval_dataloader:
        batched_image_patches = batched_image_patches.to(device)
        batched_gt_patches = batched_gt_patches.to(device)

        pred_maps = model(batched_image_patches.view(-1, *batched_image_patches.shape[-3:]))
        
        pred_counts = pred_maps.sum(dim=[1,2,3]).view(batched_image_patches.shape[0], -1).sum(dim=1)
        gt_counts = batched_gt_patches.sum(dim=[2,3,4]).squeeze(-1).sum(dim=1)
        
        # Calculate absolute errors for whole images
        batch_abs_error = torch.abs(pred_counts - gt_counts).sum().item()
        total_abs_error += batch_abs_error
        
        # Calculate MAPE for each image and sum them up
        batch_mape = (torch.abs(pred_counts - gt_counts) / (gt_counts + epsilon)).sum().item()
        total_mape += batch_mape
        
        total_images += batched_image_patches.size(0)  # Add actual number of images
        print(f"MAE: {total_abs_error/total_images:.2f} | MAPE: {(total_mape/total_images)*100:.2f}%")

    print(f"Total Images Evaluated: {total_images}")
    print(f"Final MAE: {total_abs_error/total_images:.2f}")
    print(f"Final MAPE: {(total_mape/total_images)*100:.2f}%")

