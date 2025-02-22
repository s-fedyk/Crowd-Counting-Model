
from CLIPGCC import *
from CLIP.tokenizer import tokenize
from CLIP.factory import create_model_and_transforms, create_model_from_pretrained
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import os
import zipfile
from datasets import CrowdDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, img_transforms = create_model_from_pretrained(
        "ViT-B/32", pretrained="openai")

    clip_model.to(device)
    clip_model.eval()

    clipgcc_model = CLIPGCC(clip_model).to(device)
    clipgcc_model.train()

    dataset_root = "./data/TEAMVISION"  # Adjust to your dataset root
    dataset = CrowdDataset(root=dataset_root, transform=img_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    loss_fn = CrowdCountingLoss()

    optimizer = optim.Adam(clipgcc_model.parameters(), lr=1e-4)

    num_epochs = 900
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, gt_maps in dataloader:
            # Move data to device.
            images = images.to(device)   # Shape: [B, 3, 224, 224]
            # Shape: [B, 1, 224, 224] (binary point maps)
            gt_maps = gt_maps.to(device)

            optimizer.zero_grad()
            # Forward pass: get predicted "density" (or point) map.
            # Expected shape: [B, 1, 224, 224]
            pred_map = clipgcc_model(images)

            # Compute the count loss: compare the total count per image.
            loss = loss_fn(pred_map, gt_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
