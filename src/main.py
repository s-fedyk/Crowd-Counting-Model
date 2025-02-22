
import kagglehub
from CLIPGCC import *
from CLIP.tokenizer import tokenize
from CLIP.factory import create_model_and_transforms, create_model_from_pretrained

from torch.utils.data import Dataset, DataLoader
import os
import zipfile
from datasets import ShanghaiTechDataset

print("Downloading ShanghaiTech dataset from KaggleHub...")
dataset_path = kagglehub.dataset_download("tthien/shanghaitech")

# Check if dataset_path is a directory or a zip file.
if os.path.isdir(dataset_path):
    print(f"Dataset already downloaded and extracted at: {dataset_path}")
    dataset_dir = dataset_path
else:
    # If it's not a directory, assume it's a zip file and extract it.
    dataset_dir = "shanghaitech_extracted"

    def extract_zip(zip_path, extract_to):
        if os.path.exists(extract_to):
            print(f"{extract_to} already exists, skipping extraction.")
            return extract_to
        print(f"Extracting {zip_path} to {extract_to} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in tqdm(zf.infolist(), desc="Extracting", unit="file"):
                zf.extract(member, extract_to)
        return extract_to

    dataset_dir = extract_zip(dataset_path, dataset_dir)

# --- Dataset Class ---


# --- Main Execution ---
if __name__ == "__main__":
    # Load the CLIP model and its associated preprocessing pipeline.
    # This loads the ViT-B/32 weights with the "openai" configuration.
    model, preprocess = create_model_from_pretrained(
        "ViT-B/32", pretrained="openai")
    print("CLIP model and transforms are ready.")

    # Determine dataset root (check for inner "ShanghaiTech" folder)
    potential_subdir = os.path.join(dataset_dir, "ShanghaiTech")
    if os.path.exists(potential_subdir):
        dataset_root = potential_subdir
    else:
        dataset_root = dataset_dir

    # Create the dataset with the CLIP preprocessing transform.
    shtech_dataset = ShanghaiTechDataset(dataset_root, transform=preprocess)
    dataloader = DataLoader(shtech_dataset, batch_size=4, shuffle=True)

    # For demonstration, iterate through one batch and print info.
    for images, paths in dataloader:
        print("Batch image tensor shape:", images.shape)
        print("Image paths:", paths)
        break

    # --- Test Standard CLIP Inference ---
    # Use the first image to test image-to-text similarity.
    if len(shtech_dataset) == 0:
        raise RuntimeError(
            "No images found in the dataset. Check your dataset directory structure.")

    sample_image, sample_path = shtech_dataset[2]
    sample_image = sample_image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image_features, _ = model.encode_image(sample_image)
        print(image_features)
    print(
        f"\nCLIP image features for {sample_path}: shape {image_features.shape}")

    prompt = "a busy street"
    text_tokens = tokenize([prompt])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    print(
        f"CLIP text features for prompt '{prompt}': shape {text_features.shape}")

    # Normalize features and compute cosine similarity.
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).item()
    print(
        f"\nCosine similarity between the image and prompt '{prompt}': {similarity:.4f}")

    # --- Integrate and Test CLIPGCC for Crowd Counting ---
    # Instantiate the CLIP-guided crowd counting model using the CLIP model.
    model.visual.output_tokens = True
    clipgcc_model = CLIPGCC(model)
    with torch.no_grad():
        density_map = clipgcc_model(sample_image)
    print(
        f"\nDensity map from CLIPGCC for {sample_path}: shape {density_map.shape}")
