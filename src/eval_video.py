import argparse
import os
import torch
import numpy as np
from CLIPGCC import UNet
from PIL import Image
import cv2
from torchvision import transforms
from tqdm import tqdm


def load_model(checkpoint_path, device):
    """Load UNet model from checkpoint"""
    model = UNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


def get_transform(resize_size, mean, std):
    """Create image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def process_frame(frame_path, model, transform, device):
    """Process a single frame through the model"""
    image = Image.open(frame_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Apply transformations
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        density = model(input_tensor)

    # Process density map
    density = density.squeeze().cpu().numpy()
    density_resized = cv2.resize(
        density, original_size, interpolation=cv2.INTER_LINEAR)
    count = density_resized.sum()

    return image, density_resized, count


def overlay_heatmap(image, density):
    """Create heatmap overlay visualization"""
    img_np = np.array(image)
    # Normalize density map
    density_normalized = (density - density.min()) / \
        (density.max() - density.min() + 1e-6)

    # Apply color map
    heatmap = cv2.applyColorMap(
        np.uint8(255 * density_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    alpha = 0.5
    overlay = cv2.addWeighted(heatmap, alpha, img_np, 1 - alpha, 0)
    return overlay


def annotate_frame(frame_np, count):
    """Add count text annotation to frame"""
    text = f"Count: {count:.1f}"
    cv2.putText(frame_np, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame_np


def create_video(frame_dir, output_path, model, transform, device, fps=30):
    """Process all frames and compile into video"""
    # Sort frames numerically
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x.split('frame')[1].split('.')[0]))

    # Get video dimensions from first frame
    first_frame = Image.open(os.path.join(frame_dir, frame_files[0]))
    width, height = first_frame.size

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_path = os.path.join(frame_dir, frame_file)

        # Get predictions
        image, density, count = process_frame(
            frame_path, model, transform, device)

        # Create visualization
        overlay = overlay_heatmap(image, density)
        annotated_frame = annotate_frame(overlay, count)

        # Convert to BGR and write to video
        frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Video crowd counting evaluation')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing input frames')
    parser.add_argument('--checkpoint', required=True,
                        help='Model checkpoint path')
    parser.add_argument('--output-video', required=True,
                        help='Output video path')
    parser.add_argument('--dataset-part', choices=['A', 'B'], default='B',
                        help='Dataset part for normalization (A/B)')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Configure normalization
    NORMALIZATIONS = {
        'A': ([0.411, 0.373, 0.364], [0.284, 0.276, 0.279]),  # SHA_NORM
        'B': ([0.451, 0.446, 0.431], [0.237, 0.229, 0.226])   # SHB_NORM
    }
    mean, std = NORMALIZATIONS[args.dataset_part]

    # Create transformation
    transform = get_transform((448, 448), mean, std)

    # Generate video
    create_video(args.input_dir, args.output_video,
                 model, transform, device, args.fps)
    print(f"Video saved to {args.output_video}")
