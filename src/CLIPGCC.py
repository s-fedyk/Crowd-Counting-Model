import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Dropout

from CLIP.tokenizer import SimpleTokenizer

# Helper: custom LayerNorm for 2D conv features.
# It permutes the tensor so that normalization is applied on the channel dimension.
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x

# ConvNeXt Block: uses a depthwise convolution, layer norm in channel-last,
# and two linear layers (point-wise convolutions) with GELU activation.
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise convolution
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # applied on channel-last tensor
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/linear layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # A learnable scale parameter initialized to a small value
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)  # (B, C, H, W)
        # Permute to (B, H, W, C) for layer norm and linear layers
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # Add the residual (after permuting the residual to channel-last)
        x = x + residual.permute(0, 2, 3, 1)
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x

# ConvNeXt-based segmentation model.
# It is organized in multiple stages with intermediate downsampling layers.
# The final head produces prediction maps, which are then upsampled to the input resolution.
class ConvNeXtSegmentation(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        """
        Args:
            in_channels: Number of channels in the input image/features.
            num_classes: Number of prediction channels (e.g. 1 for binary segmentation).
            depths: Number of ConvNeXt blocks in each stage.
            dims: Feature dimensions for each stage.
        """
        super().__init__()
        # Stem: initial downsampling via a conv layer with kernel 4 and stride 4,
        # followed by a normalization.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Stage 1 (after stem)
        self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])]))

        # Stage 2
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm2d(dims[0]),
                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
            )
        )
        self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])]))

        # Stage 3
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm2d(dims[1]),
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
            )
        )
        self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(depths[2])]))

        # Stage 4
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm2d(dims[2]),
                nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
            )
        )
        self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(depths[3])]))

        # Segmentation head: a few conv layers to produce prediction maps.
        self.head = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1], num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # x: input tensor of shape [B, in_channels, H, W]
        x = self.stem(x)  # downsample by factor of 4
        x = self.stages[0](x)
        # Each downsample layer further halves the spatial dimensions.
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i+1](x)
        x = self.head(x)
        # The overall downsampling factor is 4 * 2^3 = 32. Upsample back to input resolution.
        x = F.interpolate(x, scale_factor=32, mode="bilinear", align_corners=False)
        return x

def reshape_tokens_to_grid(tokens):
    B, N, D = tokens.shape
    grid_size = int(N ** 0.5)

    assert grid_size * grid_size == N, "Expected a square grid of patches"
    grid = tokens.view(B, grid_size, grid_size, D)
    grid = grid.permute(0, 3, 1, 2)

    return grid


BASE_PROMPTS = [
    "this image has 0-5 people",
    "this image has 5-7 people",
    "this image has 7-10 people",
    "this image has 10-12 people",
    "this image has 12-15 people",
    "this image has 15-18 people",
    "this image has 18-20 people",
    "this image has 20-22 people",
    "this image has 22-25 people",
    "this image has 25-27 people",
    "this image has 27-30 people",
    "this image has 30-35 people",
    "this image has 35-40 people",
    "this image has 40-45 people",
    "this image has 45-50 people",
    "this image has 50-55 people",
    "this image has 55-58 people",

]


class CLIPGCC(nn.Module):
    def __init__(self, clip_model, prompts=BASE_PROMPTS):
        super(CLIPGCC, self).__init__()
        self.clip_model = clip_model
        self.clip_embed_dim = clip_model.visual.output_dim
        self.num_prompts = len(prompts)
        self.tokenizer = SimpleTokenizer()

        self.feature_dim = 768
        self.projection = nn.Conv2d(
            in_channels=self.feature_dim, out_channels=self.clip_embed_dim, kernel_size=1)
        self.regressor = nn.Sequential(
            nn.Conv2d(self.num_prompts, 1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1)
        )
        self.upsampler = nn.Upsample(scale_factor=2, mode="bilinear")
        self.scale = nn.Parameter(torch.tensor(0.1))

        self.text_embeddings = self.encode_text(prompts)

    def forward(self, x):
        # Get patch tokens [batch, num_patches, visual_feature_dim]
        patch_tokens = self.get_visual_features(x)
        batch_size, num_patches, _ = patch_tokens.shape

        h = w = int(num_patches**0.5)
        visual_features = patch_tokens.permute(
            0, 2, 1).view(batch_size, -1, h, w)

        projected_visual = self.projection(visual_features)
        projected_visual = projected_visual / \
            projected_visual.norm(dim=1, keepdim=True)

        similarity = torch.einsum(
            'bchw,pc->bpwh', projected_visual, self.text_embeddings)
        similarity = similarity.permute(
            0, 1, 3, 2)

        density = self.regressor(similarity)

        for _ in range(5):
            density = self.upsampler(density)
        return torch.sigmoid(density) * self.scale

    def encode_text(self, prompts):
        with torch.no_grad():
            text_tokens = torch.cat([self.tokenizer(p) for p in prompts]).to(
                next(self.clip_model.parameters()).device)
            text_features = self.clip_model.encode_text(text_tokens)
            return text_features / text_features.norm(dim=-1, keepdim=True)

    def get_visual_features(self, x):
        _, features = self.clip_model.visual(x)
        return features
