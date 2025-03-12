import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Dropout

from CLIP.tokenizer import SimpleTokenizer


import torch
import torch.nn as nn
import torch.nn.functional as F

# A helper module that applies two consecutive convolutional layers,
# each followed by batch normalization and ReLU activation.


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down-sampling block: applies max pooling followed by a DoubleConv block.


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up-sampling block: upsamples the input, concatenates with the corresponding
# feature map from the encoder (skip connection), and then applies a DoubleConv block.


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal upsampling method; otherwise use transposed conv.
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation the number of input channels becomes in_channels,
            # so we reduce it to out_channels via DoubleConv.
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # using transposed convolution to upsample
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: upsampled feature map from the decoder.
        # x2: corresponding feature map from the encoder (skip connection).
        x1 = self.up(x1)
        # Ensure that x1 and x2 have the same spatial dimensions.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channels dimension.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Final output convolution: reduces the number of channels to the desired number of classes.


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# The complete U-Net model.


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels//2, kernel_size=2, stride=2)

        self.conv = ResidualBlock(in_channels//2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)

        # Decoder
        self.up1 = Up(1024//factor, 512, 512//factor, bilinear)
        self.up2 = Up(512//factor, 256, 256//factor, bilinear)
        self.up3 = Up(256//factor, 128, 128//factor, bilinear)
        self.up4 = Up(128//factor, 64, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)          # 64x224x224
        x2 = self.down1(x1)       # 128x112x112
        x3 = self.down2(x2)       # 256x56x56
        x4 = self.down3(x3)       # 512x28x28
        x5 = self.down4(x4)       # 1024//factor x14x14

        # Decoder
        x = self.up1(x5, x4)      # 512//factor x28x28
        x = self.up2(x, x3)       # 256//factor x56x56
        x = self.up3(x, x2)       # 128//factor x112x112
        x = self.up4(x, x1)       # 64x224x224

        logits = self.outc(x)     # 1x224x224
        return torch.abs(logits)

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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # depthwise convolution
        # applied on channel-last tensor
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/linear layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # A learnable scale parameter initialized to a small value
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(
            (dim)), requires_grad=True) if layer_scale_init_value > 0 else None

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

# Decoder block: upsamples input, concatenates with skip connection, and refines via convolution.


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels,
                               out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample x by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        # Concatenate with corresponding encoder feature (skip connection)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# ConvNeXt-based segmentation model with U-Net style decoder.


class ConvNeXtSegmentation(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        """
        Args:
            in_channels: Number of channels in the input image.
            num_classes: Number of prediction channels (e.g., 1 for binary segmentation).
            depths: Number of ConvNeXt blocks in each stage.
            dims: Feature dimensions for each stage.
        """
        super().__init__()
        # Encoder: Stem and stages.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Stage 1 (resolution: H/4)
        self.stages.append(nn.Sequential(
            *[ConvNeXtBlock(dims[0]) for _ in range(depths[0])]))

        # Stage 2 (resolution: H/8)
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm2d(dims[0]),
                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
            )
        )
        self.stages.append(nn.Sequential(
            *[ConvNeXtBlock(dims[1]) for _ in range(depths[1])]))

        # Stage 3 (resolution: H/16)
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm2d(dims[1]),
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
            )
        )
        self.stages.append(nn.Sequential(
            *[ConvNeXtBlock(dims[2]) for _ in range(depths[2])]))

        # Stage 4 (resolution: H/32)
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm2d(dims[2]),
                nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
            )
        )
        self.stages.append(nn.Sequential(
            *[ConvNeXtBlock(dims[3]) for _ in range(depths[3])]))

        # Decoder: U-Net style blocks.
        # Decoder Block 1: upsample from stage 4 (H/32) to merge with stage 3 (H/16).
        self.decoder1 = DecoderBlock(
            in_channels=dims[3], skip_channels=dims[2], out_channels=dims[2])
        # Decoder Block 2: upsample to merge with stage 2 (H/8).
        self.decoder2 = DecoderBlock(
            in_channels=dims[2], skip_channels=dims[1], out_channels=dims[1])
        # Decoder Block 3: upsample to merge with stage 1 (H/4).
        self.decoder3 = DecoderBlock(
            in_channels=dims[1], skip_channels=dims[0], out_channels=dims[0])

        # Final segmentation head: produces prediction maps.
        self.head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)                      # (B, dims[0], H/4, W/4)
        x1 = self.stages[0](x0)                # (B, dims[0], H/4, W/4)
        x2 = self.stages[1](self.downsample_layers[0](x1)
                            )  # (B, dims[1], H/8, W/8)
        x3 = self.stages[2](self.downsample_layers[1](x2)
                            )  # (B, dims[2], H/16, W/16)
        x4 = self.stages[3](self.downsample_layers[2](x3)
                            )  # (B, dims[3], H/32, W/32)

        # Decoder with skip connections
        d1 = self.decoder1(x4, x3)   # (B, dims[2], H/16, W/16)
        d2 = self.decoder2(d1, x2)   # (B, dims[1], H/8, W/8)
        d3 = self.decoder3(d2, x1)   # (B, dims[0], H/4, W/4)

        out = self.head(d3)         # (B, num_classes, H/4, W/4)
        # Upsample to match original resolution.
        out = F.interpolate(out, scale_factor=4,
                            mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


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
