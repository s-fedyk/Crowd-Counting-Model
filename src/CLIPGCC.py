import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadPointRegressor(nn.Module):
    """
    This regressor takes in a grid of features, and outputs where it thinks there should be points.
    """

    def __init__(self, in_channels):
        super(HeadPointRegressor, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            self._upsample_block(512, 256),  # 14x14
            self._upsample_block(256, 128),  # 28x28
            self._upsample_block(128, 64),   # 56x56
            self._upsample_block(64, 32),    # 112x112
            self._upsample_block(32, 16),    # 224x224
            
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # find out where the points are
        x = self.decoder(x)

        return torch.sigmoid(x)


def reshape_tokens_to_grid(tokens):
    B, N, D = tokens.shape
    grid_size = int(N ** 0.5)

    assert grid_size * grid_size == N, "Expected a square grid of patches"
    grid = tokens.view(B, grid_size, grid_size, D)
    grid = grid.permute(0, 3, 1, 2)

    return grid


class CLIPGCC(nn.Module):
    def __init__(self, clip_model, regressor_channels=64):
        super(CLIPGCC, self).__init__()
        self.clip_model = clip_model

        self.feature_dim = 768
        self.regressor = HeadPointRegressor(
            in_channels=self.feature_dim,
        )

    def forward(self, x):
        grid_features = self.get_grid_features(x)

        return self.regressor(grid_features)

    def get_grid_features(self, x):
        # tokens is a bunch of features.
        _, tokens = self.clip_model.visual(x)

        grid_of_patch_tokens = reshape_tokens_to_grid(tokens)

        return grid_of_patch_tokens
