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
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
        
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # find out where the points are
        x = self.decoder(x)

        # upscale the grid to 224x224
        for _ in range(5):  
            x = self.upsampler(x)

        # We need RELU because we can't have negatives in the output.
        return torch.relu(x) + 1e-7


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
