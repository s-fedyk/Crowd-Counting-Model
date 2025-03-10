import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Dropout

class SimpleImplicitFeaturizer(torch.nn.Module):

    def __init__(self, n_freqs=20):
        super().__init__()
        self.n_freqs = n_freqs
        self.dim_multiplier = 2

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        feat_list = [feats]
        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1)
        feats = (feats * freqs)

        feats = feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        all_feats = [torch.sin(feats), torch.cos(feats), original_image]

        return torch.cat(all_feats, dim=1)


class IFA(torch.nn.Module):

    def __init__(self, feat_dim, num_scales=20):
        super().__init__()
        self.scales = 2 * torch.exp(torch.tensor(torch.arange(1, num_scales + 1)))
        self.feat_dim = feat_dim
        self.sin_feats = SimpleImplicitFeaturizer()
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_dim + (num_scales * 4) + 2, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv2d(feat_dim, feat_dim, 1),
        )

    def forward(self, source):
        b, c, h, w = source.shape
        up_source = F.interpolate(source, (h * 2, w * 2), mode="nearest")
        assert h == w
        lr_cord = torch.linspace(0, h, steps=h, device=source.device)
        hr_cord = torch.linspace(0, h, steps=2 * h, device=source.device)
        lr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(lr_cord, lr_cord)], dim=0).unsqueeze(0)
        hr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(hr_cord, hr_cord)], dim=0).unsqueeze(0)
        up_lr_coords = F.interpolate(lr_coords, (h * 2, w * 2), mode="nearest")
        coord_diff = up_lr_coords - hr_coords
        coord_diff_feats = self.sin_feats(coord_diff)
        c2 = coord_diff_feats.shape[1]
        bcast_coord_feats = torch.broadcast_to(coord_diff_feats, (b, c2, h * 2, w * 2))
        return self.mlp(torch.cat([up_source, bcast_coord_feats], dim=1))  # + up_source


"""
        self.decoder1 = nn.Sequential(
            IFA(in_channels),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            IFA(256),
            nn.Dropout(dropout),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            IFA(128),
            nn.Dropout(dropout),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            IFA(64),
            nn.Dropout(dropout),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            IFA(1),
        )
"""

class HeadPointRegressor(nn.Module):
    """
    This regressor takes in a grid of features, and outputs where it thinks there should be points.
    """

    def __init__(self, in_channels, dropout=0.3):
        super(HeadPointRegressor, self).__init__()

        self.decoder1 = nn.Sequential(
            IFA(in_channels),
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            IFA(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            IFA(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            IFA(64),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
            IFA(1),
        )
        
        
    def forward(self, x):
        # find out where the points are
        x = self.decoder1(x)
        
        return torch.relu(x)


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
