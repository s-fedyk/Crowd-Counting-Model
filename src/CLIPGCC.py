import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Dropout

from CLIP.tokenizer import SimpleTokenizer


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

        return torch.relu(density)

    def encode_text(self, prompts):
        with torch.no_grad():
            text_tokens = torch.cat([self.tokenizer(p) for p in prompts]).to(
                next(self.clip_model.parameters()).device)
            text_features = self.clip_model.encode_text(text_tokens)
            return text_features / text_features.norm(dim=-1, keepdim=True)

    def get_visual_features(self, x):
        _, features = self.clip_model.visual(x)
        return features
