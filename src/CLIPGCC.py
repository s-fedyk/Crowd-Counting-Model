
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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
    "this image has 50-55 people",
    "this image has 55-58 people",
]

class CLIPGCC(nn.Module):
    def __init__(self, clip_model, prompts=BASE_PROMPTS, use_checkpointing=True):
        super(CLIPGCC, self).__init__()
        self.clip_model = clip_model
        self.clip_embed_dim = clip_model.visual.output_dim
        self.num_prompts = len(prompts)
        self.tokenizer = SimpleTokenizer()
        self.use_checkpointing = use_checkpointing

        self.feature_dim = 512
        self.projection = nn.Conv2d(
            in_channels=self.feature_dim, out_channels=self.clip_embed_dim, kernel_size=1)
        self.regressor = nn.Sequential(
            nn.Conv2d(self.num_prompts, 1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1)
        )

        self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False)        
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Pre-compute text embeddings (no gradient needed)
        self.text_embeddings = self.encode_text(prompts)

    def heavy_forward(self, x):
        # This part contains the operations that are memory intensive.
        visual_features = self.upsampler(x)
        projected_visual = self.projection(visual_features)
        projected_visual = F.normalize(projected_visual, dim=1)
        similarity = torch.einsum('bchw,pc->bpwh', projected_visual, self.text_embeddings)
        similarity = similarity.permute(0, 1, 3, 2)  # [BS, num_prompts, H, W]
        density = self.regressor(similarity) * self.scale
        return density

    def forward(self, x):
        if self.use_checkpointing:
            # Wrap the heavy forward computation with checkpointing.
            density = checkpoint(self.heavy_forward, x)
        else:
            density = self.heavy_forward(x)
        return torch.sigmoid(density)

    def encode_text(self, prompts):
        with torch.no_grad():
            text_tokens = torch.cat([self.tokenizer(p) for p in prompts]).to(
                next(self.clip_model.parameters()).device)
            text_features = self.clip_model.encode_text(text_tokens)
            return text_features / text_features.norm(dim=-1, keepdim=True)

    def get_visual_features(self, x):
        _, features = self.clip_model.visual(x)
        return features

