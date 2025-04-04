import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
# Code modified from https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from geomloss import SamplesLoss


class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=1, sinkhorn_blur=0.20, density_scale=10):  # Increased blur
        super().__init__()
        self.alpha = alpha
        self.density_scale = density_scale
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            backend="auto",
            blur=sinkhorn_blur,
            scaling=0.1,
            reach=0.5
        )

    def forward(self, pred_map, gt_map, gt_blur_map):
        # Add density map reconstruction loss
        gt_map = gt_map.squeeze(0).squeeze(0)
        gt_blur_map = gt_blur_map.squeeze(0).squeeze(0)

        density_loss = F.mse_loss(pred_map, gt_blur_map)

        # Scale counts appropriately
        pred_count = pred_map.sum(dim=[0, 1])
        gt_count = gt_map.sum(dim=[0, 1])

        print(pred_count, gt_count)

        count_loss = F.l1_loss(pred_count, gt_count)
        density_loss = 10 * F.mse_loss(pred_map, gt_blur_map)

        return density_loss + count_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        focal_weight = (self.alpha * (1 - p) ** self.gamma * target + 
                        (1 - self.alpha) * p ** self.gamma * (1 - target))
        return (focal_weight * bce_loss).mean()

class DensityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            backend="auto",
            blur=0.05,
            scaling=0.1,
            reach=0.5
        )
        self.focal_loss = FocalLoss()

    def forward(self, pred_map, gt_blur_map):
        ssim_loss = (1 - ssim(pred_map, gt_blur_map))

        binary_gt = (gt_blur_map > 0).float()
        pixel_differences = self.focal_loss(pred_map, binary_gt)
        return 0.05 * ssim_loss + pixel_differences



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
