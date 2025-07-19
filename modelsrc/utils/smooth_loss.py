import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss


# ===============================
# First-order edge gradient
# ===============================
def gradient_x(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(img.size(1), 1, 1, 1)  # 适配多通道
    return F.conv2d(img, sobel_x, stride=1, padding=1, groups=img.size(1))

def gradient_y(img):
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(img.size(1), 1, 1, 1)
    return F.conv2d(img, sobel_y, stride=1, padding=1, groups=img.size(1))

# ===============================
# Second-order edge-laplacian
# ===============================
def laplacian_edge(img):
    lap_kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32, device=img.device)
    lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(img.size(1), 1, 1, 1)
    return F.conv2d(img, lap_kernel, stride=1, padding=1, groups=img.size(1))

# ===============================
# Charbonnier
# ===============================
def charbonnier_penalty(x, epsilon=1e-3):
    return torch.sqrt(x ** 2 + epsilon ** 2)

# ===============================
# Saliency map smoothing loss
# ===============================
def get_saliency_smoothness(pred, gt, size_average=True, weight_1st=10.0, weight_2nd=0.0):
    alpha = 10.0  # 权重因子

    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    gt_x = gradient_x(gt)
    gt_y = gradient_y(gt)

    w_x = torch.exp(-alpha * torch.abs(gt_x))
    w_y = torch.exp(-alpha * torch.abs(gt_y))

    loss_x = charbonnier_penalty(sal_x * w_x)
    loss_y = charbonnier_penalty(sal_y * w_y)
    smooth_1st = loss_x + loss_y

    lap_pred = torch.abs(laplacian_edge(pred))
    lap_gt = torch.abs(laplacian_edge(gt))
    w_lap = torch.exp(-alpha * lap_gt)
    smooth_2nd = charbonnier_penalty(lap_pred * w_lap)

    loss = weight_1st * torch.mean(smooth_1st)
    if weight_2nd > 0:
        loss += weight_2nd * torch.mean(smooth_2nd)

    return loss

class GHXSmoothnessLoss(nn.Module):
    def __init__(self, size_average=True, weight_1st=10.0, weight_2nd=0.0):
        super(GHXSmoothnessLoss, self).__init__()
        self.size_average = size_average
        self.weight_1st = weight_1st
        self.weight_2nd = weight_2nd

    def forward(self, pred, target):
        return get_saliency_smoothness(pred, target, self.size_average,
                                       self.weight_1st, self.weight_2nd)

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice_loss = DiceLoss()
        # self.bce_loss = BCELoss()
        self.bce_loss = MSELoss()
        # self.bce_loss = BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.weight_dice * dice + self.weight_bce * bce

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)
