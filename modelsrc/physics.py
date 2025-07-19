import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import GaussianBlur

class ThermalPhysicsPrior(nn.Module):
    """
    Calculate the physical prior characteristics of thermal radiation based on Fourier's law and the heat conduction equation
    """
    def __init__(self, diffusion_steps=5, alpha=0.1):
        """
        :param diffusion_steps: The time step of thermal diffusion simulation
        :param alpha: Thermal diffusion rate coefficient
        """
        super(ThermalPhysicsPrior, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=2.0)
        self.channel_alignment = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Parameter(torch.tensor(10.0))

    def compute_temperature_gradient(self, thermal_image):
        """
        Calculate the temperature gradient of the thermal image (Fourier's Law)
        :param thermal_image: (B, 1, H, W)
        :return: (B, 1, H, W)
        """
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=thermal_image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=thermal_image.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(thermal_image, sobel_x, padding=1)
        grad_y = F.conv2d(thermal_image, sobel_y, padding=1)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude

    def compute_heat_diffusion(self, thermal_image):
        """
        Calculate the thermal diffusion characteristics (heat conduction equation)
        :param thermal_image: (B, 1, H, W)
        :return: (B, 1, H, W)
        """
        heatmap = thermal_image.clone()
        for _ in range(self.diffusion_steps):
            heatmap = self.gaussian_blur(heatmap)
        return heatmap

    def forward(self, thermal_image):
        """
        Calculate the thermophysical prior characteristics
        :param thermal_image: (B, 1, H, W)
        :return: (B, 2, H, W)
        """
        temp_grad = self.compute_temperature_gradient(thermal_image)
        temp_diffusion = self.compute_heat_diffusion(thermal_image)

        prior_features = torch.cat([temp_grad, temp_diffusion], dim=1)

        # Ensure that the number of channels is consistent with the Swin Transformer
        prior_features = self.channel_alignment(prior_features) * self.scale # (B, C, H, W)

        return prior_features

