import torch
from torch import nn


class OIODSegmentor(nn.Module):
    def __init__(self, num_output_channels: int, num_class: int):
        super().__init__()
        self.out_conv = nn.Conv2d(num_output_channels, num_class, kernel_size=1)

    def forward(self, anatomy_feature: torch.Tensor) -> torch.Tensor:
        return self.out_conv(anatomy_feature)
