import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    def forward(self, prediction, target, num_class=3, weight_map=None):
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, num_class)
        target = target.permute(0, 2, 3, 1).contiguous().view(-1, num_class)

        if weight_map is not None:
            weights = weight_map.view(-1).repeat(num_class).view_as(prediction)
            target_volume = torch.sum(weights * target, dim=0)
            intersection = torch.sum(weights * target * prediction, dim=0)
            prediction_volume = torch.sum(weights * prediction, dim=0)
        else:
            target_volume = torch.sum(target, dim=0)
            intersection = torch.sum(target * prediction, dim=0)
            prediction_volume = torch.sum(prediction, dim=0)

        dice = (2.0 * intersection) / (target_volume + prediction_volume + 1e-5)
        dice = torch.clamp(dice, min=1e-6, max=1.0)
        return torch.mean(-torch.log(dice))


def KL_divergence(logvar, mu):
    divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return divergence.mean()
