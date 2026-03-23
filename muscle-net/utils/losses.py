from torch import nn
import torch
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    """
    Main output at full resolution.
    Auxiliary branch at H/4, W/4.
    Best weight found in experiments: 0.10 for auxiliary loss.
    """
    def __init__(self, main_weight: float = 0.90, aux_weight: float = 0.10):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, main_output: torch.Tensor, aux_output: torch.Tensor, target: torch.Tensor):
        target_full = target.long()

        target_aux = F.interpolate(
            target.unsqueeze(1).float(),
            size=aux_output.shape[-2:],
            mode="nearest"
        ).squeeze(1).long()

        loss_main = self.criterion(main_output, target_full)
        loss_aux = self.criterion(aux_output, target_aux)

        return (self.main_weight * loss_main) + (self.aux_weight * loss_aux)
