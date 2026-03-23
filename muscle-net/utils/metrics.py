import torch
import torch.nn.functional as F


def calculate_iou(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int = 8):
    predictions = torch.argmax(predictions, dim=1)
    labels = F.interpolate(
        labels.unsqueeze(1).float(),
        size=predictions.shape[1:],
        mode="nearest"
    ).squeeze(1).long()

    ious = []
    for cls in range(num_classes):
        pred_cls = predictions == cls
        true_cls = labels == cls
        intersection = (pred_cls & true_cls).float().sum()
        union = (pred_cls | true_cls).float().sum()

        if union > 0:
            ious.append(intersection / (union + 1e-6))

    return sum(ious) / len(ious) if ious else torch.tensor(0.0, device=predictions.device)
