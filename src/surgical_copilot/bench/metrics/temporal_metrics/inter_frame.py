import numpy as np
import torch


class InterFrameTemporalMetric:

    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.prev_pred = None
        self.temporal_ious = []
        self.temporal_dices = []

    def __call__(self, preds: torch.Tensor):

        current = (preds > 0.5)

        if self.prev_pred is not None:

            inter = torch.sum(
                current & self.prev_pred,
                dim=(1, 2, 3)
            ).float()

            current_sum = torch.sum(
                current,
                dim=(1, 2, 3)
            ).float()

            prev_sum = torch.sum(
                self.prev_pred,
                dim=(1, 2, 3)
            ).float()

            union = current_sum + prev_sum - inter

            temporal_iou = (
                inter + self.smooth
            ) / (
                union + self.smooth
            )

            temporal_dice = (
                2.0 * inter + self.smooth
            ) / (
                current_sum + prev_sum + self.smooth
            )

            self.temporal_ious.extend(
                temporal_iou.cpu().tolist()
            )

            self.temporal_dices.extend(
                temporal_dice.cpu().tolist()
            )

        self.prev_pred = current.clone().detach()

    def aggregate(self):

        return {
            "temporal_iou": (
                float(np.mean(self.temporal_ious))
                if self.temporal_ious else 0.0
            ),
            "temporal_dice": (
                float(np.mean(self.temporal_dices))
                if self.temporal_dices else 0.0
            )
        }