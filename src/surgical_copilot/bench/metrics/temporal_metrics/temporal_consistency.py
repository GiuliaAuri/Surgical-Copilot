import numpy as np
import torch

class TemporalConsistencyMetric:
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.prev_pred = None
        self.prev_label = None
        self.ious = []
        self.dices = []

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
        # Binarizzazione
        p_bin = preds > 0.5
        l_bin = labels > 0.5

        if self.prev_pred is not None and self.prev_label is not None:
            # Calcolo dei delta (XOR logico per trovare i pixel cambiati)
            delta_pred = (p_bin ^ self.prev_pred).float()
            delta_label = (l_bin ^ self.prev_label).float()

            # Operatori insiemistici
            inter = torch.sum(delta_pred * delta_label, dim=(1, 2, 3))
            sum_parts = torch.sum(delta_pred, dim=(1, 2, 3)) + torch.sum(delta_label, dim=(1, 2, 3))
            union = sum_parts - inter

            # Calcolo metriche per ogni elemento del batch
            for i, u, s, d_gt in zip(inter, union, sum_parts, torch.sum(delta_label, dim=(1, 2, 3))):
                if u > 0:
                    iou = (i + self.smooth) / (u + self.smooth)
                    dice = (2.0 * i + self.smooth) / (s + self.smooth)
                    
                    self.ious.append(iou.item())
                    self.dices.append(dice.item())
                elif d_gt == 0 and u == 0:
                    # Perfetta consistenza: nessuno dei due è cambiato
                    self.ious.append(1.0)
                    self.dices.append(1.0)

        # Update degli stati per t-1
        self.prev_pred = p_bin.clone().detach()
        self.prev_label = l_bin.clone().detach()

    def aggregate(self):
        return {
            "temporal_iou": float(np.mean(self.ious)) if self.ious else 0.0,
            "temporal_dice": float(np.mean(self.dices)) if self.dices else 0.0
        }
   