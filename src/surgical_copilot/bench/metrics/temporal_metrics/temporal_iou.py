import torch
import numpy as np

class TemporalIoU:

    def __init__(self, threshold=0.5, from_logits=False, eps=1e-6):
        self.threshold = threshold
        self.from_logits = from_logits
        self.eps = eps

        self.reset()

    def reset(self):
        self.values = []
        self.prev_mask = None

    def reset_sequence(self):
        self.prev_mask = None

    @torch.no_grad()
    def __call__(self, preds, is_first_frame=False):

        if self.from_logits:
            preds = torch.sigmoid(preds)

        preds = preds > self.threshold

        # Primo frame della sequenza: non esiste una tIoU
        if is_first_frame:
            self.prev_mask = preds.detach()
            return

        # Prima chiamata assoluta
        if self.prev_mask is None:
            self.prev_mask = preds.detach()
            return

        inter = (preds & self.prev_mask).sum(dim=(-2, -1)).float()
        union = (preds | self.prev_mask).sum(dim=(-2, -1)).float()

        tiou = inter / (union + self.eps)

        self.values.extend(tiou.cpu().tolist())

        self.prev_mask = preds.detach()

    def aggregate(self):
        if len(self.values) == 0:
            return {"temporal_iou": 0.0}

        return {"temporal_iou": float(np.mean(self.values))}