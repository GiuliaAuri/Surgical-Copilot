import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class TemporalConsistencyMetric:

    def __init__(self, device):
        self.device = device

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()

        self.raft = raft_small(weights=weights).to(device).eval()
        for p in self.raft.parameters():
            p.requires_grad = False

        self.reset()

    def reset_sequence(self):
        self.prev_pred = None
        self.prev_image = None
        
    def reset(self):
        self.prev_pred = None
        self.prev_image = None
        self.ious = []

    def _compute_pair(
        self,
        prev_pred,
        curr_pred,
        prev_img,
        curr_img,
    ):
        """
        prev_pred : (B,1,H,W)
        curr_pred : (B,1,H,W)

        prev_img : (B,C,H,W)
        curr_img : (B,C,H,W)
        """

        # Early Fusion -> mantieni solo RGB
        if prev_img.shape[1] > 3:
            prev_img = prev_img[:, :3]

        if curr_img.shape[1] > 3:
            curr_img = curr_img[:, :3]

        prev_pred = (prev_pred > 0.5).float()
        curr_pred = (curr_pred > 0.5).float()

        prev_raft, curr_raft = self.transforms(prev_img, curr_img)

        with torch.no_grad():
            flow = self.raft(prev_raft, curr_raft)[-1]

        b, _, h, w = prev_img.shape

        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )

        grid = torch.stack((xx, yy), dim=0).float()
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)

        warped_grid = grid + flow

        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (w - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (h - 1) - 1.0

        warped_grid = warped_grid.permute(0, 2, 3, 1)

        warped_prev = F.grid_sample(
            prev_pred,
            warped_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        warped_prev = (warped_prev > 0.5).float()

        inter = (warped_prev * curr_pred).sum(dim=(-2, -1))
        union = (warped_prev + curr_pred).sum(dim=(-2, -1)) - inter

        tc = inter / (union + 1e-6)

        self.ious.extend(tc.cpu().numpy().tolist())

    def __call__(self, preds, labels, images):

        # ==========================================================
        # BASELINE / EARLY FUSION
        # images : (B,C,H,W)
        # ==========================================================

        if images.ndim == 4:

            if self.prev_image is not None:
                self._compute_pair(
                    self.prev_pred,
                    preds,
                    self.prev_image,
                    images,
                )

            self.prev_pred = preds.detach()
            self.prev_image = images.detach()

            return

        # ==========================================================
        # LATE FUSION
        # images : (B,T,C,H,W)
        # ==========================================================

        if images.ndim == 5:

            B, T, C, H, W = images.shape

            # collega con la sequenza precedente
            if self.prev_image is not None:

                self._compute_pair(
                    self.prev_pred,
                    preds[:, 0],
                    self.prev_image,
                    images[:, 0],
                )

            # confronti interni alla sequenza
            for t in range(T - 1):

                self._compute_pair(
                    preds[:, t],
                    preds[:, t + 1],
                    images[:, t],
                    images[:, t + 1],
                )

            # salva ultimo frame
            self.prev_pred = preds[:, -1].detach()
            self.prev_image = images[:, -1].detach()

            return

        raise ValueError(
            f"Unsupported image shape {images.shape}"
        )

    def aggregate(self):

        if len(self.ious) == 0:
            return {"temporal_consistency": 0.0}

        return {
            "temporal_consistency": float(np.mean(self.ious))
        }