import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class TemporalConsistencyMetric:
    def __init__(self, device):
        self.device = device
        # Carichiamo RAFT come suggerito dal paper per calcolare il warping
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.raft = raft_small(weights=weights).to(self.device).eval()
        for param in self.raft.parameters():
            param.requires_grad = False
        self.reset()

    def reset(self):
        self.prev_pred = None
        self.prev_image = None
        self.ious = []

    def __call__(self, preds, labels, images):
        """
        preds: (B, 1, H, W) - Predizione attuale al tempo t
        images: (B, C, H, W) - Immagine attuale al tempo t
        """
        # RAFT works on the current pair of frames, so we keep the previous
        # raw image and transform the pair only when we have both frames.
        p_bin = (preds > 0.5).float()

        if self.prev_image is not None and self.prev_pred is not None:
            prev_raft, curr_raft = self.transforms(self.prev_image, images)

            # 2. Calcolo Optical Flow (RAFT)
            with torch.no_grad():
                # RAFT predice il movimento da prev_image a curr_image
                flow = self.raft(prev_raft, curr_raft)[-1]

            # 3. Warping della predizione precedente (S_t-1 -> S_t)
            # Creiamo la griglia di coordinate
            b, _, h, w = images.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=0).to(self.device).float()
            grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)

            # Deformiamo la griglia con il flusso
            warped_grid = grid + flow
            
            # Normalizzazione per grid_sample (che vuole coordinate in [-1, 1])
            warped_grid[:, 0, :, :] = 2.0 * warped_grid[:, 0, :, :] / (w - 1) - 1.0
            warped_grid[:, 1, :, :] = 2.0 * warped_grid[:, 1, :, :] / (h - 1) - 1.0
            warped_grid = warped_grid.permute(0, 2, 3, 1)

            # Warp della maschera precedente: qui avviene la magia del paper
            warped_prev_pred = F.grid_sample(self.prev_pred, warped_grid, align_corners=True, padding_mode='border')
            warped_prev_pred = (warped_prev_pred > 0.5).float()

            # 4. Calcolo TC come IoU tra maschera deformata e maschera attuale
            inter = (warped_prev_pred * p_bin).sum(dim=(-2, -1))
            union = (warped_prev_pred + p_bin).sum(dim=(-2, -1)) - inter
            
            tc_iou = inter / (union + 1e-6)
            self.ious.extend(tc_iou.mean(dim=0).cpu().tolist())

        # Update stato
        self.prev_pred = p_bin.detach()
        self.prev_image = images.detach()
        

    def aggregate(self):
        return {"temporal_consistency": float(np.mean(self.ious)) if self.ious else 0.0}