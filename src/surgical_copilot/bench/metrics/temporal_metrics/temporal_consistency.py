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
        images: (B, C, H, W) - Immagine attuale al tempo t (C può essere 3 o 4)
        """
        # 1. ISOLAMENTO CANALI RGB:
        # Se siamo in EARLY_FUSION (C=4), prendiamo solo i primi 3 canali (RGB).
        # Se siamo in modalità standard (C=3), prendiamo tutto.
        if images.shape[1] > 3:
            img_to_raft = images[:, :3, :, :]
        else:
            img_to_raft = images

        p_bin = (preds > 0.5).float()

        # Usiamo img_to_raft ovunque d'ora in avanti
        if self.prev_image is not None and self.prev_pred is not None:
            # Trasformiamo solo la parte RGB
            prev_raft, curr_raft = self.transforms(self.prev_image, img_to_raft)

            # 2. Calcolo Optical Flow (RAFT)
            with torch.no_grad():
                flow = self.raft(prev_raft, curr_raft)[-1]

            # 3. Warping della predizione precedente
            b, _, h, w = images.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=0).to(self.device).float()
            grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)

            warped_grid = grid + flow
            
            # Normalizzazione per grid_sample
            warped_grid[:, 0, :, :] = 2.0 * warped_grid[:, 0, :, :] / (w - 1) - 1.0
            warped_grid[:, 1, :, :] = 2.0 * warped_grid[:, 1, :, :] / (h - 1) - 1.0
            warped_grid = warped_grid.permute(0, 2, 3, 1)

            # Warp
            warped_prev_pred = F.grid_sample(self.prev_pred, warped_grid, align_corners=True, padding_mode='border')
            warped_prev_pred = (warped_prev_pred > 0.5).float()

            # 4. Calcolo IoU
            inter = (warped_prev_pred * p_bin).sum(dim=(-2, -1))
            union = (warped_prev_pred + p_bin).sum(dim=(-2, -1)) - inter
            
            tc_iou = inter / (union + 1e-6)
            self.ious.extend(tc_iou.cpu().tolist()) # Tolto .mean(dim=0) perché vogliamo la media finale dopo

        # Update stato: salviamo sempre la versione pulita (3 canali)
        self.prev_pred = p_bin.detach()
        self.prev_image = img_to_raft.detach()

    def aggregate(self):
        return {"temporal_consistency": float(np.mean(self.ious)) if self.ious else 0.0}