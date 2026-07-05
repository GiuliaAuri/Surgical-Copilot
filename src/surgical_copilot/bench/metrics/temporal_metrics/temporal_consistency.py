import numpy as np
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torch.nn.functional as F

class TemporalConsistencyMetric:
    
    def __init__(self, device, smooth=1e-6):
        self.smooth = smooth
        self.device = device
        
        # Recuperiamo i pesi, creiamo il modello e lo spostiamo sulla GPU
        weights = Raft_Small_Weights.DEFAULT
        self.raft = raft_small(weights=weights).to(self.device)
        
        # Modalità valutazione
        self.raft.eval()
        
        # Blocchiamo i gradienti
        for param in self.raft.parameters():
            param.requires_grad = False
            
        self.reset()

    def reset(self):
        self.prev_pred = None
        self.prev_label = None
        self.prev_image = None
        self.ious = []
        self.dices = []

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor, images: torch.Tensor):
        # Binarizzazione
        p_bin = preds > 0.5
        l_bin = labels > 0.5

        if self.prev_pred is not None and self.prev_label is not None:
            # 1. Calcola il flusso (i vettori di movimento)
            flow = self.raft(self.prev_image, images)[-1] 

            # 2. Crea una "griglia" di coordinate che il modello possa deformare
            b, _, h, w = images.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=0).float().to(self.device)
            grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)

            # 3. Aggiungi il flusso alla griglia (questo sposta le coordinate dei pixel)
            # Il flusso è [Batch, 2, H, W]
            warped_grid = grid + flow

            # 4. Normalizza la griglia in [-1, 1] (requisito di grid_sample)
            warped_grid[:, 0, :, :] = 2.0 * warped_grid[:, 0, :, :] / (w - 1) - 1.0
            warped_grid[:, 1, :, :] = 2.0 * warped_grid[:, 1, :, :] / (h - 1) - 1.0
            warped_grid = warped_grid.permute(0, 2, 3, 1) # [B, H, W, 2]

            # 5. Finalmente, "deforma" la maschera passata
            warped_prev_pred = F.grid_sample(self.prev_pred.float(), warped_grid, align_corners=True)
            warped_prev_pred = (warped_prev_pred > 0.5).float() 

            # 6. Calcolo IoU tra la predizione deformata e quella attuale
            inter = torch.sum(warped_prev_pred * p_bin.float(), dim=(1, 2, 3))
            union = torch.sum(warped_prev_pred + p_bin.float(), dim=(1, 2, 3)) - inter
            
            iou = (inter + self.smooth) / (union + self.smooth)
            dice = (2.0 * inter + self.smooth) / (torch.sum(warped_prev_pred + p_bin.float(), dim=(1, 2, 3)) + self.smooth)
            
            self.ious.extend(iou.cpu().tolist())
            self.dices.extend(dice.cpu().tolist())

            # Update degli stati per t-1
            self.prev_pred = p_bin.clone().detach()
            self.prev_label = l_bin.clone().detach()
            self.prev_image = images.clone().detach()

    def aggregate(self):
        return {
            "temporal_iou": float(np.mean(self.ious)) if self.ious else 0.0,
            "temporal_dice": float(np.mean(self.dices)) if self.dices else 0.0
        }
   