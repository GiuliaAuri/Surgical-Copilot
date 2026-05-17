import torch
import torch.nn as nn
import torch.nn.functional as F

# Importa la classe che abbiamo scritto insieme nel notebook
from surgical_copilot.models.yolov8_seg import YOLOv8Segmenter 
# (Assicurati che il nome del file da cui importi sia corretto)

class CustomYOLOSemantic(nn.Module):
    def __init__(self, num_classes=1, num_masks=32):
        super().__init__()
        # Inizializza la TUA rete scritta da zero in PyTorch
        self.yolo = YOLOv8Segmenter(num_classes=num_classes, num_masks=num_masks)

    def forward(self, x):
        # 1. Passa il frame chirurgico nella tua rete
        # x shape: [B, 3, H, W]
        classes, coeffs, protos = self.yolo(x)
        
        # Le dimensioni che escono dalla tua rete sono:
        # coeffs: [B, 32, H/8, W/8]
        # protos: [B, 32, H/4, W/4]

        # 2. LA SEGMENTAZIONE SEMANTICA DENSE
        # Ingrandiamo la mappa dei coefficienti per farla combaciare con i prototipi
        coeffs_up = F.interpolate(coeffs, size=(protos.shape[2], protos.shape[3]), 
                                  mode='bilinear', align_corners=False)
        
        # Prodotto scalare spaziale: moltiplichiamo pixel-per-pixel e sommiamo i 32 canali.
        # Questo fonde le maschere astratte creando la maschera del sangue!
        # Risultato: [B, 1, H/4, W/4]
        mask_logits = torch.sum(coeffs_up * protos, dim=1, keepdim=True)
        
        # 3. Riportiamo la maschera alla risoluzione originale dell'immagine (es. 640x640)
        mask_logits_full = F.interpolate(mask_logits, size=(x.shape[2], x.shape[3]), 
                                         mode='bilinear', align_corners=False)
        
        # Restituiamo i logit. 
        return mask_logits_full