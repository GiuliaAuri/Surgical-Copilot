import torch
import torch.nn as nn
import torch.nn.functional as F
from surgical_copilot.models.yolov8_seg import YOLOv8Segmenter 
from surgical_copilot.models.yolov8_seg.conv_gru import ConvGRUCell

class CustomYOLOSemantic(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, num_masks=32):
        super().__init__()
        self.yolo = YOLOv8Segmenter(in_channels=in_channels, num_classes=num_classes, num_masks=num_masks)


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
    

class YOLOLateFusionConvGRU(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, num_masks=32):
        super().__init__()
        
        # 1. IL BACKBONE SPAZIALE (Lo YOLO standard a 3 canali)
        self.yolo = YOLOv8Segmenter(in_channels=in_channels, num_classes=num_classes, num_masks=num_masks)
        
        # 2. IL MODULO TEMPORALE (ConvGRU) 
        self.conv_gru = ConvGRUCell(input_dim=32, hidden_dim=32)

    def forward(self, x, h_prev=None):
        """
        x: Immagine RGB [B, 3, H, W]
        h_prev: Lo stato nascosto (memoria) del frame precedente [B, 32, H/8, W/8]
        """
        # 1. YOLO elabora l'immagine (Spazio)
        classes, coeffs, protos = self.yolo(x)
        
        # 2. La ConvGRU elabora i coefficienti guardando il passato (Tempo)
        coeffs_temporali = self.conv_gru(coeffs, h_prev)
        
        # 3. SEGMENTAZIONE FINALE 
        # Ingrandiamo i coefficienti temporali
        coeffs_up = F.interpolate(coeffs_temporali, size=(protos.shape[2], protos.shape[3]), 
                                  mode='bilinear', align_corners=False)
        
        # Moltiplichiamo per i prototipi per ottenere la maschera finale
        mask_logits = torch.einsum('bchw,bchw->bhw', coeffs_up, protos).unsqueeze(1)
        
        
        # Riportiamo la maschera alla grandezza originale dell'immagine
        mask_logits_full = F.interpolate(mask_logits, size=(x.shape[2], x.shape[3]), 
                                         mode='bilinear', align_corners=False)
        
        # Restituiamo SOLO la maschera finale
        return mask_logits_full