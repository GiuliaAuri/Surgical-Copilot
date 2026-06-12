import torch
import torch.nn as nn
import torch.nn.functional as F
from surgical_copilot.models.yolov8_seg import YOLOv8Segmenter 
from surgical_copilot.models.conv_gru import ConvGRUCell
from surgical_copilot.models.conv_lstm import ConvLSTMCell

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
    

class YOLOLateFusionTemporal(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, num_masks=32, recurrent_type="gru", freeze_backbone=False, warmup_epochs=5, pretrained_weights_path=None):
        super().__init__()
        
        # Salviamo entrambe le variabili per renderle accessibili dall'esterno
        self.freeze_backbone = freeze_backbone
        self.warmup_epochs = warmup_epochs
        self.pretrained_weights_path = pretrained_weights_path
        self.recurrent_type = recurrent_type.lower()
        
        # 1. IL BACKBONE SPAZIALE
        self.yolo = YOLOv8Segmenter(in_channels=in_channels, num_classes=num_classes, num_masks=num_masks)
        
        # 2. IL MODULO TEMPORALE (ConvGRU) 
        if self.recurrent_type == "gru":
            self.temporal_cell = ConvGRUCell(input_dim=32, hidden_dim=32)
        elif self.recurrent_type == "lstm":
            self.temporal_cell = ConvLSTMCell(in_channels=32, hidden_channels=32, kernel_size=3)
        

    def forward(self, x, h_prev=None):
        """
        x: Immagine RGB [B, 3, H, W]
        h_prev: Lo stato nascosto (memoria) del frame precedente [B, 32, H/8, W/8]
        """
        # 1. Spazio
        classes, coeffs, protos = self.yolo(x)
        
        # 2. Tempo (Gestione differenziata dei ritorni)
        if self.recurrent_type == "gru":
            # La GRU restituisce solo 1 tensore
            state_next = self.temporal_cell(coeffs, h_prev)
            h_next = state_next 
        elif self.recurrent_type == "lstm":
            # La LSTM restituisce una tupla (H, C)
            h_next, c_next = self.temporal_cell(coeffs, h_prev)
            state_next = (h_next, c_next)
            
        # 3. Maschera (Usiamo SEMPRE h_next, che sia uscito dalla GRU o dalla LSTM)
        coeffs_up = F.interpolate(h_next, size=(protos.shape[2], protos.shape[3]), 
                                  mode='bilinear', align_corners=False)
        
        mask_logits = torch.einsum('bchw,bchw->bhw', coeffs_up, protos).unsqueeze(1)
        mask_logits_full = F.interpolate(mask_logits, size=(x.shape[2], x.shape[3]), 
                                         mode='bilinear', align_corners=False)
        
        # Restituisce logit e lo stato (che sarà singolo per GRU o tupla per LSTM)
        return mask_logits_full, state_next