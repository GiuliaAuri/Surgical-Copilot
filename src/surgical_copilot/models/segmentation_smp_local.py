import torch.nn as nn
import segmentation_models_pytorch as smp

from surgical_copilot.models.Recurrent_U_Net.recurrent_u_net import RecurrentWrapper


class SMPUNet(nn.Module):
    """ U-Net Spaziale Base con Encoder ResNet """
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1, **kwargs):
        super().__init__()
        self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)

    def forward(self, x):
        return self.model(x)


class SMPUNetPlusPlus(nn.Module):
    """ U-Net++ Spaziale Base con Encoder ResNet """
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1, **kwargs):
        super().__init__()
        self.model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)

    def forward(self, x):
        return self.model(x)


class RecurrentSMPUNet(nn.Module):
    """ U-Net Temporale con modulo ricorrente (GRU/LSTM) al Bottleneck """
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1, recurrent_type="gru", bottleneck_dim=512, freeze_backbone=False, warmup_epochs=5, pretrained_weights_path=None, **kwargs):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.warmup_epochs = warmup_epochs
        self.pretrained_weights_path = pretrained_weights_path
        
        # Costruisce la U-Net
        self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
        
        # Innesca la ricorrenza al bottleneck
        self.recurrent = RecurrentWrapper(recurrent_type=recurrent_type, channels=bottleneck_dim)

    def forward(self, x, h_prev=None):
        b, t, c, h, w = x.shape
        
        # Encoder (Time-Distributed)
        x_flat = x.view(b * t, c, h, w)
        features_flat = self.model.encoder(x_flat)
        
        # Trasformazione sequenziale
        features_seq = []
        for f in features_flat:
            features_seq.append(f.view(b, t, f.shape[1], f.shape[2], f.shape[3]))
            
        bottleneck = features_seq[-1]
        
        # Elaborazione temporale
        bottleneck, current_state = self.recurrent(bottleneck, h_prev)
        
        # Preparazione per il decoder
        # Il decoder di smp si aspetta una LISTA di tensori (B*T, C, H, W)
        decoder_features = []
        for i in range(len(features_seq) - 1):
            f = features_seq[i]
            decoder_features.append(f.view(b * t, f.shape[2], f.shape[3], f.shape[4]))
        
        # Aggiungiamo il bottleneck processato dalla RNN
        decoder_features.append(bottleneck.view(b * t, bottleneck.shape[2], bottleneck.shape[3], bottleneck.shape[4]))
        
        # Il decoder si aspetta gli argomenti spacchettati
        decoder_output = self.model.decoder(decoder_features) 
        
        logits = self.model.segmentation_head(decoder_output)
        
        return logits.view(b, t, 1, h, w), current_state