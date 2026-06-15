import torch
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
        # Estrae le feature spaziali
        features = self.model.encoder(x)
        bottleneck = features[-1]
        
        # Elaborazione temporale
        bottleneck, current_state = self.recurrent(bottleneck, h_prev)
        
        # Ripristino e decodifica
        features = list(features)
        features[-1] = bottleneck
        decoder_output = self.model.decoder(*features)
        logits = self.model.segmentation_head(decoder_output)
        
        return logits, current_state