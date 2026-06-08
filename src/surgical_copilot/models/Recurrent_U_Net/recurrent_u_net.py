from monai.networks.nets.unet import UNet
from surgical_copilot.models.yolov8_seg.conv_gru import ConvGRUCell
import torch.nn as nn
import torch
import os

class InjectedBottleneck(nn.Module):
    def __init__(self, spatial_layer, parent_model):
        super().__init__()
        self.spatial_layer = spatial_layer
        self.parent_model = parent_model
        
    def forward(self, x):
        # 1. Feature extraction spaziale
        x_spatial = self.spatial_layer(x)
        # 2. Propagazione temporale
        h_next = self.parent_model.conv_gru(x_spatial, self.parent_model.h_state)
        self.parent_model.h_state = h_next
        return h_next

class RecurrentUNet(nn.Module):

    def __init__(
        self, 
        spatial_dims=2, 
        in_channels=3, 
        out_channels=1, 
        channels=[32, 64, 128, 256, 512], 
        strides=[2, 2, 2, 2], 
        num_res_units=2,
        freeze_backbone=False,
        warmup_epochs=5,
        pretrained_weights_path=None
    ):
        
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.warmup_epochs = warmup_epochs
        self.pretrained_weights_path = pretrained_weights_path
        
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

        bottleneck_dim = channels[-1]
        self.conv_gru = ConvGRUCell(input_dim=bottleneck_dim, hidden_dim=bottleneck_dim)
        
        self.h_state = None
        self._is_patched = False

        if pretrained_weights_path is not None:
            self.load_spatial_weights(pretrained_weights_path, device='cpu')

    def load_spatial_weights(self, path, device):
        
        if os.path.exists(path):
            self.unet.load_state_dict(torch.load(path, map_location=device))
            print(f"[*] [RecurrentUNet]: Pesi pre-addestrati spaziali caricati con successo.")
        else:
            print(f"[!] [RecurrentUNet]: File di checkpoint {path} non trovato. Inizializzazione casuale.")
        
        self.patch_bottleneck()

    def patch_bottleneck(self):
        if self._is_patched:
            return
        self._inject_temporal_bottleneck(self.unet.model)
        self._is_patched = True
        print("[*] [RecurrentUNet]: Spatial-temporal bottleneck patch applied successfully.")

    def _inject_temporal_bottleneck(self, module):
        
        # Caso 1: Siamo nel nodo radice, ovvero il Sequential principale dell'UNet.
        # Il blocco che ci interessa per scendere in profondità è all'indice 1.
        if isinstance(module, nn.Sequential) and len(module) >= 3:
            return self._inject_temporal_bottleneck(module[1])
            
        # Caso 2: Siamo all'interno di una SkipConnection.
        if hasattr(module, 'submodule'):
            # Se il sottomodulo è un Sequential, non siamo ancora sul fondo.
            # In MONAI, l'indice [1] di questo contenitore è la SkipConnection successiva.
            if isinstance(module.submodule, nn.Sequential):
                return self._inject_temporal_bottleneck(module.submodule[1])
            
            # Se non è un Sequential, abbiamo raggiunto il vero bottleneck (il ResidualUnit).
            else:
                original_bottleneck = module.submodule 
                module.submodule = InjectedBottleneck(original_bottleneck, self)
                return True
                
        return False

    def forward(self, x, h_prev=None):
        
        if not self._is_patched:
            self.patch_bottleneck()
            
        self.h_state = h_prev
        logits = self.unet(x)
        return logits, self.h_state