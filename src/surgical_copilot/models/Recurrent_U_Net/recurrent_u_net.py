from monai.networks.nets.unet import UNet

from surgical_copilot.models.conv_gru import ConvGRUCell
from surgical_copilot.models.conv_lstm import ConvLSTMCell

import torch
import torch.nn as nn
import os

class RecurrentWrapper(nn.Module):
    """
    Uniform interface:

        output, new_state = recurrent(x, state)

    GRU:
        state = h_t

    LSTM:
        state = (h_t, c_t)
    """

    def __init__(self, recurrent_type, channels):
        super().__init__()

        self.recurrent_type = recurrent_type

        if recurrent_type == "gru":
            self.cell = ConvGRUCell(
                input_dim=channels,
                hidden_dim=channels
            )

        elif recurrent_type == "lstm":
            self.cell = ConvLSTMCell(
                in_channels=channels,
                hidden_channels=channels,
                kernel_size=3
            )

        else:
            raise ValueError(
                f"Unsupported recurrent_type: {recurrent_type}"
            )

    def forward(self, x, state=None):

        if self.recurrent_type == "gru":
            h_t = self.cell(x, state)
            return h_t, h_t

        elif self.recurrent_type == "lstm":
            h_t, c_t = self.cell(x, state)
            return h_t, (h_t, c_t)
        
class InjectedBottleneck(nn.Module):
    def __init__(self, spatial_layer, recurrent_wrapper):
        super().__init__()

        self.spatial_layer = spatial_layer
        self.recurrent_wrapper = recurrent_wrapper
        self.h_state = None 

    def forward(self, x):

        # Spatial Feature extraction 
        x_spatial = self.spatial_layer(x)

        # Temporal propagation
        output, new_state = self.recurrent_wrapper(x_spatial, self.h_state)
        
        # State updatre
        self.h_state = new_state

        return output

class RecurrentUNet(nn.Module):

    def __init__(
        self, 
        spatial_dims=2, 
        in_channels=3, 
        out_channels=1, 
        channels=[32, 64, 128, 256, 512], 
        strides=[2, 2, 2, 2], 
        num_res_units=2,
        recurrent_type="gru",
        freeze_backbone=False,
        warmup_epochs=5,
        pretrained_weights_path=None
    ):
        
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.warmup_epochs = warmup_epochs
        self.pretrained_weights_path = pretrained_weights_path

        self.recurrent_type = recurrent_type
        
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

        bottleneck_dim = channels[-1]

        self.recurrent = RecurrentWrapper(
            recurrent_type=recurrent_type,
            channels=bottleneck_dim
        )

        self.recurrent_state = None

        self._is_patched = False

        if pretrained_weights_path is not None:
            self.load_spatial_weights(pretrained_weights_path, device='cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # we are in the root node of the UNet a Sequential with 5 ResidualUnits, and the bottleneck is the second one (index 1).
        if isinstance(module, nn.Sequential) and len(module) >= 3:
            return self._inject_temporal_bottleneck(module[1])
            
        # we are in a ResidualUnit, we check if it is the bottleneck (the one with the highest number of channels, which is the last one in the UNet).
        if hasattr(module, 'submodule'):

            # if submodule is a Sequential, we are not yet at the bottom.
            # In MONAI, index [1] of this container is the next SkipConnection.
            if isinstance(module.submodule, nn.Sequential):
                return self._inject_temporal_bottleneck(module.submodule[1])
            
            # if not a Sequential, we have reached the actual bottleneck (the ResidualUnit).
            else:
                original_bottleneck = module.submodule 
                self.injected_module = InjectedBottleneck(original_bottleneck, self.recurrent)
                module.submodule = self.injected_module

                return True
                
        return False

    def forward(self, x, h_prev=None):
        
        if not self._is_patched:
            self.patch_bottleneck()
        
        # if the frame is the first of the sequence, we reset the hidden state of the 
        # recurrent module to avoid information leakage between sequences.
        if h_prev is None and hasattr(self, 'injected_module'):
            self.injected_module.h_state = None
         
        if hasattr(self, 'injected_module'):
             self.injected_module.h_state = h_prev
             
        logits = self.unet(x)
        current_state = self.injected_module.h_state if hasattr(self, 'injected_module') else None
        
        return logits, current_state