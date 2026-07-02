import torch
import torch.nn as nn
import os
import logging
from monai.networks.nets.unet import UNet

from src.surgical_copilot.models.conv_gru import ConvGRU
from src.surgical_copilot.models.conv_lstm import ConvLSTM

# Configurazione del logger
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: [%(name)s] %(message)s')
logger = logging.getLogger("RecurrentUNet")

class RecurrentWrapper(nn.Module):

    def __init__(self, recurrent_type, in_channels, hidden_channels=[256], kernel_sizes=[3]):
        super().__init__()

        self.recurrent_type = recurrent_type

        if recurrent_type == "gru":
            self.rnn = ConvGRU(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_sizes=kernel_sizes,
                return_sequence=True
            )
        elif recurrent_type == "lstm":
            self.rnn = ConvLSTM(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_sizes=kernel_sizes,
                return_sequence=True
            )
        else:
            raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")

    def forward(self, x, states=None):
        # x shape attesa: (B, T, C, H, W)
        logger.debug(f"[RecurrentWrapper] Input shape (B, T, C, H, W): {x.shape}")
        
        output_seq, _ = self.rnn(x, states)
        
        logger.debug(f"[RecurrentWrapper] Output shape (B, T, C, H, W): {output_seq.shape}")
        return output_seq


class InjectedBottleneck(nn.Module):
    def __init__(self, spatial_layer, recurrent_wrapper):
        super().__init__()

        self.spatial_layer = spatial_layer
        self.recurrent_wrapper = recurrent_wrapper
                
        self.current_B = 1
        self.current_T = 1

    def forward(self, x):
        logger.debug(f"[InjectedBottleneck] Input flat shape (B*T, C, H, W): {x.shape}")
        
        x_spatial = self.spatial_layer(x)
        logger.debug(f"[InjectedBottleneck] Post-spatial layer shape (B*T, C_out, H_out, W_out): {x_spatial.shape}")

        B, C, H, W = x_spatial.shape
        
        x_seq = x_spatial.view(self.current_B, self.current_T, C, H, W)
        logger.debug(f"[InjectedBottleneck] Sequence unflattening per RNN (B={self.current_B}, T={self.current_T}): {x_seq.shape}")

        output_seq = self.recurrent_wrapper(x_seq)

        output_flat = output_seq.contiguous().view(B, C, H, W)
        logger.debug(f"[InjectedBottleneck] Output flat shape (ritorno alla U-Net): {output_flat.shape}")

        return output_flat


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
        recurrent_layers=1,      
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
            in_channels=bottleneck_dim,
            hidden_channels=[bottleneck_dim] * recurrent_layers,
            kernel_sizes=[3] * recurrent_layers
        )

        self._is_patched = False

        if pretrained_weights_path is not None:
            self.load_spatial_weights(pretrained_weights_path, device='cuda' if torch.cuda.is_available() else 'cpu')

        self.patch_bottleneck()


    def load_spatial_weights(self, path, device):
        if os.path.exists(path):
            self.unet.load_state_dict(torch.load(path, map_location=device))
            logger.info(f"Pesi pre-addestrati spaziali caricati con successo da {path}.")
        else:
            logger.warning(f"File di checkpoint {path} non trovato. Inizializzazione casuale.")
        
    def patch_bottleneck(self):
        if self._is_patched:
            return
        
        patched = self._inject_temporal_bottleneck(self.unet.model)
        self._is_patched = True
        
        if patched:
            logger.info("Spatial-temporal bottleneck patch applicata con successo.")
        else:
            logger.error("Impossibile applicare la patch. Struttura del sottomodulo non riconosciuta.")

    def _inject_temporal_bottleneck(self, module):
        if isinstance(module, nn.Sequential) and len(module) >= 3:
            return self._inject_temporal_bottleneck(module[1])
            
        if hasattr(module, 'submodule'):
            if isinstance(module.submodule, nn.Sequential):
                return self._inject_temporal_bottleneck(module.submodule[1])
            else:
                original_bottleneck = module.submodule 
                self.injected_module = InjectedBottleneck(original_bottleneck, self.recurrent)
                module.submodule = self.injected_module
                return True
                
        return False

    def forward(self, x, seq_len=None):
        
        assert seq_len is not None, "seq_len must be provided for the forward pass."
        
        logger.debug(f"[RecurrentUNet Forward] Input originale shape: {x.shape} con seq_len={seq_len}")
        
        BT, C, H, W = x.shape
        B = BT // seq_len 
        T = seq_len

        # Passaggio di stato per informare il bottleneck delle dimensioni sequenziali
        self.injected_module.current_B = B
        self.injected_module.current_T = T

        x_flat = x.view(B * T, C, H, W)
        
        # Forward dell'intera U-Net (che intercetterà il bottleneck modificato)
        logits_flat = self.unet(x_flat)
        logger.debug(f"[RecurrentUNet Forward] Logits flat da U-Net: {logits_flat.shape}")

        _, C_out, H_out, W_out = logits_flat.shape
        logits_seq = logits_flat.view(B, T, C_out, H_out, W_out)
        
        logger.debug(f"[RecurrentUNet Forward] Output sequenziale finale (B, T, C, H, W): {logits_seq.shape}")

        return logits_seq