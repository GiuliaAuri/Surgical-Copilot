import torch
import torch.nn as nn
import os
from monai.networks.nets.unet import UNet

from surgical_copilot.models.conv_gru import ConvGRU
from surgical_copilot.models.conv_lstm import ConvLSTM

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
        # x shape: (B, T, C, H, W)
        output_seq, _ = self.rnn(x, states)
        return output_seq


class InjectedBottleneck(nn.Module):
    def __init__(self, spatial_layer, recurrent_wrapper):
        super().__init__()

        self.spatial_layer = spatial_layer
        self.recurrent_wrapper = recurrent_wrapper
                
        self.current_B = 1
        self.current_T = 1

    def forward(self, x):

        x_spatial = self.spatial_layer(x)

        B, C, H, W = x_spatial.shape
        x_seq = x_spatial.view(self.current_B, self.current_T, C, H, W)

        output_seq = self.recurrent_wrapper(x_seq)[0]

        output_flat = output_seq.contiguous().view(B, C, H, W)

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
            print(f"[*] [RecurrentUNet]: Pesi pre-addestrati spaziali caricati con successo.")
        else:
            print(f"[!] [RecurrentUNet]: File di checkpoint {path} non trovato. Inizializzazione casuale.")
        
    def patch_bottleneck(self):
        if self._is_patched:
            return
        
        self._inject_temporal_bottleneck(self.unet.model)
        self._is_patched = True
        print("[*] [RecurrentUNet]: Spatial-temporal bottleneck patch applied successfully.")

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

    def forward(self, x):
        B, T, C, H, W = x.shape

        self.injected_module.current_B = B
        self.injected_module.current_T = T

        x_flat = x.view(B * T, C, H, W)

        logits_flat = self.unet(x_flat)

        _, C_out, H_out, W_out = logits_flat.shape
        logits_seq = logits_flat.view(B, T, C_out, H_out, W_out)

        return logits_seq

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 1. Config modello
    # =========================
    model = RecurrentUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=[32, 64, 128, 256, 512],
        strides=[2, 2, 2, 2],
        num_res_units=2,
        recurrent_type="gru",
        recurrent_layers=1,
    ).to(device)

    model.eval()

    # =========================
    # 2. Fake input sequence
    # =========================
    B = 2
    T = 5
    C = 3
    H = 256
    W = 256

    x = torch.randn(B, T, C, H, W).to(device)

    # =========================
    # 3. Forward pass
    # =========================
    with torch.no_grad():
        logits = model(x)

    # =========================
    # 4. Check output
    # =========================
    print("\n===== OUTPUT SHAPE =====")
    print("Input :", x.shape)
    print("Output:", logits.shape)

    # sanity check
    assert logits.shape[0] == B
    assert logits.shape[1] == T

    print("\nOK: forward pass completato correttamente.")


if __name__ == "__main__":
    main()