import torch
import torch.nn as nn
from typing import List, Tuple, Optional

import logging

logger = logging.getLogger("ConvGRU")

class ConvGRUCell(nn.Module):
    """
    Una singola cella ConvGRU. 
    Prende in input le feature spaziali correnti e la memoria del frame precedente.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 2, kernel_size, padding=padding)
        self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.shape[0], self.hidden_dim, x.shape[2], x.shape[3], device=x.device)

        combined = torch.cat([x, h_prev], dim=1)
        
        gates = self.conv_gates(combined)
        z_gate, r_gate = torch.split(gates, self.hidden_dim, dim=1)
        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)
        
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_candidate(combined_reset))
        
        h_next = (1 - z) * h_prev + z * h_candidate
        
        return h_next
    

# ---------------------------------------------------------------------------
# ConvGRU — multi-layer, multi-step
# ---------------------------------------------------------------------------

class ConvGRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_sizes: List[int | Tuple[int, int]],
        bias: bool = True,   # kept for API compatibility (unused)
        return_sequence: bool = True,
    ):
        super().__init__()

        assert len(hidden_channels) == len(kernel_sizes)

        self.num_layers = len(hidden_channels)
        self.return_sequence = return_sequence

        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            c_in = in_channels if l == 0 else hidden_channels[l - 1]
            self.cells.append(
                ConvGRUCell(c_in, hidden_channels[l], kernel_sizes[l])
            )

    def forward(
        self,
        X: torch.Tensor,
        initial_states: Optional[List[torch.Tensor]] = None,
    ):
        B, T, C, H, W = X.shape
        logger.debug(f"[ConvGRU] Input shape: {X.shape}")

        if initial_states is None:
            states = [None] * self.num_layers
            logger.debug("[ConvGRU] No initial states provided, initializing with zeros.")
        else:
            states = list(initial_states)

        layer_outputs = []
        current_input = X

        for l, cell in enumerate(self.cells):
            h = states[l]
            if h is None:
                h = torch.zeros(
                    B, cell.hidden_dim, H, W,
                    device=X.device,
                    dtype=X.dtype
                )
            
            logger.debug(f"[ConvGRU] Layer {l} processing. Input: {current_input.shape}, Hidden: {h.shape}")

            h_list = []
            for t in range(T):
                h = cell(current_input[:, t], h)
                h_list.append(h)

            seq = torch.stack(h_list, dim=1)
            
            if not self.return_sequence:
                seq = seq[:, -1]
                logger.debug(f"[ConvGRU] Layer {l} returned last frame: {seq.shape}")
            else:
                logger.debug(f"[ConvGRU] Layer {l} returned sequence: {seq.shape}")

            layer_outputs.append(seq)
            current_input = seq
            states[l] = h

        last_layer_output = layer_outputs[-1]
        logger.debug(f"[ConvGRU] Final output shape: {last_layer_output.shape}")
        
        return last_layer_output, states