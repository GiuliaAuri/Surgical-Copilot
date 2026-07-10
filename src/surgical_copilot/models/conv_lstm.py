"""
Convolutional LSTM Network
Implementation following:
  Shi et al., "Convolutional LSTM Network: A Machine Learning Approach
  for Precipitation Nowcasting", NeurIPS 2015 (arXiv:1506.04214)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

import logging
logger = logging.getLogger("ConvLSTM")


class ConvLSTMCell(nn.Module):
    """
    All inputs X_t, cell outputs C_t and hidden states H_t are 3-D tensors
    with shape (batch, channels, height, width).

    Peephole connections (W_ci, W_cf, W_co) are implemented as element-wise
    products with learnable vectors broadcast over the spatial dimensions,
    exactly as in eq. (3).

    Zero-padding on the hidden states is used so
    that the spatial dimensions are preserved after every convolution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int | Tuple[int, int],
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # Same-padding so that H and W are preserved after convolution
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Input-to-state convolution: produces gates i, f, c_tilde, o
        # Four gates concatenated → 4 * hidden_channels output channels
        self.conv_x = nn.Conv2d(
            in_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # State-to-state convolution (W_hi, W_hf, W_hc, W_ho)
        self.conv_h = nn.Conv2d(
            hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False, # bias already added by conv_x
        )

        # Peephole weights: shape (1, hidden_channels, 1, 1) → broadcast
        # W_ci and W_cf are used before the cell update (depend on C_{t-1})
        # W_co is used after  the cell update (depends on C_t)
        self.W_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.W_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.W_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def forward(
        self,
        X: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X     : (B, C_in, H, W)
        state : (H_{t-1}, C_{t-1}), each (B, hidden, H, W)
                If None the cell is initialised to all-zeros ("total ignorance").

        Returns
        -------
        H_t, C_t : each (B, hidden, H, W)
        """
        B, _, H, W = X.shape

        if state is None:
            H_prev = X.new_zeros(B, self.hidden_channels, H, W)
            C_prev = X.new_zeros(B, self.hidden_channels, H, W)
        else:
            H_prev, C_prev = state

        # Combined linear transform
        gates = self.conv_x(X) + self.conv_h(H_prev)           # (B, 4*hid, H, W)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1) # each (B, hid, H, W)

        # Peephole connections from C_{t-1}
        i_t = torch.sigmoid(i_gate + self.W_ci * C_prev)       # input  gate
        f_t = torch.sigmoid(f_gate + self.W_cf * C_prev)       # forget gate

        C_t = f_t * C_prev + i_t * torch.tanh(g_gate)          # cell state

        # Peephole connection from C_t (note: C_t, not C_{t-1})
        o_t = torch.sigmoid(o_gate + self.W_co * C_t)          # output gate

        H_t = o_t * torch.tanh(C_t)                            # hidden state

        return H_t, C_t


# ---------------------------------------------------------------------------
# ConvLSTM — multi-layer, multi-step
# ---------------------------------------------------------------------------

class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM that processes a full temporal sequence.

    Each layer is a ConvLSTMCell.  Hidden states are propagated along the
    time axis; outputs of layer l are the inputs of layer l+1.

    Parameters
    ----------
    in_channels     : C of the input tensor X_t
    hidden_channels : list of ints, one per layer
    kernel_sizes    : list of kernel sizes (int or tuple), one per layer
    bias            : whether to include bias in convolutions
    return_sequence : if True  → return all H_t for each layer
                      if False → return only the last H_t
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_sizes: List[int | Tuple[int, int]],
        bias: bool = True,
        return_sequence: bool = True,
    ) -> None:
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes), (
            "hidden_channels and kernel_sizes must have the same length"
        )

        self.num_layers = len(hidden_channels)
        self.return_sequence = return_sequence

        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            c_in = in_channels if l == 0 else hidden_channels[l - 1]
            self.cells.append(
                ConvLSTMCell(c_in, hidden_channels[l], kernel_sizes[l], bias)
            )

    def forward(
        self,
        X: torch.Tensor,
        initial_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Parameters
        ----------
        X              : (B, T, C_in, H, W)  — sequence of T frames
        initial_states : optional list of (H_0, C_0) per layer

        Returns
        -------
        layer_outputs  : list (one per layer) of tensors (B, T, C_hid, H, W)
                         or (B, C_hid, H, W) if return_sequence=False
        last_states    : list (one per layer) of (H_T, C_T)
        """

        B, T, C, H, W = X.shape
        #print(f"[ConvLSTM] Input shape: {X.shape}")

        if initial_states is None:
            states = [None] * self.num_layers
            #print("[ConvLSTM] No initial states, using None (lazy initialization in cell).")
        else:
            states = list(initial_states)
            #print(f"[ConvLSTM] Initial states provided for {len(states)} layers.")

        layer_outputs: List[torch.Tensor] = []
        current_input = X

        for l, cell in enumerate(self.cells):
            h_list: List[torch.Tensor] = []
            state = states[l]

            #print(f"[ConvLSTM] Layer {l} processing. Input: {current_input.shape}")
            #print(f"[ConvLSTM] state type: {type(state)}")
            #print(f"[ConvLSTM] is tensor: {isinstance(state, torch.Tensor)}")

            #if isinstance(state, torch.Tensor):
            #    print(f"[ConvLSTM] state.shape: {state.shape}")
            #elif isinstance(state, tuple):
            #    print(f"[ConvLSTM] tuple shapes: {[x.shape for x in state]}")

            #for i, s in enumerate(states):
            #    print(f"Layer index: {i}, State type: {type(s)}")

            for t in range(T):
                x_t = current_input[:, t]
                h_t, c_t = cell(x_t, state)
                state = (h_t, c_t)
                h_list.append(h_t)

            #print("DEBUG STATE AFTER LAYER")
            #print(type(states[l]))

            #if isinstance(states[l], tuple):
            #    print("tuple length:", len(states[l]))
            #    print(states[l][0].shape)
            #    print(states[l][1].shape)

            states[l] = state
            
            # Stack along time: (B, T, C_hid, H, W)
            seq = torch.stack(h_list, dim=1)
            
            if not self.return_sequence:
                seq = seq[:, -1]
                #print(f"[ConvLSTM] Layer {l} returned last frame: {seq.shape}")
            #else:
            #    print(f"[ConvLSTM] Layer {l} returned sequence: {seq.shape}")
            #    pass

            layer_outputs.append(seq)
            current_input = seq

        last_layer_output = layer_outputs[-1]
        #last_states = states[-1]

        #print(f"[ConvLSTM] Final output: {last_layer_output.shape}")
        #print(f"[ConvLSTM] Final states: {[s[0].shape for s in last_states]}")
        
        return last_layer_output, states
