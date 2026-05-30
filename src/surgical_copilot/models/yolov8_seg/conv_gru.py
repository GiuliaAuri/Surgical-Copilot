import torch
import torch.nn as nn

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