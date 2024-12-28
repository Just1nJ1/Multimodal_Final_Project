import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension
        self.register_buffer('positional_encoding', self.encoding)  # Register as buffer

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)  # Sequence length
        return x + self.encoding[:, :T, :].to(x.device)  # Add positional encoding
