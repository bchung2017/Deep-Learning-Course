import torch
import torch.nn as nn
from multiheadattention import MultiheadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None):
        attn_output = self.attention(x, src_mask)
        x = self.layer_norm1(x + attn_output)

        feedforward_output = self.feedforward(x)
        x = self.layer_norm2(x + feedforward_output)
        return x
    
