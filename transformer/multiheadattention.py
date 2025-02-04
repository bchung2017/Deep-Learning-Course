import torch.nn as nn
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        
        query = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        query = query * self.scaling

        scores = torch.einsum("bhqd, bhkd -> bhqk", query, key)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            if torch.isinf(scores).all(dim=-1).any():
                print("-inf after masking")

        attn_weights = torch.softmax(scores, dim=-1)

        attention_output = torch.einsum("bhqk, bhvd -> bhqd", attn_weights, value)

        attention_output = attention_output.contiguous().view(batch_size, seq_length, embed_dim)
        return self.output(attention_output)
