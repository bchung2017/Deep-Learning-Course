import torch
import torch.nn as nn
import math
from transformerblock import TransformerBlock
from positional_encoding import PositionalEncoding


class NextWordPredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, hidden_dim):
        super(NextWordPredictionModel, self).__init__()
        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, x, src_mask):
        embedded = self.embedding(x) * 0.1
        batch_size, seq_length, embed_dim = embedded.size()
        
        for transformer_block in self.layers:
            embedded = transformer_block(embedded, src_mask)
        
        return self.output(embedded)

    @staticmethod
    def create_mask(seq_length, batch_size, num_heads):
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        
        mask = mask.unsqueeze(0).unsqueeze(0)

        mask = mask.repeat(batch_size, num_heads, 1, 1)
        
        return mask
