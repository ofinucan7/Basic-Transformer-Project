import torch
import torch.nn as nn
from attention import Self_Attention 

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads, ffn_hidden_dims, dropout_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dims)
        self.attn = Self_Attention(embedding_dims, num_heads, dropout_rate)
        self.drop1 = nn.Dropout(dropout_rate)

        self.ln2 = nn.LayerNorm(embedding_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dims, ffn_hidden_dims),
            nn.GELU(),
            nn.Linear(ffn_hidden_dims, embedding_dims),
        )
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x