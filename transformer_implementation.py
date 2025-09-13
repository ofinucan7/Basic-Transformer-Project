import torch
import math
import torch.nn as nn
from transformer_block import TransformerBlock
from bpe_position_encoding import get_positional_encoding

class TransformerImplementation(nn.Module):
    def __init__(self, vocab_size, embedding_dims, num_heads, ffn_hidden_dims, dropout_rate, num_layers, max_seq_len=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.embed_dropout = nn.Dropout(dropout_rate)

        pe = get_positional_encoding(max_seq_len, embedding_dims, device="cpu")  # (max_seq_len, d_model)
        self.register_buffer("pos_enc", pe, persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dims, num_heads, ffn_hidden_dims, dropout_rate)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(embedding_dims)
        self.lm_head = nn.Linear(embedding_dims, vocab_size, bias=True)
        self.lm_head.weight = self.embedding.weight
        self.scale = math.sqrt(embedding_dims)

    def forward(self, token_ids):
        x = self.embedding(token_ids) * self.scale  # (batch, seq_len) ; scale by sqrt(d_model)

        seq_len = x.size(1)
        x = x + self.pos_enc[:seq_len, :].unsqueeze(0).to(x.device)  # (1, seq_len, d)

        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        return self.lm_head(x)  # (batch, seq_len, vocab_size)