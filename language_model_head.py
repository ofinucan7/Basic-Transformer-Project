import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class LanguageModelHead(nn.Module):
    def __init__(self, vocab_size, embedding_dims, num_heads, ffn_hidden_dims, dropout_rate, num_layers):
        # Inputs: vocab_size (int), embedding_dims (int), num_heads (int), ffn_hidden_dims (int), dropout_rate (float; decimal percent), num_layers (int)
        super().__init__()

        # create num_layers number of TransformerBlocks
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dims, num_heads, ffn_hidden_dims, dropout_rate) for _ in range(num_layers)])

        self.final_layer_norm = nn.LayerNorm(embedding_dims)

        # project back to vocab size
        self.language_model_head = nn.Linear(embedding_dims, vocab_size)

    def forward(self, x):
        # loop over all the TranformerBlocks and do the transformer block steps
        for block in self.blocks:
            x = block(x)

        # Do one final layer normalization
        x = self.final_layer_norm(x)

        # get the unscaled probs of all next potential tokens (softmax still needs to be applied)
        unscaled_probs = self.language_model_head(x)

        return unscaled_probs