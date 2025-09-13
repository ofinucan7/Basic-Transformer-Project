import torch
import torch.nn as nn

class TransformerAddNorm(nn.Module):
    def __init__(self, embedding_dims, ffn_hidden_dims, dropout_rate):
        # init function
        # Inputs: 1.) self
        #         2.) embedding_dims (int) - number of embedding dims
        #         3.) ffn_hidden_dims (int) - hidden size for FFN
        #         4.) dropout_rate (float) - dropout rate as decimal

        super().__init__()

        # define layer normalization layers
        self.layer_normal_1 = nn.LayerNorm(embedding_dims)
        self.layer_normal_2 = nn.LayerNorm(embedding_dims)

        # make 2-layer position-wise FFN
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dims, ffn_hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dims, embedding_dims),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, attention_out):
        # forward pass through Add & Norm Block
        # Inputs: 1.) self
        #         2.) x (tensor of shape [batch, seq_length, embedding_dims]) - input to attention layer
        #         3.) attention_out (tensor of shape [batch, seq_len, embedding_dims]) - output from attention layer
        # Returns: 1.) tensor after residual, form, ffn, residual, norm w/ shape [batch, seq_length, embedding_dims]

        # add x to attention out then take layer norm
        x = self.layer_normal_1(x + attention_out)

        # plug x into the FFN
        ffn_out = self.ffn(x)

        # second layer norm but w/ x plus output of FFN
        x = self.layer_normal_2(x +  ffn_out)

        return x