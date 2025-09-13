import torch
import torch.nn as nn

class Self_Attention(nn.Module):

    def __init__(self, embedding_dims, num_heads, dropout_rate):
        # init function
        # Inputs: 1.) self
        #         2.) embedding_dims (int) - number of embedding dimensions
        #         3.) num_heads (int) - number of heads
        #         4.) dropout_rate (float) - decimal of how often to dropout
        
        super().__init__()

        # double check embedding_dims is divisible evenly by num_heads
        assert embedding_dims % num_heads == 0

        # define embedding_dims, num_heads, and head_dims
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.head_dims = embedding_dims // num_heads

        # instead of having 3 seperate q, k, and v tensors --> combined into 1 projection to be split later (hence 3*embedding_dims)
        self.qkv_proj = nn.Linear(embedding_dims, 3*embedding_dims)

        # Final output projection
        self.out_proj = nn.Linear(embedding_dims, embedding_dims)

        # set the dropout rates for the attention dropouts
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # forward pass through attention block
        # Inputs: 1.) self
        #         2.) x (tensor of shape [batch, seq_len, embedding_dims])
        # Returns: 1.) linear projection of attention_output (tensor of shape [batch, seq_len, embedding_dims])

        # unpack input shape
        batch, seq_len, _ = x.shape # note: x's shape is (batch, seq_len, embedding_dims)

        # single linear layer to compute concatinated Q, K, V for all tokens (3.2.2 attention is all you need)
        qkv = self.qkv_proj(x) # (batch, seq_len, 3 * embedding_dims)

        # reorder and reshape to go from (batch, seq_len, 3*embedding_dims) to (batch, seq_len, 3*head_dims) after .view to (batch, heads, seq_len, 3*head_dims) after transpose
        # (3.2.2 attention is all you need) 
        qkv = qkv.view(batch, seq_len, self.num_heads, 3 * self.head_dims).transpose(1, 2) # (batch, num_heads, seq_len, 3 * head_dims)

        # seperate out query, key, and value
        q, k, v = qkv.chunk(3, dim=-1) # each one is (batch, heads, seq_len, head_dims)

        # inside of the softmax equation in 3.2.1 attention is all you need
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dims ** 0.5) # (batch, heads, seq_len, seq_len)

        # make a triangular matrix with 1s above the diagonal (mask future positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()  # (seq_len, seq_len)
        
        # where the mask is true, set the score to -inf so when applying softmax sets equal to ~0
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # apply softmax to the attention scores (still formula from 3.2.1)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1) # (batch, heads, seq_len, seq_len)

        # do first dropout
        attention_weights = self.attention_dropout(attention_weights) # (batch, heads, seq_len, seq_len)

        # multiply by v (formula 3.2.1)
        attention_output = torch.matmul(attention_weights, v) # (batch, heads, seq_len, head_dims)

        # concatinate (need to reshape first)
        # (batch, heads, seq_len, head_dims) --> transpose to (batch, seq_len, heads, head_dims) --> .view to reshape to (batch, seq_len, embedding_dims) by flattening (embed_dims = heads*head_dims)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch, seq_len, self.embedding_dims)

        # do final linear projection and after do last dropout
        return self.output_dropout(self.out_proj(attention_output))