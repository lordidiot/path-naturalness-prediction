# A lot taken from https://github.com/lightmatmul/Transformer-from-scratch/
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention Module as described in the paper "Attention is All You Need".
    It takes in model size and number of heads.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculate the attention weights and return the weighted sum of values.
        """
        # Compute dot product of Q and K transposed for each head (scaled by sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x, batch_size):
        """
        Reverses the operation performed by split_heads.
        """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Perform linear operations and split into num_heads
        Q = self.split_heads(self.W_q(Q), batch_size)
        K = self.split_heads(self.W_k(K), batch_size)
        V = self.split_heads(self.W_v(V), batch_size)

        # Apply scaled dot product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine the attention output heads into a single matrix
        attn_output = self.combine_heads(attn_output, batch_size)

        # Final linear layer
        output = self.W_o(attn_output)
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network described in "Attention is All You Need".
    This consists of two dense layers with a ReLU activation in between.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Parameters:
            d_model (int): The size of the input and output dimensions.
            d_ff (int): The size of the hidden layer dimensions.
            dropout (float): Dropout rate.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        The forward method for PositionWiseFeedForward.
        Applies two linear transformations with a ReLU activation in between,
        with dropout applied after the first linear and the ReLU activation.
        """
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor to introduce information about the position of tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that they can be summed.
    Uses sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEncoding layer.
        Parameters:
            d_model (int): The dimension of the embeddings.
            max_len (int): The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Create a long enough 'pe' matrix that can be sliced according to actual sequence lengths.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # .transpose(0, 1)
        # Register pe as a buffer that is not a model parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Applies the positional encoding to the input embeddings.
        Arguments:
            x (Tensor): The input embeddings (batch_size, seq_len, d_model).
        Returns:
            Tensor: The embeddings with positional encoding added, with dropout applied.
        """
        # Add positional encoding to each embedding and apply dropout.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """
    Represents one layer of the Transformer encoder stack.
    Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        Perform forward pass of the encoder layer.
        Parameters:
            src (Tensor): Input tensor to the encoder layer.
            src_mask (Tensor): Mask to be applied on the input tensor.
        Returns:
            Tensor: Output tensor of the encoder layer.
        """
        # Apply self attention
        attn_output = self.self_attn(src, src, src, src_mask)
        # Add & norm
        src = self.norm1(src + self.dropout(attn_output))
        # Apply feed forward network
        output = self.feed_forward(src)
        # Second add & norm
        src = self.norm2(src + self.dropout(output))
        return src

class Transformer(nn.Module):
    """
    The Transformer model follows the architecture described in "Attention is All You Need".
    It includes an encoder and a decoder, each composed of a stack of layers.
    """

    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        """
        Initialize the Transformer model.
        Parameters:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): The dimensionality of the input/output tokens.
            num_heads (int): The number of heads in the multi-head attention models.
            num_layers (int): The number of encoder and decoder layers.
            d_ff (int): The dimensionality of the feed-forward layer.
            max_seq_length (int): The maximum length of the input sequences.
            dropout (float): The dropout rate.
        """
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.max_seq_length = max_seq_length
        # self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, lengths):
        """
        Creates a mask for the source sequence to ignore the padding tokens.
        """
        src_mask = torch.arange(self.max_seq_length).unsqueeze(0) < lengths.unsqueeze(1)
        src_mask = src_mask.unsqueeze(1).unsqueeze(3)
        return src_mask

    def forward(self, src, lengths):
        """
        Forward pass of the Transformer model.
        """
        src_mask = self.make_src_mask(lengths)
        src = self.dropout(self.positional_encoding(src))
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        pooled = enc_output.mean(dim=1)
        return pooled