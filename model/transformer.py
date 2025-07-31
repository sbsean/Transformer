import torch
import torch.nn as nn
import torch.nn.functional as F
import math

" Mask Creation Utilities "
'''
배치 처리 시 서로 다른 길이의 시퀀스를 동일한 길이로 맞추기 위해 패딩을 사용
패딩 토큰이 attention 계산에 영향을 주지 않도록 마스킹
padding mask : 패딩 토큰을 0으로 처리하고, 패딩 토큰이 아닌 경우 1로 처리

causal mask : 현재 토큰 이후의 토큰은 참조할 수 없도록 마스크 처리
'''
def create_padding_mask(seq, pad=0):
    """Create padding mask for sequences"""
    '''
    seq != 0 : 패딩 토큰이 아닌 경우 1(True)로 처리
    '''
    return (seq != pad).unsqueeze(1).unsqueeze(2) # (B, seq_len) --> (B, 1, 1, seq_len)

def create_causal_mask(size, device=None):
    """Create causal mask for decoder self-attention"""
    '''
    [[0, 1, 1],
     [0, 0, 1],
     [0, 0, 0]]
    '''
    if device is None:
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool() # 상삼각행렬 생성 (대각선 위쪽만 1, 나머지는 0)
    else:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool() # 상삼각행렬 생성 (대각선 위쪽만 1, 나머지는 0)
    return mask


"Scaled Dot-Product Attention"
'''
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
Q, K, V : 입력 토큰들의 임베딩 텐서 (B, sequence_length, d_model)
mask : 패딩 토큰이 아닌 경우 1로 처리
'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model #model의 embedding 차원

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q)

        K = self.W_k(K)

        V = self.W_v(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)

        return output


" Multi-Head Attention "
'''
Ensemble 효과 : 여러 모델을 사용해서 더 다양한 표현을 획득
다양한 관점 : 각 head가 서로 다른 가중치를 학습하여 다양한 관점에서 attention 계산
Concatenation : 모든 head의 결과를 연결하여 풍부한 표현 생성성
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)

        self.W_k = nn.Linear(d_model, d_model)

        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q)

        K = self.W_k(K)

        V = self.W_v(V)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        K = K.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        V = V.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model // self.num_heads)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(output)

        return output


" Position-wise Feed-Forward Networks (MLP)"
'''
512 --> 2048 --> 512 (풍부한 표현 생성)
'''
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff        

        self.W_1 = nn.Linear(d_model, d_ff) 
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.W_2(self.dropout(F.relu(self.W_1(x))))
    
" Embeddings and Softmax "
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  

class Softmax(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Softmax, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        logits = self.proj(x)
        return F.softmax(logits, dim=-1)
    
" Positional Encoding 정리"
'''
PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
e^ln(A) = A 
div_term : exp(X*C) : exp(2i*(-ln(10000)/d_model)) = (10000)^(-2i/d_model)
2i : torch.arange(0, d_model, 2).float() : [0, 2, 4, ..., d_model - 2]

'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        self.max_len = max_len #model이 처리할 수 있는 시퀀스의 최대 길이

        self.positional_encoding = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # positional encoding 벡터의 짝수 차원 채우기
        # 0::2 0부터 두칸씩
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # encoding 벡터의 홀수 차원 채우기
        # 1::2 1부터 두칸씩
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        #(max_len, d_model) --> (1, max_len, d_model)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

        
    def forward(self, x):
        # x : 입력 토큰들의 임베딩 텐서 (B, sequence_length, d_model)
        # x + ... : 
        return x + self.positional_encoding[:, :x.size(1), :].expand(x.size(0), -1, -1).to(x.device).detach()

"Encoder Layer"

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection(Gradient Vanishing 방지)
        attn_output = self.attention(x, x, x, mask)
        # 논문 Section 3.1 (the output of each sub-layer)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
" Encoder " # a = nn.TransformerEncoder
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers # 6
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        
        self.embedding = Embeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.softmax = Softmax(d_model, vocab_size)
        
    
        
    def forward(self, src, src_mask):   
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

" Decoder "
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Self-attention sub-layer
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # Cross-attention sub-layer  
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        # Feed-forward sub-layer
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Layer normalization for each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-attention sub-layer with residual connection
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention sub-layer with residual connection
        attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward sub-layer with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
    
" Decoder "
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        
        self.embedding = Embeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.softmax = Softmax(d_model, vocab_size)
        
    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        for layer in self.layers:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
" Transformer "
class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6, src_vocab_size=30000, tgt_vocab_size=30000, max_len=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.dropout = dropout
        
        # Encoder: N=6 identical layers
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size, max_len, dropout)
        # Decoder: N=6 identical layers  
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len, dropout)
        # Output projection layer
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = create_padding_mask(tgt) & create_causal_mask(tgt.size(1), device=tgt.device)
            
        # Encoder forward pass
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder forward pass with causal masking
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        
        return output
    
    











