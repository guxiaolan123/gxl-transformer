import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码（保留原有实现）"""

    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力（无修改）"""

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力（无修改）"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output, attn_weights = ScaledDotProductAttention()(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_O(output)
        return output, attn_weights


class PositionWiseFFN(nn.Module):
    """位置-wise前馈网络（无修改）"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Encoder层（无修改）"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_output)
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)
        return x


class DecoderLayer(nn.Module):
    """Decoder层（无修改）"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        x_norm = self.norm1(x)
        masked_attn_output, _ = self.masked_self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(masked_attn_output)
        x_norm = self.norm2(x)
        enc_dec_attn_output, _ = self.enc_dec_attn(x_norm, enc_output, enc_output, src_mask)
        x = x + self.dropout2(enc_dec_attn_output)
        x_norm = self.norm3(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout3(ffn_output)
        return x


class Transformer(nn.Module):
    """完整Transformer（新增use_pos_encoding参数控制位置编码）"""

    def __init__(self, vocab_size, d_model=128, num_heads=4, num_enc_layers=2, num_dec_layers=2, d_ff=512,
                 max_seq_len=5000, dropout=0.1, use_pos_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding  # 控制是否启用位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout) if use_pos_encoding else None
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_enc_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_dec_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        """生成掩码（无修改）"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        future_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=src.device), diagonal=1)
        tgt_mask = tgt_mask & (~future_mask)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 编码器前向（根据参数决定是否添加位置编码）
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        if self.use_pos_encoding:
            src_emb = self.dropout(self.pos_encoding(src_emb))
        else:
            src_emb = self.dropout(src_emb)
        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # 解码器前向（根据参数决定是否添加位置编码）
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.use_pos_encoding:
            tgt_emb = self.dropout(self.pos_encoding(tgt_emb))
        else:
            tgt_emb = self.dropout(tgt_emb)
        dec_output = tgt_emb
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, src_mask)

        output = self.fc_out(dec_output)
        return output