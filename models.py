import torch
import torch.nn as nn

class PhysicsBiLSTMAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, out_steps=60, n_targets=4, num_layers=2, dropout=0.3, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        d_model = hidden_dim * 2
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        dropout_feed = dropout + 0.2
        self.ff_dropout = nn.Dropout(dropout_feed)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(negative_slope=0.01),
            self.ff_dropout
        )
        self.out_proj = nn.Linear(d_model, out_steps * n_targets)
        self.out_steps = out_steps
        self.n_targets = n_targets

    def forward(self, x):
        h, _ = self.lstm(x)
        h_norm_lstm = self.norm(h)
        attn_out, _ = self.mha(h_norm_lstm, h_norm_lstm, h_norm_lstm)
        h = h + attn_out
        h_norm_attn = self.norm(h)
        ctx_attn = h_norm_attn.mean(dim=1)
        ctx_ff = self.ff(ctx_attn)
        ctx = ctx_attn + ctx_ff
        ctx_norm_ff = self.norm(ctx)
        out = self.out_proj(ctx_norm_ff)
        return out.view(-1, self.out_steps, self.n_targets)