import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F


class GatingAttention(nn.Module):
    def __init__(self, alpha_size=None, attn_dropout_alpha=0.1, attn_dropout_data=0.1, is_sparse=True, topk_ratio=0.1):
        super().__init__()
        H, S, F = alpha_size
        self.alpha_size = alpha_size
        self.is_sparse = is_sparse
        self.topk_ratio = topk_ratio
        self.attn_dropout_alpha = nn.Dropout(attn_dropout_alpha)
        self.attn_dropout_data = nn.Dropout(attn_dropout_data)

        self.alpha = nn.Parameter(torch.empty(H, S, F), requires_grad=True)
        self.temp = nn.Parameter(torch.ones(H, 1))
        self.gamma_hs = nn.Parameter(torch.zeros(H, S, 1))

        # bilinear：U in R^{H,S,r}, V in R^{H,r,F}
        rank = 12
        self.U = nn.Parameter(torch.randn(H, S, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(H, rank, F) * 0.01)

        # LayerNorm
        self.ln_f = nn.LayerNorm(F)

        self.reset_parameters()

    def reset_parameters(self):
        h, s, f = self.alpha.shape
        flat = torch.empty(h, s * f, dtype=self.alpha.dtype, device=self.alpha.device)
        nn.init.orthogonal_(flat, gain=1.0)
        self.alpha.data.copy_(flat.view(h, s, f))

    def _build_data_logits(self, values):
        # values: [B,F,H,D] -> [B,H,F,D]
        w = values.transpose(1, 2).contiguous()
        energy = (w ** 2).mean(dim=-1)  # [B,H,F]
        rms = energy.mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        score = energy / rms  # [B,H,F]

        gain = F.softplus(self.temp).squeeze(-1)  # [H]
        score = score * gain.unsqueeze(0).unsqueeze(-1)

        score = self.ln_f(score)  # [B,H,F]

        # bilinear[h] = U[h] @ V[h]  → [S,F]
        # broadcast to batch： [B,H,S,F]
        bilinear = torch.einsum("hsr,hrf->hsf", self.U, self.V)  # [H,S,F]
        data_logits = score.unsqueeze(2) + self.gamma_hs + bilinear.unsqueeze(0)
        return data_logits  # [B,H,S,F]

    @staticmethod
    def _topk_on_logits(logits, k):
        topv, topi = logits.topk(k, dim=-1)
        masked = torch.full_like(logits, -float("inf")).scatter_(-1, topi, topv)
        return masked

    def forward(self, values, data_logits=None, need_weights=False, return_both=False):
        H, S, F_ = self.alpha_size
        scale = 1.0 / sqrt(F_)
        alpha_logits = (self.alpha * scale).unsqueeze(0)  # [1,H,S,F]

        if data_logits is None:
            data_logits = self._build_data_logits(values)  # [B,H,S,F]

        if self.is_sparse and self.topk_ratio is not None:
            k = max(1, int(self.topk_ratio * F_))
            data_logits = self._topk_on_logits(data_logits, k)
            alpha_logits = self._topk_on_logits(alpha_logits, k)

        attn_data = torch.softmax(data_logits, dim=-1)  # [B,H,S,F]
        attn_data = self.attn_dropout_data(attn_data)
        attn_alpha = torch.softmax(alpha_logits, dim=-1)  # [B,H,S,F]
        attn_alpha = self.attn_dropout_alpha(attn_alpha)
        
        # attn_mix = w1b * attn_data + w2b * attn_alpha  # [B,H,S,F]
        attn_mix = attn_data + attn_alpha  # [B,H,S,F]

        out = torch.einsum("bhsf,bfhd->bshd", attn_mix, values).contiguous()
        self.attn = attn_mix
        return out, (self.attn if need_weights else None)


class GatingAttentionLayer(nn.Module):
    """
    Full attention layer combining projections and asymmetric attention.

    Args:
        embed_dim (int): Data embedding dimension.
        num_heads (int): Number of attention heads.
        enc_in (int): num variables.
        alpha_size : Shape (head, ori_time_dim, new_time_dim) for alpha projection.
        d_values (int, optional): Value projection size per head.
        cross_attention (bool): Whether to use cross-attention by concat query and value.
        output_attention (bool): Whether to output attention weights.
        combined_batch (bool): batch management.
    """

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), d_values=None,
                 cross_attention=False, output_attention=False, dropout_alpha=0.1, dropout_data=0.1,
                 is_sparse=True, topk_ratio=0.1):
        super().__init__()

        self.d_values = d_values or (embed_dim // num_heads)
        self.n_heads = num_heads
        self.enc_in = enc_in
        self.cross_attention = cross_attention
        self.output_attention = output_attention
        self.alpha_size = alpha_size

        self.inner_attention = GatingAttention(
            alpha_size=alpha_size,
            attn_dropout_alpha=dropout_alpha,
            attn_dropout_data=dropout_data, 
            is_sparse=is_sparse,
            topk_ratio=topk_ratio
        )

        self.value_projection = nn.Linear(embed_dim, self.d_values * num_heads)
        self.out_projection = nn.Linear(self.d_values * num_heads, embed_dim)

    def forward(self, query, key, value, attn_mask=None, tau=None, delta=None):
        if self.cross_attention:
            value = torch.cat((query, value), dim=1)
        B, S, E = value.shape

        values = self.value_projection(value).view(B, S, self.n_heads, self.d_values)

        out, attn = self.inner_attention(values, need_weights=self.output_attention)

        output = self.out_projection(out.view(B, -1, self.n_heads * self.d_values))

        return (output, attn) if self.output_attention else (output, None)
