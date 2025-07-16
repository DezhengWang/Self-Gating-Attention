import torch
import torch.nn as nn
from math import sqrt


class SelfGatingAttention(nn.Module):
    """
    Computes self-gating attention using learnable static weights and dynamic alpha.

    Args:
        alpha_size (Tuple[int]): Shape of the learnable alpha tensor (H, new_time_dim, ori_time_dim).
    """

    def __init__(self, alpha_size=None, attention_dropout=0.1, is_sparse=False):
        super().__init__()
        assert alpha_size is not None, "alpha_size must be provided"

        self.alpha_size = alpha_size
        h, s, f = alpha_size
        self.alpha = nn.Parameter(torch.empty(h, s, f), requires_grad=True)
        # init.xavier_uniform_(self.alpha)

        self.temp = nn.Parameter(torch.ones(h, 1))
        self.denom_bias = nn.Parameter(torch.zeros(h, f, 1))
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.is_sparse = is_sparse

        # orthogonal initialization
        self.reset_parameters()

    def reset_parameters(self):
        h, s, f = self.alpha.shape
        flat = torch.empty(h, s * f,
                           dtype=self.alpha.dtype,
                           device=self.alpha.device)
        nn.init.orthogonal_(flat, gain=1.0)
        self.alpha.data.copy_(flat.view(h, s, f))

    def forward(self, values, need_weights=False):
        scale_alpha = 1.0 / sqrt(self.alpha_size[-1])
        # alpha: b, s, h, f
        w = values.transpose(1, 2)  # b, h, f, d
        w_sq = w ** 2
        denom = (torch.cumsum(w_sq, dim=-2)).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias
        tmp = torch.sum(w_normed, dim=-1).mul(self.temp)  # b, h, f

        if self.is_sparse:
            # Top-k sparse
            k = int(tmp.shape[-1]*0.8)
            topk_vals, _ = tmp.topk(k, dim=-1)
            threshold = topk_vals[:, :, -1].unsqueeze(-1)
            tmp = tmp.masked_fill(tmp < threshold, float('-inf'))

        Pi = torch.softmax(tmp, dim=-1)  # b, h, f
        Pi = self.attn_dropout(Pi)
        values = - w.mul(Pi.unsqueeze(-1))  # .mul(attn)

        self.attn = torch.softmax(self.alpha.mul(scale_alpha), dim=-1)  # h, s, f
        # values: b, h, f, d
        values = torch.matmul(self.attn, values).transpose(2, 1)
        # values: b, s, h, d
        return values.contiguous(), self.attn if need_weights else None


class SelfGatingAttentionLayer(nn.Module):
    """
    Args:
        embed_dim (int): Data embedding dimension.
        num_heads (int): Number of attention heads.
        enc_in (int): num variables.
        alpha_size (Tuple[int, int]): Shape (head, ori_time_dim, new_time_dim) for alpha projection.
        d_values (int, optional): Value projection size per head.
        cross_attention (bool): Whether to use cross-attention by concat query and value.
        output_attention (bool): Whether to output attention weights.
    """

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), d_values=None,
                 cross_attention=False, output_attention=False, dropout=0.1):
        super().__init__()

        self.d_values = d_values or (embed_dim // num_heads)
        self.n_heads = num_heads
        self.enc_in = enc_in
        self.cross_attention = cross_attention
        self.output_attention = output_attention
        self.alpha_size = alpha_size

        self.inner_attention = SelfGatingAttention(
            alpha_size=alpha_size,
            attention_dropout=dropout
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
