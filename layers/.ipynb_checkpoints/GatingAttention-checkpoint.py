import torch
import torch.nn as nn
from math import sqrt


class GatingAttention(nn.Module):
    def __init__(self, alpha_size=None, attn_dropout_alpha=0.1, attn_dropout_data=0.1, is_sparse=True, beta=0.6, topk_ratio=0.1):
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

        # bilinear： per-head 的 SxF 结构耦合
        # bilinear[h] = U[h] @ V[h]  → [S,F]
        # broadcast 到 batch： [B,H,S,F]
        bilinear = torch.einsum("hsr,hrf->hsf", self.U, self.V)  # [H,S,F]
        data_logits = score.unsqueeze(2) + self.gamma_hs + bilinear.unsqueeze(0)
        return data_logits  # [B,H,S,F]

    @staticmethod
    def _topk_on_logits(logits, k):
        topv, topi = logits.topk(k, dim=-1)
        masked = torch.full_like(logits, -float("inf")).scatter_(-1, topi, topv)
        return masked

    def forward(self, values, need_weights=False, return_both=False):
        """
        returns:
          out: [B,S,H,D]
          attn (optional):
              return_both=False → 返回融合后注意力
              return_both=True  → 返回 (attn_data, attn_alpha, attn_mix)
        """
        H, S, F_ = self.alpha_size
        scale = 1.0 / sqrt(F_)
        alpha_logits = (self.alpha * scale).unsqueeze(0)  # [1,H,S,F]

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
        alpha_size (Tuple[int, int]): Shape (head, ori_time_dim, new_time_dim) for alpha projection.
        beta_size (Tuple[int], optional): Shape (head, ori_var_dim, new_var_dim) for beta projection.
        d_values (int, optional): Value projection size per head.
        cross_attention (bool): Whether to use cross-attention by concat query and value.
        output_attention (bool): Whether to output attention weights.
        combined_batch (bool): batch management.
    """

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), d_values=None,
                 cross_attention=False, output_attention=False, dropout_alpha=0.1, dropout_data=0.1,
                 is_sparse=True, beta=0.6, topk_ratio=0.1):
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
            beta=beta,
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
        # output = out.view(B, -1, self.n_heads * self.d_values)

        return (output, attn) if self.output_attention else (output, None)


class GatingAttentionWmlp(nn.Module):
    """
    Computes self-gating attention using learnable static weights and dynamic alpha.
    values: [B, F, H, D]
    alpha:  [H, S, F]
    """

    def __init__(self, alpha_size=None, attention_dropout=0.1, is_sparse=True, beta=0.6, topk_ratio=0.1):
        super().__init__()
        assert alpha_size is not None, "alpha_size must be provided"
        self.alpha_size = alpha_size
        H, S, F = alpha_size

        self.alpha = nn.Parameter(torch.empty(H, S, F), requires_grad=True)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.is_sparse = True
        self.topk_ratio = topk_ratio

        self.beta_param = nn.Parameter(torch.full((H, S, 1), float(self._inv_softplus(beta))))

        # 正交初始化
        self.reset_parameters()

        self.sf_affine_w = nn.Parameter(torch.zeros(H, S, F))  # (h,s,f) 缩放
        self.sf_affine_b = nn.Parameter(torch.zeros(H, S, 1))  # (h,s) 偏置

    def reset_parameters(self):
        h, s, f = self.alpha.shape
        flat = torch.empty(h, s * f, dtype=self.alpha.dtype, device=self.alpha.device)
        nn.init.orthogonal_(flat, gain=1.0)
        self.alpha.data.copy_(flat.view(h, s, f))

    @staticmethod
    def _topk_on_logits(logits, k):
        """
        对 logits 做 top-k 掩码（其它置为 -inf），再交给 softmax。
        logits: [B, H, S, F]
        """
        topv, topi = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topi, 1.0)
        masked = logits.masked_fill(mask < 0.5, float('-inf'))
        return masked

    @staticmethod
    def _inv_softplus(y):
        # y: float
        y_t = torch.as_tensor(y, dtype=torch.float32)  # 保持设备在创建 Parameter 时再统一
        return y_t + torch.log1p(-torch.exp(-y_t).clamp_min(1e-12))

    def forward(self, values, data_logits, need_weights=False):
        """
        values: [B, F, H, D]
        returns: out [B, S, H, D], attn [B, H, S, F] (if need_weights)
        """

        scale = 1.0 / sqrt(self.alpha.shape[-1])
        alpha_logits = self.alpha * scale  # [H, S, F]

        data_logits = data_logits.permute(0, 2, 3, 1)

        mu = data_logits.mean(dim=-1, keepdim=True)
        std = data_logits.var(dim=-1, unbiased=False, keepdim=True).add(1e-6).sqrt() + 1e-6
        data_logits = (data_logits - mu) / std

        beta_eff = F.softplus(self.beta_param)  # [H,S,1]
        logits = alpha_logits.unsqueeze(0) + beta_eff * data_logits
        # logits = alpha_logits.unsqueeze(0) + self.beta * data_logits  # [B, H, S, F]

        if self.is_sparse and self.topk_ratio is not None:
            k = max(1, int(self.topk_ratio * self.alpha.shape[-1]))
            logits = self._topk_on_logits(logits, k)

        attn = torch.softmax(logits, dim=-1)  # [B, H, S, F]
        attn = self.attn_dropout(attn)
        self.attn = attn

        out = torch.einsum("bhsf,bfhd->bshd", attn, values).contiguous()
        return out, (attn if need_weights else None)


class GatingAttentionLayerWmlp(nn.Module):
    """
    Full attention layer combining projections and asymmetric attention.

    Args:
        embed_dim (int): Data embedding dimension.
        num_heads (int): Number of attention heads.
        enc_in (int): num variables.
        alpha_size (Tuple[int, int]): Shape (head, ori_time_dim, new_time_dim) for alpha projection.
        beta_size (Tuple[int], optional): Shape (head, ori_var_dim, new_var_dim) for beta projection.
        d_values (int, optional): Value projection size per head.
        cross_attention (bool): Whether to use cross-attention by concat query and value.
        output_attention (bool): Whether to output attention weights.
        combined_batch (bool): batch management.
    """

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), d_values=None,
                 cross_attention=False, output_attention=False, dropout=0.1,
                 is_sparse=True, beta=0.6, topk_ratio=0.1):
        super().__init__()

        self.d_values = d_values or (embed_dim // num_heads)
        self.n_heads = num_heads
        self.enc_in = enc_in
        self.cross_attention = cross_attention
        self.output_attention = output_attention
        self.alpha_size = alpha_size

        self.inner_attention = GatingAttentionWmlp(
            alpha_size=alpha_size,
            attention_dropout=dropout,
            is_sparse=is_sparse,
            beta=beta,
            topk_ratio=topk_ratio
        )

        self.value_projection = nn.Linear(embed_dim, self.d_values * num_heads)
        self.out_projection = nn.Linear(self.d_values * num_heads, embed_dim)

        self.data_project = nn.Sequential(
            nn.Linear(embed_dim, self.d_values * num_heads),
            nn.LayerNorm(self.d_values * num_heads),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_values * num_heads, alpha_size[-2] * num_heads)
        )

    def forward(self, query, key, value, attn_mask=None, tau=None, delta=None):
        if self.cross_attention:
            value = torch.cat((query, value), dim=1)
        B, S, E = value.shape

        values = self.value_projection(value).view(B, S, self.n_heads, self.d_values)

        data_logits = self.data_project(value).view(B, S, self.n_heads, -1)

        out, attn = self.inner_attention(values, data_logits, need_weights=self.output_attention)
        output = self.out_projection(out.view(B, -1, self.n_heads * self.d_values))
        # output = out.view(B, -1, self.n_heads * self.d_values)

        return (output, attn) if self.output_attention else (output, None)


class GatingAttentionWsa(nn.Module):
    """
    Computes self-gating attention using learnable static weights and dynamic alpha.
    values: [B, F, H, D]
    alpha:  [H, S, F]
    """

    def __init__(self, alpha_size=None, attention_dropout=0.1, is_sparse=True, beta=0.6, topk_ratio=0.1, d_value=64):
        super().__init__()
        assert alpha_size is not None, "alpha_size must be provided"
        self.alpha_size = alpha_size
        H, S, F = alpha_size

        self.d_value = d_value

        self.alpha = nn.Parameter(torch.empty(H, S, F), requires_grad=True)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.is_sparse = True
        self.topk_ratio = topk_ratio

        self.beta_param = nn.Parameter(torch.full((H, S, 1), float(self._inv_softplus(beta))))

        # 正交初始化
        self.reset_parameters()

        self.sf_affine_w = nn.Parameter(torch.zeros(H, S, F))  # (h,s,f) 缩放
        self.sf_affine_b = nn.Parameter(torch.zeros(H, S, 1))  # (h,s) 偏置

    def reset_parameters(self):
        h, s, f = self.alpha.shape
        flat = torch.empty(h, s * f, dtype=self.alpha.dtype, device=self.alpha.device)
        nn.init.orthogonal_(flat, gain=1.0)
        self.alpha.data.copy_(flat.view(h, s, f))

    @staticmethod
    def _topk_on_logits(logits, k):
        """
        对 logits 做 top-k 掩码（其它置为 -inf），再交给 softmax。
        logits: [B, H, S, F]
        """
        topv, topi = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topi, 1.0)
        masked = logits.masked_fill(mask < 0.5, float('-inf'))
        return masked

    @staticmethod
    def _inv_softplus(y):
        # y: float
        y_t = torch.as_tensor(y, dtype=torch.float32)  # 保持设备在创建 Parameter 时再统一
        return y_t + torch.log1p(-torch.exp(-y_t).clamp_min(1e-12))

    def forward(self, values, data_logits, need_weights=False):
        """
        values: [B, F, H, D]
        returns: out [B, S, H, D], attn [B, H, S, F] (if need_weights)
        """

        scale = 1.0 / sqrt(self.alpha.shape[-1])
        alpha_logits = self.alpha * scale  # [H, S, F]

        mu = data_logits.mean(dim=-1, keepdim=True)
        std = data_logits.var(dim=-1, unbiased=False, keepdim=True).add(1e-6).sqrt() + 1e-6
        data_logits = (data_logits - mu) / std

        beta_eff = F.softplus(self.beta_param)  # [H,S,1]
        # logits = alpha_logits.unsqueeze(0) + beta_eff * data_logits
        logits = data_logits  # beta_eff * data_logits
        # logits = alpha_logits.unsqueeze(0) + self.beta * data_logits  # [B, H, S, F]

        # if self.is_sparse and self.topk_ratio is not None:
        #     k = max(1, int(self.topk_ratio * self.alpha.shape[-1]))
        #     logits = self._topk_on_logits(logits, k)

        attn = torch.softmax(logits, dim=-1)  # [B, H, S, F]
        attn = self.attn_dropout(attn)
        self.attn = attn

        out = torch.einsum("bhsf,bfhd->bshd", attn, values).contiguous()
        return out, (attn if need_weights else None)


class GatingAttentionLayerWsa(nn.Module):
    """
    Full attention layer combining projections and asymmetric attention.

    Args:
        embed_dim (int): Data embedding dimension.
        num_heads (int): Number of attention heads.
        enc_in (int): num variables.
        alpha_size (Tuple[int, int]): Shape (head, ori_time_dim, new_time_dim) for alpha projection.
        beta_size (Tuple[int], optional): Shape (head, ori_var_dim, new_var_dim) for beta projection.
        d_values (int, optional): Value projection size per head.
        cross_attention (bool): Whether to use cross-attention by concat query and value.
        output_attention (bool): Whether to output attention weights.
        combined_batch (bool): batch management.
    """

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), d_values=None,
                 cross_attention=False, output_attention=False, dropout=0.1,
                 is_sparse=True, beta=0.6, topk_ratio=0.1):
        super().__init__()

        self.d_values = d_values or (embed_dim // num_heads)
        self.n_heads = num_heads
        self.enc_in = enc_in
        self.cross_attention = cross_attention
        self.output_attention = output_attention
        self.alpha_size = alpha_size

        self.inner_attention = GatingAttentionWsa(
            alpha_size=alpha_size,
            attention_dropout=dropout,
            is_sparse=is_sparse,
            beta=beta,
            topk_ratio=topk_ratio,
            d_value=self.d_values
        )

        self.value_projection = nn.Linear(embed_dim, self.d_values * num_heads)
        self.key_projection = nn.Linear(embed_dim, self.d_values * num_heads)
        self.query_projection = nn.Linear(embed_dim, self.d_values * num_heads)
        self.out_projection = nn.Linear(self.d_values * num_heads, embed_dim)

    def forward(self, query, key, value, attn_mask=None, tau=None, delta=None):
        B, S, E = value.shape
        _, L, _ = query.shape

        values = self.value_projection(value).view(B, S, self.n_heads, self.d_values)
        keys = self.key_projection(key).view(B, S, self.n_heads, -1)
        queries = self.query_projection(query).view(B, L, self.n_heads, -1)

        data_logits = torch.einsum("blhe,bshe->bhls", queries, keys)

        out, attn = self.inner_attention(values, data_logits, need_weights=self.output_attention)
        output = self.out_projection(out.view(B, -1, self.n_heads * self.d_values))
        # output = out.view(B, -1, self.n_heads * self.d_values)

        return (output, attn) if self.output_attention else (output, None)


class GatingAttentionWoPsi(nn.Module):
    """
    Computes gating attention using learnable static weights and dynamic alpha.

    Args:
        alpha_size (Tuple[int]): Shape of the learnable alpha tensor (H, new_time_dim, ori_time_dim).
        beta_size (Tuple[int], optional): Shape of the learnable beta tensor (H, new_var_dim, ori_var_dim).
    """

    def __init__(self, alpha_size=None, beta_size=None, attention_dropout=0.1):
        super().__init__()
        assert alpha_size is not None, "alpha_size must be provided"

        self.alpha_size = alpha_size
        self.beta_size = beta_size
        h, s, f = alpha_size
        self.alpha = nn.Parameter(torch.empty(h, s, f), requires_grad=True)
        # init.xavier_uniform_(self.alpha)

        self.temp = nn.Parameter(torch.ones(h, 1))
        self.denom_bias = nn.Parameter(torch.zeros(h, f, 1))
        self.reset_parameters()
        self.attn_dropout = nn.Dropout(attention_dropout)

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
        w = -values.transpose(1, 2)  # b, h, f, d

        self.attn = torch.softmax(self.alpha.mul(scale_alpha), dim=-1)  # h, s, f
        # values: b, h, f, d
        values = torch.matmul(self.attn, w).transpose(2, 1)
        # values: b, s, h, d
        return values.contiguous(), self.attn if need_weights else None


class GatingAttentionWoPsiLayer(GatingAttentionLayer):

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), beta_size=None, d_values=None,
                 cross_attention=False, output_attention=False, combined_batch=True, dropout=0.1):
        super().__init__(embed_dim, num_heads, enc_in,
                         alpha_size, beta_size, d_values,
                         cross_attention, output_attention, combined_batch, dropout)

        self.inner_attention = GatingAttentionWoPsi(
            alpha_size=alpha_size,
            beta_size=beta_size,
            attention_dropout=dropout
        )


class GatingAttentionWoAlpha(nn.Module):
    """
    Computes gating attention using learnable static weights and dynamic alpha.

    Args:
        alpha_size (Tuple[int]): Shape of the learnable alpha tensor (H, new_time_dim, ori_time_dim).
        beta_size (Tuple[int], optional): Shape of the learnable beta tensor (H, new_var_dim, ori_var_dim).
    """

    def __init__(self, alpha_size=None, beta_size=None, attention_dropout=0.1):
        super().__init__()
        assert alpha_size is not None, "alpha_size must be provided"

        self.alpha_size = alpha_size
        self.beta_size = beta_size
        h, s, f = alpha_size
        self.alpha = nn.Parameter(torch.empty(h, s, f), requires_grad=True)
        # init.xavier_uniform_(self.alpha)

        self.temp = nn.Parameter(torch.ones(h, 1))
        self.denom_bias = nn.Parameter(torch.zeros(h, f, 1))
        self.reset_parameters()
        self.attn_dropout = nn.Dropout(attention_dropout)

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

        # Top-k sparse
        # k = int(tmp.shape[-1]*0.8)
        # topk_vals, _ = tmp.topk(k, dim=-1)
        # threshold = topk_vals[:, :, -1].unsqueeze(-1)
        # tmp = tmp.masked_fill(tmp < threshold, float('-inf'))

        Pi = torch.softmax(tmp, dim=-1)  # b, h, f
        Pi = self.attn_dropout(Pi)
        values = - w.mul(Pi.unsqueeze(-1))  # .mul(attn)

        # values: b, h, f, d
        values = values.transpose(2, 1)
        # values: b, s, h, d
        return values.contiguous(), self.attn if need_weights else None


class GatingAttentionWoAlphaLayer(GatingAttentionLayer):

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), beta_size=None, d_values=None,
                 cross_attention=False, output_attention=False, combined_batch=True, dropout=0.1):
        super().__init__(embed_dim, num_heads, enc_in,
                         alpha_size, beta_size, d_values,
                         cross_attention, output_attention, combined_batch, dropout)

        self.inner_attention = GatingAttentionWoAlpha(
            alpha_size=alpha_size,
            beta_size=beta_size,
            attention_dropout=dropout
        )


class GatingAttentionWSparse(nn.Module):
    """
    Computes gating attention using learnable static weights and dynamic alpha.

    Args:
        alpha_size (Tuple[int]): Shape of the learnable alpha tensor (H, new_time_dim, ori_time_dim).
        beta_size (Tuple[int], optional): Shape of the learnable beta tensor (H, new_var_dim, ori_var_dim).
    """

    def __init__(self, alpha_size=None, beta_size=None, attention_dropout=0.1):
        super().__init__()
        assert alpha_size is not None, "alpha_size must be provided"

        self.alpha_size = alpha_size
        self.beta_size = beta_size
        h, s, f = alpha_size
        self.alpha = nn.Parameter(torch.empty(h, s, f), requires_grad=True)
        # init.xavier_uniform_(self.alpha)

        self.temp = nn.Parameter(torch.ones(h, 1))
        self.denom_bias = nn.Parameter(torch.zeros(h, f, 1))
        self.reset_parameters()
        self.attn_dropout = nn.Dropout(attention_dropout)

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

        # Top-k sparse
        k = int(tmp.shape[-1] * 0.8)
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


class GatingAttentionWSparseLayer(GatingAttentionLayer):

    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), beta_size=None, d_values=None,
                 cross_attention=False, output_attention=False, combined_batch=True, dropout=0.1):
        super().__init__(embed_dim, num_heads, enc_in,
                         alpha_size, beta_size, d_values,
                         cross_attention, output_attention, combined_batch, dropout)

        self.inner_attention = GatingAttentionWSparse(
            alpha_size=alpha_size,
            beta_size=beta_size,
            attention_dropout=dropout
        )


class GatingAttentionLayerSWT(nn.Module):
    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), beta_size=None,
                 d_values=None, dropout_alpha=0.1, dropout_data=0.1, output_attention=False, combined_batch=True, cross_attention=False,
                 requires_grad=True, wv='db2', m=2, kernel_size=None, d_channel=None, is_sparse=True, beta=0.6,
                 topk_ratio=0.1):
        super().__init__()
        self.cross_attention = cross_attention
        self.output_attention = output_attention

        self.inner_attention = GatingAttentionLayer(
            embed_dim, num_heads, enc_in,
            alpha_size=alpha_size,
            d_values=d_values, cross_attention=False,
            output_attention=output_attention,
            dropout_alpha=dropout_alpha,
            dropout_data=dropout_data,
            is_sparse=is_sparse,
            beta=beta,
            topk_ratio=topk_ratio
        )

        self.swt_out = WaveletEmbedding(d_channel=d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m,
                                        kernel_size=kernel_size)

        self.swt = WaveletEmbedding(d_channel=d_channel, swt=True,
                                    requires_grad=requires_grad, wv=wv, m=m,
                                    kernel_size=kernel_size)

    def forward(self, query, key, value, attn_mask=None, tau=None, delta=None):
        if self.cross_attention:
            value = torch.cat((query, value), dim=1)

        value = self.swt(value)
        b, v, n, d = value.shape
        value = torch.reshape(value, (-1, n, d))

        out, attn = self.inner_attention(value, value, value, attn_mask, tau, delta)

        out = torch.reshape(out, (b, v, n, d))
        output = self.swt_out(out)

        return (output, attn) if self.output_attention else (output, None)


class GatingAttentionLayerSWTwoPsi(nn.Module):
    def __init__(self, embed_dim, num_heads, enc_in=1,
                 alpha_size=(96, 96), beta_size=None,
                 d_values=None, dropout=0.1, output_attention=False, combined_batch=True, cross_attention=False,
                 requires_grad=True, wv='db2', m=2, kernel_size=None, d_channel=None):
        super().__init__()
        self.cross_attention = cross_attention
        self.output_attention = output_attention

        self.inner_attention = GatingAttentionWoPsiLayer(
            embed_dim, num_heads, enc_in,
            alpha_size=alpha_size, beta_size=beta_size,
            d_values=d_values, cross_attention=False,
            combined_batch=combined_batch, output_attention=output_attention,
            dropout=dropout
        )

        self.swt_out = WaveletEmbedding(d_channel=d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m,
                                        kernel_size=kernel_size)

        self.swt = WaveletEmbedding(d_channel=d_channel, swt=True,
                                    requires_grad=requires_grad, wv=wv, m=m,
                                    kernel_size=kernel_size)

    def forward(self, query, key, value, attn_mask=None, tau=None, delta=None):
        if self.cross_attention:
            value = torch.cat((query, value), dim=1)

        value = self.swt(value)
        b, v, n, d = value.shape
        value = torch.reshape(value, (-1, n, d))

        out, attn = self.inner_attention(value, value, value, attn_mask, tau, delta)

        out = torch.reshape(out, (b, v, n, d))
        output = self.swt_out(out)

        return (output, attn) if self.output_attention else (output, None)


import torch.nn.functional as F
import pywt


class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:, :, 0, :]
        detail_coeffs = coeffs[:, :, 1:, :]

        for i in range(m):
            detail_coeff = detail_coeffs[:, :, i, :]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")

            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2

        return approx_coeff
