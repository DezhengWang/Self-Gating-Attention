import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from layers.Embed import DataEmbedding_inverted
from layers.SWTAttention_Family import WaveletEmbedding


class CustAttentionLayer(nn.Module):
    def __init__(self, attentionLayer,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, ):
        super(CustAttentionLayer, self).__init__()

        self.d_channel = d_channel

        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m,
                                    kernel_size=kernel_size)
        self.att_layer = attentionLayer

        self.out_projection = nn.Sequential(
            WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m,
                             kernel_size=kernel_size),
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch_size = queries.size(0)

        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        queries = torch.reshape(queries, (-1, queries.shape[2], queries.shape[3]))
        keys = torch.reshape(keys, (-1, keys.shape[2], keys.shape[3]))
        values = torch.reshape(values, (-1, values.shape[2], values.shape[3]))

        out, attn = self.att_layer(
            queries,
            keys,
            values,
        )

        out = torch.reshape(out, (batch_size, -1, out.shape[1], out.shape[2]))

        out = self.out_projection(out)

        return out, attn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = configs.kernel_size

        enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 
                                               configs.embed, configs.freq, configs.dropout)
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [  
                EncoderLayer(
                    CustAttentionLayer(
                        AttentionLayer(
                            ProbAttention(False, configs.factor,
                                            attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        requires_grad=configs.requires_grad, 
                        wv=configs.wv, 
                        m=configs.m, 
                        d_channel=configs.dec_in, 
                        kernel_size=self.kernel_size,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder = encoder

        projector = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.projector = projector


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_enc /= stdev
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # Linear Projection             B L N -> B L' (pseudo temporal tokens) N 
        enc_out = enc_embedding(x_enc, x_mark_enc) 

        # SimpleTM Layer                B L' N -> B L' N 
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # Output Projection             B L' N -> B H (Horizon) N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N] 

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        self.attns = attns
        return dec_out