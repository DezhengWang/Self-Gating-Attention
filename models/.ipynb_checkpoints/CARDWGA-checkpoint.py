import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from layers.GatingAttention import GatingAttentionLayer

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)




class Model(nn.Module):
    def __init__(self, config, **kwargs):
        
        super().__init__()    
        self.model = CARDformer(config)
        self.task_name = config.task_name
    def forward(self, x, *args, **kwargs):    

        x = x.permute(0,2,1)    

        mask = args[-1]            
        x= self.model(x,mask = mask)
        if self.task_name != 'classification':
            x = x.permute(0,2,1)   
        return x
    
    
    
    
    
class CARDformer(nn.Module):
    def __init__(self,
                 configs,**kwargs):
        
        super().__init__()
        
        self.patch_len  = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.task_name = configs.task_name
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num, configs.d_model) * 1e-2)
        self.model_token_number = 0
        
        topk_ratio = configs.topk_ratio
        dropout_alpha = configs.dropout_alpha
        dropout_data = configs.dropout_data
        
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(torch.randn(configs.enc_in, self.model_token_number, configs.d_model) * 1e-2)
        
        
        self.total_token_number = (self.patch_num  + self.model_token_number + 1)
        configs.total_token_number = self.total_token_number
             
        self.W_input_projection = nn.Linear(self.patch_len, configs.d_model)
        self.input_dropout  = nn.Dropout(configs.dropout)
        
                
        self.use_statistic = configs.use_statistic
        self.W_statistic = nn.Linear(2, configs.d_model)
        self.cls = nn.Parameter(torch.randn(1, configs.d_model) * 1e-2)
        
        
        
        if configs.task_name == 'long_term_forecast' or configs.task_name == 'short_term_forecast':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number) * configs.d_model, configs.pred_len)
        elif configs.task_name == 'imputation' or configs.task_name == 'anomaly_detection':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number) * configs.d_model, configs.seq_len)
        elif configs.task_name == 'classification':
            self.W_out = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

     
        
        
        self.Attentions_over_token = nn.ModuleList(
            [GatingAttentionLayer(configs.d_model, configs.n_heads, configs.enc_in,
                                         alpha_size=(configs.n_heads,
                                                     patch_num + 1,
                                                     patch_num + 1),
                                         dropout_alpha=dropout_alpha,
                                         dropout_data=dropout_data,
                                         is_sparse=True,
                                         beta=0.6,
                                         topk_ratio=topk_ratio)
             for i in range(configs.e_layers)]
        )
        self.Attentions_over_channel = nn.ModuleList(
            [GatingAttentionLayer(configs.d_model, configs.n_heads, configs.enc_in,
                                         alpha_size=(configs.n_heads,
                                                     configs.enc_in,
                                                     configs.enc_in),
                                         dropout_alpha=dropout_alpha,
                                         dropout_data=dropout_data,
                                         is_sparse=True,
                                         beta=0.6,
                                         topk_ratio=topk_ratio)
             for i in range(configs.e_layers)]
        )

        self.Attentions_mlp = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model) for i in range(configs.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(configs.dropout) for i in range(configs.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model, momentum = configs.momentum), Transpose(1, 2)) for i in range(configs.e_layers)])
            
        

        

    def forward(self, z,*args, **kwargs):     

        b,c,s = z.shape
        

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            z_mean = torch.mean(z,dim = (-1),keepdims = True)
            z_std = torch.std(z,dim = (-1),keepdims = True)
            z =  (z - z_mean)/(z_std + 1e-4)
        
        elif self.task_name == 'imputation':     
            mask = kwargs['mask'].permute(0,2,1) 
            z_mean = torch.sum(z, dim=-1) / torch.sum(mask == 1, dim=-1)
            z_mean = z_mean.unsqueeze(-1)
            z = z - z_mean
            z = z.masked_fill(mask == 0, 0)
            z_std = torch.sqrt(torch.sum(z * z, dim=-1) /
                           torch.sum(mask == 1, dim=-1) + 1e-5)
            z_std = z_std.unsqueeze(-1)
            z /= z_std + 1e-4

       
        
            
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                 
        z_embed = self.input_dropout(self.W_input_projection(zcube))+ self.W_pos_embed 
        
        
        if self.use_statistic:
            
            z_stat = torch.cat((z_mean,z_std),dim = -1)
            if z_stat.shape[-2]>1:
                z_stat = (z_stat - torch.mean(z_stat,dim =-2,keepdims = True))/( torch.std(z_stat,dim =-2,keepdims = True)+1e-4)
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2),z_embed),dim = -2) 
        else:
            cls_token = self.cls.repeat(z_embed.shape[0],z_embed.shape[1],1,1)
            z_embed = torch.cat((cls_token,z_embed),dim = -2) 

        inputs = z_embed
        b,c,t,h = inputs.shape 
        for a_2,a_1,mlp,drop,norm  in zip(self.Attentions_over_token, self.Attentions_over_channel,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            inputs_re = inputs.permute(0,2,1,3).reshape(-1, c, h)
            output_1 = a_1(inputs_re, inputs_re, inputs_re)[0].reshape(b, t, c, h).permute(0,2,1,3)

            output_2 = a_2(output_1.reshape(-1, t, h), output_1.reshape(-1, t, h), output_1.reshape(-1, t, h))[0].reshape(b, c, t, h)
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1) 
            inputs = outputs
        
        if self.task_name != 'classification':
            z_out = self.W_out(outputs.reshape(b,c,-1))  
            z = z_out *(z_std+1e-4)  + z_mean 
        else:
            z = self.W_out(torch.mean(outputs[:,:,:,:],dim = -2).reshape(b,-1))
        return z
    

class Attenion(nn.Module):
    def __init__(self,config, over_hidden = False,trianable_smooth = False,untoken = False, *args, **kwargs):
        super().__init__()

        
        self.over_hidden = over_hidden
        self.untoken = untoken
        self.n_heads = config.n_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        
        
    
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        

        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear( config.d_model,  config.d_model)
        
        

        self.norm_post1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        self.norm_post2  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        
        
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                       )
        
        self.ff_2= nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                                )     
        self.merge_size = config.merge_size

        ema_size = max(config.enc_in,config.total_token_number,config.dp_rank)
        ema_matrix = torch.zeros((ema_size,ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1,config.total_token_number):
            for j in range(i):
                ema_matrix[i][j] =  ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix',ema_matrix)
 
           

       
    def ema(self,src):
        return torch.einsum('bnhad,ga ->bnhgd',src,self.ema_matrix[:src.shape[-2],:src.shape[-2]])
        
        
    def ema_trianable(self,src):
        alpha = F.sigmoid(self.alpha)
        
        weights = alpha * (1 - alpha) ** self.arange[-src.shape[-2]:]
 

        w_f = torch.fft.rfft(weights,n = src.shape[-2]*2)
        src_f = torch.fft.rfft(src.float(),dim = -2,n = src.shape[-2]*2)    
        src_f = (src_f.permute(0,1,2,4,3)*w_f)
        src1 =torch.fft.irfft(src_f.float(),dim = -1,n=src.shape[-2]*2)[...,:src.shape[-2]].permute(0,1,2,4,3)#.half()
        return src1



    def dynamic_projection(self,src,mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp,dim = -1)
        src_dp = torch.einsum('bnhef,bnhec -> bnhcf',src,src_dp)
        return src_dp
        

        
        
    def forward(self, src, *args,**kwargs):


        B,nvars, H, C, = src.shape
        


        
        
        qkv = self.qkv(src).reshape(B,nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
   

        q, k, v = qkv[0], qkv[1], qkv[2]
    
        if not self.over_hidden: 
        
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k))/ self.head_dim ** -0.5

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
   
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)


            
        else:

            v_dp,k_dp = self.dynamic_projection(v,self.dp_v) , self.dynamic_projection(k,self.dp_k)
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp))/ self.head_dim ** -0.5
            

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)

        
        
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q,k)/ q.shape[-2] ** -0.5
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1) )    
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)



        merge_size = self.merge_size
        if not self.untoken:
            output1 = rearrange(output_along_token.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        
        
            output2 = rearrange(output_along_hidden.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        

        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B,nvars, -1, self.n_heads * self.head_dim)





        src2 =  self.ff_1(output1)+self.ff_2(output2)
        
        
        src = src + src2
        src = src.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        return src