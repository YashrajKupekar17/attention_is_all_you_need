import torch
import torch.nn as nn
class Multihead_SelfAttention(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    assert cfg["emb_dim"] % cfg["n_heads"] == 0
    self.head_size = cfg["emb_dim"]
    self.dim_in = cfg["emb_dim"]
    self.num_heads = cfg["n_heads"]
    self.query = nn.Linear(self.dim_in,self.head_size,cfg["qkv_bias"])
    self.key = nn.Linear(self.dim_in,self.head_size,cfg["qkv_bias"])
    self.value = nn.Linear(self.dim_in,self.head_size,cfg["qkv_bias"])
    self.dropout = nn.Dropout(cfg["drop_rate"])
    self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
    self.register_buffer("mask", torch.tril(torch.ones(cfg["context_length"], cfg["context_length"])))




  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x);   #(B,T,head_size)
    q = self.query(x);
    v = self.value(x);

    #split the k,q,v
    self.split_head_size = self.head_size // self.num_heads

    #for k , q, v ==>  (B,T,head_size) --> (B,T,num_heads,split_head_size)
    k = k.view(B,T,self.num_heads,self.split_head_size)
    q = q.view(B,T,self.num_heads,self.split_head_size)
    v = v.view(B,T,self.num_heads,self.split_head_size)

    # q -->(B,T,num_heads,split_head_size)
    # k --> (B,T,num_heads,split_head_size)
    # k.T --> (B,T,split_head_size,num_heads)

    # i.e. q @k.T --->(B,T,num_heads,num_heads).  this is not what we want .

    k = torch.transpose(k,1,2)
    q = torch.transpose(q,1,2)
    v = torch.transpose(v,1,2)


    # q -->(B,num_heads,T,split_head_size)
    # k --> (B,num_heads,T,split_head_size)
    # k.T --> (B,num_heads,split_head_size,T)

    # i.e. q @k.T --->(B,num_heads,T,T)

    attention_scores = q @ torch.transpose(k,-2,-1)   #(B,num_heads,T,T)
    attention_scores = torch.masked_fill(attention_scores,self.mask[:T,:T]==0,-torch.inf)
    # self.attention_scores = attention_scores
    attention_weight = torch.softmax(attention_scores/k.shape[-1]**0.5,dim=-1)
    attention_weight = self.dropout(attention_weight)
    # self.attention_weight = attention_weight
    context_vector = attention_weight @ v   # (B,num_heads,T,T) @ (B,num_heads,T,split_head_size) -->(B,num_heads,T,split_head_size)

    context_vector = torch.transpose(context_vector,1,2)   # (B,T,num_heads,split_head_size)

    context_vector = context_vector.contiguous().view(B,T,self.head_size)
    context_vector = self.out_proj(context_vector)

    return context_vector # (B,T,head_size)



