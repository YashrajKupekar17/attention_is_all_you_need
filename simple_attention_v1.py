import torch.nn as nn

class Simplified_SelfAttention(nn.Module):
  def __init__(self, dim_in,head_size,qkv_bias=False):
    super().__init__()
    self.dim_in = dim_in
    self.head_size = head_size
    self.query = nn.Linear(self.dim_in,self.head_size,bias=qkv_bias)
    self.key = nn.Linear(self.dim_in,self.head_size,bias=qkv_bias)
    self.value = nn.Linear(self.dim_in,self.head_size,bias=qkv_bias)
  def forward(self,x):
    T,C = x.shape 
    k = self.key(x)
    v = self.value(x)
    q = self.query(x) # (T,head_size)

    attention_score = q @ k.T   #(T,T) 
    attention_weight = torch.softmax(attention_score/k.shape[-1]**0.5,dim=-1)
    context_vector = attention_weight @ v
    return context_vector # (T,head_size)
