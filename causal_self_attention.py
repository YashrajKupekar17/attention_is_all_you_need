class Causal_SelfAttention(nn.Module):
  def __init__(self, dim_in,head_size,context_length,dropout,qkv_bias=False):
    super().__init__()
    self.dim_in = dim_in
    self.head_size = head_size
    self.query = nn.Linear(self.dim_in,self.head_size,bias=qkv_bias)
    self.key = nn.Linear(self.dim_in,self.head_size,bias=qkv_bias)
    self.value = nn.Linear(self.dim_in,self.head_size,bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

  def forward(self,x):
    T,C = x.shape 
    k = self.key(x)
    v = self.value(x)
    q = self.query(x) # (T,head_size)

    attention_scores = q @ k.T   #(T,T) 
    attention_scores = torch.masked_fill(attention_scores,self.mask[:T,:T]==0,-torch.inf)
    # self.attention_scores = attention_scores
    attention_weight = torch.softmax(attention_scores/k.shape[-1]**0.5,dim=-1)
    attention_weight = self.dropout(attention_weight)
    # self.attention_weight = attention_weight
    context_vector = attention_weight @ v
    
    return context_vector # (T,head_size)