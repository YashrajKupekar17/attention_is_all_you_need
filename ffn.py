import torch.nn as nn
class FeedForwardNet(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(cfg["emb_dim"],cfg["emb_dim"]*4),
        nn.GELU(),
        nn.Linear(cfg["emb_dim"]*4,cfg["emb_dim"])
    )

  def forward(self,x):
    # x.shape = B,T,d_model
    return self.net(x)