import torch
import torch.nn as nn
from layer_norm import LayerNorm
from multihead_attention import Multihead_SelfAttention

from ffn import FeedForwardNet
class TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layerNorm1 = LayerNorm(cfg)
    self.layerNorm2 = LayerNorm(cfg)
    self.attention = Multihead_SelfAttention(cfg)
    self.dropout = nn.Dropout(cfg["drop_rate"])
    self.ffn = FeedForwardNet(cfg)

  def forward(self,x):
    B,T,C = x.shape       #B,T,C

    shortcut = x
    x=self.layerNorm1(x)
    x=self.attention(x)
    x=self.dropout(x);
    x=x+shortcut

    shortcut = x
    x=self.layerNorm2(x)
    x=self.ffn(x)
    x=self.dropout(x)
    x=x+shortcut

    return x

