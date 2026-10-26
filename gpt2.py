import torch
import torch.nn as nn

from transformer import TransformerBlock
from layer_norm import LayerNorm
class GPT2(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.context_length=cfg["context_length"]
    self.token_embedding = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
    self.pos_embedding = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    self.transformers = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    self.final_layerNorm = LayerNorm(cfg)
    self.linear_output_layer = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

  def forward(self,ids):
    B, T = ids.shape
    assert T <= self.context_length, f"sequence length {T} > context_length {self.context_length}"
    token_emb = self.token_embedding(ids)
    pos_emb = self.pos_embedding(torch.arange(T,device = ids.device))
    emb = token_emb+pos_emb
    emb = self.drop_emb(emb)

    out = self.transformers(emb)

    out = self.final_layerNorm(out)
    logits = self.linear_output_layer(out)

    return logits