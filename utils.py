import torch 
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from typing import Any


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


def compute_loss(worldmodel, batch_obs, batch_act, mask_fill, tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

    with torch.no_grad():
        obs_tokens = tokenizer.encode(batch_obs, should_preprocess=True).tokens  # (BL, K)

    act_tokens = rearrange(batch_act, 'b l -> b l 1')
    tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

    outputs = worldmodel(tokens)

    labels_observations = compute_labels_world_model(obs_tokens, mask_fill)

    logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
    loss_obs = F.cross_entropy(logits_observations, labels_observations)
    return loss_obs

def compute_labels_world_model(obs_tokens: torch.Tensor, mask_fill: torch.BoolTensor) -> torch.Tensor:
    labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
    return labels_observations.reshape(-1)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

