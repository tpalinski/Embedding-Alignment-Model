from abc import ABC, abstractmethod
import torch
from torch import nn

from .Data2VecModelWithSharedExtractorConfig import Data2VecModelWithSharedExtractorConfig

class Data2VecModelWithSharedExtractor(ABC, nn.Module):
  """Set of sequence transducer layers to be used in Data2Vec training after masking"""
  sequence_transducer: list[nn.Module]

  config: Data2VecModelWithSharedExtractorConfig

  def __init__(self, config: Data2VecModelWithSharedExtractorConfig) -> None:
    super().__init__()
    self.config = config

  @abstractmethod
  def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
    """ Encode sequence with masked time steps
      :param x: Masked sequence encoded by the feature encoder in format (B, T, C)
      :returns: Output of last K blocks of sequence transducer in format (K, B, T, C)
    """
    pass

  def _generate_batch_mask(self, x: torch.Tensor) -> torch.Tensor:
    """ Generate a mask for batched time sequence
    :param x: Tensor of shape (B, T, C)
    """
    B, T, C = x.shape
    device = x.device
    p = self.config['mask_p']
    span = self.config['mask_steps']

    start_mask = torch.rand(B, T, device=device) < p
    offsets = torch.arange(span, device=device)
    start_indices = start_mask.nonzero(as_tuple=False)  # (N, 2) -> (b, t)
    if start_indices.numel() == 0:
        return x
    b_idx, t_idx = start_indices[:, 0], start_indices[:, 1]

    span_t = t_idx[:, None] + offsets[None, :]
    span_t = span_t.clamp(max=T - 1)
    time_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    time_mask[b_idx[:, None], span_t] = True
    return time_mask

  """Gets input from a relevant feature encoder in shape (B, T, C)"""
  def forward(self, x: torch.Tensor, masked = True) -> (torch.Tensor, torch.Tensor|None):
    assert self.config['K'] <= len(self.sequence_transducer)
    mask = None
    if masked:
      mask = self._generate_batch_mask(x).unsqueeze(-1)
      x = x * (~mask)
    x = self._encode_sequence(x)
    return x, mask
