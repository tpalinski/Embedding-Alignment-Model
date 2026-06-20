from abc import ABC, abstractmethod
import torch
from torch import nn

from .Data2VecModelConfig import Data2VecModelConfig

class Data2VecModel(ABC, nn.Module):
  """ Part of the model performing feature encoding (e.g. CNN), after which time step masking is applied"""
  feature_encoder: nn.Module | None

  """Set of sequence transducer layers to be used in Data2Vec training after masking"""
  sequence_transducer: list[nn.Module]

  config: Data2VecModelConfig

  def __init__(self, config: Data2VecModelConfig) -> None:
    super().__init__()
    self.config = config

  @abstractmethod
  def _encode_features(self, x: torch.Tensor, attention_mask: torch.Tensor|None = None) -> torch.Tensor:
    """ Calculate forward pass of the feature encoder
      :param x: Input data in format (B, T, C)
      :returns: Encoded input data in format (B, T, C)
    """
    pass


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

  def get_skippable_params(self) -> list[str]:
    """ Get a list of paramters to be copied over to teacher model
      :returns: List of keys of paramters from state_dict
    """
    if self.feature_encoder == None:
      return []
    else:
      skip_params = []
      for key, _ in self.feature_encoder.state_dict().items():
        skip_params.append(f"feature_encoder.{key}")
      return skip_params

  def forward(self, x: torch.Tensor, masked = True) -> (torch.Tensor, torch.Tensor|None):
    assert self.config['K'] <= len(self.sequence_transducer)
    x = self._encode_features(x)
    mask = None
    if masked:
      mask = self._generate_batch_mask(x).unsqueeze(-1)
      x = x * (~mask)
    x = self._encode_sequence(x)
    return x, mask



