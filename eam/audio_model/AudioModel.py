from typing import override
import torch
from torch import nn

from .AudioModelConfig import AudioModelConfig
from eam.unsupervised_model import Data2VecModelWithSharedExtractor

class AudioModel(Data2VecModelWithSharedExtractor):
  def __init__(self, config: AudioModelConfig) -> None:
    super().__init__(config)
    self.sequence_transducer = nn.ModuleList([
      nn.TransformerEncoderLayer(
          d_model=config["feature_dims"],
          nhead=config["num_heads"],
          dim_feedforward=config["dim_ffd"],
          dropout=config["dropout"],
          batch_first=True
      )
      for _ in range(config["num_layers"])])

  @override
  def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
    K = self.config['K']
    layer_count = len(self.sequence_transducer)
    hidden_states = []
    # Run layers before top-K
    for i in range(layer_count - K):
        x = self.sequence_transducer[i](x)
    # Collect top-K layers
    for i in range(K):
        x = self.sequence_transducer[layer_count - K + i](x)
        hidden_states.append(x)
    # Stack -> [K, B, T, C]
    return torch.stack(hidden_states, dim=0)
