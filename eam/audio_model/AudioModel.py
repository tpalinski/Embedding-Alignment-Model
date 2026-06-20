import torch
from torch import nn
from transformers import WhisperModel

from eam.unsupervised_model import Data2VecModelWithSharedExtractor
from .WhisperEncoderConfig import WhisperEncoderConfig, get_default_config

class AudioModel(Data2VecModelWithSharedExtractor):
  def __init__(self, config: WhisperEncoderConfig | None) -> None:
    if config == None:
        config = get_default_config()
    super().__init__(config)
    model = WhisperModel.from_pretrained(config['whisper_variant'])
    d_extractor = model.config.d_model
    self.feature_encoder = model.get_encoder()
    self.sequence_transducer = nn.ModuleList([
      nn.TransformerEncoderLayer(
          d_model=d_extractor,
          nhead=config["num_heads"],
          dim_feedforward=config["dim_ffd"],
          dropout=config["dropout"],
          batch_first=True
      )
      for _ in range(num_layers)])

  @override
  def _encode_features(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = self.feature_encoder(x, attention_mask)
    return x

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
