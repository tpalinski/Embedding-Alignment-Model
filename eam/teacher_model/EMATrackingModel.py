import torch
import copy
from torch import nn

from eam.unsupervised_model import Data2VecModel, Data2VecModelWithSharedExtractor
from .EMATrackingModelConfig import EMATrackingModelConfig

class EMATrackingModel[T: Data2VecModel|Data2VecModelWithSharedExtractor](nn.Module):

  def __init__(self, config: EMATrackingModelConfig[T]) -> None:
    super().__init__()
    self.model = copy.deepcopy(config['student_model'])
    self.config = config
    self.config['student_model'] = None
    self.step = 1

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    self.model.eval()
    with torch.no_grad():
      return self.model(X, masked=False)

  def update(self, student_model: T):
    with torch.no_grad():
      ema_factor = self.__get_ema_factor()
      ema_old = []
      ema_new = []
      model_params   = dict(self.model.named_parameters())
      student_params = dict(student_model.named_parameters())
      for name, s_param in student_params.items():
          m_param = model_params[name]
          if name in self.config['copy_params']:
              m_param.copy_(s_param)
          else:
              ema_old.append(m_param)
              ema_new.append(s_param)
      if ema_old:
          torch._foreach_mul_(ema_old, ema_factor)
          torch._foreach_add_(ema_old, ema_new, alpha=1.0 - ema_factor)
      self.step += 1

  def __get_ema_factor(self):
    if self.step >= self.config['annealing_warmup']:
      return self.config['annealing_target_value']
    else:
      progress = (self.config['annealing_target_value'] - self.config['annealing_init_value']) / self.step
      return self.config['annealing_init_value'] + progress


