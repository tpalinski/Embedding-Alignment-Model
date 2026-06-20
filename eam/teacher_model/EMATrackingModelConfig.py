from typing import TypedDict
import torch
from torch import nn

from eam.audio_model import AudioModel
from eam.unsupervised_model import Data2VecModel


class EMATrackingModelConfig[T: Data2VecModel|AudioModel](TypedDict):
  student_model: T | None
  annealing_warmup: int
  annealing_init_value: float
  annealing_target_value: float
  copy_params: list[str]

def get_default_audio_teacher_config() -> EMATrackingModelConfig[AudioModel]:
    return EMATrackingModelConfig(
        student_model=None,
        annealing_warmup=7500,
        annealing_init_value=0.9990,
        annealing_target_value=0.9999,
        copy_params = []
    )
