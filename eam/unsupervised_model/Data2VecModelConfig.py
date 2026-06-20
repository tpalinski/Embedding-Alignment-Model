from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Data2VecModelConfig(TypedDict):
  K: int
  mask_p: float
  mask_steps: int
  output_dims: int

