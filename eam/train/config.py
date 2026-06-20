from dataclasses import dataclass
from typing import TypedDict


@dataclass
class UnsupervisedTrainConfig(TypedDict):
    epochs: int
    lr: float
    reg_term: float
    accum_steps: int
    warmup_steps: int
    distributed: bool
