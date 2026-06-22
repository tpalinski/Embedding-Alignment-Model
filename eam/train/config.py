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
    lambda_var: float
    lambda_cov: float
    gamma: float

@dataclass
class SupervisedTrainConfig(TypedDict):
    epochs: int
    lr: float
    lr_warmup: float
    reg_term: float
    temperature_start: float
    temperature_end: float
    lambda_cov: float
    lambda_guided: float
