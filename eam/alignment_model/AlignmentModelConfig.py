from typing import TypedDict


class AlignmentModelConfig(TypedDict):
    device: str
    input_dim: int
    embedding_dim: int
    num_heads: int
    dropout: float
    use_training_cross_attention: bool
