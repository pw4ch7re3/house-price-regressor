from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ModelConfig:
    model: str


@dataclass
class TrainConfig:
    X: pd.DataFrame
    y: pd.Series
    epochs: int
    lr: float
    batch_size: int | None = field(default=None)
    verbose: bool = field(default=False)