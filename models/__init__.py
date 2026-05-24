from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    model: str


@dataclass
class TrainConfig:
    X: pd.DataFrame | np.ndarray
    y: pd.Series | np.ndarray
    epochs: int
    lr: float
    batch_size: int | None = field(default=None)
    verbose: bool = field(default=False)