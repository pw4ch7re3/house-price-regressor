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
    # Early stopping (MLP). patience=0 disables it and trains the full budget.
    # When enabled, val_split of the training data is held out to monitor loss.
    patience: int = field(default=0)
    val_split: float = field(default=0.1)
    min_delta: float = field(default=0.0)