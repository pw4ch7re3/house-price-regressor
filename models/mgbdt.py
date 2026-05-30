import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field

from models import ModelConfig
from metrics.timer import timer


@dataclass
class mGBDTConfig(ModelConfig):
    input_size: int
    output_size: int
    bias: bool = field(default=True)
    learning_rate: float = field(default=0.1)
    loss: str | None = field(default=None)
    loss_params: dict | None = field(default=None)
    optimizer: str = field(default="SGD")
    activation: str | None = field(default=None)
    max_depth: int = field(default=5)
    num_boost_round: int = field(default=1)
    force_no_parallel: bool = field(default=False)