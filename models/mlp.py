import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass, field

@dataclass
class MLPConfig(object):
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float = field(default=0.2)
    activation: nn.Module = field(default=nn.ReLU)
    
    use_batch_norm: bool = field(default=False)

class MLP(nn.Module):
    def __init__(self, model_config: MLPConfig):
        input_dim = model_config.input_dim
        hidden_dims = model_config.hidden_dims
        output_dim = model_config.output_dim
        dropout = model_config.dropout
        activation = model_config.activation
        use_batch_norm = model_config.use_batch_norm
        
        layers = []
        prev = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(activation())
            
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            prev = hidden_dim
            
        layers.append(nn.Linear(prev, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)