import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataclasses import dataclass, field

from models import ModelConfig
from metrics.timer import timer


@dataclass
class MLPConfig(ModelConfig):
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float = field(default=0.0)
    activation: type[nn.Module] = field(default=nn.ReLU)
    
    use_batch_norm: bool = field(default=True)


class MLP(nn.Module):
    def __init__(self, model_config: MLPConfig):
        super().__init__()

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

    @timer
    def fit(self, train_config):
        X_train = train_config.X
        y_train = train_config.y

        if hasattr(X_train, "values"):
            X = torch.as_tensor(X_train.values, dtype=torch.float32)
        else:
            X = torch.as_tensor(X_train, dtype=torch.float32)

        if hasattr(y_train, "values"):
            y = torch.as_tensor(y_train.values, dtype=torch.float32)
        else:
            y = torch.as_tensor(y_train, dtype=torch.float32)

        if y.ndim == 1:
            y = y.unsqueeze(1)

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=train_config.batch_size or len(X),
            shuffle=True,
        )

        optimizer = optim.Adam(self.parameters(), lr=train_config.lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(train_config.epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)

            if train_config.verbose:
                print(f"epoch {epoch + 1}/{train_config.epochs} loss {epoch_loss / len(X):.4f}")

        return self

    @timer
    def predict(self, X):
        if hasattr(X, "values"):
            X = torch.as_tensor(X.values, dtype=torch.float32)
        else:
            X = torch.as_tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self(X).squeeze(1).numpy()
