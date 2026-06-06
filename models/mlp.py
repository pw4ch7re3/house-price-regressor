import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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

    @property
    def device(self):
        # The device the parameters live on; falls back to CPU before any
        # explicit .to() call has been made.
        return next(self.parameters()).device

    @timer
    def fit(self, train_config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

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

        dataset = TensorDataset(X, y)

        # Hold out a validation split to drive early stopping. Disabled when
        # patience == 0 (or the split would be empty), in which case we train
        # the full epoch budget and monitor the training loss instead.
        patience = train_config.patience
        n_val = int(len(dataset) * train_config.val_split)
        if patience > 0 and n_val > 0:
            train_ds, val_ds = random_split(
                dataset,
                [len(dataset) - n_val, n_val],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_ds, val_ds = dataset, None

        batch_size = train_config.batch_size or len(train_ds)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None
        )

        optimizer = optim.Adam(self.parameters(), lr=train_config.lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(train_config.epochs):
            self.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(self(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)

            train_loss = epoch_loss / len(train_ds)
            if val_loader is not None:
                monitor_loss = self._eval_loss(val_loader, criterion, device)
            else:
                monitor_loss = train_loss

            if train_config.verbose:
                tag = "val" if val_loader is not None else "loss"
                print(
                    f"epoch {epoch + 1}/{train_config.epochs} "
                    f"loss {train_loss:.4f} {tag} {monitor_loss:.4f}"
                )

            # Early stopping: keep the best weights seen and bail out after
            # `patience` epochs without a meaningful improvement.
            if patience > 0:
                if monitor_loss < best_loss - train_config.min_delta:
                    best_loss = monitor_loss
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if train_config.verbose:
                            print(f"early stopping at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(device)

        return self

    def _eval_loss(self, loader, criterion, device):
        self.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                total += criterion(self(x_batch), y_batch).item() * x_batch.size(0)
                count += x_batch.size(0)
        return total / count

    @timer
    def predict(self, X):
        if hasattr(X, "values"):
            X = torch.as_tensor(X.values, dtype=torch.float32)
        else:
            X = torch.as_tensor(X, dtype=torch.float32)
        device = self.device
        X = X.to(device)
        self.eval()
        with torch.no_grad():
            return self(X).squeeze(1).cpu().numpy()
