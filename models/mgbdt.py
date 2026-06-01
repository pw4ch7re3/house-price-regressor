import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from mgbdt import MGBDT
from mgbdt.model import MultiXGBModel
from mgbdt.model.linear_model import LinearModel

from models import ModelConfig
from metrics.timer import timer


@dataclass
class mGBDTConfig(ModelConfig):
    input_size: int
    output_size: int
    task: str = field(default="regression")
    bias: bool = field(default=True)
    learning_rate: float = field(default=0.1)
    loss: str | None = field(default=None)
    loss_params: dict | None = field(default=None)
    optimizer: str = field(default="SGD")
    activation: str | None = field(default=None)
    max_depth: int = field(default=5)
    num_boost_round: int = field(default=1)
    force_no_parallel: bool = field(default=False)
    
    
class MGBDTModel: 
    def __init__(
        self,
        config: mGBDTConfig,
        layer_configs: Optional[list] = None,
        target_lr: float = 0.1,
        epsilon: float = 0.3,
        verbose: bool = False,
    ):
        self.config = config
        self.target_lr = target_lr
        self.epsilon = epsilon
        self.verbose = verbose
 
        if layer_configs is None:
            layer_configs = [("tp_layer", "xgb")]
        self.layer_configs = layer_configs
 
        self.model = self._build()

    def _loss(self):
        if self.config.loss is not None:
            return self.config.loss
        if self.config.task == "regression":
            return "MSELoss"
        return None
 
    def _build(self) -> MGBDT:
        cfg = self.config
        mgbdt = MGBDT(
            loss=self._loss(),
            target_lr=self.target_lr,
            epsilon=self.epsilon,
            verbose=self.verbose,
        )
 
        n_layers = len(self.layer_configs)
        for idx, layer_config in enumerate(self.layer_configs):
            if len(layer_config) == 2:
                layer_type, model_type = layer_config
                out_size = cfg.output_size
            elif len(layer_config) == 3:
                layer_type, model_type, out_size = layer_config
            else:
                raise ValueError(
                    "layer_configs entries must be (layer_type, model_type) "
                    "or (layer_type, model_type, output_size)."
                )
            is_last = idx == n_layers - 1
            in_size = cfg.input_size if idx == 0 else prev_out_size
            if is_last and out_size != cfg.output_size:
                raise ValueError(
                    f"Last layer output_size ({out_size}) must match "
                    f"config.output_size ({cfg.output_size})."
                )
 
            F = self._build_model(model_type, in_size, out_size)
 
            if layer_type == "bp_layer":
                mgbdt.add_layer(
                    "bp_layer",
                    F=F,
                )
            else:
                Finv = self._build_model(model_type, out_size, in_size)
                mgbdt.add_layer(
                    "tp_layer",
                    F=F,
                    G=Finv,
                )
            prev_out_size = out_size
 
        return mgbdt
 
    def _build_model(self, model_type: str, input_size: int, output_size: int):
        cfg = self.config
        if model_type == "xgb":
            return MultiXGBModel(
                input_size=input_size,
                output_size=output_size,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                num_boost_round=cfg.num_boost_round,
                force_no_parallel=cfg.force_no_parallel,
            )
        elif model_type == "linear":
            return LinearModel(
                input_size=input_size,
                output_size=output_size,
                bias=cfg.bias,
                learning_rate=cfg.learning_rate,
                loss=self._loss(),
                loss_params=cfg.loss_params,
                optimizer=cfg.optimizer,
                activation=cfg.activation,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'xgb' or 'linear'.")

    def _as_array(self, data, dtype=None):
        if hasattr(data, "values"):
            data = data.values
        return np.asarray(data, dtype=dtype)

    def _prepare_X(self, X):
        X = self._as_array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _prepare_y(self, y):
        dtype = np.float64 if self.config.task == "regression" else None
        y = self._as_array(y, dtype=dtype)
        if y.ndim == 1 and (self.config.task == "regression" or self.config.output_size == 1):
            y = y.reshape(-1, 1)
        return y

    def _prepare_eval_sets(self, eval_sets):
        if eval_sets is None:
            return None
        return [(self._prepare_X(X), self._prepare_y(y)) for X, y in eval_sets]
 
    def init(self, X, n_rounds: int = 1):
        cfg = self.config
        self.model.init(
            self._prepare_X(X),
            n_rounds=n_rounds,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
        )
 
    @timer
    def fit(self, X, y, n_epochs: int = 10, eval_sets=None, **kwargs):
        X = self._prepare_X(X)
        y = self._prepare_y(y)
        eval_sets = self._prepare_eval_sets(eval_sets)
        self.model.fit(X, y, n_epochs=n_epochs, eval_sets=eval_sets, **kwargs)
        return self
 
    @timer
    def predict(self, X):
        pred = self.model.forward(self._prepare_X(X))
        if self.config.task == "regression" and pred.ndim == 2 and pred.shape[1] == 1:
            return pred[:, 0]
        return pred
 
    def __repr__(self):
        return f"MGBDTModel(config={self.config}, layer_configs={self.layer_configs})\n{self.model}"
 
 
if __name__ == "__main__":
    """ Usage Example """
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
 
    np.random.seed(42)
 
    X, y_raw = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5, random_state=42)
    y = np.eye(3)[y_raw]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    config = mGBDTConfig(
        input_size=10,
        output_size=3,
        learning_rate=0.1,
        max_depth=4,
        num_boost_round=2,
        loss="MSELoss",
    )
    wrapper = MGBDTModel(config, layer_configs=[("tp_layer", "xgb")], verbose=True)
    wrapper.init(X_train, n_rounds=1)
    wrapper.fit(X_train, y_train, n_epochs=5, eval_sets=[(X_test, y_test)])
    pred = wrapper.predict(X_test)
    print("pred.shape:", pred.shape)
 
    config2 = mGBDTConfig(
        input_size=10,
        output_size=3,
        learning_rate=0.05,
        max_depth=3,
        num_boost_round=1,
        loss="CrossEntropyLoss",
        optimizer="Adam",
    )
    wrapper2 = MGBDTModel(
        config2,
        layer_configs=[("tp_layer", "xgb"), ("bp_layer", "linear")],
        verbose=True,
    )
    wrapper2.init(X_train, n_rounds=1)
    wrapper2.fit(X_train, y_train, n_epochs=3)
    print(wrapper2)
