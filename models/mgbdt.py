import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
 
    def _build(self) -> MGBDT:
        cfg = self.config
        mgbdt = MGBDT(
            loss=cfg.loss,
            target_lr=self.target_lr,
            epsilon=self.epsilon,
            verbose=self.verbose,
        )
 
        n_layers = len(self.layer_configs)
        for idx, (layer_type, model_type) in enumerate(self.layer_configs):
            is_last = idx == n_layers - 1
            in_size = cfg.input_size if idx == 0 else cfg.output_size
            out_size = cfg.output_size
 
            F = self._build_model(model_type, in_size, out_size)
 
            if layer_type == "bp_layer":
                mgbdt.add_layer(
                    "bp_layer",
                    F=F,
                    input_size=in_size,
                    output_size=out_size,
                    loss=cfg.loss,
                    loss_params=cfg.loss_params,
                )
            else:
                Finv = self._build_model(model_type, out_size, in_size)
                mgbdt.add_layer(
                    "tp_layer",
                    F=F,
                    Finv=Finv,
                    input_size=in_size,
                    output_size=out_size,
                )
 
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
                loss=cfg.loss,
                loss_params=cfg.loss_params,
                optimizer=cfg.optimizer,
                activation=cfg.activation,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'xgb' or 'linear'.")
 
    def init(self, X, n_rounds: int = 1):
        cfg = self.config
        self.model.init(
            X,
            n_rounds=n_rounds,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
        )
 
    @timer
    def fit(self, X, y, n_epochs: int = 10, eval_sets=None, **kwargs):
        self.model.fit(X, y, n_epochs=n_epochs, eval_sets=eval_sets, **kwargs)
 
    @timer
    def predict(self, X):
        return self.model.forward(X)
 
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