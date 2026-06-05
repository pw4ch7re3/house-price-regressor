from dataclasses import dataclass, field
from typing import Literal

from sklearn.ensemble import RandomForestRegressor

from models import ModelConfig
from metrics.timer import timer


@dataclass
class RandomForestConfig(ModelConfig):
    n_estimators: int = field(default=100)
    max_depth: int | None = field(default=None)
    min_samples_split: int = field(default=2)
    min_samples_leaf: int = field(default=1)
    max_features: float | Literal["auto", "sqrt", "log2"] | None = field(default=None)


class RandomForest:
    def __init__(self, model_config: RandomForestConfig):
        self.forest = RandomForestRegressor(
            n_estimators=model_config.n_estimators,
            max_depth=model_config.max_depth,
            min_samples_split=model_config.min_samples_split,
            min_samples_leaf=model_config.min_samples_leaf,
            max_features=model_config.max_features,
        )

    @timer
    def fit(self, train_config):
        X_train = train_config.X
        y_train = train_config.y

        self.forest.fit(X_train, y_train)

        return self

    @timer
    def predict(self, X):
        return self.forest.predict(X)

    def score(self, X, y):
        return self.forest.score(X, y)

    def state_dict(self):
        return {"forest": self.forest}
