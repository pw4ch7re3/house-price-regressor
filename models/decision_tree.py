from dataclasses import dataclass, field
from typing import Literal

from sklearn.tree import DecisionTreeRegressor

from models import ModelConfig


@dataclass
class DecisionTreeConfig(ModelConfig):
    max_depth: int | None = field(default=None)
    min_samples_split: int = field(default=2)
    min_samples_leaf: int = field(default=1)
    max_features: float | Literal["auto", "sqrt", "log2"] | None = field(default=None)


class DecisionTree:
    def __init__(self, model_config: DecisionTreeConfig):
        self.tree = DecisionTreeRegressor(
            max_depth=model_config.max_depth,
            min_samples_split=model_config.min_samples_split,
            min_samples_leaf=model_config.min_samples_leaf,
            max_features=model_config.max_features,
        )

    def fit(self, train_config):
        X_train = train_config.X
        y_train = train_config.y

        self.tree.fit(X_train, y_train)

        if train_config.verbose:
            score = self.tree.score(X_train, y_train)
            print(f"training R² score: {score:.4f}")

        return self

    def predict(self, X):
        return self.tree.predict(X)

    def score(self, X, y):
        return self.tree.score(X, y)

    def state_dict(self):
        return {"tree": self.tree}
