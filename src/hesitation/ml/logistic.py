from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class StandardScaler:
    means: list[float]
    stds: list[float]

    @classmethod
    def fit(cls, features: list[list[float]]) -> "StandardScaler":
        n_features = len(features[0]) if features else 0
        means: list[float] = []
        stds: list[float] = []
        for i in range(n_features):
            values = [row[i] for row in features]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
            std = math.sqrt(var) if var > 1e-12 else 1.0
            means.append(mean)
            stds.append(std)
        return cls(means=means, stds=stds)

    def transform(self, features: list[list[float]]) -> list[list[float]]:
        transformed: list[list[float]] = []
        for row in features:
            transformed.append([(v - m) / s for v, m, s in zip(row, self.means, self.stds)])
        return transformed


class BinaryLogisticRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.05, l2: float = 1e-4) -> None:
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.l2 = l2

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, x: list[list[float]], y: list[int], epochs: int = 80) -> None:
        for _ in range(epochs):
            for row, target in zip(x, y):
                logit = sum(w * v for w, v in zip(self.weights, row)) + self.bias
                pred = self._sigmoid(logit)
                err = pred - float(target)
                for i in range(len(self.weights)):
                    grad = err * row[i] + self.l2 * self.weights[i]
                    self.weights[i] -= self.learning_rate * grad
                self.bias -= self.learning_rate * err

    def predict_proba(self, x: list[list[float]]) -> list[float]:
        out: list[float] = []
        for row in x:
            logit = sum(w * v for w, v in zip(self.weights, row)) + self.bias
            out.append(self._sigmoid(logit))
        return out


@dataclass(slots=True)
class OVRLogisticModel:
    classes: list[str]
    models: dict[str, BinaryLogisticRegression]

    def predict_proba(self, x: list[list[float]]) -> list[dict[str, float]]:
        raw: dict[str, list[float]] = {c: m.predict_proba(x) for c, m in self.models.items()}
        results: list[dict[str, float]] = []
        for idx in range(len(x)):
            denom = sum(raw[c][idx] for c in self.classes) + 1e-9
            probs = {c: raw[c][idx] / denom for c in self.classes}
            results.append(probs)
        return results

    def predict(self, x: list[list[float]]) -> list[str]:
        probs = self.predict_proba(x)
        return [max(p.items(), key=lambda kv: kv[1])[0] for p in probs]
