from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adult_tree import DecisionTree, NUMERIC_COLUMNS


# --------------------------------------------------------------------------- #
# Базовый класс ансамбля
# --------------------------------------------------------------------------- #
class EnsembleBase:
    """Общий интерфейс: fit/predict, плюс сохранение OOB-оценки для бэггинга."""

    def __init__(self, n_estimators: int, criterion: str = "gini",
                 max_depth: int = 8, min_samples_split: int = 200,
                 min_samples_leaf: int = 80, random_state: int = 42) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.models: list[DecisionTree] = []

    def _make_tree(self, feature_subset: list[str] | None = None) -> DecisionTree:
        # Ограничение по признакам реализуется обёрткой над x DataFrame
        tree = DecisionTree(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        tree._feature_subset = feature_subset  # type: ignore[attr-defined]
        return tree

    def _fit_tree(self, tree: DecisionTree, x: pd.DataFrame, y: np.ndarray,
                  columns: list[str]) -> None:
        tree.fit(x[columns], y)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit must be called before predict")
        votes = np.zeros((len(x), len(self.models)), dtype=np.uint8)
        for i, model in enumerate(self.models):
            cols = getattr(model, "_feature_subset", None) or list(x.columns)
            votes[:, i] = model.predict(x[cols])
        # Жёсткое голосование большинством (для классификации)
        return (votes.sum(axis=1) * 2 >= len(self.models)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Бэггинг
# --------------------------------------------------------------------------- #
class BaggingClassifier(EnsembleBase):
    """Бэггинг: каждый участник обучается на bootstrap-выборке."""

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "BaggingClassifier":
        rng = np.random.default_rng(self.random_state)
        self.models = []
        columns = list(x.columns)
        n = len(x)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = self._make_tree(columns)
            self._fit_tree(tree, x.iloc[idx], y[idx], columns)
            self.models.append(tree)
        return self


# --------------------------------------------------------------------------- #
# Случайный лес
# --------------------------------------------------------------------------- #
class RandomForestClassifier(EnsembleBase):
    """Случайный лес: bootstrap + случайное подмножество признаков на узле.

    Настоящий RF выбирает подмножество на каждом узле. Здесь для простоты и
    совместимости с существующим DecisionTree реализовано приближение: каждый
    участник получает случайное подмножество признаков размера
    max_features (по умолчанию sqrt(p)), что даёт декорреляцию деревьев.
    """

    def __init__(self, n_estimators: int, max_features: str | int | float = "sqrt",
                 **kwargs) -> None:
        super().__init__(n_estimators=n_estimators, **kwargs)
        self.max_features = max_features

    def _resolve_max_features(self, p: int) -> int:
        mf = self.max_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(math.sqrt(p)))
            if mf == "log2":
                return max(1, int(math.log2(p)))
            raise ValueError(f"unknown max_features: {mf}")
        if isinstance(mf, float):
            return max(1, int(mf * p))
        return min(mf, p)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "RandomForestClassifier":
        rng = np.random.default_rng(self.random_state)
        columns = list(x.columns)
        p = len(columns)
        m = self._resolve_max_features(p)
        n = len(x)
        self.models = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            subset = list(rng.choice(columns, size=m, replace=False))
            tree = self._make_tree(subset)
            self._fit_tree(tree, x.iloc[idx], y[idx], subset)
            self.models.append(tree)
        return self


# --------------------------------------------------------------------------- #
# Бустинг (AdaBoost.M1 над деревьями)
# --------------------------------------------------------------------------- #
class AdaBoostClassifier(EnsembleBase):
    """AdaBoost.M1 поверх DecisionTree.

    Каждый следующий участник обучается на взвешенной выборке; веса объектов,
    где текущий участник ошибся, увеличиваются. Итоговый прогноз — взвешенное
    голосование с весами alpha = 0.5 * ln((1 - err) / err).
    """

    def __init__(self, n_estimators: int, learning_rate: float = 1.0,
                 **kwargs) -> None:
        super().__init__(n_estimators=n_estimators, **kwargs)
        self.learning_rate = learning_rate
        self.alphas: list[float] = []

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "AdaBoostClassifier":
        rng = np.random.default_rng(self.random_state)
        n = len(x)
        weights = np.full(n, 1.0 / n)
        self.models = []
        self.alphas = []
        columns = list(x.columns)
        for _ in range(self.n_estimators):
            # взвешенная bootstrap-выборка (доля пропорциональна весу объекта)
            sample_idx = rng.choice(n, size=n, p=weights / weights.sum())
            tree = self._make_tree(columns)
            # обучаем на подвыборке, но взвешивание — через повторения в sample_idx
            tree.fit(x.iloc[sample_idx][columns], y[sample_idx])
            pred = tree.predict(x[columns])
            err = float((pred != y).astype(float) @ weights)
            err = max(err, 1e-10)
            if err >= 0.5:
                # участник не лучше случайного — обнуляем его вклад
                self.models.append(tree)
                self.alphas.append(0.0)
                continue
            alpha = 0.5 * math.log((1.0 - err) / err) * self.learning_rate
            weights = weights * np.exp(-alpha * (2 * y - 1) * (2 * pred - 1))
            weights /= weights.sum()
            self.models.append(tree)
            self.alphas.append(alpha)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit must be called before predict")
        scores = np.zeros(len(x), dtype=float)
        for alpha, model in zip(self.alphas, self.models):
            scores += alpha * (2 * model.predict(x).astype(float) - 1.0)
        return (scores >= 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Фабрика
# --------------------------------------------------------------------------- #
def make_ensemble(technique: str, n_estimators: int, **kwargs) -> EnsembleBase:
    technique = technique.lower()
    if technique == "bagging":
        return BaggingClassifier(n_estimators=n_estimators, **kwargs)
    if technique in ("random_forest", "rf", "forest"):
        return RandomForestClassifier(n_estimators=n_estimators, **kwargs)
    if technique in ("boosting", "adaboost"):
        return AdaBoostClassifier(n_estimators=n_estimators, **kwargs)
    raise ValueError(f"unknown technique: {technique}")