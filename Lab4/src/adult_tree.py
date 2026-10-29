from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]
NUMERIC_COLUMNS = {
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week",
}
CRITERIA = ("information_gain", "gain_ratio", "gini")


def load_adult(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        names=COLUMNS,
        skipinitialspace=True,
        comment="|",
        na_filter=False,
    )
    frame = frame[frame["income"].astype(str).str.len() > 0].copy()
    for column in frame.columns:
        if column not in NUMERIC_COLUMNS:
            frame[column] = frame[column].astype(str).str.strip()
    frame["income"] = frame["income"].str.rstrip(".")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame[column].median())
    return frame.reset_index(drop=True)


def _entropy(positive: int, total: int) -> float:
    if total == 0 or positive == 0 or positive == total:
        return 0.0
    p = positive / total
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _gini(positive: int, total: int) -> float:
    if total == 0:
        return 0.0
    p = positive / total
    return 2.0 * p * (1.0 - p)


def _split_score(y: np.ndarray, left: np.ndarray, criterion: str) -> float:
    n = y.size
    nl = int(left.sum())
    nr = n - nl
    if nl == 0 or nr == 0:
        return -np.inf
    pos = int(y.sum())
    pos_l = int(y[left].sum())
    pos_r = pos - pos_l
    impurity = _gini if criterion == "gini" else _entropy
    parent = impurity(pos, n)
    children = (nl / n) * impurity(pos_l, nl) + (nr / n) * impurity(pos_r, nr)
    gain = parent - children
    if criterion != "gain_ratio":
        return gain
    split_info = _entropy(nl, n)
    return gain / split_info if split_info > 0 else -np.inf


@dataclass
class Node:
    prediction: int
    samples: int
    positive: int
    depth: int
    feature: str | None = None
    threshold: float | None = None
    category: str | None = None
    score: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 8,
        min_samples_split: int = 200,
        min_samples_leaf: int = 80,
        max_thresholds: int = 32,
        min_gain: float = 1e-7,
    ) -> None:
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.min_gain = min_gain
        self.root: Node | None = None
        self.feature_names: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> "DecisionTree":
        self.feature_names = list(x.columns)
        arrays = {c: x[c].to_numpy() for c in x.columns}
        labels = np.asarray(y, dtype=np.uint8)
        self.root = self._grow(arrays, labels, np.arange(len(labels)), 0)
        return self

    def _grow(self, x: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray, depth: int) -> Node:
        current_y = y[idx]
        positive = int(current_y.sum())
        node = Node(int(positive * 2 >= len(idx)), len(idx), positive, depth)
        if (
            depth >= self.max_depth
            or len(idx) < self.min_samples_split
            or positive == 0
            or positive == len(idx)
        ):
            return node

        best_score = -np.inf
        best: tuple[str, float | None, str | None, np.ndarray] | None = None
        for feature in self.feature_names:
            values = x[feature][idx]
            if feature in NUMERIC_COLUMNS:
                unique = np.unique(values.astype(float))
                if unique.size < 2:
                    continue
                if unique.size - 1 > self.max_thresholds:
                    quantiles = np.linspace(0.03, 0.97, self.max_thresholds)
                    candidates = np.unique(np.quantile(values.astype(float), quantiles))
                else:
                    candidates = (unique[:-1] + unique[1:]) / 2.0
                for threshold in candidates:
                    mask = values.astype(float) <= threshold
                    nl = int(mask.sum())
                    if nl < self.min_samples_leaf or len(idx) - nl < self.min_samples_leaf:
                        continue
                    score = _split_score(current_y, mask, self.criterion)
                    if score > best_score:
                        best_score = score
                        best = (feature, float(threshold), None, mask)
            else:
                categories, counts = np.unique(values.astype(str), return_counts=True)
                for category, count in zip(categories, counts):
                    if count < self.min_samples_leaf or len(idx) - count < self.min_samples_leaf:
                        continue
                    mask = values.astype(str) == category
                    score = _split_score(current_y, mask, self.criterion)
                    if score > best_score:
                        best_score = score
                        best = (feature, None, str(category), mask)

        if best is None or best_score < self.min_gain:
            return node
        feature, threshold, category, mask = best
        node.feature = feature
        node.threshold = threshold
        node.category = category
        node.score = float(best_score)
        node.left = self._grow(x, y, idx[mask], depth + 1)
        node.right = self._grow(x, y, idx[~mask], depth + 1)
        return node

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("fit must be called before predict")
        result = np.empty(len(x), dtype=np.uint8)
        for row_number, (_, row) in enumerate(x.iterrows()):
            node = self.root
            while not node.is_leaf:
                if node.threshold is not None:
                    go_left = float(row[node.feature]) <= node.threshold
                else:
                    go_left = str(row[node.feature]) == node.category
                node = node.left if go_left else node.right
            result[row_number] = node.prediction
        return result

    def to_dict(self) -> dict:
        def convert(node: Node) -> dict:
            data = {
                "prediction": ">50K" if node.prediction else "<=50K",
                "samples": node.samples,
                "positive": node.positive,
                "depth": node.depth,
            }
            if not node.is_leaf:
                data.update({
                    "feature": node.feature,
                    "threshold": node.threshold,
                    "category": node.category,
                    "score": node.score,
                    "left": convert(node.left),
                    "right": convert(node.right),
                })
            return data
        if self.root is None:
            raise RuntimeError("fit must be called first")
        return convert(self.root)

    def export_dot(self, path: str | Path, display_depth: int = 4) -> None:
        if self.root is None:
            raise RuntimeError("fit must be called first")
        lines = [
            "digraph Tree {", "rankdir=TB;", "node [shape=box, style=\"rounded,filled\", fontname=\"Arial\", fontsize=10];",
            "edge [fontname=\"Arial\", fontsize=9];",
        ]
        counter = 0

        def add(node: Node) -> int:
            nonlocal counter
            node_id = counter
            counter += 1
            probability = node.positive / node.samples
            fill = "#f7b267" if node.prediction else "#74a9cf"
            if node.is_leaf or node.depth >= display_depth:
                condition = "leaf" if node.is_leaf else "subtree hidden"
            elif node.threshold is not None:
                condition = f"{node.feature} <= {node.threshold:.3g}"
            else:
                safe_category = node.category.replace('"', "'")
                condition = f"{node.feature} = {safe_category}"
            label = f"{condition}\\nsamples = {node.samples}\\nP(>50K) = {probability:.3f}\\nclass = {'>50K' if node.prediction else '<=50K'}"
            lines.append(f'{node_id} [label="{label}", fillcolor="{fill}"];')
            if not node.is_leaf and node.depth < display_depth:
                left_id = add(node.left)
                right_id = add(node.right)
                lines.append(f'{node_id} -> {left_id} [label="yes"];')
                lines.append(f'{node_id} -> {right_id} [label="no"];')
            return node_id

        add(self.root)
        lines.append("}")
        Path(path).write_text("\n".join(lines), encoding="utf-8")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def stratified_split(frame: pd.DataFrame, train_ratio: float, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, group in frame.groupby("income"):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        cut = int(len(indices) * train_ratio)
        train_indices.extend(indices[:cut])
        test_indices.extend(indices[cut:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return frame.loc[train_indices].reset_index(drop=True), frame.loc[test_indices].reset_index(drop=True)


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return frame.drop(columns="income"), (frame["income"] == ">50K").to_numpy(dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify UCI Adult income with a decision tree")
    parser.add_argument("dataset", type=Path, help="CSV file: adult.data or adult.test")
    parser.add_argument("--test", type=Path, help="Optional separate test CSV")
    parser.add_argument("--criterion", choices=CRITERIA, default="gini")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    data = load_adult(args.dataset)
    if args.test:
        train, test = data, load_adult(args.test)
    else:
        if not 0.0 < args.train_ratio < 1.0:
            parser.error("--train-ratio must be between 0 and 1")
        train, test = stratified_split(data, args.train_ratio)
    x_train, y_train = xy(train)
    x_test, y_test = xy(test)
    tree = DecisionTree(criterion=args.criterion, max_depth=args.max_depth).fit(x_train, y_train)
    metrics = classification_metrics(y_test, tree.predict(x_test))
    result = {"criterion": args.criterion, "train_size": len(train), "test_size": len(test), **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
