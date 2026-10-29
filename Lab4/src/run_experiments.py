from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from adult_tree import (
    DecisionTree,
    classification_metrics,
    load_adult,
    xy,
)
from ensembles import make_ensemble


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"accuracy": "#264653", "precision": "#e76f51",
          "recall": "#2a9d8f", "f1": "#e9c46a"}
WINDOWS_FONTS = Path(r"C:\Windows\Fonts")

TECHNIQUES = ("bagging", "random_forest", "boosting")
N_ESTIMATORS = [50, 60, 70, 80, 90, 100]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = WINDOWS_FONTS / ("arialbd.ttf" if bold else "arial.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Диаграмма: метрики vs число участников, с одиночным деревом как опорной линией
# --------------------------------------------------------------------------- #
def plot_metrics(rows: list[dict], single_tree: dict, output: Path) -> None:
    width, height = 1500, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 185, 140, 1420, 760

    # диапазон Y с запасом
    all_vals = []
    for row in rows:
        for m in COLORS:
            all_vals.append(row[m])
    for m in COLORS:
        all_vals.append(single_tree[m])
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.15 or 0.05
    lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)

    draw.text((width // 2, 42),
              "Качество классификации: ансамбли vs одиночное дерево",
              fill="#1d3557", font=font(30, True), anchor="ma")

    steps = 6
    for i in range(steps + 1):
        value = lo + (hi - lo) * i / steps
        y = bottom - (value - lo) / (hi - lo) * (bottom - top)
        draw.line((left, y, right, y), fill="#d9e2ec", width=2)
        draw.text((left - 18, y), f"{value:.3f}", fill="#334e68",
                  font=font(20), anchor="rm")
    draw.line((left, top, left, bottom), fill="#334e68", width=3)
    draw.line((left, bottom, right, bottom), fill="#334e68", width=3)

    xs_units = [row["n_estimators"] for row in rows]
    n_min, n_max = min(xs_units), max(xs_units)
    xs = [left + (u - n_min) / (n_max - n_min) * (right - left) for u in xs_units]
    for x, u in zip(xs, xs_units):
        draw.line((x, bottom, x, bottom + 10), fill="#334e68", width=3)
        draw.text((x, bottom + 24), str(u), fill="#334e68",
                  font=font(22), anchor="ma")

    # опорные линии одиночного дерева
    for metric, color in COLORS.items():
        y = bottom - (single_tree[metric] - lo) / (hi - lo) * (bottom - top)
        # пунктир
        x0 = left
        while x0 < right:
            x1 = min(x0 + 18, right)
            draw.line((x0, y, x1, y), fill=color, width=3)
            x0 += 30
        draw.text((right + 6, y), f"{single_tree[metric]:.3f}",
                  fill=color, font=font(18, True), anchor="lm")

    for metric, color in COLORS.items():
        points = [(x, bottom - (row[metric] - lo) / (hi - lo) * (bottom - top))
                  for x, row in zip(xs, rows)]
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9),
                         fill=color, outline="white", width=2)
            draw.text((x + 10, y - 14), f"{row[metric]:.3f}",
                      fill="#243b53", font=font(16))

    draw.text(((left + right) // 2, 858),
              "Количество участников ансамбля",
              fill="#334e68", font=font(24, True), anchor="mm")
    draw.text((92, (top + bottom) // 2), "Значение",
              fill="#334e68", font=font(24, True), anchor="mm")

    legend_x = 220
    for metric, color in COLORS.items():
        draw.line((legend_x, 110, legend_x + 45, 110), fill=color, width=6)
        draw.text((legend_x + 58, 110), metric, fill="#243b53",
                  font=font(21), anchor="lm")
        legend_x += 260
    draw.line((legend_x, 110, legend_x + 45, 110), fill="#7b8794", width=3)
    draw.text((legend_x + 58, 110),
              "одиночное дерево (пунктир)", fill="#243b53",
              font=font(21), anchor="lm")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


# --------------------------------------------------------------------------- #
# Основной сценарий
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ансамблевая классификация Adult (bagging / RF / boosting)"
    )
    parser.add_argument("--train", type=Path,
                        default=ROOT / "data" / "adult.data.txt")
    parser.add_argument("--test", type=Path,
                        default=ROOT / "data" / "adult.test.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results")
    parser.add_argument("--criterion", default="gini",
                        choices=("information_gain", "gain_ratio", "gini"))
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0,
                        help="learning rate для бустинга")
    parser.add_argument("--max-features", default="sqrt",
                        help="max_features для случайного леса (sqrt/log2/int)")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    official_train = load_adult(args.train)
    official_test = load_adult(args.test)
    x_train, y_train = xy(official_train)
    x_test, y_test = xy(official_test)

    # 1. Опорная точка: одиночное дерево (из lab3)
    print("Single tree (reference)...", flush=True)
    single = DecisionTree(criterion=args.criterion,
                          max_depth=args.max_depth).fit(x_train, y_train)
    single_metrics = classification_metrics(y_test, single.predict(x_test))
    (out / "single_tree_metrics.json").write_text(
        json.dumps(single_metrics, indent=2), encoding="utf-8"
    )

    # 2. Основной эксперимент: 3 техники × 6 размеров
    common_kwargs = dict(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=200,
        min_samples_leaf=80,
        random_state=42,
    )
    all_rows: list[dict] = []
    for technique in TECHNIQUES:
        for n in N_ESTIMATORS:
            print(f"Training {technique}, n_estimators={n}...", flush=True)
            kwargs = dict(common_kwargs)
            if technique == "boosting":
                kwargs["learning_rate"] = args.learning_rate
            if technique == "random_forest":
                kwargs["max_features"] = _parse_max_features(args.max_features)
            model = make_ensemble(technique, n_estimators=n, **kwargs)
            model.fit(x_train, y_train)
            metrics = classification_metrics(y_test, model.predict(x_test))
            all_rows.append({
                "technique": technique,
                "n_estimators": n,
                "train_size": len(official_train),
                "test_size": len(official_test),
                **metrics,
            })

    with (out / "ensemble_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # 3. Отдельная диаграмма на каждую технику + сводная
    for technique in TECHNIQUES:
        subset = [r for r in all_rows if r["technique"] == technique]
        plot_metrics(subset, single_metrics,
                     out / f"metrics_{technique}.png")

    # сводная: три техники на одном графике
    plot_all_techniques(all_rows, single_metrics, out / "metrics_all.png")

    summary = {
        "criterion": args.criterion,
        "single_tree": single_metrics,
        "ensembles": all_rows,
    }
    (out / "summary_ensemble.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _parse_max_features(value: str) -> str | int | float:
    v = value.strip().lower()
    if v in ("sqrt", "log2"):
        return v
    try:
        return int(v)
    except ValueError:
        return float(v)


def plot_all_techniques(rows: list[dict], single_tree: dict,
                        output: Path) -> None:
    """Сводный график: F1 и accuracy для трёх техник на одних осях."""
    width, height = 1500, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 185, 140, 1420, 760

    metric = "f1"
    vals = [r[metric] for r in rows] + [single_tree[metric]]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.25 or 0.05
    lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)

    draw.text((width // 2, 42),
              f"Сравнение техник ансамблей по {metric.upper()}",
              fill="#1d3557", font=font(30, True), anchor="ma")

    steps = 6
    for i in range(steps + 1):
        value = lo + (hi - lo) * i / steps
        y = bottom - (value - lo) / (hi - lo) * (bottom - top)
        draw.line((left, y, right, y), fill="#d9e2ec", width=2)
        draw.text((left - 18, y), f"{value:.3f}", fill="#334e68",
                  font=font(20), anchor="rm")
    draw.line((left, top, left, bottom), fill="#334e68", width=3)
    draw.line((left, bottom, right, bottom), fill="#334e68", width=3)

    n_min, n_max = min(N_ESTIMATORS), max(N_ESTIMATORS)
    xs = [left + (u - n_min) / (n_max - n_min) * (right - left)
          for u in N_ESTIMATORS]
    for x, u in zip(xs, N_ESTIMATORS):
        draw.line((x, bottom, x, bottom + 10), fill="#334e68", width=3)
        draw.text((x, bottom + 24), str(u), fill="#334e68",
                  font=font(22), anchor="ma")

    technique_colors = {"bagging": "#264653",
                        "random_forest": "#e76f51",
                        "boosting": "#2a9d8f"}
    for technique, color in technique_colors.items():
        subset = sorted([r for r in rows if r["technique"] == technique],
                        key=lambda r: r["n_estimators"])
        points = [(left + (row["n_estimators"] - n_min) /
                   (n_max - n_min) * (right - left),
                   bottom - (row[metric] - lo) / (hi - lo) * (bottom - top))
                  for row in subset]
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8),
                         fill=color, outline="white", width=2)

    # опорная линия одиночного дерева
    y0 = bottom - (single_tree[metric] - lo) / (hi - lo) * (bottom - top)
    x0 = left
    while x0 < right:
        x1 = min(x0 + 18, right)
        draw.line((x0, y0, x1, y0), fill="#7b8794", width=3)
        x0 += 30
    draw.text((right + 6, y0), f"single {single_tree[metric]:.3f}",
              fill="#7b8794", font=font(18, True), anchor="lm")

    draw.text(((left + right) // 2, 858),
              "Количество участников ансамбля",
              fill="#334e68", font=font(24, True), anchor="mm")
    draw.text((92, (top + bottom) // 2), metric.upper(),
              fill="#334e68", font=font(24, True), anchor="mm")

    legend_x = 260
    for technique, color in technique_colors.items():
        draw.line((legend_x, 110, legend_x + 45, 110), fill=color, width=6)
        draw.text((legend_x + 58, 110), technique, fill="#243b53",
                  font=font(21), anchor="lm")
        legend_x += 280
    draw.line((legend_x, 110, legend_x + 45, 110), fill="#7b8794", width=3)
    draw.text((legend_x + 58, 110), "single tree",
              fill="#243b53", font=font(21), anchor="lm")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()