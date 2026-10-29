# Классификация с помощью дерева решений

Проект классифицирует записи UCI Adult по годовому доходу (`<=50K` или `>50K`) с помощью собственного бинарного дерева решений. Поддерживаются критерии `information_gain`, `gain_ratio` и `gini`, изменение доли обучающей выборки и расчёт accuracy, precision, recall и F1.

## Установка зависимостей

```bash
python3 -m pip install -r requirements.txt
```

Для визуализации деревьев требуется Graphviz (команда `dot`). В macOS: `brew install graphviz`.

## Запуск программы

```bash
python3 src/adult_tree.py data/adult.data.txt \
  --test data/adult.test.txt \
  --criterion information_gain \
  --output results/single_run.json
```

Без отдельного тестового файла можно указать долю обучения:

```bash
python3 src/adult_tree.py data/adult.data.txt \
  --criterion gain_ratio \
  --train-ratio 0.8 \
  --output results/single_run.json
```

## Эксперименты и отчёт

```bash
python3 src/run_experiments.py
python3 src/build_report.py
```

Полный запуск одной командой:

```bash
./scripts/run_all.sh
```

Альтернативная сборка LaTeX-версии отчёта:

```bash
./scripts/build_latex.sh
```

Готовый отчёт: `output/pdf/report_decision_tree.pdf`.

## Структура

- `src/adult_tree.py` — реализация дерева и интерфейс командной строки;
- `src/run_experiments.py` — эксперименты и визуализация;
- `src/build_report.py` — сборка PDF-отчёта через ReportLab;
- `latex/report.tex` — альтернативный LaTeX-исходник отчёта;
- `scripts/` — сценарии автоматической сборки;
- `data/` — исходные файлы набора Adult;
- `results/` — CSV, JSON и диаграммы;
- `output/pdf/` — готовый PDF-отчёт.
