#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/output/pdf"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/decision-tree-latex.XXXXXX")"
trap 'rm -rf "${BUILD_DIR}"' EXIT

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="${BUILD_DIR}" latex/report.tex
else
  xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="${BUILD_DIR}" latex/report.tex
  xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="${BUILD_DIR}" latex/report.tex
fi

cp "${BUILD_DIR}/report.pdf" "${OUTPUT_DIR}/report_decision_tree_latex.pdf"
echo "Создан файл: ${OUTPUT_DIR}/report_decision_tree_latex.pdf"
